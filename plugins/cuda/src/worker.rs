use crate::{Error, NonceGenEnum};
use cust::context::CurrentContext;
use cust::device::DeviceAttribute;
use cust::function::Function;
use cust::memory::DeviceCopy;
use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::*;
use karlsen_miner::xoshiro256starstar::Xoshiro256StarStar;
use karlsen_miner::Worker;
use log::{error, info};
use memmap::MmapOptions;
use rand::{Fill, RngCore};
use std::ffi::CString;
use std::fs::OpenOptions;
use std::ops::BitXor;
use std::path::Path;
use std::sync::{Arc, Weak};
use tiny_keccak::Hasher;

static BPS: f32 = 0.5;

static PTX_86: &str = include_str!("../resources/karlsen-cuda-sm86.ptx");
static PTX_75: &str = include_str!("../resources/karlsen-cuda-sm75.ptx");
static PTX_61: &str = include_str!("../resources/karlsen-cuda-sm61.ptx");
static PTX_30: &str = include_str!("../resources/karlsen-cuda-sm30.ptx");
static PTX_20: &str = include_str!("../resources/karlsen-cuda-sm20.ptx");

pub struct Kernel<'kernel> {
    func: Arc<Function<'kernel>>,
    block_size: u32,
    grid_size: u32,
}

impl<'kernel> Kernel<'kernel> {
    pub fn new(module: Weak<Module>, name: &'kernel str) -> Result<Kernel<'kernel>, Error> {
        let func = Arc::new(unsafe {
            module.as_ptr().as_ref().unwrap().get_function(name).inspect_err(|&e| {
                error!("Error loading function: {}", e);
            })?
        });
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

        let device = CurrentContext::get_device()?;
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let grid_size = sm_count * func.max_active_blocks_per_multiprocessor(block_size.into(), 0)?;

        Ok(Self { func, block_size, grid_size })
    }

    pub fn get_workload(&self) -> u32 {
        //self.block_size * self.grid_size
        //we force workload to 1 for the moment
        1
    }

    pub fn set_workload(&mut self, workload: u32) {
        self.grid_size = workload.div_ceil(self.block_size)
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)] // Allow dead code for unused unions
pub union hash256 {
    pub word64s: [u64; 4usize],
    pub word32s: [u32; 8usize],
    pub bytes: [u8; 32usize],
    pub str_: [::std::os::raw::c_char; 32usize],
}

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)] // Allow dead code for unused unions
pub union hash512 {
    pub word64s: [u64; 8usize],
    pub word32s: [u32; 16usize],
    pub bytes: [u8; 64usize],
    pub str_: [::std::os::raw::c_char; 64usize],
}

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)] // Allow dead code for unused unions
pub union hash1024 {
    pub hash512s: [hash512; 2usize],
    pub word64s: [u64; 16usize],
    pub word32s: [u32; 32usize],
    pub bytes: [u8; 128usize],
    pub str_: [::std::os::raw::c_char; 128usize],
}

const SIZE_U32: usize = std::mem::size_of::<u32>();
#[allow(dead_code)]
const SIZE_U64: usize = std::mem::size_of::<u64>();

pub trait HashData {
    fn new() -> Self;
    fn as_bytes(&self) -> &[u8];
    fn as_bytes_mut(&mut self) -> &mut [u8];

    fn get_as_u32(&self, index: usize) -> u32 {
        u32::from_le_bytes(self.as_bytes()[index * SIZE_U32..index * SIZE_U32 + SIZE_U32].try_into().unwrap())
    }

    fn set_as_u32(&mut self, index: usize, value: u32) {
        self.as_bytes_mut()[index * SIZE_U32..index * SIZE_U32 + SIZE_U32].copy_from_slice(&value.to_le_bytes())
    }

    #[allow(dead_code)]
    fn get_as_u64(&self, index: usize) -> u64 {
        u64::from_le_bytes(self.as_bytes()[index * SIZE_U64..index * SIZE_U64 + SIZE_U64].try_into().unwrap())
    }

    #[allow(dead_code)]
    fn set_as_u64(&mut self, index: usize, value: u64) {
        self.as_bytes_mut()[index * SIZE_U64..index * SIZE_U64 + SIZE_U64].copy_from_slice(&value.to_le_bytes())
    }
}

#[derive(Debug)]
pub struct Hash256([u8; 32]);

impl HashData for Hash256 {
    fn new() -> Self {
        Self([0; 32])
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, DeviceCopy)]
pub struct Hash512([u8; 64]);

impl HashData for Hash512 {
    fn new() -> Self {
        Self([0; 64])
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl BitXor<&Hash512> for &Hash512 {
    type Output = Hash512;

    fn bitxor(self, rhs: &Hash512) -> Self::Output {
        let mut hash = Hash512::new();

        for i in 0..64 {
            hash.0[i] = self.0[i] ^ rhs.0[i]
        }

        hash
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, DeviceCopy)]
pub struct Hash1024([u8; 128]);

impl HashData for Hash1024 {
    fn new() -> Self {
        Self([0; 128])
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl Hash1024 {
    fn from_512s(first: &Hash512, second: &Hash512) -> Self {
        let mut hash = Self::new();
        let (first_half, second_half) = hash.0.split_at_mut(first.0.len());
        first_half.copy_from_slice(&first.0);
        second_half.copy_from_slice(&second.0);

        hash
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut array = [0u8; 128];
        array.copy_from_slice(bytes); // Ensure this does not panic by validating input length.
        Hash1024(array)
    }
}

const FNV_PRIME: u32 = 0x01000193;
const FULL_DATASET_ITEM_PARENTS: u32 = 512;
const LIGHT_CACHE_ROUNDS: i32 = 3;

const LIGHT_CACHE_NUM_ITEMS: u32 = 1179641;
const FULL_DATASET_NUM_ITEMS: u32 = 37748717;
const SEED: Hash256 = Hash256([
    0xeb, 0x01, 0x63, 0xae, 0xf2, 0xab, 0x1c, 0x5a, 0x66, 0x31, 0x0c, 0x1c, 0x14, 0xd6, 0x0f, 0x42, 0x55, 0xa9, 0xb3,
    0x9b, 0x0e, 0xdf, 0x26, 0x53, 0x98, 0x44, 0xf1, 0x17, 0xad, 0x67, 0x21, 0x19,
]);

pub struct CudaGPUWorker {
    // NOTE: The order is important! context must be closed last
    heavy_hash_kernel: Kernel<'static>,
    stream: Stream,
    start_event: Event,
    stop_event: Event,
    _module: Arc<Module>,

    rand_state: DeviceBuffer<u64>,
    final_nonce_buff: DeviceBuffer<u64>,

    cache2: DeviceBuffer<Hash512>,
    dataset2: DeviceBuffer<Hash1024>,
    //cache2_ptr: DevicePointer<Hash512>,
    //dataset2_ptr: DevicePointer<Hash1024>,
    device_id: u32,
    pub workload: usize,
    _context: Context,

    random: NonceGenEnum,
    //pub full_dataset: *mut Hash1024,
    //pub light_cache: *mut Hash512,
}

impl Worker for CudaGPUWorker {
    fn id(&self) -> String {
        let device = CurrentContext::get_device().unwrap();
        format!("#{} ({})", self.device_id, device.name().unwrap())
    }

    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]) {
        //info!("load_block_constants: debug1 ");
        let u8matrix: Arc<[[u8; 64]; 64]> = Arc::new(matrix.map(|row| row.map(|v| v as u8)));
        //info!("load_block_constants: debug2 ");
        let mut hash_header_gpu = self._module.get_global::<[u8; 72]>(&CString::new("hash_header").unwrap()).unwrap();
        //info!("load_block_constants: debug3 ");
        hash_header_gpu.copy_from(hash_header).map_err(|e| e.to_string()).unwrap();
        //info!("load_block_constants: debug4 ");

        let mut matrix_gpu = self._module.get_global::<[[u8; 64]; 64]>(&CString::new("matrix").unwrap()).unwrap();
        //info!("load_block_constants: debug5 ");
        matrix_gpu.copy_from(&u8matrix).map_err(|e| e.to_string()).unwrap();
        //info!("load_block_constants: debug6 ");

        let mut target_gpu = self._module.get_global::<[u64; 4]>(&CString::new("target").unwrap()).unwrap();
        //info!("load_block_constants: debug7 ");
        target_gpu.copy_from(target).map_err(|e| e.to_string()).unwrap();
        //info!("load_block_constants: debug8 ");

        //let mut data = DeviceBuffer::from_slice(&vec![hash1024 { bytes: [0; 128] }; FULL_DATASET_NUM_ITEMS]);

        //let u8cache: Arc<[u8; 10]> = Arc::new([0; 10]);
        //let mut data = DeviceBuffer::from_slice(&vec![hash512 { bytes: [0; 64] }; LIGHT_CACHE_NUM_ITEMS]);
        //self.cache = DeviceBuffer::from_slice(&vec![hash512 { bytes: [0; 64] }; LIGHT_CACHE_NUM_ITEMS]);
        /*
        info!("load_block_constants: debug8.1 ");
        let mut cache_gpu = self._module.get_global::<[DeviceBuffer<hash512>; LIGHT_CACHE_NUM_ITEMS]>(&CString::new("cache_test").unwrap()).unwrap();
        info!("load_block_constants: debug9 ");
        cache_gpu.copy_from(&data).map_err(|e| e.to_string()).unwrap();
        info!("load_block_constants: debug10 ");
        */
    }

    #[inline(always)]
    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64) {
        //info!("calculate_hash: debug1 ");
        let func = &self.heavy_hash_kernel.func;
        let stream = &self.stream;
        let random: u8 = match self.random {
            NonceGenEnum::Lean => {
                self.rand_state.copy_from(&[rand::thread_rng().next_u64()]).unwrap();
                0
            }
            NonceGenEnum::Xoshiro => 1,
        };

        //self.light_cache = vec![Hash512::new(); LIGHT_CACHE_NUM_ITEMS as usize].into_boxed_slice();
        //self.full_dataset = vec![Hash1024::new(); FULL_DATASET_NUM_ITEMS as usize].into_boxed_slice();

        //info!("calculate_hash: debug2 ");
        self.start_event.record(stream).unwrap();
        //info!("calculate_hash: debug3 cache size : {}", self.cache2.len());
        //info!("calculate_hash: debug3 dataset size : {}", self.dataset2.len());

        //info!("calculate_hash: debug3 dataset[10] : {:?}", self.dataset.index(10));
        unsafe {
            launch!(
                func<<<
                    self.heavy_hash_kernel.grid_size, self.heavy_hash_kernel.block_size,
                    0, stream
                >>>(
                    nonce_mask,
                    nonce_fixed,
                    self.workload,
                    random,
                    self.rand_state.as_device_ptr(),
                    self.final_nonce_buff.as_device_ptr(),
                    self.dataset2.as_device_ptr(),
                    self.cache2.as_device_ptr(),
                    //self.cache2_ptr.as_raw(),
                    //self.dataset2_ptr.as_raw(),
                )
            )
            .unwrap(); // We see errors in sync
        }
        //info!("calculate_hash: debug4 ");
        self.stop_event.record(stream).unwrap();
        //info!("calculate_hash: debug5 ");
    }

    #[inline(always)]
    fn sync(&self) -> Result<(), Error> {
        //self.stream.synchronize()?;
        self.stop_event.synchronize()?;
        if self.stop_event.elapsed_time_f32(&self.start_event)? > 1000. / BPS {
            return Err("Cuda takes longer then block rate. Please reduce your workload.".into());
        }
        Ok(())
    }

    fn get_workload(&self) -> usize {
        self.workload
    }

    #[inline(always)]
    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error> {
        self.final_nonce_buff.copy_to(nonces)?;
        Ok(())
    }
}

pub fn keccak_in_place(data: &mut [u8]) {
    let mut hasher = tiny_keccak::Keccak::v512();
    hasher.update(data);
    hasher.finalize(data);
}

pub fn keccak(out: &mut [u8], data: &[u8]) {
    let mut hasher = tiny_keccak::Keccak::v512();
    hasher.update(data);
    hasher.finalize(out);
}

fn build_light_cache(cache: &mut [Hash512]) {
    let mut item: Hash512 = Hash512::new();
    keccak(&mut item.0, &SEED.0);
    cache[0] = item;

    for cache_item in cache.iter_mut().take(LIGHT_CACHE_NUM_ITEMS as usize).skip(1) {
        keccak_in_place(&mut item.0);
        *cache_item = item;
    }

    for _ in 0..LIGHT_CACHE_ROUNDS {
        for i in 0..LIGHT_CACHE_NUM_ITEMS {
            // First index: 4 first bytes of the item as little-endian integer
            let t: u32 = cache[i as usize].get_as_u32(0);
            let v: u32 = t % LIGHT_CACHE_NUM_ITEMS;

            // Second index
            let w: u32 = (LIGHT_CACHE_NUM_ITEMS.wrapping_add(i.wrapping_sub(1))) % LIGHT_CACHE_NUM_ITEMS;

            let x = &cache[v as usize] ^ &cache[w as usize];
            keccak(&mut cache[i as usize].0, &x.0);
        }
    }
}

fn prebuild_dataset(full_dataset: &mut Box<[Hash1024]>, light_cache: &[Hash512], num_threads: usize) {
    //let full_dataset = full_dataset_opt.as_mut().unwrap();

    if num_threads > 1 {
        std::thread::scope(|scope| {
            let mut threads = Vec::with_capacity(num_threads);

            let light_cache_slice = &light_cache[0..];
            let batch_size = full_dataset.len() / num_threads;
            let chunks = full_dataset.chunks_mut(batch_size);

            for (index, chunk) in chunks.enumerate() {
                let start = index * batch_size;

                let thread_handle = scope.spawn(move || build_dataset_segment(chunk, light_cache_slice, start));
                threads.push(thread_handle);
            }

            for handle in threads {
                handle.join().unwrap();
            }
        });
    } else {
        build_dataset_segment(&mut full_dataset[0..], light_cache, 0);
    }
}

fn build_dataset_segment(dataset_slice: &mut [Hash1024], light_cache: &[Hash512], offset: usize) {
    for (index, item) in dataset_slice.iter_mut().enumerate() {
        *item = calculate_dataset_item_1024(light_cache, offset + index);
    }
}

fn fnv1(u: u32, v: u32) -> u32 {
    (u * FNV_PRIME) ^ v
}

fn fnv1_512(u: Hash512, v: Hash512) -> Hash512 {
    let mut r = Hash512::new();

    for i in 0..r.0.len() / SIZE_U32 {
        r.set_as_u32(i, fnv1(u.get_as_u32(i), v.get_as_u32(i)));
    }

    r
}

fn calculate_dataset_item_1024(light_cache: &[Hash512], index: usize) -> Hash1024 {
    let seed0 = (index * 2) as u32;
    let seed1 = seed0 + 1;

    let mut mix0 = light_cache[(seed0 % LIGHT_CACHE_NUM_ITEMS) as usize];
    let mut mix1 = light_cache[(seed1 % LIGHT_CACHE_NUM_ITEMS) as usize];

    let mix0_seed = mix0.get_as_u32(0) ^ seed0;
    let mix1_seed = mix1.get_as_u32(0) ^ seed1;

    mix0.set_as_u32(0, mix0_seed);
    mix1.set_as_u32(0, mix1_seed);

    keccak_in_place(&mut mix0.0);
    keccak_in_place(&mut mix1.0);

    let num_words: u32 = (std::mem::size_of_val(&mix0) / SIZE_U32) as u32;
    for j in 0..FULL_DATASET_ITEM_PARENTS {
        let t0 = fnv1(seed0 ^ j, mix0.get_as_u32((j % num_words) as usize));
        let t1 = fnv1(seed1 ^ j, mix1.get_as_u32((j % num_words) as usize));
        mix0 = fnv1_512(mix0, light_cache[(t0 % LIGHT_CACHE_NUM_ITEMS) as usize]);
        mix1 = fnv1_512(mix1, light_cache[(t1 % LIGHT_CACHE_NUM_ITEMS) as usize]);
    }

    keccak_in_place(&mut mix0.0);
    keccak_in_place(&mut mix1.0);

    Hash1024::from_512s(&mix0, &mix1)
}

fn save_dataset_to_file(full_dataset_unwrap: &[Hash1024], filename: &str) {
    let total_size = std::mem::size_of_val(full_dataset_unwrap);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(filename)
        .unwrap_or_else(|_| panic!("Failed to open or create the file"));

    file.set_len(total_size as u64).expect("Failed to set file length");

    let mut mmap = unsafe { MmapOptions::new().map_mut(&file).expect("Failed to memory-map the file") };

    for (i, hash) in full_dataset_unwrap.iter().enumerate() {
        let offset = i * std::mem::size_of::<Hash1024>();
        mmap[offset..offset + std::mem::size_of::<Hash1024>()].copy_from_slice(hash.as_bytes());
    }

    mmap.flush().expect("Failed to flush memory map to disk");
}

fn read_dataset_from_file(filename: &str, full_dataset_unwrap: &mut Box<[Hash1024]>) {
    if !Path::new(filename).exists() {
        panic!("File does not exist");
    }

    let file =
        OpenOptions::new().read(true).open(filename).unwrap_or_else(|_| panic!("Failed to open the file for reading"));

    let mmap = unsafe { MmapOptions::new().map(&file).expect("Failed to memory-map the file for reading") };
    let item_size = std::mem::size_of::<Hash1024>();
    if mmap.len() % item_size != 0 {
        panic!("File size is not a multiple of Hash1024 size");
    }

    let num_items = mmap.len() / item_size;
    if num_items != full_dataset_unwrap.len() {
        panic!("Mismatch between file data size and provided buffer size");
    }

    for i in 0..num_items {
        let start = i * item_size;
        let end = start + item_size;
        full_dataset_unwrap[i] = Hash1024::from_bytes(&mmap[start..end]);
    }
}

impl CudaGPUWorker {
    pub fn new(
        device_id: u32,
        workload: f32,
        is_absolute: bool,
        blocking_sync: bool,
        random: NonceGenEnum,
    ) -> Result<Self, Error> {
        info!("Starting a CUDA worker");
        let sync_flag = match blocking_sync {
            true => ContextFlags::SCHED_BLOCKING_SYNC,
            false => ContextFlags::SCHED_AUTO,
        };
        let device = Device::get_device(device_id).unwrap();
        let _context = Context::new(device)?;
        _context.set_flags(sync_flag)?;

        let mut light_cache = vec![Hash512::new(); LIGHT_CACHE_NUM_ITEMS as usize].into_boxed_slice();
        build_light_cache(&mut light_cache);
        //cache.copy_from(&light_cache)?;
        let cache2 = DeviceBuffer::from_slice(&light_cache).unwrap();

        info!("light_cache[10] : {:x?}", &light_cache[10].as_bytes());
        info!("light_cache[42] : {:x?}", &light_cache[42].as_bytes());

        let mut full_dataset = Some(vec![Hash1024::new(); FULL_DATASET_NUM_ITEMS as usize].into_boxed_slice());
        let full_dataset_uwrap = full_dataset.as_mut().unwrap();
        //build_dataset_segment(&mut full_dataset_uwrap[0..], &light_cache, 0);
        if Path::new("dataset.bin").exists() {
            read_dataset_from_file("dataset.bin", full_dataset_uwrap);
        } else {
            prebuild_dataset(full_dataset_uwrap, &light_cache, 8);
            //save_dataset_to_file(&full_dataset_uwrap, "hashes.dat")
            save_dataset_to_file(full_dataset_uwrap, "dataset.bin");
        }

        info!("dataset[10] : {:x?}", full_dataset_uwrap[10].as_bytes());
        info!("dataset[42] : {:x?}", full_dataset_uwrap[42].as_bytes());
        info!("dataset[12345] : {:x?}", full_dataset_uwrap[12345].as_bytes());

        //dataset.copy_from(&full_dataset_uwrap)?;
        let dataset2 = DeviceBuffer::from_slice(full_dataset_uwrap).unwrap();
        //let dataset2_ptr: DevicePointer<Hash1024> = dataset2.as_device_ptr();

        let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
        let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
        let _module: Arc<Module>;
        info!("Device #{} compute version is {}.{}", device_id, major, minor);
        if major > 8 || (major == 8 && minor >= 6) {
            _module =
                Arc::new(Module::from_ptx(PTX_86, &[ModuleJitOption::OptLevel(OptLevel::O4)]).inspect_err(|_e| {
                    error!("Error loading PTX. Make sure you have the updated driver for you devices");
                })?);
        } else if major > 7 || (major == 7 && minor >= 5) {
            _module =
                Arc::new(Module::from_ptx(PTX_75, &[ModuleJitOption::OptLevel(OptLevel::O4)]).inspect_err(|_e| {
                    error!("Error loading PTX. Make sure you have the updated driver for you devices");
                })?);
        } else if major > 6 || (major == 6 && minor >= 1) {
            _module =
                Arc::new(Module::from_ptx(PTX_61, &[ModuleJitOption::OptLevel(OptLevel::O4)]).inspect_err(|_e| {
                    error!("Error loading PTX. Make sure you have the updated driver for you devices");
                })?);
        } else if major >= 3 {
            _module =
                Arc::new(Module::from_ptx(PTX_30, &[ModuleJitOption::OptLevel(OptLevel::O4)]).inspect_err(|_e| {
                    error!("Error loading PTX. Make sure you have the updated driver for you devices");
                })?);
        } else if major >= 2 {
            _module =
                Arc::new(Module::from_ptx(PTX_20, &[ModuleJitOption::OptLevel(OptLevel::O4)]).inspect_err(|_e| {
                    error!("Error loading PTX. Make sure you have the updated driver for you devices");
                })?);
        } else {
            return Err("Cuda compute version not supported".into());
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let mut heavy_hash_kernel = Kernel::new(Arc::downgrade(&_module), "heavy_hash")?;

        let mut chosen_workload = 0u32;
        if is_absolute {
            chosen_workload = 1;
        } else {
            let cur_workload = heavy_hash_kernel.get_workload();
            if chosen_workload == 0 || chosen_workload < cur_workload {
                chosen_workload = cur_workload;
            }
        }
        chosen_workload = (chosen_workload as f32 * workload) as u32;
        info!("GPU #{} Chosen workload: {}", device_id, chosen_workload);
        heavy_hash_kernel.set_workload(chosen_workload);

        let final_nonce_buff = vec![0u64; 1].as_slice().as_dbuf()?;

        let rand_state: DeviceBuffer<u64> = match random {
            NonceGenEnum::Xoshiro => {
                info!("Using xoshiro for nonce-generation");
                let mut buffer = DeviceBuffer::<u64>::zeroed(4 * (chosen_workload as usize)).unwrap();
                info!("GPU #{} is generating initial seed. This may take some time.", device_id);
                let mut seed = [1u64; 4];
                seed.try_fill(&mut rand::thread_rng())?;
                buffer.copy_from(
                    Xoshiro256StarStar::new(&seed)
                        .iter_jump_state()
                        .take(chosen_workload as usize)
                        .flatten()
                        .collect::<Vec<u64>>()
                        .as_slice(),
                )?;
                info!("GPU #{} initialized", device_id);
                buffer
            }
            NonceGenEnum::Lean => {
                info!("Using lean nonce-generation");
                let mut buffer = DeviceBuffer::<u64>::zeroed(1).unwrap();
                let seed = rand::thread_rng().next_u64();
                buffer.copy_from(&[seed])?;
                buffer
            }
        };
        Ok(Self {
            device_id,
            _context,
            _module,
            start_event: Event::new(EventFlags::DEFAULT)?,
            stop_event: Event::new(EventFlags::DEFAULT)?,
            workload: chosen_workload as usize,
            stream,
            rand_state,
            final_nonce_buff,
            cache2,
            dataset2,
            heavy_hash_kernel,
            random,
        })
    }
}
