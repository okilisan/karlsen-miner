#[macro_use]
extern crate karlsen_miner;

use clap::{ArgMatches, FromArgMatches};
use karlsen_miner::{Plugin, Worker, WorkerSpec};
use log::LevelFilter;
use std::error::Error as StdError;

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

mod cli;
mod worker;

use crate::cli::KarlsenGptOpt;
use crate::worker::KarlsenGptGPUWorker;

//const DEFAULT_WORKLOAD_SCALE: f32 = 1024.;

pub struct KarlsenGptPlugin {
    spec: KarlsenGptWorkerSpec,
    //#[cfg(feature = "overclock")]
    //nvml_instance: Nvml,
    _enabled: bool,
}

impl KarlsenGptPlugin {
    fn new() -> Result<Self, Error> {
        println!("KarlsenGptPlugin - new");
        //cust::init(CudaFlags::empty())?;
        env_logger::builder().filter_level(LevelFilter::Info).parse_default_env().init();
        Ok(Self {
            spec: KarlsenGptWorkerSpec {
                device_id:42,
                model_path: String::new()
            },
            _enabled: true,
            //#[cfg(feature = "overclock")]
            //nvml_instance: Nvml::init()?,
        })
    }
}

impl Plugin for KarlsenGptPlugin {
    fn name(&self) -> &'static str {
        "KarlsenGpt Cuda Worker"
    }

    fn enabled(&self) -> bool {
        println!("KarlsenGptPlugin - enabled {}", self._enabled);
        self._enabled
    }

    fn get_worker_specs(&self) -> Vec<Box<dyn WorkerSpec>> {
        println!("KarlsenGptPlugin - get_worker_specs");
        //self.specs.iter().map(|spec| Box::new(*spec) as Box<dyn WorkerSpec>).collect::<Vec<Box<dyn WorkerSpec>>>()
        vec![Box::new(self.spec.clone()) as Box<dyn WorkerSpec>]
    }

    /* */
    fn process_option(&mut self, matches: &ArgMatches) -> Result<usize, karlsen_miner::Error> {
        let opts: KarlsenGptOpt = KarlsenGptOpt::from_arg_matches(matches)?;
        println!("KarlsenGptPlugin - process_option");
        
        let _model_path = opts.karlsengpt_model_path;

        //TODO: initialize the llama backend here

        self.spec.device_id = 42;
        //self.spec.model_path = opts.karlsengpt_model_path;

        Ok(1)
    }

    //noinspection RsTypeCheck
    /*
    fn process_option(&mut self, matches: &ArgMatches) -> Result<usize, karlsen_miner::Error> {
        let opts: KarlsenGptOpt = KarlsenGptOpt::from_arg_matches(matches)?;

        self._enabled = !opts.cuda_disable;
        if self._enabled {
            let gpus: Vec<u16> = match &opts.cuda_device {
                Some(devices) => devices.clone(),
                None => {
                    let gpu_count = Device::num_devices().unwrap() as u16;
                    (0..gpu_count).collect()
                }
            };

            

            self.specs = (0..gpus.len())
                .map(|i| KarlsenGptWorkerSpec {
                    device_id: gpus[i] as u32,
                    workload: match &opts.cuda_workload {
                        Some(workload) if i < workload.len() => workload[i],
                        Some(workload) if !workload.is_empty() => *workload.last().unwrap(),
                        _ => DEFAULT_WORKLOAD_SCALE,
                    },
                    is_absolute: opts.cuda_workload_absolute,
                    blocking_sync: !opts.cuda_no_blocking_sync,
                    random: opts.cuda_nonce_gen,
                })
                .collect();
        }
        Ok(self.specs.len())
    }
    */
    
}

#[derive(Clone)]
struct KarlsenGptWorkerSpec {
    device_id: u32,
    model_path: String,
}

impl WorkerSpec for KarlsenGptWorkerSpec {
    fn id(&self) -> String {
        //let device = Device::get_device(self.device_id).unwrap();
        println!("KarlsenGptPlugin - id");
        format!("#{}", self.device_id)
    }

    fn build(&self) -> Box<dyn Worker> {
        println!("KarlsenGptPlugin - build");
        Box::new(
            KarlsenGptGPUWorker::new(self.device_id, self.model_path.clone())
                .unwrap(),
        )
    }
}

declare_plugin!(KarlsenGptPlugin, KarlsenGptPlugin::new, KarlsenGptOpt);
