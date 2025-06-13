#![cfg_attr(all(test, feature = "bench"), feature(test))]

use std::env::consts::DLL_EXTENSION;
use std::env::current_exe;
use std::error::Error as StdError;
use std::ffi::OsStr;

use clap::{App, FromArgMatches, IntoApp};
use karlsen_miner::PluginManager;
use log::{error, info};
use rand::{rng, RngCore};
use std::array::TryFromSliceError;
use std::fs;
use std::sync::atomic::AtomicU16;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use crate::cli::Opt;
use crate::client::grpc::KarlsendHandler;
use crate::client::stratum::StratumHandler;
use crate::client::Client;
use crate::miner::MinerManager;
use crate::target::Uint256;

mod cli;
mod client;
mod karlsend_messages;
mod miner;
mod pow;
mod target;
mod watch;

//remove the opencl plugin support for the moment
const WHITELIST: [&str; 3] = ["libkarlsencuda", "karlsencuda", "karlsengpt"];

pub mod proto {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("protowire");
}

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

type Hash = Uint256;

pub const HASH_SIZE: usize = 32;
pub struct HashKls([u8; HASH_SIZE]);

//serde_impl_ser_fixed_bytes_ref!(HashKls, HASH_SIZE);
//serde_impl_deser_fixed_bytes_ref!(HashKls, HASH_SIZE);

impl std::fmt::Debug for HashKls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HashKls({})", hex::encode(self.as_bytes_ref()))
    }
}

impl From<[u8; HASH_SIZE]> for HashKls {
    fn from(value: [u8; HASH_SIZE]) -> Self {
        HashKls(value)
    }
}

impl TryFrom<&[u8]> for HashKls {
    type Error = TryFromSliceError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        HashKls::try_from_slice(value)
    }
}

impl HashKls {
    #[inline(always)]
    pub fn as_bytes_ref(&self) -> &[u8; 32] {
        &self.0
    }

    #[inline(always)]
    pub const fn from_bytes(bytes: [u8; HASH_SIZE]) -> Self {
        HashKls(bytes)
    }

    #[inline(always)]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    #[inline(always)]
    /// # Panics
    /// Panics if `bytes` length is not exactly `HASH_SIZE`.
    pub fn from_slice(bytes: &[u8]) -> Self {
        Self(<[u8; HASH_SIZE]>::try_from(bytes).expect("Slice must have the length of HashKls"))
    }

    #[inline(always)]
    pub fn try_from_slice(bytes: &[u8]) -> Result<Self, TryFromSliceError> {
        Ok(Self(<[u8; HASH_SIZE]>::try_from(bytes)?))
    }

    #[inline(always)]
    pub fn to_le_u64(self) -> [u64; 4] {
        let mut out = [0u64; 4];
        out.iter_mut().zip(self.iter_le_u64()).for_each(|(out, word)| *out = word);
        out
    }

    #[inline(always)]
    pub fn iter_le_u64(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.0.chunks_exact(8).map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
    }

    #[inline(always)]
    pub fn from_le_u64(arr: [u64; 4]) -> Self {
        let mut ret = [0; HASH_SIZE];
        ret.chunks_exact_mut(8).zip(arr.iter()).for_each(|(bytes, word)| bytes.copy_from_slice(&word.to_le_bytes()));
        Self(ret)
    }

    #[inline(always)]
    pub fn from_u64_word(word: u64) -> Self {
        Self::from_le_u64([0, 0, 0, word])
    }
}

#[cfg(target_os = "windows")]
fn adjust_console() -> Result<(), Error> {
    let console = win32console::console::WinConsole::input();
    let mut mode = console.get_mode()?;
    mode = (mode & !win32console::console::ConsoleMode::ENABLE_QUICK_EDIT_MODE)
        | win32console::console::ConsoleMode::ENABLE_EXTENDED_FLAGS;
    console.set_mode(mode)?;
    Ok(())
}

fn filter_plugins(dirname: &str) -> Vec<String> {
    match fs::read_dir(dirname) {
        Ok(readdir) => readdir
            .map(|entry| entry.unwrap().path())
            .filter(|fname| {
                fname.is_file()
                    && fname.extension().is_some()
                    && fname.extension().and_then(OsStr::to_str).unwrap_or_default().starts_with(DLL_EXTENSION)
            })
            .filter(|fname| WHITELIST.iter().any(|lib| *lib == fname.file_stem().and_then(OsStr::to_str).unwrap()))
            .map(|path| path.to_str().unwrap().to_string())
            .collect::<Vec<String>>(),
        _ => Vec::<String>::new(),
    }
}

async fn get_client(
    karlsend_address: String,
    mining_address: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,
) -> Result<Box<dyn Client + 'static>, Error> {
    if karlsend_address.starts_with("stratum+tcp://") {
        let (_schema, address) = karlsend_address.split_once("://").unwrap();
        Ok(StratumHandler::connect(
            address.to_string().clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
        )
        .await?)
    } else if karlsend_address.starts_with("grpc://") {
        Ok(KarlsendHandler::connect(
            karlsend_address.clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
        )
        .await?)
    } else {
        Err("Did not recognize pool/grpc address schema".into())
    }
}

async fn client_main(
    opt: &Opt,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
) -> Result<(), Error> {
    let mut client = get_client(
        opt.karlsend_address.clone(),
        opt.mining_address.clone(),
        opt.mine_when_not_synced,
        block_template_ctr.clone(),
    )
    .await?;

    if opt.devfund_percent > 0 {
        client.add_devfund(opt.devfund_address.clone(), opt.devfund_percent);
    }
    client.register().await?;
    let mut miner_manager = MinerManager::new(client.get_block_channel(), opt.num_threads, plugin_manager);
    client.listen(&mut miner_manager).await?;
    info!("drop from main.rs");
    drop(miner_manager);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    #[cfg(target_os = "windows")]
    adjust_console().unwrap_or_else(|e| {
        eprintln!("WARNING: Failed to protect console ({}). Any selection in console will freeze the miner.", e)
    });
    let mut path = current_exe().unwrap_or_default();
    path.pop(); // Getting the parent directory
    let plugins = filter_plugins(path.to_str().unwrap_or("."));
    let (app, mut plugin_manager): (App, PluginManager) = karlsen_miner::load_plugins(Opt::into_app(), &plugins)?;

    let matches = app.get_matches();

    let worker_count = plugin_manager.process_options(&matches)?;
    let mut opt: Opt = Opt::from_arg_matches(&matches)?;
    opt.process()?;
    env_logger::builder().filter_level(opt.log_level()).parse_default_env().init();
    info!("=================================================================================");
    info!("                 karlsen-miner GPU {}", env!("CARGO_PKG_VERSION"));
    info!(" Mining for: {}", opt.mining_address);
    info!("=================================================================================");
    info!("Found plugins: {:?}", plugins);
    info!("Plugins found {} workers", worker_count);
    if worker_count == 0 && opt.num_threads.unwrap_or(0) == 0 {
        error!("No workers specified");
        return Err("No workers specified".into());
    }

    let block_template_ctr = Arc::new(AtomicU16::new((rng().next_u64() % 10_000u64) as u16));
    if opt.devfund_percent > 0 {
        info!(
            "devfund enabled, mining {}.{}% of the time to devfund address: {} ",
            opt.devfund_percent / 100,
            opt.devfund_percent % 100,
            opt.devfund_address
        );
    }
    loop {
        match client_main(&opt, block_template_ctr.clone(), &plugin_manager).await {
            Ok(_) => info!("Client closed gracefully"),
            Err(e) => error!("Client closed with error {:?}", e),
        }
        info!("Client closed, reconnecting");
        sleep(Duration::from_millis(100));
    }
}
