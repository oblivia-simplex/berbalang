#![cfg_attr(feature = "cargo-clippy", allow(clippy::option_map_unit_fn))]
#![cfg_attr(feature = "cargo-clippy", allow(clippy::needless_range_loop))]

use std::sync::atomic;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use configure::Config;

pub mod configure;
#[allow(dead_code)] // FIXME
mod disassembler;
#[allow(dead_code)] // FIXME
pub mod emulator;
pub mod error;
pub mod evolution;
pub mod examples;
#[allow(dead_code)] // FIXME
pub mod fitness;
pub mod logger;
pub mod macros;
pub mod observer;
pub mod ontogenesis;
pub mod roper;
#[allow(dead_code)] // FIXME
pub mod util;

pub static EPOCH_COUNTER: AtomicUsize = AtomicUsize::new(0);
pub static KEEP_GOING: AtomicBool = AtomicBool::new(true);
pub static WINNING_ISLAND: AtomicUsize = AtomicUsize::new(0xbaad_f00d);
pub static STARTING_TIMESTAMP: AtomicUsize = AtomicUsize::new(0);
pub static TIMEOUT: AtomicUsize = AtomicUsize::new(0);

pub fn set_starting_timestamp() {
    let now: usize = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Error getting timestamp")
        .as_secs() as usize;
    STARTING_TIMESTAMP.store(now, atomic::Ordering::Relaxed);
}

pub fn set_timeout(timeout: &str) {
    let timeout = parse_duration::parse(timeout).expect("Failed to parse timeout string");
    let seconds = timeout.as_secs() as usize;
    TIMEOUT.store(seconds, atomic::Ordering::Relaxed);
}

pub fn uptime() -> Duration {
    let started = STARTING_TIMESTAMP.load(atomic::Ordering::Relaxed) as u64;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Error getting timestamp")
        .as_secs();
    if now <= started {
        Duration::from_secs(0)
    } else {
        Duration::from_secs(now - started)
    }
}

pub fn timeout_expired() -> bool {
    let timeout = Duration::from_secs(TIMEOUT.load(atomic::Ordering::Relaxed) as u64);
    let up_for = uptime();
    log::debug!("Uptime: {:?}", up_for);
    let expired = up_for > timeout;
    if expired {
        log::error!("Berbalang has timed out, at {:?}", up_for);
    }
    expired
}

pub fn keep_going() -> bool {
    !timeout_expired() && KEEP_GOING.load(atomic::Ordering::Relaxed)
}

pub fn stop_everything(island: usize, champion: bool) {
    let prior = KEEP_GOING.swap(false, atomic::Ordering::Relaxed);
    if prior {
        if champion {
            let msg = format!("Island {} has produced a champion!", island);
            let mut msg = ansi_colors::ColouredStr::new(&msg);
            msg.bold();
            msg.light_blue();
            println!("{}", msg);
            WINNING_ISLAND.store(island, atomic::Ordering::Relaxed);
        } else {
            let msg = format!("Evolution completed on Island {}.", island);
            let mut msg = ansi_colors::ColouredStr::new(&msg);
            msg.bold();
            msg.blue();
            println!("{}", msg);
        }
    };
}

pub fn get_epoch_counter() -> usize {
    EPOCH_COUNTER.load(atomic::Ordering::Relaxed)
}

pub fn increment_epoch_counter() {
    EPOCH_COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
}

pub fn limit_threads(threads: usize, config: &mut Config) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    config.num_islands = threads;
    config.roper.num_emulators = threads;
    config.roper.num_workers = threads;
}
