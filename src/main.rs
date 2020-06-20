#![cfg_attr(feature = "cargo-clippy", allow(clippy::option_map_unit_fn))]
#![cfg_attr(feature = "cargo-clippy", allow(clippy::needless_range_loop))]

use std::sync::atomic;
use std::sync::atomic::{AtomicBool, AtomicUsize};

use configure::Config;

use crate::configure::Job;
use crate::examples::{hello_world, linear_gp};

mod configure;
#[allow(dead_code)] // FIXME
mod disassembler;
#[allow(dead_code)] // FIXME
mod emulator;
mod error;
mod evolution;
mod examples;
#[allow(dead_code)] // FIXME
mod fitness;
mod logger;
mod macros;
mod observer;
mod ontogenesis;
mod roper;
#[allow(dead_code)] // FIXME
mod util;

pub static EPOCH_COUNTER: AtomicUsize = AtomicUsize::new(0);
pub static KEEP_GOING: AtomicBool = AtomicBool::new(true);
pub static WINNING_ISLAND: AtomicUsize = AtomicUsize::new(0xbaad_f00d);

pub fn keep_going() -> bool {
    KEEP_GOING.load(atomic::Ordering::Relaxed)
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

fn main() {
    // TODO add standard cli
    let config_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./config.toml".to_string());
    let population_name = std::env::args().nth(2);
    let config = Config::from_path(&config_file, population_name)
        .unwrap_or_else(|e| panic!("Failed to generate Config from {:?}: {:?}", &config_file, e));

    logger::init(&config.observer.population_name);

    match config.job {
        Job::LinearGp => {
            linear_gp::run(config);
        }
        Job::Hello => {
            hello_world::run(config);
        }
        Job::Roper => {
            use unicorn::Arch::*;

            match config.roper.arch {
                X86 => roper::run::<unicorn::CpuX86<'_>>(config),
                ARM => roper::run::<unicorn::CpuARM<'_>>(config),
                ARM64 => roper::run::<unicorn::CpuARM64<'_>>(config),
                MIPS => roper::run::<unicorn::CpuMIPS<'_>>(config),
                SPARC => roper::run::<unicorn::CpuSPARC<'_>>(config),
                M68K => roper::run::<unicorn::CpuM68K<'_>>(config),
                _ => unimplemented!("architecture unimplemented"),
            }
        }
    }

    log::info!("Waiting 3 seconds for file writes to complete...");
    std::thread::sleep(std::time::Duration::from_secs(3));
}
