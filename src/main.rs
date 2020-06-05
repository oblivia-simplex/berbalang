#![cfg_attr(feature = "cargo-clippy", allow(clippy::option_map_unit_fn))]
#![cfg_attr(feature = "cargo-clippy", allow(clippy::needless_range_loop))]

use configure::Config;

use crate::configure::Job;
use crate::examples::{hello_world, linear_gp};

mod configure;
#[allow(dead_code)] // FIXME
mod disassembler;
#[allow(dead_code)] // FIXME
mod emulator;
mod error;
mod evaluator;
mod evolution;
mod examples;
#[allow(dead_code)] // FIXME
mod fitness;
mod logger;
mod macros;
mod observer;
mod roper;
#[allow(dead_code)] // FIXME
mod util;

fn main() {
    let mut config: Config = toml::from_str(
        &std::fs::read_to_string("./config.toml").expect("Failed to open config.toml"),
    )
    .expect("Failed to parse config.toml");
    config.assert_invariants();
    config.set_data_directory();

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
}
