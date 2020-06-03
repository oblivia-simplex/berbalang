#![cfg_attr(feature = "cargo-clippy", allow(clippy::option_map_unit_fn))]
#![cfg_attr(feature = "cargo-clippy", allow(clippy::needless_range_loop))]

use pretty_env_logger as logger;
//
// #[cfg(not(feature = "linear_gp"))]
// use examples::hello_world as example;
// #[cfg(feature = "linear_gp")]
// use examples::linear_gp as example;

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
mod macros;
mod observer;
#[allow(unused_imports, dead_code)] // FIXME
mod roper;
#[allow(dead_code)] // FIXME
mod util;

use configure::Config;
use unicorn::Arch;

fn main() {
    logger::init();

    // TODO: develop a proper CLI
    let args = std::env::args().collect::<String>();
    if args.contains("hello_world") {
        let config: Config = toml::from_str(
            &std::fs::read_to_string("./config.toml").expect("Failed to open config.toml"),
        )
        .expect("Failed to parse config.toml");
        config.assert_invariants();
        hello_world::run(config);
    } else if args.contains("linear_gp") {
        let config: Config = toml::from_str(
            &std::fs::read_to_string("./config.toml").expect("Failed to open config.toml"),
        )
        .expect("Failed to parse config.toml");
        config.assert_invariants();
        linear_gp::run(config);
    } else {
        // ROPER TIME
        let config: Config = toml::from_str(
            &std::fs::read_to_string("./config.toml").expect("Failed to open config.toml"),
        )
        .expect("Failed to parse config.toml");
        config.assert_invariants();

        log::info!("Config: {:#x?}", config);
        // now switch off on architecture, I guess
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
