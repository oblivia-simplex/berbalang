#![cfg_attr(feature = "cargo-clippy", allow(clippy::option_map_unit_fn))]

use pretty_env_logger as logger;

#[cfg(not(feature = "linear_gp"))]
use examples::hello_world as example;
#[cfg(feature = "linear_gp")]
use examples::linear_gp as example;

use crate::configure::Configure;

mod scratch; // NOTE: tinkering files

mod configure;
mod creature;
#[allow(dead_code)] // FIXME
mod disassembler;
#[allow(dead_code)] // FIXME
mod emulator;
mod evaluator;
mod evolution;
mod examples;
mod fitness;
mod observer;
#[allow(dead_code)] // FIXME
mod roper;
#[allow(dead_code)] // FIXME
mod util;

fn main() {
    logger::init();

    let config: example::Config = toml::from_str(
        &std::fs::read_to_string("./config.toml").expect("Failed to open config.toml"),
    )
    .expect("Failed to parse config.toml");

    config.assert_invariants();

    example::run(config);
}
