#![cfg_attr(feature = "cargo-clippy", allow(clippy::option_map_unit_fn))]
#![cfg_attr(feature = "cargo-clippy", allow(clippy::needless_range_loop))]

use pretty_env_logger as logger;
//
// #[cfg(not(feature = "linear_gp"))]
// use examples::hello_world as example;
// #[cfg(feature = "linear_gp")]
// use examples::linear_gp as example;

use crate::examples::{hello_world, linear_gp};

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

    // TODO: maybe just define a single, shared config struct. Simple enough to do.
    // with sub-fields.
    let args = std::env::args().collect::<String>();
    if args.contains("hello_world") {
        let config: hello_world::Config = toml::from_str(
            &std::fs::read_to_string("./config.toml").expect("Failed to open config.toml"),
        )
        .expect("Failed to parse config.toml");
        config.assert_invariants();
        hello_world::run(config);
    } else {
        let config: linear_gp::Config = toml::from_str(
            &std::fs::read_to_string("./config.toml").expect("Failed to open config.toml"),
        )
        .expect("Failed to parse config.toml");
        config.assert_invariants();
        linear_gp::run(config);
    }
}
