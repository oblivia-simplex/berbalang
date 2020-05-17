use pretty_env_logger as logger;

use crate::configure::Configure;
use crate::examples::hello_world;

mod configure;
mod evaluator;
mod evolution;
mod examples;
mod observer;

fn main() {
    logger::init();

    let config: hello_world::Config = toml::from_str(
        &std::fs::read_to_string("./config.toml").expect("Failed to open config.toml"),
    )
    .expect("Failed to parse config.toml");
    config.assert_invariants();

    hello_world::run(config);
}
