use pretty_env_logger as logger;

use examples::linear_gp as example;

use crate::configure::Configure;

mod configure;
mod evaluator;
mod evolution;
mod examples;
mod fitness;
mod observer;

fn main() {
    logger::init();

    let config: example::Config = toml::from_str(
        &std::fs::read_to_string("./config.toml").expect("Failed to open config.toml"),
    )
    .expect("Failed to parse config.toml");
    config.assert_invariants();

    example::run(config);
}
