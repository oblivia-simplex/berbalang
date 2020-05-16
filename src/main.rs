mod evolution;
mod examples;
mod observer;

use crate::examples::hello_world;
use pretty_env_logger as logger;

fn main() {
    logger::init();

    const TARGET: &str = "Wanna know how I got these scars?";

    let config = hello_world::Config {
        mut_rate: 0.3,
        init_len: TARGET.len() * 4,
        pop_size: 1000,
        target: TARGET.to_string(),
    };

    hello_world::run(config);
}
