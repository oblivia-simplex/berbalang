mod evolution;
mod examples;

use examples::basic;

fn main() {
    const TARGET: &str = "This is Lucca's genetic algorithm example.";

    let config = basic::Config {
        mut_rate: 0.1,
        init_len: TARGET.len(),
        pop_size: 100,
        target: TARGET.to_string(),
    };

    basic::run(config);
}
