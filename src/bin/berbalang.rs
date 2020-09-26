use berbalib::configure::{Config, Job};
use berbalib::examples::{hello_world, linear_gp};
use berbalib::{limit_threads, logger, roper, set_starting_timestamp, set_timeout};

fn main() {
    // TODO add standard cli
    let config_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./config.toml".to_string());
    let population_name = std::env::args().nth(2);
    let mut config = Config::from_path(&config_file, population_name)
        .unwrap_or_else(|e| panic!("Failed to generate Config from {:?}: {:?}", &config_file, e));
    logger::init(&config.observer.population_name);
    set_starting_timestamp();
    if let Some(ref timeout) = config.timeout {
        set_timeout(timeout);
    }

    // config.roper.parse_register_patterns();
    if let Ok(n) = std::env::var("BERBALANG_LIMIT_THREADS") {
        limit_threads(
            n.parse()
                .expect("Invalid value for BERBALANG_LIMIT_THREADS"),
            &mut config,
        );
    } else if cfg!(feature = "disassemble_trace") {
        limit_threads(1, &mut config);
    }

    match config.job {
        Job::LinearGp => {
            linear_gp::run(config);
        }
        Job::Hello => {
            hello_world::run(config);
        }
        Job::Roper => {
            roper::run(config);
        }
    }

    log::info!("Waiting 3 seconds for file writes to complete...");
    std::thread::sleep(std::time::Duration::from_secs(3));
}
