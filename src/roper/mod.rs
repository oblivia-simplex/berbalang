use std::fs::File;
use std::hash::Hash;
use std::hash::Hasher;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::thread::spawn;

use non_dominated_sort::DominanceOrd;
use rand::Rng;
use unicorn::Cpu;

use crate::configure::{Config, Selection};
use crate::error::Error;
use crate::evolution::metropolis::Metropolis;
use crate::evolution::pareto_roulette::Roulette;
use crate::evolution::population::pier::Pier;
use crate::fitness::Weighted;
use crate::observer::Observer;
use crate::ontogenesis::FitnessFn;
use crate::util::architecture::Perms;
use crate::util::count_min_sketch::CountMinSketch;
use crate::util::random::hash_seed_rng;
use crate::{
    emulator::loader,
    evolution::{tournament::Tournament, Phenome},
};

/// The `analysis` module contains the reporting function passed to the observation
/// window. Population saving, soup dumping, statistical assessment, etc., happens there.
mod analysis;

/// Generic fitness functions, which can be used for either push or bare
/// mode ROPER.
mod fitness_functions;

/// The `creature` module contains the implementation of the `Genome` and `Phenome`
/// traits associated with `roper` mode.
pub mod bare;

/// A ROPER-specific implementation of Spector's PUSH VM.
pub mod push;

/// load binary before calling this function
pub fn init_soup(config: &mut Config) -> Result<(), Error> {
    let mut soup = Vec::new();
    //might as well take the constants from the register pattern
    for pattern in config.roper.register_patterns() {
        pattern.0.values().for_each(|w| soup.push(w.val))
    }
    if let Some(gadget_file) = config.roper.gadget_file.as_ref() {
        // parse the gadget file
        let reader = File::open(gadget_file).map(BufReader::new)?;

        if gadget_file.ends_with(".json") {
            log::info!("Deserializing soup from {}", gadget_file);
            soup = serde_json::from_reader(reader)?;
        } else {
            log::info!("Parsing soup from {}", gadget_file);
            for line in reader.lines() {
                let word = line?.parse::<u64>()?;
                soup.push(word)
            }
        }
    } else if let Some(soup_size) = config.roper.soup_size.as_ref() {
        let memory = loader::get_static_memory_image();
        for addr in (0..(*soup_size)).map(|i| {
            let mut hasher = fnv::FnvHasher::default();
            i.hash(&mut hasher);
            config.random_seed.hash(&mut hasher);
            let seed = hasher.finish();
            memory.random_address(Some(Perms::EXEC), seed)
        }) {
            soup.push(addr)
        }
    }
    config.roper.soup = Some(soup);
    Ok(())
}

pub struct Sketches {
    pub register_error: CountMinSketch,
    pub memory_writes: CountMinSketch,
    pub addresses_visited: CountMinSketch,
}

impl Sketches {
    pub fn new(config: &Config) -> Self {
        Self {
            register_error: CountMinSketch::new(config),
            memory_writes: CountMinSketch::new(config),
            addresses_visited: CountMinSketch::new(config),
        }
    }
}

type Fitness<'a> = Weighted<'a>; //Pareto<'static>;

fn prepare_bare<C: 'static + Cpu<'static>>(
    config: &Config,
) -> (Observer<bare::Creature>, bare::evaluation::Evaluator<C>) {
    let fitness_function: FitnessFn<bare::Creature, Sketches, Config> =
        fitness_functions::get_fitness_function(&config.fitness.function);
    let observer = Observer::spawn(&config, Box::new(analysis::report_fn));
    let evaluator = bare::evaluation::Evaluator::spawn(&config, fitness_function);
    (observer, evaluator)
}

fn prepare_push<C: 'static + Cpu<'static>>(
    config: &Config,
) -> (Observer<push::Creature>, push::evaluation::Evaluator<C>) {
    let fitness_function: FitnessFn<push::Creature, Sketches, Config> =
        fitness_functions::get_fitness_function(&config.fitness.function);
    let observer: Observer<push::Creature> =
        Observer::spawn(&config, Box::new(analysis::report_fn));
    let evaluator = push::evaluation::Evaluator::spawn(&config, fitness_function);
    (observer, evaluator)
}

pub struct CreatureDominanceOrd;

impl DominanceOrd<bare::Creature> for CreatureDominanceOrd {
    fn dominance_ord(&self, a: &bare::Creature, b: &bare::Creature) -> std::cmp::Ordering {
        a.fitness()
            .partial_cmp(&b.fitness())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl DominanceOrd<push::Creature> for CreatureDominanceOrd {
    fn dominance_ord(&self, a: &push::Creature, b: &push::Creature) -> std::cmp::Ordering {
        a.fitness()
            .partial_cmp(&b.fitness())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl DominanceOrd<&bare::Creature> for CreatureDominanceOrd {}

impl DominanceOrd<&push::Creature> for CreatureDominanceOrd {}

pub fn run(mut config: Config) {
    let _ = loader::falcon_loader::load_from_path(&mut config, true)
        .expect("Failed to load binary image");
    init_soup(&mut config).expect("Failed to initialize the soup");

    use unicorn::Arch::*;
    match config.roper.arch {
        X86 => launch::<unicorn::CpuX86<'_>>(config),
        ARM => launch::<unicorn::CpuARM<'_>>(config),
        ARM64 => launch::<unicorn::CpuARM64<'_>>(config),
        MIPS => launch::<unicorn::CpuMIPS<'_>>(config),
        SPARC => launch::<unicorn::CpuSPARC<'_>>(config),
        M68K => launch::<unicorn::CpuM68K<'_>>(config),
        _ => unimplemented!("architecture unimplemented"),
    }
}

pub fn launch<C: 'static + Cpu<'static>>(config: Config) {
    match config.selection {
        Selection::Tournament => {
            // TODO: Refactor this!!
            let num_islands = config.num_islands;
            if config.roper.use_push {
                let pier: Arc<Pier<push::Creature>> = Arc::new(Pier::new(config.num_islands));
                let mut handles = Vec::new();
                let mut rng = hash_seed_rng(&config.random_seed);
                for i in 0..num_islands {
                    let mut config = config.clone();
                    config.island_identifier = i;
                    config.set_data_directory();
                    config.random_seed = rng.gen::<u64>();
                    let (observer, evaluator) = prepare_push(&config);
                    let pier = pier.clone();
                    let h = spawn(move || {
                        let mut world =
                            Tournament::<push::evaluation::Evaluator<C>, push::Creature>::new(
                                &config, observer, evaluator, pier,
                            );
                        while crate::keep_going() {
                            world = world.evolve();
                        }
                    });
                    handles.push(h);
                }
                for h in handles.into_iter() {
                    h.join().expect("Failed to join thread");
                }
            } else {
                let pier: Arc<Pier<bare::Creature>> = Arc::new(Pier::new(config.num_islands));
                let mut handles = Vec::new();
                let mut rng = hash_seed_rng(&config.random_seed);
                for i in 0..num_islands {
                    let mut config = config.clone();
                    config.island_identifier = i;
                    config.set_data_directory();
                    config.random_seed = rng.gen::<u64>();
                    let (observer, evaluator) = prepare_bare(&config);
                    let pier = pier.clone();
                    let h = spawn(move || {
                        let mut world =
                            Tournament::<bare::evaluation::Evaluator<C>, bare::Creature>::new(
                                &config, observer, evaluator, pier,
                            );
                        while crate::keep_going() {
                            world = world.evolve();
                        }
                    });
                    handles.push(h);
                }
                for h in handles.into_iter() {
                    h.join().expect("Failed to join thread");
                }
            }
        }
        Selection::Roulette => {
            let (observer, evaluator) = prepare_bare(&config);
            let mut world = Roulette::<
                bare::evaluation::Evaluator<C>,
                bare::Creature,
                CreatureDominanceOrd,
            >::new(&config, observer, evaluator, CreatureDominanceOrd);
            while crate::keep_going() {
                world = world.evolve();
            }
        }
        Selection::Metropolis => {
            let (observer, evaluator) = prepare_bare(&config);
            let mut world = Metropolis::<bare::evaluation::Evaluator<C>, bare::Creature>::new(
                &config, observer, evaluator,
            );
            while crate::keep_going() {
                world = world.evolve();
            }
        }
        Selection::Lexicase => unimplemented!("Probably needs an overhaul"),
        // Selection::Lexicase => {
        //     let fitness_function: FitnessFn<bare::Creature, Sketches, Config> =
        //         match config.fitness.function.as_str() {
        //             "register_pattern" => Box::new(bare::evaluation::register_pattern_ff),
        //             "register_conjunction" => Box::new(bare::evaluation::register_conjunction_ff),
        //             "code_coverage" => Box::new(bare::evaluation::code_coverage_ff),
        //             s => unimplemented!("No such fitness function as {}", s),
        //         };
        //     let pier: Arc<Pier<bare::Creature>> = Arc::new(Pier::new(config.num_islands));
        //     let evaluator = bare::evaluation::BareEvaluator::spawn(&config, fitness_function);
        //     let observer = Observer::spawn(
        //         &config,
        //         Box::new(bare::analysis::lexicase::report_fn),
        //         CreatureDominanceOrd,
        //     );
        //     let mut cases = config
        //         .roper
        //         .register_pattern
        //         .as_ref()
        //         .map(|rp| {
        //             let pattern: RegisterPattern = rp.into();
        //             pattern
        //                 .features()
        //                 .into_iter()
        //                 .map(lexi::Task::Reg)
        //                 .collect::<Vec<lexi::Task>>()
        //         })
        //         .expect("No register pattern specified");
        //     cases.push(lexi::Task::UniqExec(2));
        //     cases.push(lexi::Task::UniqExec(3));
        //     log::info!("Register Feature cases: {:#x?}", cases);
        //     let mut world =
        //         Lexicase::<lexi::Task, bare::evaluation::BareEvaluator<C>, bare::Creature>::new(
        //             &config, observer, evaluator, pier, cases,
        //         );
        //     while crate::keep_going() {
        //         world = world.evolve();
        //     }
        // }
    }
}
