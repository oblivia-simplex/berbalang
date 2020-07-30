use std::sync::Arc;
use std::thread::spawn;

use non_dominated_sort::DominanceOrd;
use rand::Rng;
use unicorn::Cpu;

use crate::configure::{Config, Selection};
/// This is where the ROP-evolution-specific code lives.
use crate::{
    emulator::loader,
    evolution::{tournament::Tournament, Phenome},
};
// the runner
use crate::emulator::register_pattern::RegisterPattern;
use crate::evolution::lexicase::Lexicase;
use crate::evolution::metropolis::Metropolis;
use crate::evolution::pareto_roulette::Roulette;
use crate::evolution::population::pier::Pier;
use crate::fitness::Weighted;
use crate::observer::Observer;
use crate::ontogenesis::FitnessFn;
use crate::roper::bare::evaluation::lexi;
use crate::util::count_min_sketch::CountMinSketch;
use crate::util::random::hash_seed_rng;

/// The `creature` module contains the implementation of the `Genome` and `Phenome`
/// traits associated with `roper` mode.
mod bare;

/// A ROPER-specific implementation of Spector's PUSH VM.
#[allow(dead_code)]
mod push;

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
) -> (Observer<bare::Creature>, bare::evaluation::BareEvaluator<C>) {
    let fitness_function: FitnessFn<bare::Creature, Sketches, Config> =
        match config.fitness.function.as_str() {
            "register_pattern" => Box::new(bare::evaluation::register_pattern_ff),
            "register_conjunction" => Box::new(bare::evaluation::register_conjunction_ff),
            "register_entropy" => Box::new(bare::evaluation::register_entropy_ff),
            "code_coverage" => Box::new(bare::evaluation::code_coverage_ff),
            "just_novelty" => Box::new(bare::evaluation::just_novelty_ff),
            s => unimplemented!("No such fitness function as {}", s),
        };
    let observer = Observer::spawn(
        &config,
        Box::new(bare::analysis::report_fn),
        CreatureDominanceOrd,
    );
    let evaluator = bare::evaluation::BareEvaluator::spawn(&config, fitness_function);
    (observer, evaluator)
}

// fn prepare_push<C: 'static + Cpu<'static>>(
//     config: &Config,
// ) -> (Observer<push::Creature>, push::evaluation::PushEvaluator<C>) {
//     let fitness_function: FitnessFn<push::Creature, Sketches, Config> =
//         match config.fitness.function.as_str() {
//             // "register_pattern" => Box::new(push::evaluation::register_pattern_ff),
//             // "register_conjunction" => Box::new(push::evaluation::register_conjunction_ff),
//             // "register_entropy" => Box::new(push::evaluation::register_entropy_ff),
//             // "code_coverage" => Box::new(push::evaluation::code_coverage_ff),
//             // "just_novelty" => Box::new(push::evaluation::just_novelty_ff),
//             s => unimplemented!("No such fitness function as {}", s),
//         };
//     let observer: Observer<push::Creature> = Observer::spawn(
//         &config,
//         Box::new(push::analysis::report_fn),
//         CreatureDominanceOrd,
//     );
//     let evaluator = push::evaluation::PushEvaluator::spawn(&config, fitness_function);
//     (observer, evaluator)
// }

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
    let _ = loader::falcon_loader::load_from_path(&mut config.roper, true)
        .expect("Failed to load binary image");
    bare::init_soup(&mut config).expect("Failed to initialize the soup");

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
            // first, crude shot at islands...
            // TODO: factor this out into its own module, so that it works with
            // any job and selection method.
            let num_islands = config.num_islands;
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
                        Tournament::<bare::evaluation::BareEvaluator<C>, bare::Creature>::new(
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
        Selection::Roulette => {
            let (observer, evaluator) = prepare_bare(&config);
            let mut world = Roulette::<
                bare::evaluation::BareEvaluator<C>,
                bare::Creature,
                CreatureDominanceOrd,
            >::new(&config, observer, evaluator, CreatureDominanceOrd);
            while crate::keep_going() {
                world = world.evolve();
            }
        }
        Selection::Metropolis => {
            let (observer, evaluator) = prepare_bare(&config);
            let mut world = Metropolis::<bare::evaluation::BareEvaluator<C>, bare::Creature>::new(
                &config, observer, evaluator,
            );
            while crate::keep_going() {
                world = world.evolve();
            }
        }
        Selection::Lexicase => {
            let fitness_function: FitnessFn<bare::Creature, Sketches, Config> =
                match config.fitness.function.as_str() {
                    "register_pattern" => Box::new(bare::evaluation::register_pattern_ff),
                    "register_conjunction" => Box::new(bare::evaluation::register_conjunction_ff),
                    "code_coverage" => Box::new(bare::evaluation::code_coverage_ff),
                    s => unimplemented!("No such fitness function as {}", s),
                };
            let pier: Arc<Pier<bare::Creature>> = Arc::new(Pier::new(config.num_islands));
            let evaluator = bare::evaluation::BareEvaluator::spawn(&config, fitness_function);
            let observer = Observer::spawn(
                &config,
                Box::new(bare::analysis::lexicase::report_fn),
                CreatureDominanceOrd,
            );
            let mut cases = config
                .roper
                .register_pattern
                .as_ref()
                .map(|rp| {
                    let pattern: RegisterPattern = rp.into();
                    pattern
                        .features()
                        .into_iter()
                        .map(lexi::Task::Reg)
                        .collect::<Vec<lexi::Task>>()
                })
                .expect("No register pattern specified");
            cases.push(lexi::Task::UniqExec(2));
            cases.push(lexi::Task::UniqExec(3));
            log::info!("Register Feature cases: {:#x?}", cases);
            let mut world =
                Lexicase::<lexi::Task, bare::evaluation::BareEvaluator<C>, bare::Creature>::new(
                    &config, observer, evaluator, pier, cases,
                );
            while crate::keep_going() {
                world = world.evolve();
            }
        }
    }
}
