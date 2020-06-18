use std::sync::Arc;
use std::thread::spawn;

use non_dominated_sort::DominanceOrd;
use rand::Rng;
use unicorn::Cpu;

use creature::*;

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
use crate::roper::evaluation::{lexi, Sketches};
use crate::util::random::hash_seed_rng;

/// The `analysis` module contains the reporting function passed to the observation
/// window. Population saving, soup dumping, statistical assessment, etc., happens there.
mod analysis;
/// The `creature` module contains the implementation of the `Genome` and `Phenome`
/// traits associated with `roper` mode.
mod creature;

/// The `evaluation` module contains the various fitness functions, and the construction
/// of the `Evaluator` structure that maps genotype to phenotype, and assigns fitness
/// scores to each member of the population.
mod evaluation;

/// A ROPER-specific implementation of Spector's PUSH VM.
#[allow(dead_code)]
mod push;

type Fitness<'a> = Weighted<'a>; //Pareto<'static>;

fn prepare<'a, C: 'static + Cpu<'static>>(
    config: &Config,
) -> (Observer<Creature<u64>>, evaluation::Evaluator<C>) {
    let fitness_function: FitnessFn<Creature<u64>, Sketches, Config> =
        match config.fitness.function.as_str() {
            "register_pattern" => Box::new(evaluation::register_pattern_ff),
            "register_conjunction" => Box::new(evaluation::register_conjunction_ff),
            "code_coverage" => Box::new(evaluation::code_coverage_ff),
            s => unimplemented!("No such fitness function as {}", s),
        };
    let observer = Observer::spawn(&config, Box::new(analysis::report_fn), CreatureDominanceOrd);
    let evaluator = evaluation::Evaluator::spawn(&config, fitness_function);
    (observer, evaluator)
}

pub struct CreatureDominanceOrd;

impl DominanceOrd<Creature<u64>> for CreatureDominanceOrd {
    fn dominance_ord(&self, a: &Creature<u64>, b: &Creature<u64>) -> std::cmp::Ordering {
        a.fitness()
            .partial_cmp(&b.fitness())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl DominanceOrd<&Creature<u64>> for CreatureDominanceOrd {}

pub fn run<C: 'static + Cpu<'static>>(mut config: Config) {
    let _ = loader::load_from_path(
        &config.roper.binary_path,
        0x1000,
        config.roper.arch,
        config.roper.mode,
    )
    .expect("Failed to load binary image");
    init_soup(&mut config).expect("Failed to initialize the soup");

    match config.selection {
        Selection::Tournament => {
            // first, crude shot at islands...
            // TODO: factor this out into its own module, so that it works with
            // any job and selection method.
            let num_islands = config.num_islands;
            let pier: Arc<Pier<Creature<u64>>> = Arc::new(Pier::new(config.num_islands));
            let mut handles = Vec::new();
            let mut rng = hash_seed_rng(&config.random_seed);
            for i in 0..num_islands {
                let mut config = config.clone();
                config.island_identifier = i;
                config.set_data_directory();
                config.random_seed = rng.gen::<u64>();
                let (observer, evaluator) = prepare(&config);
                let pier = pier.clone();
                let h = spawn(move || {
                    let mut world = Tournament::<evaluation::Evaluator<C>, Creature<u64>>::new(
                        &config, observer, evaluator, pier,
                    );
                    while world.observer.keep_going() {
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
            let (observer, evaluator) = prepare(&config);
            let mut world =
                Roulette::<evaluation::Evaluator<C>, Creature<u64>, CreatureDominanceOrd>::new(
                    &config,
                    observer,
                    evaluator,
                    CreatureDominanceOrd,
                );
            while world.observer.keep_going() {
                world = world.evolve();
            }
        }
        Selection::Metropolis => {
            let (observer, evaluator) = prepare(&config);
            let mut world = Metropolis::<evaluation::Evaluator<C>, Creature<u64>>::new(
                &config, observer, evaluator,
            );
            while world.observer.keep_going() {
                world = world.evolve();
            }
        }
        Selection::Lexicase => {
            let fitness_function: FitnessFn<Creature<u64>, Sketches, Config> =
                match config.fitness.function.as_str() {
                    "register_pattern" => Box::new(evaluation::register_pattern_ff),
                    "register_conjunction" => Box::new(evaluation::register_conjunction_ff),
                    "code_coverage" => Box::new(evaluation::code_coverage_ff),
                    s => unimplemented!("No such fitness function as {}", s),
                };
            let pier: Arc<Pier<Creature<u64>>> = Arc::new(Pier::new(config.num_islands));
            let evaluator = evaluation::Evaluator::spawn(&config, fitness_function);
            let observer = Observer::spawn(
                &config,
                Box::new(analysis::lexicase::report_fn),
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
            let mut world = Lexicase::<lexi::Task, evaluation::Evaluator<C>, Creature<u64>>::new(
                &config, observer, evaluator, pier, cases,
            );
            while world.observer.keep_going() {
                world = world.evolve();
            }
        }
    }
}
