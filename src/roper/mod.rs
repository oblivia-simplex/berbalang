use std::cmp::Ordering;

use non_dominated_sort::DominanceOrd;
use serde::{Deserialize, Serialize};
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
use crate::ontogenesis::{Develop, FitnessFn};
use crate::roper::evaluation::lexi;
use crate::util::count_min_sketch::CountMinSketch;

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
mod push;

type Fitness<'a> = Weighted<'a>; //Pareto<'static>;

fn prepare<'a, C: 'static + Cpu<'static>>(
    config: Config,
) -> (Observer<Creature<u64>>, evaluation::Evaluator<C>) {
    let fitness_function: FitnessFn<Creature<u64>, CountMinSketch, Config> =
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

    let (observer, evaluator) = prepare(config.clone());
    let pier = Pier::spawn(&config);

    match config.selection {
        Selection::Tournament => {
            let mut world = Tournament::<evaluation::Evaluator<C>, Creature<u64>>::new(
                config, observer, evaluator, pier,
            );
            let mut counter = 0;
            while world.observer.keep_going() {
                world = world.evolve();
                counter += 1;
                if counter % 0x1000 == 0 {
                    log::info!("best: {:#x?}", world.best);
                }
            }
        }
        Selection::Roulette => {
            let mut world =
                Roulette::<evaluation::Evaluator<C>, Creature<u64>, CreatureDominanceOrd>::new(
                    config,
                    observer,
                    evaluator,
                    CreatureDominanceOrd,
                );
            while world.observer.keep_going() {
                world = world.evolve();
            }
        }
        Selection::Metropolis => {
            let mut world = Metropolis::<evaluation::Evaluator<C>, Creature<u64>>::new(
                config, observer, evaluator,
            );
            while world.observer.keep_going() {
                world = world.evolve();
            }
        }
        Selection::Lexicase => {
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
                config, observer, evaluator, pier, cases,
            );
            while world.observer.keep_going() {
                world = world.evolve();
            }
        }
    }
}
