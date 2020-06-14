use std::cmp::Ordering;

use unicorn::Cpu;

use creature::*;

use crate::configure::{Config, Selection};
/// This is where the ROP-evolution-specific code lives.
use crate::{
    emulator::loader,
    evolution::{tournament::Tournament, Phenome},
};
// the runner
use crate::evaluator::{Evaluate, FitnessFn};
use crate::evolution::metropolis::Metropolis;
use crate::evolution::pareto_roulette::Roulette;
use crate::evolution::population::pier::Pier;
use crate::fitness::Weighted;
use crate::observer::Observer;
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

type Fitness<'a> = Weighted<'a>; //Pareto<'static>;

crate::make_phenome_heap_friendly!(Creature);

fn prepare<C: 'static + Cpu<'static>>(
    config: Config,
) -> (Observer<Creature>, evaluation::Evaluator<C>) {
    let fitness_function: FitnessFn<Creature, CountMinSketch, Config> =
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

crate::impl_dominance_ord_for_phenome!(Creature, CreatureDominanceOrd);

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

    match config.selection {
        Selection::Tournament => {
            let pier = Pier::spawn(&config);
            let mut world = Tournament::<evaluation::Evaluator<C>, Creature>::new(
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
                Roulette::<evaluation::Evaluator<C>, Creature, CreatureDominanceOrd>::new(
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
            let mut world =
                Metropolis::<evaluation::Evaluator<C>, Creature>::new(config, observer, evaluator);
            while world.observer.keep_going() {
                world = world.evolve();
            }
        }
    }
}
