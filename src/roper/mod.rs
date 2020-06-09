use std::cmp::Ordering;

use unicorn::Cpu;

use crate::configure::{Config, Selection};
/// This is where the ROP-evolution-specific code lives.
use crate::{
    emulator::loader,
    evolution::{tournament::Tournament, Phenome},
};
// the runner
use crate::evaluator::Evaluate;
use crate::evolution::metropolis::Metropolis;
use crate::evolution::roulette::Roulette;
use crate::fitness::Pareto;
use crate::observer::Observer;

/// The `creature` module contains the implementation of the `Genome` and `Phenome`
/// traits associated with `roper` mode.
mod creature;
use creature::*;

/// The `analysis` module contains the reporting function passed to the observation
/// window. Population saving, soup dumping, statistical assessment, etc., happens there.
mod analysis;

/// The `evaluation` module contains the various fitness functions, and the construction
/// of the `Evaluator` structure that maps genotype to phenotype, and assigns fitness
/// scores to each member of the population.
mod evaluation;

type Fitness<'a> = Pareto<'a>; //Pareto<'static>;

crate::make_phenome_heap_friendly!(Creature);

fn prepare<C: 'static + Cpu<'static>>(
    config: Config,
) -> (Observer<Creature>, evaluation::Evaluator<C>) {
    let observer = Observer::spawn(&config, Box::new(analysis::report_fn), CreatureDominanceOrd);
    let evaluator =
        evaluation::Evaluator::spawn(&config, Box::new(evaluation::register_pattern_fitness_fn));
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
    init_soup(&mut config.roper).expect("Failed to initialize the soup");

    let (observer, evaluator) = prepare(config.clone());

    match config.selection {
        Selection::Tournament => {
            let mut world =
                Tournament::<evaluation::Evaluator<C>, Creature>::new(config, observer, evaluator);
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
