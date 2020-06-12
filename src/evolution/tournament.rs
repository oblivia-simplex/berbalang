use std::cmp::{Ordering, PartialOrd};
use std::iter;
use std::sync::Arc;

use rand::{thread_rng, Rng};

use crate::configure::Config;
use crate::evaluator::Evaluate;
use crate::evolution::population::trivial_geography::TrivialGeography;
use crate::evolution::{Genome, Phenome};
use crate::observer::Observer;
use crate::util::count_min_sketch::{SeasonalSketch, Sketch};
use rayon::prelude::*;

// consider an island-pier structure

pub struct Tournament<E: Evaluate<P, SeasonalSketch>, P: Phenome + 'static> {
    pub population: TrivialGeography<P>, //BinaryHeap<P>,
    pub config: Arc<Config>,
    pub best: Option<P>,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
}

// TODO factor TrivialGeography to its own module
// maybe create a Population trait.

impl<E: Evaluate<P, SeasonalSketch>, P: Phenome + Genome + 'static> Tournament<E, P> {
    pub fn new(config: Config, observer: Observer<P>, evaluator: E) -> Self
    where
        Self: Sized,
    {
        log::debug!("Initializing population");
        let mut population: TrivialGeography<P> = (0..config.pop_size)
            .into_par_iter()
            .map(|i| {
                log::debug!("creating phenome {}/{}", i, config.pop_size);
                P::random(&config)
            })
            .collect();
        population.set_radius(config.tournament.geographic_radius);
        log::debug!("population initialized");

        Self {
            population,
            config: Arc::new(config),
            best: None,
            iteration: 0,
            observer,
            evaluator,
        }
    }
}

impl<E: Evaluate<P, SeasonalSketch>, P: Phenome + Genome> Tournament<E, P> {
    pub fn evolve(self) -> Self {
        // destruct the Epoch
        let Self {
            mut population,
            mut best,
            observer,
            mut evaluator,
            config,
            iteration,
        } = self;

        let mut rng = thread_rng();
        // let combatants: Vec<P> = iter::repeat(())
        //     .take(tournament_size)
        //     .filter_map(|()| population.pop())
        //     .collect();
        let combatants: Vec<P> =
            population.choose_combatants(config.tournament.tournament_size, &mut rng);

        let mut combatants = evaluator
            .eval_pipeline(combatants.into_iter())
            .into_iter()
            .map(|mut e| {
                e.set_tag(rng.gen::<u64>());
                e
            })
            .map(|e| {
                observer.observe(e.clone());
                e
            })
            .collect::<Vec<P>>();

        combatants.sort_by(|a, b| {
            a.fitness()
                .partial_cmp(&b.fitness())
                .unwrap_or(Ordering::Equal)
        });
        // the best are now at the beginning of the vec

        //log::debug!("combatants' fitnesses: {:?}", combatants.iter().map(|c| c.fitness()).collect::<Vec<_>>());
        best = Self::update_best(best, &combatants[0]);

        // kill one off for every offspring to be produced
        for _ in 0..config.num_offspring() {
            let _ = combatants.pop();
        }

        // replace the combatants that will neither breed nor die
        let bystanders = config.tournament.tournament_size - (config.num_offspring() + 2);
        for _ in 0..bystanders {
            if let Some(c) = combatants.pop() {
                population.insert(c).unwrap();
            }
        }
        // TODO implement breeder, similar to observer, etc?
        //let mother = combatants.pop().unwrap();
        //let father = combatants.pop().unwrap();
        let parents = combatants.iter().collect::<Vec<&P>>();
        let offspring: Vec<P> = iter::repeat(())
            .take(config.num_offspring)
            .map(|()| Genome::mate(&parents, &config))
            .collect::<Vec<_>>();

        // return everyone to the population
        //population.push(mother);
        //population.push(father);
        for other_guy in combatants.into_iter() {
            population.insert(other_guy).unwrap()
        }
        for child in offspring.into_iter() {
            population.insert(child).unwrap()
        }

        // put the epoch back together
        crate::increment_epoch_counter();
        Self {
            population,
            config,
            best,
            iteration: iteration + 1,
            observer,
            evaluator,
        }
    }

    pub fn update_best(best: Option<P>, champ: &P) -> Option<P> {
        match best {
            Some(ref best) if champ.scalar_fitness() < best.scalar_fitness() => {
                log::info!("new champ with fitness {:?}:\n{:?}", champ.fitness(), champ);
                Some(champ.clone())
            }
            None => {
                log::info!("new champ with fitness {:?}\n{:?}", champ.fitness(), champ);
                Some(champ.clone())
            }
            _ => best,
        }
    }

    // pub fn target_reached(&self, target: &<P as Phenome>::Fitness) -> bool {
    //     self.best
    //         .as_ref()
    //         .and_then(|b| b.fitness())
    //         .map_or(false, |f| f[0] <= target)
    // }
}
