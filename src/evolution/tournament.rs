use std::cmp::{Ordering, PartialOrd};
use std::iter;
use std::sync::Arc;

use rand::Rng;
use rayon::prelude::*;

use crate::configure::Config;
use crate::evolution::population::pier::Pier;
use crate::evolution::population::trivial_geography::TrivialGeography;
use crate::evolution::{Genome, Phenome};
use crate::observer::Observer;
use crate::ontogenesis::Develop;
use crate::util::count_min_sketch::CountMinSketch;
use crate::util::random::hash_seed_rng;

type SketchType = CountMinSketch;
// consider an island-pier structure

pub struct Tournament<E: Develop<P, SketchType>, P: Phenome + 'static> {
    pub population: TrivialGeography<P>,
    //BinaryHeap<P>,
    pub config: Arc<Config>,
    pub best: Option<P>,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
    pub pier: Pier<P>,
}

// TODO factor TrivialGeography to its own module
// maybe create a Population trait.

impl<E: Develop<P, SketchType>, P: Phenome + Genome + 'static> Tournament<E, P> {
    pub fn new(config: Config, observer: Observer<P>, evaluator: E, pier: Pier<P>) -> Self
    where
        Self: Sized,
    {
        log::debug!("Initializing population");
        let mut population: TrivialGeography<P> = (0..config.pop_size)
            .into_par_iter()
            .map(|i| {
                log::debug!("creating phenome {}/{}", i, config.pop_size);
                P::random(&config, i)
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
            pier,
        }
    }

    pub fn evolve(self) -> Self {
        // destruct the Epoch
        let Self {
            mut population,
            best,
            observer,
            mut evaluator,
            config,
            iteration,
            pier,
        } = self;

        let mut rng = hash_seed_rng(&population);

        let combatants: Vec<P> =
            population.choose_combatants(config.tournament.tournament_size, &mut rng);

        let mut combatants = evaluator
            .development_pipeline(combatants.into_iter())
            .into_iter()
            .map(|p| evaluator.apply_fitness_function(p))
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
        let best = Self::update_best(best, &combatants[0]);

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

        let mut survivors = combatants; //combatants.into_iter().collect::<Vec<P>>();
                                        // there should be some small chance that the parents migrate

        // A generation should be considered to have elapsed once
        // `pop_size` offspring have been spawned.
        if iteration % (config.pop_size / config.num_offspring) == 0 {
            crate::increment_epoch_counter();
            if rng.gen_range(0.0, 1.0) < config.tournament.migration_rate {
                log::info!("Attempting migration...");
                if let Some(immigrant) = pier.disembark() {
                    log::info!("Found immigrant on pier");
                    let emigrant = survivors.pop().unwrap();
                    if let Err(_emigrant) = pier.embark(emigrant) {
                        log::error!("emigration failure, do something!");
                    }
                    survivors.push(immigrant);
                }
            }
        }

        let parents = survivors
            .iter()
            .take(config.num_parents)
            .collect::<Vec<&P>>();

        let offspring: Vec<P> = iter::repeat(())
            .take(config.num_offspring)
            .map(|()| Genome::mate(&parents, &config))
            .collect::<Vec<_>>();

        // return everyone to the population
        for other_guy in survivors.into_iter() {
            population.insert(other_guy).unwrap()
        }
        for child in offspring.into_iter() {
            population.insert(child).unwrap()
        }

        Self {
            population,
            config,
            best: Some(best),
            iteration: iteration + 1,
            observer,
            evaluator,
            pier,
        }
    }

    pub fn update_best(best: Option<P>, champ: &P) -> P {
        match best {
            Some(ref best) if champ.scalar_fitness() < best.scalar_fitness() => {
                log::info!("new champ with fitness {:?}:\n{:?}", champ.fitness(), champ);
                champ.clone()
            }
            None => {
                log::info!("new champ with fitness {:?}\n{:?}", champ.fitness(), champ);
                champ.clone()
            }
            best => best.unwrap(),
        }
    }

    // pub fn target_reached(&self, target: &<P as Phenome>::Fitness) -> bool {
    //     self.best
    //         .as_ref()
    //         .and_then(|b| b.fitness())
    //         .map_or(false, |f| f[0] <= target)
    // }
}
