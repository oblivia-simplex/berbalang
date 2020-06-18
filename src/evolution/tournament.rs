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
use crate::util::random::hash_seed_rng;

pub struct Tournament<E: Develop<P>, P: Phenome + 'static> {
    pub population: TrivialGeography<P>,
    pub config: Config,
    pub best: Option<P>,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
    pub pier: Arc<Pier<P>>,
}

// TODO factor TrivialGeography to its own module
// maybe create a Population trait.

impl<E: Develop<P>, P: Phenome + Genome + 'static> Tournament<E, P> {
    pub fn new(config: &Config, observer: Observer<P>, evaluator: E, pier: Arc<Pier<P>>) -> Self
    where
        Self: Sized,
    {
        let config = config.clone();
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
            config,
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
        for _ in 0..config.tournament.num_offspring {
            let _ = combatants.pop();
        }

        let mut survivors = combatants; //combatants.into_iter().collect::<Vec<P>>();
                                        // there should be some small chance that the parents migrate

        // A generation should be considered to have elapsed once
        // `pop_size` offspring have been spawned.
        if iteration % (config.pop_size / config.tournament.num_offspring) == 0 {
            crate::increment_epoch_counter();
            // NOTE: migration relies on tournaments being at least 1 larger than
            // the number of parents plus the number of children
            if survivors.len() > config.tournament.num_parents {
                if rng.gen_range(0.0, 1.0) < config.tournament.migration_rate {
                    log::info!("Attempting migration...");
                    let emigrant = survivors.pop().unwrap();
                    if let Err(emigrant) = pier.embark(emigrant) {
                        log::debug!("Pier full, returning emigrant to population");
                        survivors.push(emigrant);
                    }
                }
                if rng.gen_range(0.0, 1.0) < config.tournament.migration_rate {
                    if let Some(immigrant) = pier.disembark() {
                        log::info!("Found immigrant on pier");
                        survivors.push(immigrant);
                    }
                }
            }
        }

        debug_assert!(survivors.len() >= config.tournament.num_parents);

        let parents = survivors
            .iter_mut()
            .take(config.tournament.num_parents)
            .map(|p| {
                p.incr_num_offspring(config.tournament.num_offspring);
                &*p
            })
            .collect::<Vec<&P>>();

        // TODO: Experimental: tuning mutation rate by soup size
        let offspring: Vec<P> = iter::repeat(())
            .take(config.tournament.num_offspring)
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
