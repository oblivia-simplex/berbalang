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
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
    pub pier: Arc<Pier<P>>,
}

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
            observer,
            mut evaluator,
            config,
            iteration,
            pier,
        } = self;
        log::debug!(
            "population size in island {}: {}",
            config.island_id,
            population.len()
        );

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

        // kill one off for every offspring to be produced
        for _ in 0..config.tournament.num_offspring {
            let _ = combatants.pop();
        }

        let mut survivors = combatants;

        // A generation should be considered to have elapsed once
        // `pop_size` offspring have been spawned.
        // For now, only Island 0 can increment the epoch. We can weigh the
        // pros and cons of letting each island have its own epoch, later.
        if config.island_id == 0
            && iteration > 0
            && iteration % (config.pop_size / config.tournament.num_offspring) == 0
        {
            crate::increment_epoch_counter();
            log::info!(
                "New global epoch. Island #{} epoch is {}",
                config.island_id,
                Self::island_epoch(iteration, &config)
            );
        }
        // NOTE: migration relies on tournaments being at least 1 larger than
        // the number of parents plus the number of children
        if survivors.len() > config.tournament.num_parents {
            let mut migrated = false;
            if rng.gen_range(0.0, 1.0) < config.tournament.migration_rate {
                log::debug!("Attempting migration...");
                let emigrant = survivors.pop().unwrap();
                if let Err(emigrant) = pier.embark(emigrant) {
                    log::debug!("Pier full, returning emigrant to population");
                    survivors.push(emigrant);
                } else {
                    migrated = true;
                }
            }
            if !migrated {
                if let Some(immigrant) = pier.disembark() {
                    log::debug!(
                        "{} has arrived from the pier of island {}",
                        immigrant.name(),
                        config.island_id
                    );
                    survivors.push(immigrant);
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
            iteration: iteration + 1,
            observer,
            evaluator,
            pier,
        }
    }

    fn island_epoch(iteration: usize, config: &Config) -> usize {
        iteration / (config.pop_size / config.tournament.num_offspring)
    }
}
