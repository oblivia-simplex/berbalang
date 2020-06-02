use std::cmp::{Ordering, PartialOrd};
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::iter;
use std::sync::Arc;

use rand::{thread_rng, Rng};

use crate::configure::Config;
use crate::evaluator::Evaluate;
use crate::evolution::{Genome, Phenome};
use crate::observer::Observer;

pub struct Tournament<E: Evaluate<P>, P: Phenome + Debug + Send + Clone + Ord + 'static> {
    pub population: BinaryHeap<P>,
    pub config: Arc<Config>,
    pub best: Option<P>,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
}

impl<E: Evaluate<P>, P: Phenome + Genome> Tournament<E, P> {
    pub fn evolve(self) -> Self {
        // destruct the Epoch
        let Self {
            mut population,
            mut best,
            observer,
            evaluator,
            config,
            iteration,
        } = self;

        let tournament_size = config.tournament_size();
        let mut rng = thread_rng();
        let mut combatants = iter::repeat(())
            .take(tournament_size)
            .filter_map(|()| population.pop())
            .map(|mut e| {
                e.set_tag(rng.gen::<u64>());
                e
            })
            .map(|e| evaluator.evaluate(e))
            .map(|e| {
                observer.observe(e.clone());
                e
            })
            .collect::<Vec<P>>();

        combatants.sort_by(|a, b| {
            a.fitness().partial_cmp(&b.fitness()).unwrap_or_else(|| {
                a.scalar_fitness()
                    .partial_cmp(&b.scalar_fitness())
                    .unwrap_or(Ordering::Equal)
            })
        });
        // the best are now at the beginning of the vec

        //log::debug!("combatants' fitnesses: {:?}", combatants.iter().map(|c| c.fitness()).collect::<Vec<_>>());
        best = Self::update_best(best, &combatants[0]);

        // kill one off for every offspring to be produced
        for _ in 0..config.num_offspring() {
            let _ = combatants.pop();
        }

        // replace the combatants that will neither breed nor die
        let bystanders = config.tournament_size() - (config.num_offspring() + 2);
        for _ in 0..bystanders {
            if let Some(c) = combatants.pop() {
                population.push(c);
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
            population.push(other_guy)
        }
        for child in offspring.into_iter() {
            population.push(child)
        }

        // put the epoch back together
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
