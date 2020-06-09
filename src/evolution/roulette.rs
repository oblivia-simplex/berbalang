use std::iter;
use std::sync::Arc;

use rand::distributions::WeightedIndex;
use rand::thread_rng;
use rand_distr::Distribution;

use non_dominated_sort::{non_dominated_sort, DominanceOrd};

use crate::configure::Config;
use crate::evaluator::Evaluate;
use crate::evolution::{Genome, Phenome};
use crate::increment_epoch_counter;
use crate::observer::Observer;

pub struct Roulette<E: Evaluate<P>, P: Phenome + Genome + 'static, D: DominanceOrd<P>> {
    pub population: Vec<P>,
    pub config: Arc<Config>,
    pub observer: Observer<P>,
    pub evaluator: E,
    pub iteration: usize,
    pub dominance_order: D,
}

impl<E: Evaluate<P>, P: Phenome + Genome + 'static, D: DominanceOrd<P>> Roulette<E, P, D> {
    pub fn new(config: Config, observer: Observer<P>, evaluator: E, dominance_order: D) -> Self {
        let population = iter::repeat(())
            .map(|()| P::random(&config))
            .take(config.population_size())
            .collect();

        Self {
            population,
            config: Arc::new(config),
            iteration: 0,
            observer,
            evaluator,
            dominance_order,
        }
    }
}

impl<E: Evaluate<P>, P: Phenome + Genome + Sized, D: DominanceOrd<P>> Roulette<E, P, D> {
    // pub fn new(config: Config) -> Self {
    //     Self {
    //         population
    //     }
    // }
    pub fn evolve(self) -> Self {
        let Self {
            population,
            observer,
            mut evaluator,
            config,
            iteration,
            dominance_order,
        } = self;

        // measure and assign fitness scores to entire population
        let mut population = evaluator.eval_pipeline(population.into_iter());
        // we're going to need to clone the population to send to the observer, anyway
        // so we might as well do that now. this lets us get around certain awkward
        // lifetime constraints imposed on us by the `Front` struct.
        let cloned_population = population.clone();

        // we want the odds of breeding to be relative to the front on which the
        // individuals occur. the lower the front, the better the chances.
        let mut cur_weight = 1.0;
        let mut indices_weights: Vec<(usize, f64)> = Vec::new();
        let mut elite_fronts: Vec<Vec<usize>> = Vec::new();
        {
            let mut front = non_dominated_sort(&cloned_population, &dominance_order);
            elite_fronts.push(front.current_front_indices().to_vec());
            while !front.is_empty() {
                front.current_front_indices().iter().for_each(|i| {
                    population[*i].set_front(front.rank());
                    indices_weights.push((*i, cur_weight));
                });

                front = front.next_front();
                cur_weight *= config.roulette.weight_decay;
            }
        }
        indices_weights.sort_by_key(|p| p.0);
        let (indices, weights): (Vec<usize>, Vec<f64>) = indices_weights.iter().cloned().unzip();
        // Now, create a weighted random distribution where the odds of an index being
        // drawn are proportionate to the rank of the front on which the creature that
        // index points to appears.
        let dist = WeightedIndex::new(&weights).expect("failed to create weighted index");

        // send the old copy of the population to the observer
        // these have each been marked with the rank of their front
        population.into_iter().for_each(|p| {
            observer.observe(p);
        });

        let mut new_population: Vec<P> = Vec::new();

        // Transfer the elites -- the members of the 0th front -- to the
        // new population as-is.
        for idxs in &elite_fronts {
            for idx in idxs.iter() {
                new_population.push(cloned_population[*idx].clone());
            }
        }

        let mut rng = thread_rng();
        while new_population.len() < config.pop_size {
            let parents: Vec<&P> = iter::repeat(())
                .take(config.num_parents)
                .map(|()| indices[dist.sample(&mut rng)])
                .map(|i| &cloned_population[i])
                .collect::<Vec<&P>>();
            let child: P = Genome::mate(&parents, &config);
            new_population.push(child)
        }

        increment_epoch_counter();

        Self {
            population: new_population,
            config,
            observer,
            evaluator,
            iteration: iteration + 1,
            dominance_order,
        }
    }
}
