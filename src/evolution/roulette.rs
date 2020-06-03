use std::cmp::{Ordering, PartialOrd};

use non_dominated_sort::{non_dominated_sort, DominanceOrd};
use rand::{thread_rng, Rng};
use rand_distr::Distribution;

use crate::configure::Config;
use crate::evaluator::Evaluate;
use crate::evolution::{Genome, Phenome};
use crate::observer::Observer;
use rand::distributions::WeightedIndex;
use std::fmt::Debug;
use std::iter;
use std::sync::Arc;

pub struct Roulette<E: Evaluate<P>, P: Phenome + Genome + 'static, D: DominanceOrd<T = P>> {
    pub population: Vec<P>,
    pub config: Arc<Config>,
    pub best: Option<P>,
    pub observer: Observer<P>,
    pub evaluator: E,
    pub iteration: usize,
    pub dominance_order: D,
}

impl<E: Evaluate<P>, P: Phenome + Genome + 'static, D: DominanceOrd<T = P>> Roulette<E, P, D> {
    pub fn new(config: Config, observer: Observer<P>, evaluator: E, dominance_order: D) -> Self {
        let population = iter::repeat(())
            .map(|()| P::random(&config))
            .take(config.population_size())
            .collect();

        Self {
            population,
            config: Arc::new(config),
            best: None,
            iteration: 0,
            observer,
            evaluator,
            dominance_order,
        }
    }
}

impl<E: Evaluate<P>, P: Phenome + Genome + Sized, D: DominanceOrd<T = P>> Roulette<E, P, D> {
    // pub fn new(config: Config) -> Self {
    //     Self {
    //         population
    //     }
    // }
    pub fn evolve(self) -> Self {
        let Self {
            population,
            mut best,
            observer,
            evaluator,
            config,
            iteration,
            dominance_order,
        } = self;

        // measure and assign fitness scores to entire population
        let population = evaluator.eval_pipeline(population.into_iter());
        population.iter().for_each(|p| {
            observer.observe(p.clone());
        });

        let mut front = non_dominated_sort(&population, &dominance_order);
        // we want the odds of breeding to be relative to the front on which the
        // individuals occur. the lower the front, the better the chances.
        let mut cur_weight = 1.0;
        let mut indices_weights: Vec<(usize, f64)> = Vec::new();
        while !front.is_empty() {
            //log::info!("Weighting front {} at {}", front.rank(), cur_weight);
            front.current_front_indices().iter().for_each(|i| {
                indices_weights.push((*i, cur_weight));
            });

            // update the best, if applicable
            if front.rank() == 0 {
                log::info!(
                    "iteration #{}, front 0 contains {} individuals",
                    iteration,
                    front.len()
                );
                let champ = front.iter().fold(None, |champ: Option<&P>, (p, _)| {
                    if let Some(champ) = champ {
                        if p.fitness().as_ref().unwrap() < champ.fitness().as_ref().unwrap() {
                            Some(p)
                        } else {
                            Some(champ)
                        }
                    } else {
                        Some(p)
                    }
                });

                match (champ, &best) {
                    (Some(c), Some(ref b)) if c.scalar_fitness() < b.scalar_fitness() => {
                        log::info!(
                            "updating best to {:?}\nold fitness: {:?}, new fitness: {:?}",
                            c,
                            b.fitness(),
                            c.fitness()
                        );
                        best = champ.cloned()
                    }
                    (Some(c), None) => {
                        log::info!("updating best to {:?}\nfitness: {:?}", c, c.fitness());
                        best = champ.cloned()
                    }
                    (_, _) => {}
                }
            }
            front = front.next_front();
            cur_weight = cur_weight.tanh();
        }
        indices_weights.sort_by_key(|p| p.0);
        let (indices, weights): (Vec<usize>, Vec<f64>) = indices_weights.iter().cloned().unzip();
        let dist = WeightedIndex::new(&weights).expect("failed to create weighted index");
        let mut rng = thread_rng();

        let mut new_population: Vec<P> = Vec::new();
        // keep the best
        // maybe condition this on an `elitism` parameter
        if let Some(ref best) = best {
            new_population.push(best.clone())
        }
        while new_population.len() < config.pop_size {
            let parents: Vec<&P> = iter::repeat(())
                .take(config.num_parents)
                .map(|()| indices[dist.sample(&mut rng)])
                .map(|i| &population[i])
                .collect::<Vec<&P>>();
            let child: P = Genome::mate(&parents, &config);
            new_population.push(child)
        }

        log::info!("Roulette iteration {} complete", iteration);
        Self {
            population: new_population,
            config,
            best,
            observer,
            evaluator,
            iteration: iteration + 1,
            dominance_order,
        }
    }
}
