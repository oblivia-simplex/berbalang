//! Implementation of the Lexicase selection algorithm, as described by Helmuth and Spector.
//!
//! Note that this is currently somewhat tailored to cases where we only need to develop (execute)
//! the genotype _once_, thereby acquiring a phenotype that we can evaluate against a sequence of
//! fitness cases. This is what we need, for instance, for ROPER's "register_pattern" task. It
//! shouldn't be hard to generalize the algorithm beyond that.
//!
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::configure::Config;
use crate::evolution::population::pier::Pier;
use crate::evolution::population::shuffling_heap::ShufflingHeap;
use crate::evolution::{Genome, Phenome};
use crate::increment_epoch_counter;
use crate::observer::Observer;
use crate::ontogenesis::Develop;
use crate::util::count_min_sketch::CountMinSketch;
use crate::util::random::{hash_seed, hash_seed_rng};

pub struct Lexicase<Q: Hash + Debug, E: Develop<P, CountMinSketch>, P: Phenome + 'static> {
    pub population: ShufflingHeap<P>,
    pub problems: ShufflingHeap<Q>,
    pub config: Arc<Config>,
    pub best: Option<P>,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
    pub pier: Pier<P>,
}

impl<
        Q: Hash + Debug,
        E: Develop<P, CountMinSketch>,
        P: Phenome<Problem = Q> + Genome + 'static,
    > Lexicase<Q, E, P>
{
    pub fn new(
        config: Config,
        observer: Observer<P>,
        mut evaluator: E,
        pier: Pier<P>,
        problems: Vec<Q>,
    ) -> Self
    where
        Self: Sized,
    {
        log::debug!("Initializing population");
        let population: ShufflingHeap<P> = (0..config.pop_size)
            .map(|i| {
                log::debug!("creating phenome {}/{}", i, config.pop_size);
                let phenome = P::random(&config, i);
                log::debug!("Developing phenome {}/{}", i, config.pop_size);
                let phenome = evaluator.develop(phenome);
                observer.observe(phenome.clone());
                phenome
            })
            .collect();
        log::debug!("population initialized");

        let problems: ShufflingHeap<Q> = problems.into_iter().collect();

        Self {
            population,
            problems,
            config: Arc::new(config),
            best: None,
            iteration: 0,
            observer,
            evaluator,
            pier,
        }
    }

    pub fn evolve(self) -> Self {
        let Self {
            mut population,
            mut problems,
            config,
            best,
            iteration,
            mut observer,
            evaluator,
            pier,
        } = self;

        let mut rng = hash_seed_rng(&(iteration as u64 ^ config.random_seed));

        // The idea here is to go through the problems in random order, and incrementally
        // filter out any members of the population that fail to optimally solve each problem.

        let mut next_population = ShufflingHeap::new(&rng.gen::<u64>());
        let mut next_problems = ShufflingHeap::new(&rng.gen::<u64>());
        while population.len() > config.num_parents {
            while problems.len() > 0 {
                if let Some(problem) = problems.pop() {
                    // In most cases, the developed phenotypes won't actually need re-development,
                    // and so `development_pipeline` will act like a simple pass-through.
                    let (mut fail, pass): (ShufflingHeap<P>, ShufflingHeap<P>) = evaluator
                        .development_pipeline(population.into_iter())
                        .into_iter()
                        // Here, we split the population into those that pass the test, and those
                        // that fail it.
                        .partition(|mut creature| creature.fails(&problem));
                    log::debug!(
                        "iteration {}, problem {:?}, # fail: {}, # pass: {}",
                        iteration,
                        problem,
                        fail.len(),
                        pass.len()
                    );
                    next_problems.push(problem);

                    population = pass;
                    if problems.len() == 0 && population.len() > 0 {
                        // if we just popped the last problem
                        // and we have some creatures who have passed,
                        // then the evolutionary process is complete!
                        log::info!("Solution(s) found!");
                        while let Some(champion) = population.pop() {
                            log::info!("{:?}", champion);
                            // TODO: Log this! have the observer handle it.
                        }
                        observer.stop_evolution();
                    }

                    while population.len() < config.num_parents {
                        population.push(fail.pop().unwrap());
                    }
                    next_population.extend(fail.into_iter());
                    if population.len() == config.num_parents {
                        break;
                    }
                }
            }
        }
        // TODO: I don't really know the best way to pass everything off
        // to the observer. Cloning the entire population each generation
        // feels excessive. Is excessive. I could just clone and observe the
        // offspring, but then they're unevaluated. But what if I evaluate
        // each offspring as it's sprung?

        // let next_population = next_population
        //     .into_iter()
        //     .map(|p| observer.observe(p.clone()))
        //     .collect::<ShufflingHeap<P>>();

        if observer.keep_going() {
            debug_assert_eq!(
                population.len(),
                config.num_parents,
                "not enough left in the population to breed"
            );

            let mut parents = Vec::new();
            for _ in 0..config.num_parents {
                let p = population.pop().unwrap();
                // Here's what we'll do: we'll restrict our observation to the individuals
                // that are eventually selected as parents. This feels like an okay compromise.
                // But it needs to be noted in any experimental reports. Can we filter our
                // tournament, etc., observations in a similar fashion?
                // NOTE: we'll observe the dead, too. No need to clone them.
                // Maybe the observer should hold onto some validation/testing data!
                // That would be the best way to do things. We could plot progress in
                // terms of *that* data.
                //observer.observe(p.clone());
                parents.push(p)
            }
            for _ in 0..config.num_offspring {
                let parents = parents.iter().collect::<Vec<&P>>();
                let offspring = Genome::mate(&parents, &config);
                // I could evaluate and observe the offspring here, as they're generated.
                // Then add a pass for the initial seed population.
                // But here, the observer doesn't see the case evaluations!

                // NOTE: Non-elitist replacement
                // Should we observe the dead?
                //observer.observe(next_population.pop());
                let _dead = next_population.pop();
                let offspring = evaluator.develop(offspring);
                observer.observe(offspring.clone());
                next_population.push(offspring);
            }
            // A generation should be considered to have elapsed once
            // `pop_size` offspring have been spawned.
            if iteration % (config.pop_size / config.num_offspring) == 0 {
                crate::increment_epoch_counter();
                if rng.gen_range(0.0, 1.0) < config.tournament.migration_rate {
                    log::info!("Attempting migration...");
                    if let Some(immigrant) = pier.disembark() {
                        log::info!("Found immigrant on pier");
                        let emigrant = next_population.pop().unwrap();
                        if let Err(_emigrant) = pier.embark(emigrant) {
                            log::error!("emigration failure, do something!");
                        }
                        next_population.push(immigrant);
                    }
                }
            }
        }

        Self {
            population: next_population,
            problems: next_problems,
            config,
            best,
            iteration: iteration + 1,
            observer,
            evaluator,
            pier,
        }
    }
}
