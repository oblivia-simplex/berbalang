use std::collections::BinaryHeap;
use std::hash::Hash;
use std::sync::Arc;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::configure::Config;
use crate::evaluator::Evaluate;
use crate::evolution::population::pier::Pier;
use crate::evolution::population::shuffling_heap::ShufflingHeap;
use crate::evolution::{Genome, Phenome};
use crate::observer::Observer;
use crate::util::count_min_sketch::CountMinSketch;

pub struct Lexicase<Q: Hash, E: Evaluate<P, CountMinSketch, Q>, P: Phenome + 'static> {
    pub population: ShufflingHeap<P>,
    pub problems: ShufflingHeap<Q>,
    pub config: Arc<Config>,
    pub best: Option<P>,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
    pub pier: Pier<P>,
}

impl<Q: Hash, E: Evaluate<P, CountMinSketch, Q>, P: Phenome + Genome + 'static> Lexicase<Q, E, P> {
    pub fn new(
        config: Config,
        observer: Observer<P>,
        evaluator: E,
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
                P::random(&config, i)
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
            observer,
            evaluator,
            pier,
        } = self;

        let mut parents: Vec<P> = Vec::new();
        let mut dead: Vec<P> = Vec::new();
        let mut survivors: Vec<P> = Vec::new();

        while let Some(guy) = population.pop() {
            while let Some(problem) = problems.pop() {
                // TODO: find a way to parallelize this later
            }
        }

        Self {
            population,
            problems,
            config,
            best,
            iteration,
            observer,
            evaluator,
            pier,
        }
    }
}
