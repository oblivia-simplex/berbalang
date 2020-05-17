use std::cmp::PartialOrd;

use serde::Serialize;

use crate::configure::Configure;
use crate::evaluator::Evaluate;
use crate::observer::Observe;

pub trait Epochal {
    type Observer;
    type Evaluator;

    /// The evolve function turns the crank once on the evolutionary
    /// process.
    fn evolve(self) -> Self;
}

#[derive(Debug)]
pub struct Epoch<O: Observe, E: Evaluate, G: Genome, C: Configure> {
    pub population: Vec<G>,
    pub config: C,
    pub best: Option<G>,
    pub iteration: usize,
    pub observer: O,
    pub evaluator: E,
}

/// Implement partial order
pub trait FitnessScore: PartialOrd + Serialize {}

pub trait Genome {
    fn crossover(&self, mate: &Self) -> Vec<Self>
    where
        Self: Sized;
    fn mutate(&mut self);
}

pub trait Phenome {
    type Fitness;

    /// This method is intended for reporting, not measuring, fitness.
    fn fitness(&self) -> Option<Self::Fitness>;
}
