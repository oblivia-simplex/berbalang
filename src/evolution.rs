use std::cmp::PartialOrd;

use serde::Serialize;

pub trait Epochal {
    type Observer;
    type Evaluator;

    /// The evolve function turns the crank once on the evolutionary
    /// process.
    fn evolve(self, observer: &Self::Observer, evaluator: &Self::Evaluator) -> Self;
}

/// Implement partial order
pub trait Fitness: PartialOrd + Serialize {}

pub trait Genotype {
    fn crossover(&self, mate: &Self) -> Vec<Self>
    where
        Self: Sized;
    fn mutate(&mut self);
}
