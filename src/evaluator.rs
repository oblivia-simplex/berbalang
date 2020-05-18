use std::sync::Arc;

use crate::configure::Configure;
use crate::evolution::{FitnessScore, Phenome};

pub type FitnessFn<P: Phenome, C, F: FitnessScore> =
Box<dyn Fn(&P, Arc<C>) -> F + Sync + Send + 'static>;

pub trait Evaluate {
    type Phenotype: Phenome;
    type Params;
    type Fitness: FitnessScore;

    /// We're assuming that the Phenotype contains a binding to
    /// the resulting fitness score, and that this method sets
    /// that score before returning the phenotype.
    ///
    /// NOTE: nothing guarantees that the returned phenotype is
    /// the same one that was passed in. Keep this in mind. This
    /// is to allow the use of asynchronous evaluation pipelines.
    fn evaluate(&self, ob: Self::Phenotype) -> Self::Phenotype;

    fn spawn(
        params: Arc<Self::Params>,
        fitness_fn: FitnessFn<Self::Phenotype, Self::Params, Self::Fitness>,
    ) -> Self;
}
