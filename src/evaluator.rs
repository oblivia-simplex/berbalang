use std::sync::Arc;

use crate::evolution::Phenome;

pub type FitnessFn<P, C> = Box<dyn Fn(P, Arc<C>) -> P + Sync + Send + 'static>;

// TODO: Consider replicating design seen in observer
// using a generic struct instead of a trait

pub trait Evaluate<P: Phenome> {
    type Params;

    /// We're assuming that the Phenotype contains a binding to
    /// the resulting fitness score, and that this method sets
    /// that score before returning the phenotype.
    ///
    /// NOTE: nothing guarantees that the returned phenotype is
    /// the same one that was passed in. Keep this in mind. This
    /// is to allow the use of asynchronous evaluation pipelines.
    fn evaluate(&self, ob: P) -> P;

    fn eval_pipeline<I: 'static + Iterator<Item = P> + Send>(&self, inbound: I) -> Vec<P>;

    fn spawn(params: &Self::Params, fitness_fn: FitnessFn<P, Self::Params>) -> Self;
}
