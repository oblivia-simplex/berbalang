use std::sync::Arc;

use crate::evolution::Phenome;

pub type FitnessFn<Pheno, State, Conf> =
    Box<dyn Fn(Pheno, &mut State, Arc<Conf>) -> Pheno + Sync + Send + 'static>;

// TODO: Consider replicating design seen in observer
// using a generic struct instead of a trait

pub trait Evaluate<P: Phenome> {
    type Params;
    type State;

    /// We're assuming that the Phenotype contains a binding to
    /// the resulting fitness score, and that this method sets
    /// that score before returning the phenotype.
    ///
    /// NOTE: nothing guarantees that the returned phenotype is
    /// the same one that was passed in. Keep this in mind. This
    /// is to allow the use of asynchronous evaluation pipelines.
    fn evaluate(&mut self, ob: P) -> P;

    fn eval_pipeline<I: 'static + Iterator<Item = P> + Send>(&mut self, inbound: I) -> Vec<P>;

    fn spawn(params: &Self::Params, fitness_fn: FitnessFn<P, Self::State, Self::Params>) -> Self;
}
