use std::sync::Arc;

use crate::evolution::Phenome;

pub type FitnessFn<Pheno, State, Conf> =
    Box<dyn Fn(Pheno, &mut State, Arc<Conf>) -> Pheno + Sync + Send + 'static>;

// TODO: Consider replicating design seen in observer
// using a generic struct instead of a trait

pub trait Develop<P: Phenome> {
    /// We're assuming that the Phenotype contains a binding to
    /// the resulting fitness score, and that this method sets
    /// that score before returning the phenotype.
    ///
    /// NOTE: nothing guarantees that the returned phenotype is
    /// the same one that was passed in. Keep this in mind. This
    /// is to allow the use of asynchronous evaluation pipelines.
    ///

    fn develop(&self, ob: P) -> P;

    fn apply_fitness_function(&mut self, ob: P) -> P;

    fn development_pipeline<I: 'static + Iterator<Item = P> + Send>(&self, inbound: I) -> Vec<P>;
}
