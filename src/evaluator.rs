pub trait Evaluate {
    type Phenotype;
    type Params;
    type Fitness;

    /// We're assuming that the Phenotype contains a binding to
    /// the resulting fitness score, and that this method sets
    /// that score before returning the phenotype.
    ///
    /// NOTE: nothing guarantees that the returned phenotype is
    /// the same one that was passed in. Keep this in mind. This
    /// is to allow the use of asynchronous evaluation pipelines.
    fn evaluate(&self, ob: Self::Phenotype) -> Self::Phenotype;

    fn spawn(params: &Self::Params) -> Self;
}
