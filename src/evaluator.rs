pub trait Evaluate {
    type Phenotype;
    type Params;

    /// The observe method should take a clone of the observable
    /// and store in something like a sliding observation window.
    fn evaluate(&self, ob: Self::Phenotype) -> Self::Phenotype;

    fn spawn(params: &Self::Params) -> Self;
}
