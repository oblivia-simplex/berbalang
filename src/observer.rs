// A Logger needs to asynchronously gather and periodically
// record information on the evolutionary process.

pub trait Observe {
    type Observable;
    type Params;
    type Error;

    /// The observe method should take a clone of the observable
    /// and store in something like a sliding observation window.
    fn observe(&self, ob: Self::Observable);

    fn spawn(params: &Self::Params) -> Self;
}

pub trait ObservationWindow {
    type Observable;

    fn insert(&mut self, ob: Self::Observable);

    fn report(&self);
}
