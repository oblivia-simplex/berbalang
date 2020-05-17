pub trait Configure {
    fn assert_invariants(&self);
    fn mutation_rate(&self) -> f32;
    fn tournament_size(&self) -> usize;
    fn population_size(&self) -> usize;
    fn observer_window_size(&self) -> usize;
}

pub trait ConfigureObserver {
    fn window_size(&self) -> usize;
}
