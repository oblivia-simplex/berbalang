use crate::observer::ObserverConfig;

pub trait Configure: Send + Sync {
    fn assert_invariants(&self);
    fn mutation_rate(&self) -> f32;
    fn tournament_size(&self) -> usize;
    fn population_size(&self) -> usize;
    fn observer_config(&self) -> ObserverConfig;
    fn observer_window_size(&self) -> usize;
    fn num_offspring(&self) -> usize;
    fn max_length(&self) -> usize;
}

// pub trait ConfigureObserver {
//     fn window_size(&self) -> usize;
// }
