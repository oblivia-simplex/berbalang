use crate::observer::ObserverConfig;
use std::fmt::Debug;

pub trait Configure: Send + Sync + Clone + Debug {
    fn assert_invariants(&self);
    fn crossover_rate(&self) -> f32;
    fn max_length(&self) -> usize;
    fn mutation_rate(&self) -> f32;
    fn num_offspring(&self) -> usize;
    fn observer_config(&self) -> ObserverConfig;
    fn observer_window_size(&self) -> usize;
    fn population_size(&self) -> usize;
    fn tournament_size(&self) -> usize;
}
