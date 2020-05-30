use crate::observer::ObserverConfig;
use serde::Deserialize;
use std::cmp::Ordering;
use std::fmt::Debug;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct DataConfig {
    pub path: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    pub mut_rate: f32,
    pub max_init_len: usize,
    pub min_init_len: usize,
    pub pop_size: usize,
    pub tournament_size: usize,
    pub num_offspring: usize,
    pub target: String,
    pub target_fitness: usize,
    pub observer: ObserverConfig,
    pub data: DataConfig,
    pub problems: Option<Vec<Problem>>,
    pub max_length: usize,
    #[serde(default = "Default::default")]
    pub num_registers: usize,
    #[serde(default = "Default::default")]
    pub return_registers: usize,
    pub crossover_period: f64,
    pub crossover_rate: f32,
    #[serde(default = "Default::default")]
    pub roper: RoperConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RoperConfig {
    pub binary_file: String,
    pub gadget_file: String,
    #[serde(default = "Default::default")]
    pub soup: Vec<u64>,
    pub arch: unicorn::Arch,
    pub mode: unicorn::Mode,
}

impl Default for RoperConfig {
    fn default() -> Self {
        Self {
            binary_file: "/bin/sh".to_string(),
            gadget_file: "".to_string(),
            soup: Vec::new(),
            arch: unicorn::Arch::X86,
            mode: unicorn::Mode::MODE_64,
        }
    }
}

impl Config {
    pub fn assert_invariants(&self) {
        assert!(self.tournament_size >= self.num_offspring + 2);
        //assert_eq!(self.num_offspring, 2); // all that's supported for now
    }

    pub fn mutation_rate(&self) -> f32 {
        self.mut_rate
    }

    pub fn crossover_rate(&self) -> f32 {
        self.crossover_rate
    }

    pub fn population_size(&self) -> usize {
        self.pop_size
    }

    pub fn tournament_size(&self) -> usize {
        self.tournament_size
    }

    pub fn observer_config(&self) -> ObserverConfig {
        self.observer.clone()
    }

    pub fn num_offspring(&self) -> usize {
        self.num_offspring
    }
}

#[derive(Debug, Clone, Deserialize, Eq, PartialEq, Hash)]
pub struct Problem {
    pub input: Vec<i32>,
    // TODO make this more generic
    pub output: i32,
    // Ditto
    pub tag: u64,
}

impl PartialOrd for Problem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.tag.partial_cmp(&other.tag)
    }
}

impl Ord for Problem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tag.cmp(&other.tag)
    }
}
