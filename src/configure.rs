use indexmap::map::IndexMap;
use serde::Deserialize;
use std::cmp::Ordering;
use std::fmt::Debug;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct DataConfig {
    pub path: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    pub crossover_period: f64,
    pub crossover_rate: f32,
    pub data: DataConfig,
    pub max_init_len: usize,
    pub max_length: usize,
    pub min_init_len: usize,
    pub mut_rate: f32,
    pub num_offspring: usize,
    pub observer: ObserverConfig,
    pub pop_size: usize,
    pub problems: Option<Vec<Problem>>,
    pub target_fitness: usize,
    pub tournament_size: usize,
    #[serde(default = "Default::default")]
    pub roper: RoperConfig,
    #[serde(default = "Default::default")]
    pub machine: MachineConfig,
    #[serde(default = "Default::default")]
    pub hello: HelloConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObserverConfig {
    pub window_size: usize,
    pub report_every: usize,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            window_size: 0x1000,
            report_every: 0x1000,
        }
    }
}

#[derive(Default, Clone, Debug, Deserialize)]
pub struct HelloConfig {
    pub target: String,
}

#[derive(Default, Clone, Debug, Deserialize)]
pub struct MachineConfig {
    pub max_steps: usize,
    // NOTE: these register values will be overridden if a data
    // file has been supplied, in order to accommodate that data
    pub num_registers: Option<usize>,
    pub return_registers: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct RoperConfig {
    pub gadget_file: Option<String>,
    pub use_registers: Vec<String>, // deserialize to Register<C>
    pub register_pattern: Option<IndexMap<String, u64>>,
    #[serde(default = "Default::default")]
    pub soup: Option<Vec<u64>>,
    pub soup_size: Option<usize>, // if no gadget file given
    pub arch: unicorn::Arch,
    pub mode: unicorn::Mode,
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,
    #[serde(default = "default_num_workers")]
    pub num_emulators: usize,
    #[serde(default = "default_wait_limit")]
    pub wait_limit: u64,
    pub max_emu_steps: Option<usize>,
    pub millisecond_timeout: Option<u64>,
    #[serde(default = "Default::default")]
    pub record_basic_blocks: bool,
    #[serde(default = "Default::default")]
    pub record_memory_writes: bool,
    #[serde(default = "default_stack_size")]
    pub emulator_stack_size: usize,
    pub binary_path: Option<String>,
}

const fn default_num_workers() -> usize {
    8
}
const fn default_wait_limit() -> u64 {
    200
}

const fn default_stack_size() -> usize {
    0x1000
}

impl Default for RoperConfig {
    fn default() -> Self {
        Self {
            gadget_file: None,
            use_registers: vec![],
            register_pattern: None,
            soup: None,
            soup_size: None,
            arch: unicorn::Arch::X86,
            mode: unicorn::Mode::MODE_64,
            num_workers: 0,
            num_emulators: 0,
            wait_limit: 0,
            max_emu_steps: None,
            millisecond_timeout: None,
            record_basic_blocks: false,
            record_memory_writes: false,
            emulator_stack_size: 0,
            binary_path: Some("/bin/sh".to_string()),
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
