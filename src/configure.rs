use crate::emulator::register_pattern::{RegisterPattern, RegisterPatternConfig};
use crate::error::Error;
use chrono::prelude::*;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct DataConfig {
    pub path: String,
}

#[derive(Clone, Debug, Copy, Deserialize)]
pub enum Job {
    Roper,
    Hello,
    LinearGp,
}

impl Default for Job {
    fn default() -> Self {
        Self::Roper
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub job: Job,
    pub selection: Selection,
    pub crossover_period: f64,
    pub crossover_rate: f32,
    pub season_length: usize,
    pub data: DataConfig,
    pub max_init_len: usize,
    pub max_length: usize,
    pub min_init_len: usize,
    pub mut_rate: f32,
    pub num_offspring: usize,
    pub num_parents: usize,
    pub observer: ObserverConfig,
    pub pop_size: usize,
    pub problems: Option<Vec<Problem>>,
    pub target_fitness: usize,
    #[serde(default)]
    pub roulette: RouletteConfig,
    #[serde(default)]
    pub tournament: TournamentConfig,
    #[serde(default = "Default::default")]
    pub roper: RoperConfig,
    #[serde(default = "Default::default")]
    pub linear_gp: LinearGpConfig,
    #[serde(default = "Default::default")]
    pub hello: HelloConfig,
    pub num_epochs: usize,
    pub fitness_weights: Arc<HashMap<String, String>>,
}

fn default_tournament_size() -> usize {
    4
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct TournamentConfig {
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,
    pub geographic_radius: usize,
}

fn default_weight_decay() -> f64 {
    0.75
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct RouletteConfig {
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
}

fn random_population_name() -> String {
    crate::util::name::random(2)
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObserverConfig {
    pub dump_population: bool,
    pub dump_soup: bool,
    pub window_size: usize,
    pub report_every: usize,
    pub dump_every: usize,
    #[serde(default)]
    pub full_data_directory: String,
    data_directory: String,
    #[serde(default = "random_population_name")]
    pub population_name: String,
}

impl Config {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut config: Self = toml::from_str(&std::fs::read_to_string(&path)?)?;
        config.assert_invariants();
        config.set_data_directory();
        // copy the config file to the data directory for posterity
        std::fs::copy(&path, &format!("{}/config.toml", config.data_directory()))?;

        println!("{:#?}", config);

        Ok(config)
    }

    /// Returns the path to the full data directory, creating it if necessary.
    pub fn set_data_directory(&mut self) {
        let local_date: DateTime<Local> = Local::now();

        let mut data_dir = self.observer.data_directory.clone();
        if data_dir.starts_with('~') {
            let home = std::env::var("HOME")
                .expect("No HOME environment variable found. Please set this.");
            data_dir.replace_range(0..1, &home);
        };

        let path = format!(
            "{data_dir}/berbalang/{job:?}/{selection:?}/{year}/{month}/{day}/{pop_name}",
            data_dir = data_dir,
            job = self.job,
            selection = self.selection,
            year = local_date.year(),
            month = local_date.month(),
            day = local_date.day(),
            pop_name = self.observer.population_name,
        );

        for sub in ["", "soup", "population"].iter() {
            let d = format!("{}/{}", path, sub);
            std::fs::create_dir_all(&d)
                .map_err(|e| log::error!("Error creating {}: {:?}", path, e))
                .expect("Failed to create data directory");
        }

        self.observer.full_data_directory = path;
    }

    pub fn data_directory(&self) -> &str {
        &self.observer.full_data_directory
    }
}

#[derive(Default, Clone, Debug, Deserialize)]
pub struct HelloConfig {
    pub target: String,
}

#[derive(Default, Clone, Debug, Deserialize)]
pub struct LinearGpConfig {
    pub max_steps: usize,
    // NOTE: these register values will be overridden if a data
    // file has been supplied, in order to accommodate that data
    pub num_registers: Option<usize>,
    pub return_registers: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct RoperConfig {
    pub fitness_function: String,
    pub gadget_file: Option<String>,
    #[serde(default)]
    pub output_registers: Vec<String>,
    pub register_pattern: Option<RegisterPatternConfig>,
    #[serde(skip)]
    pub parsed_register_pattern: Option<RegisterPattern>,
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
    pub binary_path: String,
}

impl RoperConfig {
    pub fn parse_register_pattern(&mut self) {
        if let Some(ref rp) = self.register_pattern {
            self.parsed_register_pattern = Some(rp.into());
        }
    }

    pub fn register_pattern(&self) -> Option<&RegisterPattern> {
        self.parsed_register_pattern.as_ref()
    }
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
            fitness_function: "register_pattern".to_string(),
            gadget_file: None,
            output_registers: vec![],
            register_pattern: None,
            parsed_register_pattern: None,
            soup: None,
            soup_size: None,
            arch: unicorn::Arch::X86,
            mode: unicorn::Mode::MODE_64,
            num_workers: 8,
            num_emulators: 8,
            wait_limit: 500,
            max_emu_steps: Some(0x10_000),
            millisecond_timeout: Some(500),
            record_basic_blocks: false,
            record_memory_writes: false,
            emulator_stack_size: 0x1000,
            binary_path: "/bin/sh".to_string(),
        }
    }
}

impl Config {
    pub fn assert_invariants(&self) {
        assert!(self.tournament.tournament_size >= self.num_offspring + 2);
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

    pub fn num_offspring(&self) -> usize {
        self.num_offspring
    }
}

#[derive(Debug, Clone, Deserialize, Eq, PartialEq, Hash, Serialize)]
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

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum Selection {
    Tournament,
    Roulette,
    Metropolis,
}

impl Default for Selection {
    fn default() -> Self {
        Self::Tournament
    }
}
