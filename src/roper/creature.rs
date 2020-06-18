use std::fmt;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use hashbrown::HashMap;
use rand::seq::IteratorRandom;
use rand::Rng;
use rand_distr::{Distribution, Standard};
use serde::{Deserialize, Serialize};
use unicorn::Protection;

use crate::configure::Config;
use crate::emulator::loader;
use crate::emulator::pack::Pack;
use crate::emulator::profiler::Profile;
use crate::error::Error;
use crate::evolution::{Genome, Phenome};
use crate::fitness::{HasScalar, MapFit};
use crate::roper::evaluation::lexi;
use crate::roper::Fitness;
use crate::util::architecture::{read_integer, write_integer};
use crate::util::levy_flight::levy_decision;
use crate::util::random::hash_seed_rng;
use crate::util::{self, architecture::Endian};

#[derive(Clone, Serialize, Deserialize)]
pub struct Creature<T: Clone + Serialize + Deserialize<'static>> {
    //pub crossover_mask: u64,
    pub chromosome: Vec<T>,
    pub chromosome_parentage: Vec<usize>,
    pub chromosome_mutation: Vec<Option<Mutation>>,
    pub tag: u64,
    pub name: String,
    pub parents: Vec<String>,
    pub generation: usize,
    pub profile: Option<Profile>,
    #[serde(borrow)]
    pub fitness: Option<Fitness<'static>>,
    pub front: Option<usize>,
    pub num_offspring: usize,
}

impl<T: Clone + Serialize + Deserialize<'static>> Hash for Creature<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tag.hash(state)
    }
}

impl Creature<u64> {
    /// Returns the number of alleles executed.
    pub fn num_uniq_alleles_executed(&self) -> usize {
        if let Some(ref profile) = self.profile {
            profile.gadgets_executed.len()
        } else {
            0
        }
    }

    /// Returns the ratio of executed to non-executed, but *executable*, alleles.
    /// If the `Creature` hasn't been executed yet, then this
    /// will always return `0.0`.
    /// FIXME: this seems to be returning n > 1 sometimes! Why?
    pub fn execution_ratio(&self) -> f64 {
        let memory = loader::get_static_memory_image();
        let mut executable_alleles = self
            .chromosome()
            .iter()
            .filter(|a| {
                memory
                    .perm_of_addr(**a)
                    .map(|p| p.intersects(Protection::EXEC))
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        if executable_alleles.is_empty() {
            return 0.0;
        };
        executable_alleles.dedup();
        let uniq_count = executable_alleles.len();
        let exec_count = self.num_uniq_alleles_executed();
        exec_count as f64 / uniq_count as f64
    }
}

impl Pack for Creature<u64> {
    fn pack(&self, word_size: usize, endian: Endian) -> Vec<u8> {
        let packer = |&word, mut bytes: &mut [u8]| match (endian, word_size) {
            (Endian::Little, 8) => LittleEndian::write_u64(&mut bytes, word),
            (Endian::Big, 8) => BigEndian::write_u64(&mut bytes, word),
            (Endian::Little, 4) => LittleEndian::write_u32(&mut bytes, word as u32),
            (Endian::Big, 4) => BigEndian::write_u32(&mut bytes, word as u32),
            (Endian::Little, 2) => LittleEndian::write_u16(&mut bytes, word as u16),
            (Endian::Big, 2) => BigEndian::write_u16(&mut bytes, word as u16),
            (_, _) => unimplemented!("I think we've covered the bases"),
        };
        let chromosome = self.chromosome();
        let mut buffer = vec![0_u8; chromosome.len() * word_size];
        let mut ptr = 0;
        for word in self.chromosome() {
            packer(word, &mut buffer[ptr..]);
            ptr += word_size;
        }
        buffer
    }

    fn as_code_addrs(&self, _word_size: usize, _endian: Endian) -> Vec<u64> {
        let memory = loader::get_static_memory_image();
        self.chromosome()
            .iter()
            .filter(|a| {
                memory
                    .perm_of_addr(**a)
                    .map(|p| p.intersects(Protection::EXEC))
                    .unwrap_or(false)
            })
            .cloned()
            .collect::<Vec<_>>()
    }
}

impl fmt::Debug for Creature<u64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Name: {}", self.name)?;
        writeln!(f, "Generation: {}", self.generation)?;
        let memory = loader::get_static_memory_image();
        for i in 0..self.chromosome.len() {
            let parent = if self.parents.is_empty() {
                "seed"
            } else {
                &self.parents[self.chromosome_parentage[i]]
            };
            let allele = self.chromosome[i];
            let perms = memory
                .perm_of_addr(allele)
                .map(|p| format!(" ({:?})", p))
                .unwrap_or_else(|| "".to_string());
            let was_it_executed = self
                .profile
                .as_ref()
                .map(|p| p.gadgets_executed.contains(&allele))
                .unwrap_or(false);
            let mutation = self.chromosome_mutation[i];
            writeln!(
                f,
                "[{}][{}] 0x{:010x}{}{} {}",
                i,
                parent,
                allele,
                perms,
                if was_it_executed { " *" } else { "" },
                mutation
                    .map(|m| format!("{:?}", m))
                    .unwrap_or("".to_string()),
            )?;
        }
        if let Some(ref profile) = self.profile {
            writeln!(f, "Trace:")?;
            for path in profile.disas_paths() {
                writeln!(f, "{}", path)?;
            }
            //writeln!(f, "Register state: {:#x?}", profile.registers)?;
            for state in &profile.registers {
                writeln!(f, "\nSpidered register state:\n{:?}", state)?;
            }
            writeln!(f, "CPU Error code(s): {:?}", profile.cpu_errors)?;
        }
        // writeln!(
        //     f,
        //     "Scalar fitness: {:?}",
        //     self.fitness().as_ref().map(|f| f.scalar())
        // )?;
        writeln!(f, "Fitness: {:#?}", self.fitness())?;
        Ok(())
    }
}

/// load binary before calling this function
pub fn init_soup(config: &mut Config) -> Result<(), Error> {
    let mut soup = Vec::new();
    //might as well take the constants from the register pattern
    if let Some(pattern) = config.roper.register_pattern() {
        pattern.0.values().for_each(|w| soup.push(w.val))
    }
    if let Some(gadget_file) = config.roper.gadget_file.as_ref() {
        // parse the gadget file
        let reader = File::open(gadget_file).map(BufReader::new)?;

        if gadget_file.ends_with(".json") {
            log::info!("Deserializing soup from {}", gadget_file);
            let map: HashMap<u64, usize> = serde_json::from_reader(reader)?;
            soup = map.keys().copied().collect::<Vec<u64>>();
        } else {
            log::info!("Parsing soup from {}", gadget_file);
            for line in reader.lines() {
                let word = line?.parse::<u64>()?;
                soup.push(word)
            }
        }
    } else if let Some(soup_size) = config.roper.soup_size.as_ref() {
        let memory = loader::get_static_memory_image();
        for addr in (0..(*soup_size)).map(|i| {
            let mut hasher = fnv::FnvHasher::default();
            i.hash(&mut hasher);
            config.random_seed.hash(&mut hasher);
            let seed = hasher.finish();
            memory.random_address(Some(Protection::EXEC), seed)
        }) {
            soup.push(addr)
        }
    }
    config.roper.soup = Some(soup);
    Ok(())
}

impl Genome for Creature<u64> {
    type Allele = u64;

    fn incr_num_offspring(&mut self, n: usize) {
        self.num_offspring += n
    }

    fn chromosome(&self) -> &[Self::Allele] {
        &self.chromosome
    }

    fn chromosome_mut(&mut self) -> &mut [Self::Allele] {
        &mut self.chromosome
    }

    fn random<H: Hash>(config: &Config, salt: H) -> Self {
        let mut hasher = fnv::FnvHasher::default();
        salt.hash(&mut hasher);
        config.random_seed.hash(&mut hasher);
        let seed = hasher.finish();
        let mut rng = hash_seed_rng(&seed);
        let length = rng.gen_range(config.min_init_len, config.max_init_len);
        let chromosome = config
            .roper
            .soup
            .as_ref()
            .expect("No soup?!")
            .iter()
            .choose_multiple(&mut rng, length)
            .into_iter()
            .copied()
            .collect::<Vec<u64>>();
        let name = util::name::random(4, &salt);
        //let crossover_mask = rng.gen::<u64>();
        let tag = rng.gen::<u64>();
        Self {
            //crossover_mask,
            chromosome,
            chromosome_parentage: vec![],
            chromosome_mutation: vec![None; length],
            tag,
            name,
            parents: vec![],
            generation: 0,
            profile: None,
            fitness: None,
            front: None,
            num_offspring: 0,
        }
    }

    fn crossover(mates: &Vec<&Self>, config: &Config) -> Self
    where
        Self: Sized,
        // note code duplication between this and linear_gp TODO
    {
        // NOTE: this bitmask schema implements an implicit incest prohibition
        let min_mate_len = mates.iter().map(|p| p.len()).min().unwrap();
        let lambda = min_mate_len as f64 / config.crossover_period;
        let distribution =
            rand_distr::Exp::new(lambda).expect("Failed to create random distribution");
        let parental_chromosomes = mates.iter().map(|m| m.chromosome()).collect::<Vec<_>>();
        let mut rng = hash_seed_rng(mates);
        let (chromosome, chromosome_parentage, parent_names) =
            // Check to see if we're performing a crossover or just cloning
            if rng.gen_range(0.0, 1.0) < config.crossover_rate() {
                let (c, p) = Self::crossover_by_distribution(&distribution, &parental_chromosomes);
                let names = mates.iter().map(|p| p.name.clone()).collect::<Vec<String>>();
                (c, p, names)
            } else {
                let parent = parental_chromosomes[rng.gen_range(0, 2)];
                let chromosome = parent.to_vec();
                let parentage =
                    chromosome.iter().map(|_| 0).collect::<Vec<usize>>();
                (chromosome, parentage, vec![mates[0].name.clone()])
            };
        let generation = mates.iter().map(|p| p.generation).max().unwrap() + 1;
        let name = util::name::random(4, &chromosome);
        let len = chromosome.len();
        Self {
            chromosome,
            chromosome_parentage,
            chromosome_mutation: vec![None; len],
            tag: rand::random::<u64>(),
            parents: parent_names,
            generation,
            name,
            profile: None,
            fitness: None,
            front: None,
            num_offspring: 0,
        }
    }

    fn mutate(&mut self, config: &Config) {
        let memory = loader::get_static_memory_image();
        let word_size = memory.word_size;
        let endian = memory.endian;
        let mut rng = hash_seed_rng(&self);
        for i in 0..self.len() {
            if !levy_decision(&mut rng, self.len(), config.mutation_exponent) {
                continue;
            }
            let mutation = rand::random::<Mutation>();
            self.chromosome_mutation[i] = Some(mutation);
            match mutation {
                Mutation::Dereference => {
                    if let Some(bytes) = memory.try_dereference(self.chromosome[i], None) {
                        if bytes.len() > 8 {
                            if let Some(word) = read_integer(bytes, endian, word_size) {
                                self.chromosome[i] = word;
                            }
                        }
                    }
                }
                Mutation::Indirection => {
                    let mut bytes = vec![0; word_size];
                    let word = self.chromosome[i];
                    write_integer(endian, word_size, word, &mut bytes);
                    if let Some(address) = memory.seek_from_random_address(&bytes, &self) {
                        self.chromosome[i] = address;
                    }
                }
                Mutation::AddressAdd => {
                    self.chromosome[i] = self.chromosome[i].wrapping_add(rng.gen_range(0, 0x100));
                }
                Mutation::AddressSub => {
                    self.chromosome[i] = self.chromosome[i].wrapping_sub(rng.gen_range(0, 0x100));
                }
                Mutation::BitFlip => {
                    self.chromosome[i] ^= 1 << rng.gen_range(0, word_size as u64 * 8)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Mutation {
    Dereference,
    Indirection,
    AddressAdd,
    AddressSub,
    BitFlip,
}

impl Distribution<Mutation> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Mutation {
        use Mutation::*;
        match rng.gen_range(0, 5) {
            0 => Dereference,
            1 => Indirection,
            2 => AddressAdd,
            3 => AddressSub,
            4 => BitFlip,
            n => unreachable!("no, can't get {}", n),
        }
    }
}

impl Phenome for Creature<u64> {
    type Fitness = Fitness<'static>;
    type Problem = lexi::Task;

    fn fitness(&self) -> Option<&Self::Fitness> {
        self.fitness.as_ref()
    }

    fn scalar_fitness(&self) -> Option<f64> {
        self.fitness.as_ref().map(HasScalar::scalar)
    }

    fn set_fitness(&mut self, f: Self::Fitness) {
        self.fitness = Some(f)
    }

    fn tag(&self) -> u64 {
        self.tag
    }

    fn set_tag(&mut self, tag: u64) {
        self.tag = tag
    }

    fn answers(&self) -> Option<&Vec<Self::Problem>> {
        unimplemented!()
    }

    fn store_answers(&mut self, _results: Vec<Self::Problem>) {
        unimplemented!()
    }

    fn front(&self) -> Option<usize> {
        self.front
    }

    fn set_front(&mut self, rank: usize) {
        self.front = Some(rank)
    }

    fn is_goal_reached(&self, config: &Config) -> bool {
        let priority = config.fitness.priority.clone();
        (self
            .fitness()
            .as_ref()
            .and_then(|f| f.get(&priority))
            .unwrap_or(std::f64::MAX)
            - config.fitness.target)
            <= std::f64::EPSILON
    }

    fn mature(&self) -> bool {
        self.profile.is_some()
    }

    fn fails(&self, case: &Self::Problem) -> bool {
        !case.check_creature(self)
        // if let Some(ref profile) = self.profile {
        //     !profile
        //         .registers
        //         .iter()
        //         .all(|reg_state| case.check_state(reg_state))
        // } else {
        //     true
        // }
    }
}
