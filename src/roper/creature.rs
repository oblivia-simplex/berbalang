use std::fmt;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use hashbrown::HashMap;
use rand::seq::IteratorRandom;
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Standard};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::configure::Config;
use crate::emulator::loader;
use crate::emulator::loader::get_static_memory_image;
use crate::emulator::pack::Pack;
use crate::emulator::profiler::Profile;
use crate::error::Error;
use crate::evolution::{Genome, LinearChromosome, Mutation, Phenome};
use crate::fitness::{HasScalar, MapFit};
use crate::roper::evaluation::lexi;
use crate::roper::Fitness;
use crate::util::architecture::{endian, read_integer, write_integer, Perms};
use crate::util::levy_flight::levy_decision;
use crate::util::random::hash_seed_rng;
use crate::util::{self, architecture::Endian};

#[derive(Clone, Serialize)]
pub struct Creature {
    // pub chromosome: Vec<T>,
    // pub chromosome_parentage: Vec<usize>,
    // pub chromosome_mutation: Vec<Option<WordMutation>>,
    pub chromosome: LinearChromosome<u64, WordMutation>,
    pub tag: u64,
    pub profile: Option<Profile>,
    #[serde(borrow)]
    pub fitness: Option<Fitness<'static>>,
    pub front: Option<usize>,
    pub num_offspring: usize,
    pub native_island: usize,
    pub description: Option<String>,
}

impl Hash for Creature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tag.hash(state)
    }
}

impl Creature {
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
                    .map(|p| p.intersects(Perms::EXEC))
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

impl Pack for Creature {
    fn pack(
        &self,
        word_size: usize,
        endian: Endian,
        byte_filter: Option<&HashMap<u8, u8>>,
    ) -> Vec<u8> {
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
        if let Some(byte_filter) = byte_filter {
            buffer
                .into_iter()
                .map(|b| {
                    if let Some(x) = byte_filter.get(&b) {
                        *x
                    } else {
                        b
                    }
                })
                .collect::<Vec<u8>>()
        } else {
            buffer
        }
    }

    fn as_code_addrs(&self, _word_size: usize, _endian: Endian) -> Vec<u64> {
        let memory = loader::get_static_memory_image();
        self.chromosome()
            .iter()
            .filter(|a| {
                memory
                    .perm_of_addr(**a)
                    .map(|p| p.intersects(Perms::EXEC))
                    .unwrap_or(false)
            })
            .cloned()
            .collect::<Vec<_>>()
    }
}

impl fmt::Debug for Creature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Name: {}, from island {}",
            self.chromosome.name, self.native_island
        )?;
        writeln!(f, "Generation: {}", self.chromosome.generation)?;
        let memory = loader::get_static_memory_image();
        for i in 0..self.chromosome.len() {
            let parent = if self.chromosome.parent_names.is_empty() {
                "seed"
            } else {
                &self.chromosome.parent_names[self.chromosome.parentage[i]]
            };
            let allele = self.chromosome.chromosome[i];
            let perms = memory
                .perm_of_addr(allele)
                .map(|p| format!(" ({:?})", p))
                .unwrap_or_else(|| "".to_string());
            let was_it_executed = self
                .profile
                .as_ref()
                .map(|p| p.gadgets_executed.contains(&allele))
                .unwrap_or(false);
            let mutation = self.chromosome.mutations[i];
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
                    .unwrap_or_else(String::new),
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
            soup = serde_json::from_reader(reader)?;
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
            memory.random_address(Some(Perms::EXEC), seed)
        }) {
            soup.push(addr)
        }
    }
    config.roper.soup = Some(soup);
    Ok(())
}

fn try_to_ensure_exec(mut chromosome: Vec<u64>) -> Vec<u64> {
    // Now make sure that the head of the chromosome is executable
    chromosome.reverse();
    let mut tail = Vec::new();
    let memory = get_static_memory_image();
    while let Some(h) = chromosome.last() {
        if let Some(perm) = memory.perm_of_addr(*h) {
            if !perm.intersects(Perms::EXEC) {
                tail.push(chromosome.pop().unwrap());
            } else {
                break;
            }
        } else {
            tail.push(chromosome.pop().unwrap());
        }
    }
    if chromosome.is_empty() {
        log::warn!("Failed to ensure executability for chromosome");
        tail
    } else {
        chromosome.reverse();
        chromosome.extend_from_slice(&tail);
        chromosome
    }
}

impl Genome for Creature {
    type Allele = u64;

    fn chromosome(&self) -> &[Self::Allele] {
        &self.chromosome.chromosome
    }

    fn chromosome_mut(&mut self) -> &mut [Self::Allele] {
        &mut self.chromosome.chromosome
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
        let chromosome = try_to_ensure_exec(chromosome);
        let len = chromosome.len();
        let name = util::name::random(4, &salt);
        //let crossover_mask = rng.gen::<u64>();
        let tag = rng.gen::<u64>();
        Self {
            //crossover_mask,
            chromosome: LinearChromosome {
                chromosome,
                mutations: vec![None; len],
                parentage: vec![],
                parent_names: vec![],
                name,
                generation: 0,
            },
            tag,
            profile: None,
            fitness: None,
            front: None,
            num_offspring: 0,
            native_island: config.island_identifier,
            description: None,
        }
    }

    fn crossover(mates: &[&Self], config: &Config) -> Self {
        let parents = mates
            .iter()
            .map(|x| &x.chromosome)
            .collect::<Vec<&LinearChromosome<_, _>>>();
        let chromosome = LinearChromosome::crossover(&parents, config);
        Self {
            chromosome,
            tag: thread_rng().gen::<u64>(),
            profile: None,
            fitness: None,
            front: None,
            num_offspring: 0,
            native_island: 0,
            description: None,
        }
    }

    fn mutate(&mut self, config: &Config) {
        self.chromosome.mutate(config)
    }

    fn incr_num_offspring(&mut self, n: usize) {
        self.num_offspring += n
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Hash)]
pub enum WordMutation {
    Dereference,
    Indirection,
    AddressAdd,
    AddressSub,
    BitFlip,
}

impl Mutation for WordMutation {
    type Allele = u64;

    fn mutate_point(allele: &mut Self::Allele) -> Self {
        let mut rng = thread_rng();
        let mutation = rand::random::<WordMutation>();
        let memory = get_static_memory_image();
        let endian = memory.endian;
        let word_size = memory.word_size;
        // TODO: add a mutation that picks a random address from the soup
        match mutation {
            WordMutation::Dereference => {
                if let Some(bytes) = memory.try_dereference(*allele, None) {
                    if bytes.len() > 8 {
                        if let Some(word) = read_integer(bytes, endian, word_size) {
                            *allele = word;
                        }
                    }
                }
            }
            WordMutation::Indirection => {
                let mut bytes = vec![0; word_size];
                let word = *allele;
                write_integer(endian, word_size, word, &mut bytes);
                if let Some(address) = memory.seek_from_random_address(&bytes, rng.gen::<u64>()) {
                    *allele = address;
                }
            }
            WordMutation::AddressAdd => {
                let word = allele.wrapping_add(rng.gen_range(0, 0x100));
                *allele = word;
            }
            WordMutation::AddressSub => {
                let word = allele.wrapping_sub(rng.gen_range(0, 0x100));
                *allele = word;
            }
            WordMutation::BitFlip => {
                let word = *allele ^ (1 << rng.gen_range(0, word_size as u64 * 8));
                *allele = word;
            }
        }
        mutation
    }
}

impl Distribution<WordMutation> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> WordMutation {
        use WordMutation::*;
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

impl Phenome for Creature {
    type Fitness = Fitness<'static>;
    type Problem = lexi::Task;

    fn generate_description(&mut self) {
        self.description = Some(format!("{:#?}", self))
    }

    fn fitness(&self) -> Option<&Self::Fitness> {
        self.fitness.as_ref()
    }

    fn scalar_fitness(&self) -> Option<f64> {
        self.fitness.as_ref().map(HasScalar::scalar)
    }

    fn name(&self) -> &str {
        self.chromosome.name.as_str()
    }

    fn priority_fitness(&self, config: &Config) -> Option<f64> {
        let priority = &config.fitness.priority;
        self.fitness().as_ref().and_then(|f| f.get(priority))
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
        self.priority_fitness(config)
            .map(|p| p - config.fitness.target <= std::f64::EPSILON)
            .unwrap_or(false)
    }

    fn fails(&self, case: &Self::Problem) -> bool {
        !case.check_creature(self)
    }

    fn mature(&self) -> bool {
        self.profile.is_some()
    }
}
