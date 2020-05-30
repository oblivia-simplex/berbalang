use std::cmp::Ordering;
use std::fmt::Formatter;
use std::sync::Arc;
use std::{fmt, iter};

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use indexmap::map::IndexMap;
use rand::seq::IteratorRandom;
use rand::{thread_rng, Rng};
use serde_derive::Deserialize;
use unicorn::{Cpu, CpuARM, CpuARM64, CpuM68K, CpuMIPS, CpuSPARC, CpuX86};

use crate::configure::{Config, Problem};
use crate::emulator::executor;
use crate::emulator::pack::Pack;
use crate::fitness::Pareto;
/// This is where the ROP-evolution-specific code lives.
use crate::{
    emulator::executor::{Hatchery, HatcheryParams, Register},
    emulator::{loader, profiler::Profile},
    evolution::{Epoch, Genome, Phenome},
    fitness::FitnessScore,
    util,
    util::architecture::{endian, word_size, Endian},
    util::bitwise::bit,
};

type Fitness = Pareto;

#[derive(Debug, Deserialize, Clone)]
pub struct RegisterPatternConfig(pub IndexMap<String, u64>);

#[derive(Debug)]
pub struct RegisterPattern<C: 'static + Cpu<'static>>(pub IndexMap<Register<C>, u64>);

macro_rules! register_pattern_converter {
    ($cpu:ty) => {
        impl From<RegisterPatternConfig> for RegisterPattern<$cpu> {
            fn from(rp: RegisterPatternConfig) -> Self {
                let mut map: IndexMap<Register<$cpu>, u64> = IndexMap::new();
                for (reg, num) in rp.0.iter() {
                    let r: Register<$cpu> =
                        toml::from_str(&reg).expect("Failed to parse register pattern");
                    map.insert(r, *num);
                }
                Self(map)
            }
        }
    };
}

register_pattern_converter!(CpuX86<'static>);
register_pattern_converter!(CpuARM<'static>);
register_pattern_converter!(CpuARM64<'static>);
register_pattern_converter!(CpuMIPS<'static>);
register_pattern_converter!(CpuSPARC<'static>);
register_pattern_converter!(CpuM68K<'static>);

#[derive(Clone)]
pub struct Creature {
    //pub crossover_mask: u64,
    pub chromosome: Vec<u64>,
    pub chromosome_parentage: Vec<usize>,
    pub tag: u64,
    pub name: String,
    pub parents: Vec<String>,
    pub generation: usize,
    pub profile: Option<Profile>,
}

impl Pack for Creature {
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
}

impl fmt::Debug for Creature {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Name: {}", self.name)?;
        let memory = loader::get_static_memory_image();
        for i in 0..self.chromosome.len() {
            let parent = &self.parents[self.chromosome_parentage[i]];
            let allele = self.chromosome[i];
            let flag = memory
                .perm_of_addr(allele)
                .map(|p| format!("({:?})", p))
                .unwrap_or_else(|| "".to_string());
            write!(f, "[{}][{}] 0x{:010x} {}", i, parent, allele, flag)?;
        }
        Ok(())
    }
}

impl Genome for Creature {
    type Allele = u64;

    fn chromosome(&self) -> &[Self::Allele] {
        &self.chromosome
    }

    fn chromosome_mut(&mut self) -> &mut [Self::Allele] {
        &mut self.chromosome
    }

    fn random(params: &Config) -> Self {
        let mut rng = rand::thread_rng();
        let length = rng.gen_range(params.min_init_len, params.max_init_len);
        let chromosome = params
            .roper
            .soup
            .iter()
            .choose_multiple(&mut rng, length)
            .into_iter()
            .copied()
            .collect::<Vec<u64>>();
        let name = crate::util::name::random(4);
        //let crossover_mask = rng.gen::<u64>();
        let tag = rng.gen::<u64>();
        Self {
            //crossover_mask,
            chromosome,
            chromosome_parentage: vec![],
            tag,
            name,
            parents: vec![],
            generation: 0,
            profile: None,
        }
    }

    fn crossover(mates: &[Self], params: &Config) -> Self
    where
        Self: Sized,
        // note code duplication between this and linear_gp TODO
    {
        // NOTE: this bitmask schema implements an implicit incest prohibition
        let distribution = rand_distr::Exp::new(params.crossover_period)
            .expect("Failed to create random distribution");
        let parental_chromosomes = mates.iter().map(Genome::chromosome).collect::<Vec<_>>();
        let mut rng = thread_rng();
        let (chromosome, chromosome_parentage, parent_names) =
                // Check to see if we're performing a crossover or just cloning
                if rng.gen_range(0.0, 1.0) < params.crossover_rate() {
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
        Self {
            chromosome,
            chromosome_parentage,
            tag: 0,
            name: util::name::random(4),
            parents: parent_names,
            generation,
            profile: None,
        }
    }

    fn mutate(&mut self, params: &Config) {
        let mut rng = thread_rng();
        let i = rng.gen_range(0, self.chromosome.len());

        match rng.gen_range(0, 5) {
            0 => {
                // Dereference mutation
                let memory = loader::get_static_memory_image();
                if let Some(bytes) = memory.try_dereference(self.chromosome[i]) {
                    if bytes.len() > 8 {
                        let endian = endian(params.roper.arch, params.roper.mode);
                        let word_size = word_size(params.roper.arch, params.roper.mode);
                        let word = match (endian, word_size) {
                            (Endian::Little, 8) => LittleEndian::read_u64(bytes),
                            (Endian::Big, 8) => BigEndian::read_u64(bytes),
                            (Endian::Little, 4) => LittleEndian::read_u32(bytes) as u64,
                            (Endian::Big, 4) => BigEndian::read_u32(bytes) as u64,
                            (Endian::Little, 2) => LittleEndian::read_u16(bytes) as u64,
                            (Endian::Big, 2) => BigEndian::read_u16(bytes) as u64,
                            (_, _) => unimplemented!("I think we've covered the bases"),
                        };
                        self.chromosome[i] = word;
                    }
                }
            }
            1 => {
                // Indirection mutation
                let memory = loader::get_static_memory_image();
                let word_size = word_size(params.roper.arch, params.roper.mode);
                let mut bytes = vec![0; word_size];
                let word = self.chromosome[i];
                match (endian(params.roper.arch, params.roper.mode), word_size) {
                    (Endian::Little, 8) => LittleEndian::write_u64(&mut bytes, word),
                    (Endian::Big, 8) => BigEndian::write_u64(&mut bytes, word),
                    (Endian::Little, 4) => LittleEndian::write_u32(&mut bytes, word as u32),
                    (Endian::Big, 4) => BigEndian::write_u32(&mut bytes, word as u32),
                    (Endian::Little, 2) => LittleEndian::write_u16(&mut bytes, word as u16),
                    (Endian::Big, 2) => BigEndian::write_u16(&mut bytes, word as u16),
                    (_, _) => unimplemented!("I think we've covered the bases"),
                }
                if let Some(address) = memory.seek_from_random_address(&bytes) {
                    self.chromosome[i] = address;
                }
            }
            3 => {
                self.chromosome[i] = self.chromosome[i].wrapping_add(rng.gen_range(0, 0x100));
            }
            4 => {
                self.chromosome[i] = self.chromosome[i].wrapping_sub(rng.gen_range(0, 0x100));
            }
            // 5 => {
            //     self.crossover_mask ^= 1 << rng.gen_range(0, 64);
            // }
            _ => unimplemented!("out of range"),
        }
    }
}

impl Phenome for Creature {
    type Fitness = Fitness;

    fn fitness(&self) -> Option<&Self::Fitness> {
        unimplemented!()
    }

    fn set_fitness(&mut self, f: Self::Fitness) {
        unimplemented!()
    }

    fn tag(&self) -> u64 {
        self.tag
    }

    fn set_tag(&mut self, tag: u64) {
        self.tag = tag
    }

    fn problems(&self) -> Option<&Vec<Problem>> {
        unimplemented!()
    }

    fn store_answers(&mut self, results: Vec<Problem>) {
        unimplemented!()
    }

    fn len(&self) -> usize {
        self.chromosome().len()
    }
}

crate::make_phenome_heap_friendly!(Creature);

mod evaluation {
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::Arc;
    use std::thread::{spawn, JoinHandle};

    use unicorn::Cpu;

    use crate::evaluator::FitnessFn;
    use crate::{
        configure::Config,
        emulator::executor::{self, Hatchery, HatcheryParams, Register},
        emulator::loader,
        evaluator::Evaluate,
        evolution::{Epoch, Genome, Phenome},
        fitness::FitnessScore,
        util,
        util::architecture::{endian, word_size, Endian},
        util::bitwise::bit,
        util::count_min_sketch::DecayingSketch,
    };

    use super::Creature;
    use crate::emulator::profiler::Profiler;
    use byteorder::{BigEndian, LittleEndian};
    use indexmap::map::IndexMap;

    pub struct Evaluator<C: Cpu<'static>> {
        hatchery: Hatchery<C>,
        inputs: Arc<Vec<IndexMap<Register<C>, u64>>>,
        output_registers: Arc<Vec<Register<C>>>,
        sketch: DecayingSketch,
    }

    impl<C: 'static + Cpu<'static>> Evaluate<Creature> for Evaluator<C> {
        type Params = Config;

        fn evaluate(&self, ob: Creature) -> Creature {
            unimplemented!()
        }

        fn eval_batch<I: 'static + Iterator<Item = Creature> + Send>(
            &self,
            inbound: I,
        ) -> Vec<Creature> {
            let inputs = self.inputs.clone();
            let output_registers = self.output_registers.clone();
            self.hatchery
                .pipeline(
                    inbound,
                    inputs,
                    output_registers,
                    None, // use default
                )
                .iter()
                .map(|(creature, profile)| {
                    // TODO
                    // measure fitness on the basis of the profile
                    // assign fitness to the creature
                    // gauge frequency with the sketch
                    creature
                })
                .collect::<Vec<Creature>>()
        }

        fn spawn(params: &Self::Params, fitness_fn: FitnessFn<Creature, Self::Params>) -> Self {
            let hatch_params = Arc::new(params.roper.clone());
            let hatchery: Hatchery<C> = Hatchery::new(hatch_params.clone());

            Self {
                hatchery,
                inputs: Arc::new(vec![]),
                output_registers: Arc::new(vec![]),
                sketch: DecayingSketch::default(), // TODO parameterize
            }
        }
    }

    // impl Evaluate<Creature> for Evaluator {
    //     type Params = Config;
    //
    //     fn evaluate(&self, creature: Creature) -> Creature {
    //
    //
    //     }
    // }
}
