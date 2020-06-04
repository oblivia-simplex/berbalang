use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt::Formatter;
use std::fs::File;
use std::io::{self, BufRead, BufReader, ErrorKind};
use std::sync::Arc;
use std::{fmt, iter};

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use indexmap::map::IndexMap;
use rand::seq::IteratorRandom;
use rand::{thread_rng, Rng};
use serde_derive::Deserialize;
use unicorn::{Cpu, CpuARM, CpuARM64, CpuM68K, CpuMIPS, CpuSPARC, CpuX86, Protection};

use crate::configure::{Config, Problem, RoperConfig, Selection};
use crate::emulator::executor;
use crate::emulator::pack::Pack;
/// This is where the ROP-evolution-specific code lives.
use crate::{
    emulator::executor::{Hatchery, HatcheryParams},
    emulator::{loader, profiler::Profile},
    error::Error,
    evolution::{tournament::Tournament, Genome, Phenome},
    fitness::FitnessScore,
    util,
    util::architecture::{endian, word_size_in_bytes, Endian},
    util::bitwise::bit,
};
// the runner
use crate::evaluator::Evaluate;
use crate::evolution::metropolis::Metropolis;
use crate::evolution::roulette::Roulette;
use crate::fitness::Pareto;
use crate::observer::{default_report_fn, Observer};
use crate::util::architecture::{read_integer, write_integer};

type Fitness = Pareto;

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
    pub fitness: Option<Fitness>,
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

    fn as_addrs(&self, _word_size: usize, _endian: Endian) -> &[u64] {
        self.chromosome()
    }
}

impl fmt::Debug for Creature {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Name: {}\nFitness: {:?}", self.name, self.fitness())?;
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
            writeln!(
                f,
                "[{}][{}] 0x{:010x}{}{}",
                i,
                parent,
                allele,
                perms,
                if was_it_executed { " *" } else { "" }
            )?;
        }
        if let Some(ref profile) = self.profile {
            //writeln!(f, "Register state: {:#x?}", profile.registers)?;
            writeln!(f, "\nSpidered register state:")?;
            for state in profile.registers.iter() {
                writeln!(f, "{:#x?}", state.spider())?;
            }
            writeln!(f, "CPU Error code(s): {:?}", profile.cpu_errors)?;
        }
        Ok(())
    }
}

/// load binary before calling this function
pub fn init_soup(params: &mut RoperConfig) -> Result<(), Error> {
    let mut soup = Vec::new();
    if let Some(gadget_file) = params.gadget_file.as_ref() {
        // parse the gadget file
        let reader = File::open(gadget_file).map(BufReader::new)?;

        for line in reader.lines() {
            let word = line?.parse::<u64>()?;
            soup.push(word)
        }
    } else if let Some(soup_size) = params.soup_size.as_ref() {
        let memory = loader::get_static_memory_image();
        for addr in iter::repeat(())
            .take(*soup_size)
            .map(|()| memory.random_address(Some(Protection::EXEC)))
        {
            soup.push(addr)
        }
    }
    params.soup = Some(soup);
    Ok(())
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
            .as_ref()
            .expect("No soup?!")
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
            fitness: None,
        }
    }

    fn crossover(mates: &[&Self], params: &Config) -> Self
    where
        Self: Sized,
        // note code duplication between this and linear_gp TODO
    {
        // NOTE: this bitmask schema implements an implicit incest prohibition
        let min_mate_len = mates.iter().map(|p| p.len()).min().unwrap();
        let lambda = min_mate_len as f64 / params.crossover_period;
        let distribution =
            rand_distr::Exp::new(lambda).expect("Failed to create random distribution");
        let parental_chromosomes = mates.iter().map(|m| m.chromosome()).collect::<Vec<_>>();
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
            fitness: None,
        }
    }

    fn mutate(&mut self, params: &Config) {
        let mut rng = thread_rng();
        let i = rng.gen_range(0, self.chromosome.len());

        match rng.gen_range(0, 4) {
            0 => {
                // Dereference mutation
                let memory = loader::get_static_memory_image();
                if let Some(bytes) = memory.try_dereference(self.chromosome[i]) {
                    if bytes.len() > 8 {
                        let endian = endian(params.roper.arch, params.roper.mode);
                        let word_size = word_size_in_bytes(params.roper.arch, params.roper.mode);
                        if let Some(word) = read_integer(bytes, endian, word_size) {
                            self.chromosome[i] = word;
                        }
                    }
                }
            }
            1 => {
                // Indirection mutation
                let memory = loader::get_static_memory_image();
                let word_size = word_size_in_bytes(params.roper.arch, params.roper.mode);
                let mut bytes = vec![0; word_size];
                let word = self.chromosome[i];
                write_integer(
                    endian(params.roper.arch, params.roper.mode),
                    word_size,
                    word,
                    &mut bytes,
                );
                if let Some(address) = memory.seek_from_random_address(&bytes) {
                    self.chromosome[i] = address;
                }
            }
            2 => {
                self.chromosome[i] = self.chromosome[i].wrapping_add(rng.gen_range(0, 0x100));
            }
            3 => {
                self.chromosome[i] = self.chromosome[i].wrapping_sub(rng.gen_range(0, 0x100));
            }
            // 5 => {
            //     self.crossover_mask ^= 1 << rng.gen_range(0, 64);
            // }
            m => unimplemented!("mutation {} out of range, but this should never happen", m),
        }
    }
}

impl Phenome for Creature {
    type Fitness = Fitness;

    fn fitness(&self) -> Option<&Self::Fitness> {
        self.fitness.as_ref()
    }

    fn scalar_fitness(&self) -> Option<f64> {
        self.fitness.as_ref().map(|v| v.0.iter().sum())
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

    fn problems(&self) -> Option<&Vec<Problem>> {
        unimplemented!()
    }

    fn store_answers(&mut self, _results: Vec<Problem>) {
        unimplemented!()
    }

    fn len(&self) -> usize {
        self.chromosome().len()
    }
}

crate::make_phenome_heap_friendly!(Creature);

mod evaluation {
    use std::convert::TryInto;
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::Arc;
    use std::thread::{spawn, JoinHandle};

    use byteorder::{BigEndian, LittleEndian};
    use indexmap::map::IndexMap;
    use unicorn::{Cpu, Unicorn};

    use crate::emulator::profiler::{Block, Profiler};
    use crate::emulator::register_pattern::{Register, RegisterPattern, UnicornRegisterState};
    use crate::evaluator::FitnessFn;
    use crate::fitness::Pareto;
    use crate::{
        configure::Config,
        emulator::executor::{self, Hatchery, HatcheryParams},
        emulator::loader,
        evaluator::Evaluate,
        evolution::{tournament::Tournament, Genome, Phenome},
        fitness::FitnessScore,
        util,
        util::architecture::{endian, word_size_in_bytes, Endian},
        util::bitwise::bit,
        util::count_min_sketch::DecayingSketch,
    };

    use super::Creature;

    pub fn register_pattern_fitness_fn(
        mut creature: Creature,
        sketch: &mut DecayingSketch,
        params: Arc<Config>,
    ) -> Creature {
        // measure fitness
        // for now, let's just handle the register pattern task
        if let Some(ref profile) = creature.profile {
            sketch.insert(&profile.registers);
            let reg_freq = sketch.query(&profile.registers);
            if let Some(pattern) = params.roper.register_pattern() {
                // assuming that when the register pattern task is activated, there's only one register state
                // to worry about. this may need to be adjusted in the future. bit sloppy now.
                let mut register_pattern_distance = pattern.distance(&profile.registers[0]);
                register_pattern_distance.push(reg_freq);

                let longest_path = profile
                    .bb_path_iter()
                    .map(|v: Vec<Block>| v.len())
                    .max()
                    .unwrap_or(0) as f64;
                register_pattern_distance.push(-(longest_path).log2()); // let's see what happens when we use negative vals
                creature.set_fitness(Pareto(register_pattern_distance)); //vec![register_pattern_distance.iter().sum()]));
                                                                         //log::debug!("fitness: {:?}", creature.fitness());
            } else {
                log::error!("No register pattern?");
            }
        }
        creature
    }
    pub struct Evaluator<C: 'static + Cpu<'static>> {
        params: Arc<Config>,
        hatchery: Hatchery<C, Creature>,
        sketch: DecayingSketch,
        fitness_fn: Box<FitnessFn<Creature, DecayingSketch, Config>>,
    }

    impl<C: 'static + Cpu<'static>> Evaluate<Creature> for Evaluator<C> {
        type Params = Config;
        type State = DecayingSketch;

        fn evaluate(&mut self, creature: Creature) -> Creature {
            let (mut creature, profile) = self
                .hatchery
                .execute(creature)
                .expect("Failed to evaluate creature");

            creature.profile = Some(profile);
            (self.fitness_fn)(creature, &mut self.sketch, self.params.clone())
        }

        fn eval_pipeline<I: 'static + Iterator<Item = Creature> + Send>(
            &mut self,
            inbound: I,
        ) -> Vec<Creature> {
            self.hatchery
                .execute_batch(inbound)
                .expect("execute batch failure")
                .into_iter()
                .map(|(mut creature, profile)| {
                    creature.profile = Some(profile);
                    (self.fitness_fn)(creature, &mut self.sketch, self.params.clone())
                })
                .collect::<Vec<_>>()
        }

        fn spawn(
            params: &Self::Params,
            fitness_fn: FitnessFn<Creature, Self::State, Self::Params>,
        ) -> Self {
            let mut params = params.clone();
            params.roper.parse_register_pattern();
            let hatch_params = Arc::new(params.roper.clone());
            let inputs = vec![IndexMap::new()]; // TODO: if dealing with data, fill this in
            let register_pattern = params.roper.register_pattern();
            let output_registers: Vec<Register<C>> = {
                let mut out_reg: Vec<Register<C>> = params
                    .roper
                    .output_registers
                    .iter()
                    .map(|s| s.parse().ok().expect("Failed to parse output register"))
                    .collect::<Vec<_>>();
                if let Some(pat) = register_pattern {
                    let arch_specific_pat: UnicornRegisterState<C> =
                        pat.try_into().expect("Failed to parse register pattern");
                    let regs_in_pat = arch_specific_pat.0.keys().cloned().collect::<Vec<_>>();
                    out_reg.extend_from_slice(&regs_in_pat);
                    out_reg.dedup();
                    // sort alphabetically (lame)
                    // out_reg.sort_by_key(|r| format!("{:?}", r));
                    out_reg
                } else {
                    todo!("implement a conversion method from problem sets to register maps");
                    //out_reg
                }
            };
            let hatchery: Hatchery<C, Creature> = Hatchery::new(
                hatch_params,
                Arc::new(inputs),
                Arc::new(output_registers),
                None,
            );

            Self {
                params: Arc::new(params),
                hatchery,
                sketch: DecayingSketch::default(), // TODO parameterize
                fitness_fn: Box::new(fitness_fn),
            }
        }
    }
}

fn prepare<C: 'static + Cpu<'static>>(
    config: Config,
) -> (Observer<Creature>, evaluation::Evaluator<C>) {
    let observer = Observer::spawn(&config, Box::new(default_report_fn));
    let evaluator =
        evaluation::Evaluator::spawn(&config, Box::new(evaluation::register_pattern_fitness_fn));
    (observer, evaluator)
}

crate::impl_dominance_ord_for_phenome!(Creature, CreatureDominanceOrd);

pub fn run<C: 'static + Cpu<'static>>(mut config: Config) {
    let _ = loader::load_from_path(
        &config.roper.binary_path,
        0x1000,
        config.roper.arch,
        config.roper.mode,
    )
    .expect("Failed to load binary image");
    init_soup(&mut config.roper).expect("Failed to initialize the soup");

    let (observer, evaluator) = prepare(config.clone());

    match config.selection {
        Selection::Tournament => {
            let mut world =
                Tournament::<evaluation::Evaluator<C>, Creature>::new(config, observer, evaluator);
            let mut counter = 0;
            loop {
                world = world.evolve();
                counter += 1;
                if counter % 0x1000 == 0 {
                    log::info!("best: {:#x?}", world.best);
                }
            }
        }
        Selection::Roulette => {
            let mut world =
                Roulette::<evaluation::Evaluator<C>, Creature, CreatureDominanceOrd>::new(
                    config,
                    observer,
                    evaluator,
                    CreatureDominanceOrd,
                );
            loop {
                world = world.evolve();
            }
        }
        Selection::Metropolis => {
            let mut world =
                Metropolis::<evaluation::Evaluator<C>, Creature>::new(config, observer, evaluator);
            loop {
                world = world.evolve();
            }
        }
    }
}
