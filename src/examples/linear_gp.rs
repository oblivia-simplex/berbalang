use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::{fmt, iter};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::configure::{ClassificationProblem, Config, Selection};
use crate::evolution::metropolis::Metropolis;
use crate::evolution::pareto_roulette::Roulette;
use crate::evolution::population::pier::Pier;
use crate::evolution::{tournament::Tournament, Genome, Phenome};
use crate::fitness::Weighted;
use crate::observer::{Observer, ReportFn, Window};
use crate::ontogenesis::FitnessFn;
use crate::util;
use crate::util::count_min_sketch::CountMinSketch;
use crate::util::levy_flight::levy_decision;
use crate::util::random::{hash_seed, hash_seed_rng};

pub type Fitness<'a> = Weighted<'a>;
// try setting fitness to (usize, usize);
pub type MachineWord = i32;

pub mod machine {
    use std::fmt::{self, Display};
    use std::hash::Hash;

    use rand::distributions::{Distribution, Standard};
    use rand::Rng;
    use serde::{Deserialize, Serialize};

    use crate::configure::LinearGpConfig;
    use crate::examples::linear_gp::MachineWord;
    use crate::util::random::hash_seed_rng;

    use super::Mutation;

    #[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, Serialize, Deserialize)]
    pub enum Op {
        Add,
        Div,
        Mov,
        Mult,
        Sub,
        Xor,
        Set(MachineWord),
        Lsl,
        And,
        Jle,
        End,
    }

    impl Display for Op {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            use Op::*;
            match self {
                Add => write!(f, "ADD"),
                Div => write!(f, "DIV"),
                Mov => write!(f, "MOV"),
                Mult => write!(f, "MUL"),
                Sub => write!(f, "SUB"),
                Xor => write!(f, "XOR"),
                Set(n) => write!(f, "SET(0x{:X})", n),
                Lsl => write!(f, "LSL"),
                And => write!(f, "AND"),
                Jle => write!(f, "JLE"), // Jump if less than or equal to 0
                End => write!(f, "END"),
            }
        }
    }

    pub const NUM_OPS: usize = 11;

    impl Distribution<Op> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Op {
            use Op::*;
            match rng.gen_range(0, NUM_OPS) {
                0 => Add,
                1 => Div,
                2 => Mov,
                3 => Mult,
                4 => Sub,
                5 => Xor,
                6 => Set(rng.gen_range(0, 256)),
                7 => Lsl,
                8 => And,
                9 => Jle,
                10 => End,
                _ => unreachable!("out of range"),
            }
        }
    }

    pub type Register = usize;

    // TODO make this configurable
    //const FIRST_INPUT_REGISTER: usize = 1;
    //const RETURN_REGISTER: usize = 0;

    #[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, Serialize, Deserialize)]
    pub struct Inst {
        pub op: Op,
        pub a: Register,
        pub b: Register,
    }

    impl Display for Inst {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.op {
                Op::Set(n) => write!(f, "SET  R{}  0x{:X}", self.a, n),
                Op::End => write!(f, "END"),
                _ => write!(f, "{}  R{}, R{}", self.op, self.a, self.b),
            }
        }
    }

    impl Inst {
        pub fn random(config: &LinearGpConfig) -> Self {
            let num_registers = config.num_registers.unwrap();
            Self {
                op: rand::random::<Op>(),
                a: rand::random::<usize>() % num_registers,
                b: rand::random::<usize>() % num_registers,
            }
        }

        pub fn mutate<H: Hash>(&mut self, config: &LinearGpConfig, seed: H) -> Mutation {
            let num_registers = config.num_registers.unwrap();
            let mut rng = hash_seed_rng(&seed);
            let mutation = rng.gen::<u8>() % 4;

            match mutation {
                0 => self.op = rand::random(),
                1 => self.a = (self.a + 1) % num_registers,
                2 => self.b = (self.b + 1) % num_registers,
                3 => std::mem::swap(&mut self.a, &mut self.b),
                _ => unreachable!("out of range"),
            }
            mutation
        }
    }

    pub struct Machine {
        return_registers: usize,
        registers: Vec<MachineWord>,
        pc: usize,
        max_steps: usize,
    }

    impl Machine {
        pub(crate) fn new(config: &LinearGpConfig) -> Self {
            Self {
                return_registers: config.return_registers.unwrap(),
                registers: vec![0; config.num_registers.unwrap()],
                pc: 0,
                max_steps: config.max_steps,
            }
        }

        #[inline]
        fn set(&mut self, reg: usize, val: MachineWord) {
            let n = self.registers.len();
            self.registers[reg % n] = val
        }

        fn eval(&mut self, inst: Inst) {
            use Op::*;

            macro_rules! set {
                ($val:expr) => {
                    self.set(inst.a, $val)
                };
            }

            match inst.op {
                Add => set!(self.registers[inst.a].wrapping_add(self.registers[inst.b])),
                Div => {
                    let b = if self.registers[inst.b] == 0 {
                        1
                    } else {
                        self.registers[inst.b]
                    };
                    set!(self.registers[inst.a].wrapping_div(b))
                }
                Mov => set!(self.registers[inst.b]),
                Mult => set!(self.registers[inst.a].wrapping_mul(self.registers[inst.b])),
                Sub => set!(self.registers[inst.a].wrapping_sub(self.registers[inst.b])),
                Xor => set!(self.registers[inst.a] ^ self.registers[inst.b]),
                Set(n) => set!(n),
                Lsl => set!(self.registers[inst.a].wrapping_shl(self.registers[inst.b] as u32)),
                And => set!(self.registers[inst.a] & self.registers[inst.b]),
                Jle => {
                    if self.registers[inst.a] <= 0 {
                        self.pc = self.registers[inst.b] as usize
                    }
                }
                End => {}
            }
        }

        fn flush_registers(&mut self) {
            self.registers.iter_mut().for_each(|i| *i = 0);
        }

        fn load_input(&mut self, inputs: &[MachineWord]) {
            if inputs.len() > self.registers.len() - 1 {
                log::error!("Too many inputs to load into input registers. Wrapping.");
            }
            (self.return_registers..(self.registers.len()))
                .zip(inputs.iter())
                .for_each(|(r, i)| self.set(r, *i));
        }

        fn exec_insts(&mut self, code: &[Inst]) {
            let mut step = 0;
            self.pc = 0;

            let fetch = |i| code[i % code.len()];

            let max_steps = self.max_steps;
            while step < max_steps {
                let old_pc = self.pc;
                let inst = fetch(self.pc);
                if let Op::End = inst.op {
                    break;
                };
                self.pc += 1;
                self.eval(inst);
                self.pc %= code.len();
                log::trace!(
                    "[{}]\t{}\t{:X?}{}",
                    old_pc,
                    inst,
                    self.registers,
                    if old_pc + 1 == self.pc { "" } else { "*" }
                );
                step += 1;
            }
        }

        fn return_value(&self) -> &[MachineWord] {
            &self.registers[0..self.return_registers]
        }

        pub fn exec<'a>(&'a mut self, code: &[Inst], input: &[MachineWord]) -> &'a [MachineWord] {
            self.flush_registers();
            self.load_input(input);
            self.exec_insts(code);
            self.return_value()
        }
    }
}

//#[derive(Debug, Clone, Default)]
//pub struct Genotype(pub Vec<machine::Inst>);
type Genotype = Vec<machine::Inst>;

/// Produce a genotype with exactly `len` random instructions.
/// Pass a randomly generated `len` to randomize the length.
///
fn random_chromosome<H: Hash>(config: &Config, seed: H) -> Genotype {
    let mut rng = hash_seed_rng(&seed);
    let len = rng.gen_range(config.min_init_len, config.max_init_len) + 1;
    iter::repeat(())
        .map(|()| machine::Inst::random(&config.linear_gp))
        .take(len)
        .collect()
}

pub type Answer = Vec<ClassificationProblem>;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Creature {
    chromosome: Genotype,
    chromosome_parentage: Vec<usize>,
    chromosome_mutation: Vec<Option<Mutation>>,
    answers: Option<Answer>,
    #[serde(borrow)]
    pub fitness: Option<Fitness<'static>>,
    tag: u64,
    //crossover_mask: u64,
    name: String,
    parents: Vec<String>,
    generation: usize,
    native_island: usize,
    num_offspring: usize,
}

impl Hash for Creature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tag.hash(state)
    }
}

impl Debug for Creature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Name: {}, generation: {}, from island {}",
            self.name,
            self.generation,
            self.native_island()
        )?;
        for (i, inst) in self.chromosome().iter().enumerate() {
            let mutation = self.chromosome_mutation[i];
            writeln!(
                f,
                "[{}][{}]  {}{}",
                if self.parents.is_empty() {
                    "seed"
                } else {
                    &self.parents[self.chromosome_parentage[i]]
                },
                i,
                inst,
                mutation
                    .map(|m| format!(" (mutation {:?})", m))
                    .unwrap_or_else(String::new)
            )?;
        }
        writeln!(f, "Fitness: {:#?}", self.fitness)
    }
}

impl Phenome for Creature {
    type Fitness = Fitness<'static>;
    type Problem = ClassificationProblem;

    fn fitness(&self) -> Option<&Self::Fitness> {
        self.fitness.as_ref()
    }

    fn scalar_fitness(&self, weighting: &str) -> Option<f64> {
        self.fitness
            .as_ref()
            .map(|f| f.scalar_with_expression(weighting))
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
        self.answers.as_ref()
    }

    fn store_answers(&mut self, answers: Vec<Self::Problem>) {
        self.answers = Some(answers);
    }

    fn is_goal_reached<'a>(&'a self, config: &'a Config) -> bool {
        if let Some(fitness) = self.scalar_fitness(&config.fitness.priority()) {
            return fitness <= config.fitness.target;
        }
        false
    }
}

impl PartialOrd for Creature {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.tag.partial_cmp(&other.tag)
    }
}

impl Ord for Creature {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tag.cmp(&other.tag)
    }
}

impl PartialEq for Creature {
    fn eq(&self, other: &Self) -> bool {
        self.tag == other.tag
    }
}

impl Eq for Creature {}

#[derive(Debug, Clone, Copy)]
struct Frame {
    start: usize,
    end: usize,
}

// means "has a genome", not "is a genome"
impl Genome for Creature {
    type Allele = machine::Inst;

    fn generation(&self) -> usize {
        self.generation
    }

    fn num_offspring(&self) -> usize {
        self.num_offspring
    }

    fn incr_num_offspring(&mut self, n: usize) {
        self.num_offspring += n
    }

    fn native_island(&self) -> usize {
        self.native_island
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
        let chromosome = random_chromosome(config, &seed);
        let length = chromosome.len();
        Self {
            chromosome,
            tag: rng.gen::<u64>(),
            //crossover_mask: rng.gen::<u64>(),
            name: crate::util::name::random(4, &seed),
            chromosome_mutation: vec![None; length],
            ..Default::default()
        }
    }

    fn crossover(mates: &[&Self], config: &Config) -> Self {
        let distribution = rand_distr::Exp::new(config.crossover_period)
            .expect("Failed to create random distribution");
        let parental_chromosomes = mates.iter().map(|m| m.chromosome()).collect::<Vec<_>>();
        let mut rng = hash_seed_rng(&mates[0]);
        let (chromosome, chromosome_parentage, parent_names) =
            // Check to see if we're performing a crossover or just cloning
            if rng.gen_range(0.0, 1.0) < config.crossover_rate {
                let names = mates.iter().map(|p| p.name.clone()).collect::<Vec<String>>();
                let (c, p) = Self::crossover_by_distribution(&distribution, &parental_chromosomes);
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
        let length = chromosome.len();
        Self {
            chromosome,
            chromosome_parentage,
            chromosome_mutation: vec![None; length],
            answers: None,
            fitness: None,
            tag: rand::random::<u64>(),
            //crossover_mask: 0,
            name,
            parents: parent_names,
            generation,
            native_island: config.island_id,
            num_offspring: 0,
        }
    }

    fn mutate(&mut self, config: &Config) {
        let mut rng = hash_seed_rng(&self);
        //let i = rng.gen_range(0, self.len());
        //self.crossover_mask ^= 1 << rng.gen_range(0, 64);
        for i in 0..self.len() {
            if !levy_decision(&mut rng, self.len(), config.mutation_exponent) {
                continue;
            }
            let seed = hash_seed(&rng.gen::<u64>());
            let m = self.chromosome_mut()[i].mutate(&config.linear_gp, &seed);
            self.chromosome_mutation[i] = Some(m);
        }
    }
}

type Mutation = u8;

fn report(window: &Window<Creature>, counter: usize, config: &Config) {
    let frame = &window.frame;
    let avg_len = frame.iter().map(|c| c.len()).sum::<usize>() as f64 / frame.len() as f64;
    let mut sketch = CountMinSketch::new(config);
    for g in frame {
        g.record_genetic_frequency(&mut sketch);
    }
    let avg_freq = frame
        .iter()
        .map(|g| g.query_genetic_frequency(&sketch))
        .sum::<f64>()
        / frame.len() as f64;
    let avg_fit = frame
        .iter()
        .filter_map(|g| g.scalar_fitness(&window.config.fitness.weighting))
        .sum::<f64>()
        / frame.len() as f64;
    log::info!(
        "[{}] Average length: {}, average genetic frequency: {}; average fitness: {}",
        counter,
        avg_len,
        avg_freq,
        avg_fit,
    );
    let soup = window.soup();
    log::info!("soup size: {}", soup.len());
    // TODO export tsv stats here too. generalize a bit.
    if let Some(ref best) = window.best {
        log::info!("Best: {:?}", best);
    }
}

fn parse_data(path: &str) -> Option<Vec<ClassificationProblem>> {
    if let Ok(mut reader) = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path) {
        let mut problems = Vec::new();
        let mut tag = 0;
        for row in reader.records() {
            if let Ok(row) = row {
                let mut vals: Vec<MachineWord> =
                    row.deserialize(None).expect("Error parsing row in data");
                let output = vals.pop().expect("Missing output field");
                let input = vals;
                problems.push(ClassificationProblem { input, output, tag });
                tag += 1;
            }
        }
        Some(problems)
    } else {
        None
    }
}

mod evaluation {
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::Arc;
    use std::thread::{spawn, JoinHandle};

    //#[cfg(not(debug_assertions))]
    use rayon::prelude::*;

    use crate::examples::linear_gp::machine::Machine;
    use crate::ontogenesis::{Develop, FitnessFn};
    use crate::util::count_min_sketch::CountMinSketch;

    use super::*;

    // the type names can get a bit confusing, here. fix this. TODO

    pub struct Evaluator {
        pub handle: JoinHandle<()>,
        tx: Sender<Creature>,
        rx: Receiver<Creature>,
        sketch: CountMinSketch,
        fitness_fn: FitnessFn<Creature, CountMinSketch, Config>,
        config: Arc<Config>,
    }

    impl Evaluator {
        pub fn spawn(
            config: &Config,
            fitness_fn: FitnessFn<Creature, CountMinSketch, Config>,
        ) -> Self {
            let (tx, our_rx): (Sender<Creature>, Receiver<Creature>) = channel();
            let (our_tx, rx): (Sender<Creature>, Receiver<Creature>) = channel();
            let config = Arc::new(config.clone());
            let conf = config.clone();
            let handle = spawn(move || {
                // TODO: parameterize sketch construction

                for mut phenome in our_rx.iter() {
                    if phenome.fitness().is_none() {
                        phenome = execute(conf.clone(), phenome);
                    }

                    our_tx.send(phenome).expect("Channel failure");
                }
            });

            let sketch = CountMinSketch::new(&config);
            Self {
                handle,
                tx,
                rx,
                sketch,
                fitness_fn,
                config,
            }
        }
    }

    // It's important that the problems in the pheno are returned sorted
    pub fn execute(config: Arc<Config>, mut creature: Creature) -> Creature {
        let problems = config.problems.as_ref().expect("No problems!");

        //#[cfg(debug_assertions)]
        //let iterator = problems.iter();
        // #[cfg(not(debug_assertions))]
        let iterator = problems.par_iter();

        let mut results = iterator
            .map(
                |ClassificationProblem {
                     input,
                     output: _expected,
                     tag,
                 }| {
                    // TODO: note that we have the expected value here. maybe this
                    // would be a good place to call the fitness function, too.
                    // TODO: is it worth creating a new machine per-thread?
                    // Probably not when it comes to unicorn, but for this, yeah.
                    let mut machine = Machine::new(&config.linear_gp);
                    let return_regs = machine.exec(creature.chromosome(), &input);
                    let output = (0..return_regs.len())
                        .map(|i| return_regs[i])
                        .fold(0, i32::max);

                    ClassificationProblem {
                        input: input.clone(),
                        output,
                        tag: *tag,
                    }
                },
            )
            .collect::<Vec<ClassificationProblem>>();
        // Sort by tag to avoid any non-seeded randomness
        results.sort_by_key(|p| p.tag);
        creature.store_answers(results);
        creature
    }

    pub fn fitness_function(
        mut creature: Creature,
        _sketch: &mut CountMinSketch,
        config: Arc<Config>,
    ) -> Creature {
        #[allow(clippy::unnecessary_fold)]
        let score = creature
            .answers()
            .as_ref()
            .expect("Missing phenotype!")
            .iter()
            .zip(config.problems.as_ref().expect("no problems!").iter())
            .filter_map(|(result, expected)| {
                assert_eq!(result.tag, expected.tag);
                // Simply counting errors.
                // TODO: consider trying distance metrics
                if result.output == expected.output {
                    None
                } else {
                    Some(1)
                }
            })
            .fold(0, |a, b| a + b);
        let mut fitness = Weighted::new(&config.fitness.weighting);
        fitness.insert("error_rate", score as f64);
        // TODO: refactor types
        //creature.set_fitness((fitness, 0.0, 0.0, len));
        creature.set_fitness(fitness);
        creature
    }

    impl Develop<Creature> for Evaluator {
        fn develop(&self, genome: Creature) -> Creature {
            self.tx.send(genome).expect("tx failure");
            self.rx.recv().expect("rx failure")
        }

        fn development_pipeline<I: Iterator<Item = Creature> + Send>(
            &self,
            inbound: I,
        ) -> Vec<Creature> {
            inbound.map(|c| self.develop(c)).collect::<Vec<Creature>>()
        }

        fn apply_fitness_function(&mut self, creature: Creature) -> Creature {
            let mut phenome = (self.fitness_fn)(creature, &mut self.sketch, self.config.clone());

            // register and measure frequency
            // TODO: move to fitness function and refactor
            phenome.record_genetic_frequency(&mut self.sketch);
            let genetic_frequency = phenome.query_genetic_frequency(&self.sketch);
            phenome
                .fitness
                .as_mut()
                .map(|fit| fit.insert("genetic_freq", genetic_frequency));

            log::debug!("fitness: {:?}", phenome.fitness());
            phenome
        }
    }
}

fn prepare(mut config: Config) -> (Config, Observer<Creature>, evaluation::Evaluator) {
    let problems = parse_data(&config.data.path);
    assert!(problems.is_some());
    // figure out the number of return registers needed
    // FIXME: refactor duplicated code out of this constructor

    let mut return_registers = problems
        .as_ref()
        .unwrap()
        .iter()
        .map(|p| p.output)
        .collect::<std::collections::HashSet<i32>>()
        .len();
    // and how many input registers
    let input_registers = problems.as_ref().unwrap()[0].input.len();

    config.problems = problems;
    if let Some(r) = config.linear_gp.return_registers {
        return_registers = std::cmp::max(return_registers, r);
    }
    let mut num_registers = return_registers + input_registers + 2;
    if let Some(r) = config.linear_gp.num_registers {
        num_registers = std::cmp::max(num_registers, r);
    }
    // NOTE: this only makes sense for classification problems, and will do strange
    // things when it comes to regression data sets. Not a priority to fix right now.
    config.linear_gp.return_registers = Some(return_registers);
    config.linear_gp.num_registers = Some(num_registers);
    log::info!("Config: {:#?}", config);
    let report_fn: ReportFn<_> = Box::new(report);
    let fitness_fn: FitnessFn<Creature, _, _> = Box::new(evaluation::fitness_function);
    let observer = Observer::spawn(&config, report_fn);
    let evaluator = evaluation::Evaluator::spawn(&config, fitness_fn);
    (config, observer, evaluator)
}

crate::impl_dominance_ord_for_phenome!(Creature, CreatureDominanceOrd);

pub fn run(config: Config) {
    //let target_fitness = config.target_fitness;
    let selection = config.selection;
    let (config, observer, evaluator) = prepare(config);

    match selection {
        Selection::Tournament => {
            let pier = Pier::new(4); // FIXME: don't hardcode
            let mut world = Tournament::<evaluation::Evaluator, Creature>::new(
                &config,
                observer,
                evaluator,
                Arc::new(pier),
            );
            while crate::keep_going() {
                world = world.evolve();
            }
        }
        Selection::Roulette => {
            let mut world = Roulette::<evaluation::Evaluator, Creature, CreatureDominanceOrd>::new(
                &config,
                observer,
                evaluator,
                CreatureDominanceOrd,
            );
            while crate::keep_going() {
                world = world.evolve();
            }
        }
        Selection::Metropolis => {
            let mut world =
                Metropolis::<evaluation::Evaluator, Creature>::new(&config, observer, evaluator);
            while crate::keep_going() {
                world = world.evolve();
            }
        }
        sel => unimplemented!("{:?} not implemented for {:?}", sel, config.job),
    }
}
