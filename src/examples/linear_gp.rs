use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;
use std::{fmt, iter};

use rand::{thread_rng, Rng};

use crate::configure::{Config, ObserverConfig, Problem};
use crate::evaluator::{Evaluate, FitnessFn};
use crate::evolution::{Epoch, Genome, Phenome};
use crate::fitness::Pareto;
use crate::observer::{Observer, ReportFn};
use crate::util;
use crate::util::count_min_sketch::DecayingSketch;

pub type Fitness = Pareto;
// try setting fitness to (usize, usize);
pub type MachineWord = i32;

pub mod machine {
    use std::fmt::{self, Display};
    use std::hash::Hash;

    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    use crate::configure::MachineConfig;
    use crate::examples::linear_gp::MachineWord;

    #[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
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
            }
        }
    }

    pub const NUM_OPS: usize = 10;

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
                _ => unreachable!("out of range"),
            }
        }
    }

    pub type Register = usize;

    // TODO make this configurable
    //const FIRST_INPUT_REGISTER: usize = 1;
    //const RETURN_REGISTER: usize = 0;

    #[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
    pub struct Inst {
        pub op: Op,
        pub a: Register,
        pub b: Register,
    }

    impl Display for Inst {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.op {
                Op::Set(n) => write!(f, "SET  R{}  0x{:X}", self.a, n),
                _ => write!(f, "{}  R{}, R{}", self.op, self.a, self.b),
            }
        }
    }

    impl Inst {
        pub fn random(params: &MachineConfig) -> Self {
            let num_registers = params.num_registers.unwrap();
            Self {
                op: rand::random::<Op>(),
                a: rand::random::<usize>() % num_registers,
                b: rand::random::<usize>() % num_registers,
            }
        }

        pub fn mutate(&mut self, params: &MachineConfig) {
            let num_registers = params.num_registers.unwrap();
            let mut rng = thread_rng();
            let mutation = rng.gen::<u8>() % 4;

            match mutation {
                0 => self.op = rand::random(),
                1 => self.a = (self.a + 1) % num_registers,
                2 => self.b = (self.b + 1) % num_registers,
                3 => std::mem::swap(&mut self.a, &mut self.b),
                _ => unreachable!("out of range"),
            }
        }
    }

    pub struct Machine {
        return_registers: usize,
        registers: Vec<MachineWord>,
        pc: usize,
        max_steps: usize,
    }

    impl Machine {
        pub(crate) fn new(params: &MachineConfig) -> Self {
            Self {
                return_registers: params.return_registers.unwrap(),
                registers: vec![0; params.num_registers.unwrap()],
                pc: 0,
                max_steps: params.max_steps,
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
fn random_chromosome(params: &Config) -> Genotype {
    let mut rng = thread_rng();
    let len = rng.gen_range(params.min_init_len, params.max_init_len) + 1;
    iter::repeat(())
        .map(|()| machine::Inst::random(&params.machine))
        .take(len)
        .collect()
}

pub type Answer = Vec<Problem>;

#[derive(Clone, Default)]
pub struct Creature {
    chromosome: Genotype,
    chromosome_parentage: Vec<usize>,
    answers: Option<Answer>,
    pub fitness: Option<Fitness>,
    tag: u64,
    //crossover_mask: u64,
    name: String,
    parents: Vec<String>,
    generation: usize,
}

impl Debug for Creature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Name: {}, generation: {}", self.name, self.generation)?;
        for (i, inst) in self.chromosome().iter().enumerate() {
            writeln!(
                f,
                "[{}][{}]  {}",
                if self.parents.is_empty() {
                    "seed"
                } else {
                    &self.parents[self.chromosome_parentage[i]]
                },
                i,
                inst
            )?;
        }
        writeln!(f)
    }
}

impl Phenome for Creature {
    type Fitness = Fitness;

    fn fitness(&self) -> Option<&Fitness> {
        self.fitness.as_ref()
    }

    fn set_fitness(&mut self, f: Fitness) {
        self.fitness = Some(f)
    }

    fn tag(&self) -> u64 {
        self.tag
    }

    fn set_tag(&mut self, tag: u64) {
        self.tag = tag
    }

    fn problems(&self) -> Option<&Vec<Problem>> {
        self.answers.as_ref()
    }

    fn store_answers(&mut self, answers: Vec<Problem>) {
        self.answers = Some(answers);
    }

    fn len(&self) -> usize {
        self.chromosome.len()
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

    fn chromosome(&self) -> &[Self::Allele] {
        &self.chromosome
    }

    fn chromosome_mut(&mut self) -> &mut [Self::Allele] {
        &mut self.chromosome
    }

    fn random(params: &Config) -> Self {
        let mut rng = thread_rng();
        let chromosome = random_chromosome(params);
        Self {
            chromosome,
            tag: rng.gen::<u64>(),
            //crossover_mask: rng.gen::<u64>(),
            name: crate::util::name::random(4),
            ..Default::default()
        }
    }

    fn crossover(mates: &[Self], params: &Config) -> Self {
        let distribution = rand_distr::Exp::new(params.crossover_period)
            .expect("Failed to create random distribution");
        let parental_chromosomes = mates.iter().map(Genome::chromosome).collect::<Vec<_>>();
        let mut rng = thread_rng();
        let (chromosome, chromosome_parentage, parent_names) =
                // Check to see if we're performing a crossover or just cloning
                if rng.gen_range(0.0, 1.0) < params.crossover_rate() {
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
        Self {
            chromosome,
            chromosome_parentage,
            answers: None,
            fitness: None,
            tag: rand::random::<u64>(),
            //crossover_mask: 0,
            name: util::name::random(4),
            parents: parent_names,
            generation,
        }
    }

    fn mutate(&mut self, params: &Config) {
        let mut rng = thread_rng();
        let i = rng.gen_range(0, self.len());
        //self.crossover_mask ^= 1 << rng.gen_range(0, 64);
        self.chromosome_mut()[i].mutate(&params.machine);
    }
}

fn report(window: &[Creature], counter: usize, _params: &ObserverConfig) {
    let avg_len = window.iter().map(|c| c.len()).sum::<usize>() as f64 / window.len() as f64;
    let mut sketch = DecayingSketch::default();
    for g in window {
        g.record_genetic_frequency(&mut sketch, 1000).unwrap();
    }
    let avg_freq = window
        .iter()
        .map(|g| g.measure_genetic_frequency(&sketch).unwrap())
        .sum::<f64>()
        / window.len() as f64;
    let avg_fit = window
        .iter()
        .filter_map(|g| g.fitness.as_ref().map(|f| f.0[0]))
        .sum::<f64>()
        / window.len() as f64;
    log::info!(
        "[{}] Average length: {}, average genetic frequency: {}; average fitness: {}",
        counter,
        avg_len,
        avg_freq,
        avg_fit,
    );
}

fn parse_data(path: &str) -> Option<Vec<Problem>> {
    if let Ok(mut reader) = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path) {
        let mut problems = Vec::new();
        let mut tag = 0;
        for row in reader.records() {
            if let Ok(row) = row {
                let mut vals: Vec<MachineWord> =
                    row.deserialize(None).expect("Error parsing row in data");
                let output = vals.pop().expect("Missing output field");
                let input = vals;
                problems.push(Problem { input, output, tag });
                tag += 1;
            }
        }
        Some(problems)
    } else {
        None
    }
}

impl Epoch<evaluation::Evaluator, Creature> {
    pub fn new(mut config: Config) -> Self {
        let problems = parse_data(&config.data.path);
        assert!(problems.is_some());
        // figure out the number of return registers needed
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
        if let Some(r) = config.machine.return_registers {
            return_registers = std::cmp::max(return_registers, r);
        }
        let mut num_registers = return_registers + input_registers + 2;
        if let Some(r) = config.machine.num_registers {
            num_registers = std::cmp::max(num_registers, r);
        }
        config.machine.return_registers = Some(return_registers);
        config.machine.num_registers = Some(num_registers);
        log::info!("Config: {:#?}", config);
        let population = iter::repeat(())
            .map(|()| Creature::random(&config))
            .take(config.population_size())
            .collect();
        let report_fn: ReportFn<_> = Box::new(report);
        let fitness_fn: FitnessFn<Creature, _> = Box::new(evaluation::fitness_function);
        let observer = Observer::spawn(&config, report_fn);
        let evaluator = evaluation::Evaluator::spawn(&config, fitness_fn);

        Self {
            population,
            config: Arc::new(config),
            best: None,
            iteration: 0,
            observer,
            evaluator,
        }
    }
}

mod evaluation {
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::Arc;
    use std::thread::{spawn, JoinHandle};

    //#[cfg(not(debug_assertions))]
    use rayon::prelude::*;

    use crate::evaluator::{Evaluate, FitnessFn};
    use crate::examples::linear_gp::machine::Machine;
    use crate::util::count_min_sketch::DecayingSketch;

    use super::*;

    // the type names can get a bit confusing, here. fix this. TODO

    pub struct Evaluator {
        pub handle: JoinHandle<()>,
        tx: Sender<Creature>,
        rx: Receiver<Creature>,
    }

    // It's important that the problems in the pheno are returned sorted
    pub fn execute(params: Arc<Config>, mut creature: Creature) -> Creature {
        let problems = params.problems.as_ref().expect("No problems!");

        //#[cfg(debug_assertions)]
        //let iterator = problems.iter();
        // #[cfg(not(debug_assertions))]
        let iterator = problems.par_iter();

        let mut results = iterator
            .map(
                |Problem {
                     input,
                     output: _expected,
                     tag,
                 }| {
                    // TODO: note that we have the expected value here. maybe this
                    // would be a good place to call the fitness function, too.
                    // TODO: is it worth creating a new machine per-thread?
                    // Probably not when it comes to unicorn, but for this, yeah.
                    let mut machine = Machine::new(&params.machine);
                    let return_regs = machine.exec(creature.chromosome(), &input);
                    let output = (0..return_regs.len())
                        .map(|i| return_regs[i])
                        .fold(0, i32::max);

                    Problem {
                        input: input.clone(),
                        output,
                        tag: *tag,
                    }
                },
            )
            .collect::<Vec<Problem>>();
        results.sort_by_key(|p| p.tag);
        creature.store_answers(results);
        creature
    }

    pub fn fitness_function<P: Phenome<Fitness = Fitness>>(
        mut creature: P,
        params: Arc<Config>,
    ) -> P {
        #[allow(clippy::unnecessary_fold)]
        let fitness = creature
            .problems()
            .as_ref()
            .expect("Missing phenotype!")
            .iter()
            .zip(params.problems.as_ref().expect("no problems!").iter())
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
        // TODO: refactor types
        //creature.set_fitness((fitness, 0.0, 0.0, len));
        creature.set_fitness(vec![fitness as f64].into());
        creature
    }

    impl Evaluate<Creature> for Evaluator {
        type Params = Config;

        fn evaluate(&self, genome: Creature) -> Creature {
            self.tx.send(genome).expect("tx failure");
            self.rx.recv().expect("rx failure")
        }

        // fn pipeline<I: Iterator<Item = Creature> + Send>(
        //     &self,
        //     inbound: I,
        // ) -> Box<dyn Iterator<Item = Creature>> {
        //     Box::new(inbound.map(|p| {
        //         self.tx.send(p).expect("tx failure");
        //         self.rx.recv().expect("rx failure")
        //     }))
        // }

        fn spawn(params: &Self::Params, fitness_fn: FitnessFn<Creature, Self::Params>) -> Self {
            let (tx, our_rx): (Sender<Creature>, Receiver<Creature>) = channel();
            let (our_tx, rx): (Sender<Creature>, Receiver<Creature>) = channel();
            let params = Arc::new(params.clone());

            let handle = spawn(move || {
                // TODO: parameterize sketch construction
                let mut sketch = DecayingSketch::default();
                let mut counter = 0;

                for phenome in our_rx {
                    counter += 1;
                    let phenome = execute(params.clone(), phenome);
                    let mut phenome = (fitness_fn)(phenome, params.clone());
                    // register and measure frequency
                    phenome
                        .record_genetic_frequency(&mut sketch, counter)
                        .expect("Failed to record phenomic frequency");
                    let _genomic_frequency = phenome
                        .measure_genetic_frequency(&sketch)
                        .expect("Failed to measure genetic diversity. Check timestamps.");
                    sketch
                        .insert(phenome.problems(), counter)
                        .expect("Failed to update phenomic diversity");
                    let phenomic_frequency = sketch
                        .query(phenome.problems())
                        .expect("Failed to measure phenomic diversity");

                    phenome.fitness.as_mut().map(|f| {
                        f.push(phenomic_frequency);
                        //f.push(genomic_frequency)
                    });
                    //.map(|(_fit, p_freq, g_freq, len)| { *g_freq = genomic_frequency; *p_freq = phenomic_frequency } );
                    log::debug!("[{}] fitness: {:?}", counter, phenome.fitness());

                    our_tx.send(phenome).expect("Channel failure");
                }
            });

            Self { handle, tx, rx }
        }
    }
}

pub fn run(config: Config) -> Option<Creature> {
    //let target_fitness = config.target_fitness;
    let mut world = Epoch::<evaluation::Evaluator, Creature>::new(config);

    loop {
        world = world.evolve();
        // Obviously needs refactoring TODO
        // if let Some(true) = world
        //     .best
        //     .as_ref()
        //     .and_then(|b| b.fitness.map(|f| f.0 == target_fitness))
        if false
        // TODO: find way to state stop condition
        {
            println!("\n***** Success after {} epochs! *****", world.iteration);
            println!("{:#?}", world.best);
            let best = std::mem::take(&mut world.best).unwrap();
            std::env::set_var("RUST_LOG", "trace");
            let best = evaluation::execute(world.config, best);
            return Some(best);
        };
    }
}
