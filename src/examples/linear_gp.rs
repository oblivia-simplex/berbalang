use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;
use std::{fmt, iter};

use rand::{thread_rng, Rng};
use serde_derive::Deserialize;

use crate::configure::{Configure, ConfigureObserver};
use crate::evaluator::{Evaluate, FitnessFn};
use crate::evolution::{Epoch, Genome, Phenome, Problem};
use crate::observer::{Observer, ReportFn};

pub type Fitness = usize;
pub type MachineWord = i32;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ObserverConfig {
    window_size: usize,
}

impl ConfigureObserver for ObserverConfig {
    fn window_size(&self) -> usize {
        self.window_size
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct DataConfig {
    pub path: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    mut_rate: f32,
    pub init_len: usize,
    pop_size: usize,
    tournament_size: usize,
    pub num_offspring: usize,
    pub target: String,
    pub target_fitness: usize,
    observer: ObserverConfig,
    pub data: DataConfig,
    problems: Option<Vec<Problem>>,
    max_length: usize,
}

impl Configure for Config {
    fn assert_invariants(&self) {
        assert!(self.tournament_size >= self.num_offspring + 2);
        assert_eq!(self.num_offspring, 2); // all that's supported for now
    }

    fn mutation_rate(&self) -> f32 {
        self.mut_rate
    }

    fn population_size(&self) -> usize {
        self.pop_size
    }

    fn tournament_size(&self) -> usize {
        self.tournament_size
    }

    fn observer_window_size(&self) -> usize {
        self.observer.window_size
    }

    fn num_offspring(&self) -> usize {
        self.num_offspring
    }

    fn max_length(&self) -> usize {
        self.max_length
    }
}

pub mod machine {
    use std::fmt::{self, Display};

    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    use crate::examples::linear_gp::MachineWord;

    //use std::sync::atomic::AtomicUsize;

    //static CORE: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
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

    pub const NUM_REGISTERS: usize = 4;
    const MAX_STEPS: usize = 0x1000;
    // TODO make this configurable
    const FIRST_INPUT_REGISTER: usize = 1;
    const RETURN_REGISTER: usize = 0;

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
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
        pub fn random() -> Self {
            Self {
                op: rand::random::<Op>(),
                a: rand::random::<usize>() % NUM_REGISTERS,
                b: rand::random::<usize>() % NUM_REGISTERS,
            }
        }

        pub fn mutate(&mut self) {
            let mut rng = thread_rng();
            let mutation = rng.gen::<u8>() % 4;

            match mutation {
                0 => self.op = rand::random(),
                1 => self.a = (self.a + 1) % NUM_REGISTERS,
                2 => self.b = (self.b + 1) % NUM_REGISTERS,
                3 => std::mem::swap(&mut self.a, &mut self.b),
                _ => unreachable!("out of range"),
            }
        }
    }

    pub struct Machine {
        registers: [MachineWord; NUM_REGISTERS],
        pc: usize,
        max_steps: usize,
    }

    impl Machine {
        pub(crate) fn new() -> Self {
            Self {
                registers: [0; NUM_REGISTERS],
                pc: 0,
                max_steps: MAX_STEPS,
            }
        }

        #[inline]
        fn set(&mut self, reg: usize, val: MachineWord) {
            self.registers[reg % NUM_REGISTERS] = val
        }

        fn eval(&mut self, inst: Inst) {
            use Op::*;

            macro_rules! set {
                ($val:expr) => {
                    self.set(inst.a, $val)
                };
            }
            let r = self.registers;

            match inst.op {
                Add => set!(r[inst.a].wrapping_add(r[inst.b])),
                Div => {
                    let b = if r[inst.b] == 0 { 1 } else { r[inst.b] };
                    set!(r[inst.a].wrapping_div(b))
                }
                Mov => set!(r[inst.b]),
                Mult => set!(r[inst.a].wrapping_mul(r[inst.b])),
                Sub => set!(r[inst.a].wrapping_sub(r[inst.b])),
                Xor => set!(r[inst.a] ^ r[inst.b]),
                Set(n) => set!(n),
                Lsl => set!(r[inst.a].wrapping_shl(r[inst.b] as u32)),
                And => set!(r[inst.a] & r[inst.b]),
                Jle => {
                    if r[inst.a] <= 0 {
                        self.pc = r[inst.b] as usize
                    }
                }
            }
        }

        fn flush_registers(&mut self) {
            self.registers.iter_mut().for_each(|i| *i = 0);
        }

        fn load_input(&mut self, inputs: &[MachineWord]) {
            if inputs.len() > NUM_REGISTERS - 1 {
                log::error!("Too many inputs to load into input registers. Wrapping.");
            }
            (FIRST_INPUT_REGISTER..NUM_REGISTERS)
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

        fn return_value(&self) -> MachineWord {
            self.registers[RETURN_REGISTER]
        }

        pub fn exec(&mut self, code: &[Inst], input: &[MachineWord]) -> MachineWord {
            self.flush_registers();
            self.load_input(input);
            self.exec_insts(code);
            self.return_value()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Genotype(pub Vec<machine::Inst>);

impl Genotype {
    /// Produce a genotype with exactly `len` random instructions.
    /// Pass a randomly generated `len` to randomize the length.
    pub fn random(len: usize) -> Self {
        Self(
            iter::repeat(())
                .map(|()| machine::Inst::random())
                .take(len)
                .collect(),
        )
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub type Answer = Vec<Problem>;

#[derive(Clone, Default)]
pub struct Creature {
    genotype: Genotype,
    answers: Option<Answer>,
    fitness: Option<Fitness>,
    tag: u64,
}

impl Debug for Creature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, inst) in self.instructions().iter().enumerate() {
            writeln!(f, "[{}]  {}", i, inst)?;
        }
        writeln!(f)
    }
}

impl Creature {
    fn len(&self) -> usize {
        self.genotype.len()
    }

    fn instructions(&self) -> &Vec<machine::Inst> {
        &self.genotype.0
    }
}

impl Phenome for Creature {
    type Inst = machine::Inst;
    type Fitness = Fitness;

    fn fitness(&self) -> Option<Fitness> {
        self.fitness
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

// means "has a genome", not "is a genome"
impl Genome for Creature {
    type Params = Config;

    fn random(params: &Self::Params) -> Self {
        let mut rng = thread_rng();
        let len = rng.gen_range(1, params.init_len);
        let genotype = Genotype::random(len);
        Self {
            genotype,
            tag: rng.gen::<u64>(),
            ..Default::default()
        }
    }

    fn crossover<C: Configure>(&self, mate: &Self, params: Arc<C>) -> Vec<Self> {
        let mut rng = thread_rng();
        // TODO: note how similar this is to the GA crossover.
        // refactor this out into a more general method
        let split_m: usize = rng.gen::<usize>() % self.len();
        let split_f: usize = rng.gen::<usize>() % mate.len();
        let (m1, m2) = self.genotype.0.split_at(split_m);
        let (f1, f2) = mate.genotype.0.split_at(split_f);

        let half = params.max_length() / 2;
        let mut c1 = m1[0..m1.len().min(half)].to_vec();
        c1.extend(f2[0..f2.len().min(half)].iter());
        let mut c2 = f1[0..f1.len().min(half)].to_vec();
        c2.extend(m2[0..m2.len().min(half)].iter());

        vec![
            Self {
                genotype: Genotype(c1),
                tag: rng.gen::<u64>(),
                ..Default::default()
            },
            Self {
                genotype: Genotype(c2),
                tag: rng.gen::<u64>(),
                ..Default::default()
            },
        ]
    }

    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let i = rng.gen_range(0, self.len());
        self.genotype.0[i].mutate();
    }
}

fn report(_window: &[Creature]) {
    log::error!("report fn not yet implemented")
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

impl Epoch<evaluation::Evaluator, Creature, Config> {
    pub fn new(mut config: Config) -> Self {
        let problems = parse_data(&config.data.path);
        assert!(problems.is_some());
        config.problems = problems;
        let config = Arc::new(config);
        let population = iter::repeat(())
            .map(|()| Creature::random(&config))
            .take(config.population_size())
            .collect();
        let report_fn: ReportFn<_> = Box::new(report);
        let fitness_fn: FitnessFn<Creature, _> = Box::new(evaluation::fitness_function);
        let observer = Observer::spawn(config.clone(), report_fn);
        let evaluator = evaluation::Evaluator::spawn(config.clone(), fitness_fn);

        Self {
            population,
            config,
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

    #[cfg(not(debug_assertions))]
    use rayon::prelude::*;

    use crate::evaluator::{Evaluate, FitnessFn};
    use crate::examples::linear_gp::machine::Machine;

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

        #[cfg(debug_assertions)]
        let iterator = problems.iter();
        #[cfg(not(debug_assertions))]
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
                    let mut machine = Machine::new();
                    let output = machine.exec(creature.instructions(), &input);
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

    pub fn fitness_function<P: Phenome>(mut creature: P, params: Arc<Config>) -> P {
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
                    log::debug!("correct result: {:?}", result);
                    None
                } else {
                    let dif = (result.output as i64 - expected.output as i64).abs() as f64;
                    let score = dif.tanh() * 100.0;
                    Some(score as usize)
                }
            })
            .fold(0, |a, b| a + b);
        log::debug!("fitness is {}", fitness);
        creature.set_fitness(fitness.into());
        creature
    }

    impl Evaluate<Creature> for Evaluator {
        type Params = Config;
        type Fitness = Fitness;

        fn evaluate(&self, phenome: Creature) -> Creature {
            self.tx.send(phenome).expect("tx failure");
            self.rx.recv().expect("rx failure")
        }

        fn spawn(params: Arc<Self::Params>, fitness_fn: FitnessFn<Creature, Self::Params>) -> Self {
            let (tx, our_rx): (Sender<Creature>, Receiver<Creature>) = channel();
            let (our_tx, rx): (Sender<Creature>, Receiver<Creature>) = channel();

            let handle = spawn(move || {
                //let mut machine = Machine::new(); // This too could be parameterized
                for phenome in our_rx {
                    let phenome = execute(params.clone(), phenome);
                    let phenome = (fitness_fn)(phenome, params.clone());
                    our_tx.send(phenome).expect("Channel failure");
                }
            });

            Self { handle, tx, rx }
        }
    }
}

pub fn run(config: Config) -> Option<Creature> {
    let target_fitness: Fitness = config.target_fitness;
    let mut world = Epoch::<evaluation::Evaluator, Creature, Config>::new(config);

    loop {
        world = world.evolve();
        if world.target_reached(target_fitness) {
            println!("\n***** Success after {} epochs! *****", world.iteration);
            println!("{:#?}", world.best);
            let best = std::mem::take(&mut world.best).unwrap();
            std::env::set_var("RUST_LOG", "trace");
            let best = evaluation::execute(world.config, best);
            return Some(best);
        };
    }
}
