use std::iter;

use rand::{Rng, thread_rng};
use serde_derive::Deserialize;

use crate::configure::{Configure, ConfigureObserver};
use crate::evolution::{FitnessScalar, Genome, Phenome};

pub type Fitness = FitnessScalar<usize>;

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
pub struct Config {
    mut_rate: f32,
    pub init_len: usize,
    pop_size: usize,
    tournament_size: usize,
    pub num_offspring: usize,
    pub target: String,
    pub target_fitness: usize,
    observer: ObserverConfig,
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
}

mod machine {
    use rand::{Rng, thread_rng};
    use rand::distributions::{Distribution, Standard};

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub enum Op {
        Add,
        Div,
        Mov,
        Mult,
        Sub,
    }

    pub const NUM_OPS: usize = 5;

    impl Distribution<Op> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Op {
            use Op::*;
            match rng.gen_range(0, NUM_OPS) {
                0 => Add,
                1 => Div,
                2 => Mov,
                3 => Mult,
                4 => Sub,
                _ => unreachable!("out of range"),
            }
        }
    }

    pub type Register = usize;

    pub const NUM_REGISTERS: usize = 8;
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
        registers: [i32; NUM_REGISTERS],
        pc: usize,
        max_steps: usize,
    }

    impl Machine {
        fn new() -> Self {
            Self {
                registers: [0_i32; NUM_REGISTERS],
                pc: 0,
                max_steps: MAX_STEPS,
            }
        }

        #[inline]
        fn set(&mut self, reg: usize, val: i32) {
            self.registers[reg % NUM_REGISTERS] = val
        }

        fn eval(&mut self, inst: Inst) {
            use Op::*;

            macro_rules! set {
                ($val:expr) => {
                    self.set(inst.a, $val)
                };
            }
            let mut r = self.registers;

            match inst.op {
                Add => set!(r[inst.a] + r[inst.b]),
                Div => {
                    let b = if r[inst.b] == 0 { 1 } else { r[inst.b] };
                    set!(r[inst.a] / b)
                }
                Mov => set!(r[inst.b]),
                Mult => set!(r[inst.a] * r[inst.b]),
                Sub => set!(r[inst.a] - r[inst.b]),
            }
        }

        fn flush_registers(&mut self) {
            self.registers.iter_mut().for_each(|i| *i = 0);
        }

        fn load_input(&mut self, inputs: &Vec<i32>) {
            if inputs.len() > NUM_REGISTERS - 1 {
                log::error!("Too many inputs to load into input registers. Wrapping.");
            }
            (FIRST_INPUT_REGISTER..NUM_REGISTERS)
                .zip(inputs.iter())
                .for_each(|(r, i)| self.set(r, *i));
        }

        fn exec_insts(&mut self, code: &Vec<Inst>) {
            let mut step = 0;
            self.pc = 0;

            let fetch = |i| code[i % code.len()];

            let max_steps = self.max_steps;
            while step < max_steps {
                let inst = fetch(self.pc);
                self.pc += 1;
                self.eval(inst);
                step += 1;
            }
        }

        fn return_value(&self) -> i32 {
            self.registers[RETURN_REGISTER]
        }

        pub fn exec(&mut self, code: &Vec<Inst>, input: &Vec<i32>) -> i32 {
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

#[derive(Debug, Clone, Default)]
pub struct Phenotype {
    pub input: Vec<i32>,
    pub output: i32,
}

#[derive(Clone, Debug, Default)]
pub struct Creature {
    genotype: Genotype,
    phenotype: Option<Phenotype>,
    fitness: Option<Fitness>,
}

impl Creature {
    fn len(&self) -> usize {
        self.genotype.len()
    }

    pub fn fitness(&self) -> Option<Fitness> {
        self.fitness
    }
}

impl Phenome for Creature {
    type Fitness = FitnessScalar<usize>;

    fn fitness(&self) -> Option<Self::Fitness> {
        self.fitness
    }
}

// means "has a genome", not "is a genome"
impl Genome for Creature {
    type Params = Config;

    fn random(params: &Self::Params) -> Self {
        let mut rng = thread_rng();
        let len = rng.gen_range(1, params.init_len);
        let genotype = Genotype::random(len);
        Self {
            genotype,
            ..Default::default()
        }
    }

    fn crossover(&self, mate: &Self) -> Vec<Self> {
        let mut rng = thread_rng();
        // TODO: note how similar this is to the GA crossover.
        // refactor this out into a more general method
        let split_m: usize = rng.gen::<usize>() % self.len();
        let split_f: usize = rng.gen::<usize>() % mate.len();
        let (m1, m2) = self.genotype.0.split_at(split_m);
        let (f1, f2) = mate.genotype.0.split_at(split_f);
        let mut c1 = m1.to_vec();
        c1.extend(f2.into_iter());
        let mut c2 = f1.to_vec();
        c2.extend(m2.into_iter());
        vec![
            Self {
                genotype: Genotype(c1),
                ..Default::default()
            },
            Self {
                genotype: Genotype(c2),
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

