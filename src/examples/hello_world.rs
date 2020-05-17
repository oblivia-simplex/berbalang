use std::iter;
use std::iter::Iterator;

use log;
use rand::distributions::Alphanumeric;
use rand::prelude::*;
use serde_derive::Deserialize;

use crate::configure::{Configure, ConfigureObserver};
use crate::evaluator::Evaluate;
use crate::evolution::*;
use crate::observer::Observer;

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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Genotype {
    genes: String,
    fitness: Option<Fitness>,
}

// because this is a GA we identify genome and phenome
impl Phenome for Genotype {
    type Fitness = Fitness;

    fn fitness(&self) -> Option<Self::Fitness> {
        self.fitness
    }
}

impl Genotype {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn mate(&self, other: &Self, params: &Config) -> Vec<Self> {
        log::debug!("Mating {:?} and {:?}", self, other);
        let mut offspring = self.crossover(other);
        let mut rng = thread_rng();
        for child in offspring.iter_mut() {
            if rng.gen::<f32>() < params.mutation_rate() {
                child.mutate();
            }
        }
        log::debug!("Offspring: {:?}", offspring);
        offspring
    }

    pub fn fitter_than(&self, other: &Self) -> bool {
        match (self.fitness, other.fitness) {
            (Some(a), Some(b)) if a < b => true,
            _ => false,
        }
    }
}

impl Genome for Genotype {
    type Params = Config;

    fn random(params: &Self::Params) -> Self {
        let mut rng = thread_rng();
        let len = rng.gen_range(1, params.init_len);
        let s: String = iter::repeat(())
            .map(|()| rng.sample(Alphanumeric))
            .take(len)
            .collect();
        Self {
            genes: s,
            fitness: None,
        }
    }

    fn crossover(&self, mate: &Self) -> Vec<Self> {
        let mut rng = thread_rng();
        let split_m: usize = rng.gen::<usize>() % self.len();
        let split_f: usize = rng.gen::<usize>() % mate.len();
        let (m1, m2) = self.genes.split_at(split_m);
        let (f1, f2) = mate.genes.split_at(split_f);
        vec![
            Genotype {
                genes: format!("{}{}", m1, f2),
                fitness: None,
            },
            Genotype {
                genes: format!("{}{}", f1, m2),
                fitness: None,
            },
        ]
    }

    fn mutate(&mut self) {
        let mut rng: ThreadRng = thread_rng();
        let mutation = rng.gen::<u8>() % 4;
        match mutation {
            // replace some character with random character
            0 => unsafe {
                let i: usize = rng.gen::<usize>() % self.len();
                let bytes = self.genes.as_bytes_mut();
                let mut c: u8 = 0;
                while c < 0x20 || 0x7E < c {
                    c = rng.gen::<u8>();
                }
                bytes[i] = c;
            },
            // swaps halves of the string
            1 => {
                let tmp = self.genes.split_off(self.genes.len() / 2);
                self.genes = format!("{}{}", self.genes, tmp);
            }
            // swaps two random characters in the string
            2 => unsafe {
                let len = self.genes.len();
                let bytes = self.genes.as_mut_vec();
                let i = rng.gen::<usize>() % len;
                let j = rng.gen::<usize>() % len;
                if i != j {
                    bytes.swap(i, j)
                }
            },
            // reverse the string
            3 => {
                self.genes = self.genes.chars().rev().collect::<String>();
            }
            _ => unreachable!("Unreachable"),
        }
    }
}

impl Epoch<evaluation::Evaluator, Genotype, Genotype, Config> {
    pub fn new(config: Config) -> Self {
        let population = iter::repeat(())
            .map(|()| Genotype::random(&config))
            .take(config.pop_size)
            .collect();
        let observer = Observer::spawn(&config);
        let evaluator = evaluation::Evaluator::spawn(&config);
        Self {
            population,
            config,
            best: None,
            iteration: 0,
            observer,
            evaluator,
        }
    }

    pub fn update_best(best: Option<Genotype>, champ: &Genotype) -> Option<Genotype> {
        match best {
            Some(ref best) if champ.fitter_than(best) => {
                log::info!("new champ {:?}", champ);
                Some(champ.clone())
            }
            None => {
                log::info!("new champ {:?}", champ);
                Some(champ.clone())
            }
            _ => best,
        }
    }

    pub fn target_reached(&self, target: Fitness) -> bool {
        self.best
            .as_ref()
            .and_then(|b| b.fitness())
            .map_or(false, |f| f <= target)
    }
}

// Genotype doubles as Phenotype here
impl Epochal for Epoch<evaluation::Evaluator, Genotype, Genotype, Config> {
    type Observer = Observer<Genotype>;
    type Evaluator = evaluation::Evaluator;

    fn evolve(self) -> Self {
        // destruct the Epoch
        let Self {
            mut population,
            mut best,
            observer,
            evaluator,
            config,
            iteration,
        } = self;

        let tournament_size = config.tournament_size();
        let mut rng = thread_rng();
        // shuffle is O(1)!
        population.shuffle(&mut rng);
        let mut combatants = Vec::new();
        for _ in 0..tournament_size {
            if let Some(combatant) = population.pop() {
                let scored = evaluator.evaluate(combatant);
                observer.observe(scored.clone());
                combatants.push(scored)
            } else {
                log::error!("Population empty!");
            }
        }
        // none of the fitnesses will be None, here, so we can freely compare them
        combatants.sort_by(|a, b| a.fitness.cmp(&b.fitness));
        best = Self::update_best(best, &combatants[0]);

        // kill one off for every offspring to be produced
        for _ in 0..config.num_offspring {
            let _ = combatants.pop();
        }

        // replace the combatants that will neither breed nor die
        let bystanders = config.tournament_size() - (config.num_offspring + 2);
        for _ in 0..bystanders {
            if let Some(c) = combatants.pop() {
                population.push(c);
            }
        }
        // TODO implement breeder, similar to observer, etc?
        let mother = combatants.pop().unwrap();
        let father = combatants.pop().unwrap();
        let offspring: Vec<Genotype> = mother.mate(&father, &config);

        // return everyone to the population
        population.push(mother);
        population.push(father);
        for child in offspring.into_iter() {
            population.push(child)
        }

        // put the epoch back together
        Self {
            population,
            config,
            best,
            iteration,
            observer,
            evaluator,
        }
    }
}

pub fn run(config: Config) -> Option<Genotype> {
    let target_fitness: Fitness = config.target_fitness.into();
    let mut world = Epoch::new(config);

    loop {
        world = world.evolve();
        if world.target_reached(target_fitness) {
            println!("\n***** Success after {} epochs! *****", world.iteration);
            println!("{:#?}", world.best);
            return world.best;
        };
    }
}

mod evaluation {
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::thread::{JoinHandle, spawn};

    use super::*;

    type Phenotype = Genotype;

    pub struct Evaluator {
        pub handle: JoinHandle<()>,
        tx: Sender<Genotype>,
        rx: Receiver<Genotype>,
    }

    impl Evaluate for Evaluator {
        // because this is a GA, not a GP
        type Params = Config;
        type Phenotype = Phenotype;
        type Fitness = usize;

        fn evaluate(&self, phenome: Self::Phenotype) -> Self::Phenotype {
            self.tx.send(phenome).expect("tx failure");
            self.rx.recv().expect("rx failure")
        }

        fn spawn(params: &Self::Params) -> Self {
            let (tx, our_rx): (Sender<Genotype>, Receiver<Genotype>) = channel();
            let (our_tx, rx): (Sender<Genotype>, Receiver<Genotype>) = channel();

            let target = params.target.clone();

            let handle = spawn(move || {
                for mut phenome in our_rx {
                    if phenome.fitness.is_none() {
                        phenome.fitness = Some(fitness_function(&phenome, &target))
                    }
                    our_tx.send(phenome).expect("Channel failure");
                }
            });

            Self { handle, tx, rx }
        }
    }

    #[inline]
    fn fitness_function(phenome: &Genotype, target: &str) -> Fitness {
        distance::damerau_levenshtein(&phenome.genes, target).into()
    }
}

//build_observation_mod!(observation, Genotype, Config);
