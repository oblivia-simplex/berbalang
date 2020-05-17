use std::iter;
use std::iter::Iterator;

use log;
use rand::distributions::Alphanumeric;
use rand::prelude::*;
use serde_derive::Deserialize;

use crate::configure::{Configure, ConfigureObserver};
use crate::evaluator::Evaluate;
use crate::evolution::*;
use crate::observer::Observe;

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

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Genotype {
    genes: String,
    fitness: Option<usize>,
}

// because this is a GA we identify genome and phenome
impl Phenome for Genotype {
    type Fitness = usize;

    fn fitness(&self) -> Option<Self::Fitness> {
        self.fitness
    }
}

impl Genotype {
    pub fn new(len: usize) -> Self {
        let mut rng = thread_rng();
        let s: String = iter::repeat(())
            .map(|()| rng.sample(Alphanumeric))
            .take(len)
            .collect();
        Self {
            genes: s,
            fitness: None,
        }
    }

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
                genes: format!("{}{}", m2, f1),
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

impl Epoch<observation::Observer, evaluation::Evaluator, Genotype, Config> {
    pub fn new(config: Config) -> Self {
        let population = iter::repeat(())
            .map(|()| Genotype::new(config.init_len))
            .take(config.pop_size)
            .collect();
        let observer = observation::Observer::spawn(&config);
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

    pub fn target_reached(&self, target: usize) -> bool {
        self.best
            .as_ref()
            .and_then(|b| b.fitness())
            .map_or(false, |f| f <= target)
    }
}

impl Epochal for Epoch<observation::Observer, evaluation::Evaluator, Genotype, Config> {
    type Observer = observation::Observer;
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
    let target_fitness = config.target_fitness;
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
    fn fitness_function(phenome: &Genotype, target: &str) -> usize {
        distance::damerau_levenshtein(&phenome.genes, target)
    }
}

mod observation {
    use std::sync::mpsc::{channel, Sender, SendError};
    use std::thread::{JoinHandle, spawn};

    use crate::observer::{ObservationWindow, Observe};

    use super::*;

    pub struct Observer {
        pub handle: JoinHandle<()>,
        tx: Sender<Genotype>,
    }

    struct Window {
        pub frame: Vec<Option<Genotype>>,
        i: usize,
        window_size: usize,
    }

    impl ObservationWindow for Window {
        type Observable = Genotype;

        fn insert(&mut self, thing: Self::Observable) {
            self.i = (self.i + 1) % self.window_size;
            self.frame[self.i] = Some(thing);
            if self.i == 0 {
                self.report();
            }
        }

        fn report(&self) {
            let fitnesses: Vec<usize> = self
                .frame
                .iter()
                .filter_map(|t| t.as_ref().and_then(Genotype::fitness))
                .collect();
            let avg_fit = fitnesses.iter().sum::<usize>() as f32 / fitnesses.len() as f32;
            log::info!("Average fitness: {}", avg_fit);
        }
    }

    impl Window {
        pub fn new(window_size: usize) -> Self {
            assert!(window_size > 0);
            Self {
                frame: vec![None; window_size],
                i: 0,
                window_size,
            }
        }
    }

    impl Observe for Observer {
        type Observable = Genotype;
        type Params = Config;
        type Error = SendError<Self::Observable>;

        fn observe(&self, ob: Self::Observable) {
            self.tx.send(ob).expect("tx failure");
        }

        fn spawn(params: &Self::Params) -> Self {
            let (tx, rx) = channel();

            let window_size = params.observer_window_size();
            let handle = spawn(move || {
                let mut window: Window = Window::new(window_size);

                for observable in rx {
                    log::debug!("received observable {:?}", observable);
                    window.insert(observable);
                }
            });

            Self { handle, tx }
        }
    }
}
