use std::iter;
use std::iter::Iterator;

use log;
use rand::distributions::Alphanumeric;
use rand::prelude::*;
use serde_derive::Deserialize;

use crate::evaluator::Evaluate;
use crate::evolution::*;
use crate::observer::Observe;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ObserverConfig {
    pub window_size: usize,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    pub mut_rate: f32,
    pub init_len: usize,
    pub pop_size: usize,
    pub tournament_size: usize,
    pub num_offspring: usize,
    pub target: String,
    pub target_fitness: Fitness,
    pub observer: ObserverConfig,
}

impl Config {
    pub fn assert_invariants(&self) {
        assert!(self.tournament_size >= self.num_offspring + 2);
        assert_eq!(self.num_offspring, 2); // all that's supported for now
    }
}

#[derive(Clone, Debug, Default)]
pub struct Genome {
    genes: String,
    fitness: Option<Fitness>,
}

impl Genome {
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
            if rng.gen::<f32>() < params.mut_rate {
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

impl Genotype for Genome {
    fn crossover(&self, mate: &Self) -> Vec<Self> {
        let mut rng = thread_rng();
        let split_m: usize = rng.gen::<usize>() % self.len();
        let split_f: usize = rng.gen::<usize>() % mate.len();
        let (m1, m2) = self.genes.split_at(split_m);
        let (f1, f2) = mate.genes.split_at(split_f);
        vec![
            Genome {
                genes: format!("{}{}", m1, f2),
                fitness: None,
            },
            Genome {
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

pub type Fitness = usize;

#[derive(Debug)]
pub struct Epoch {
    pub population: Vec<Genome>,
    pub config: Config,
    pub best: Option<Genome>,
    pub iteration: usize,
}

impl Epoch {
    pub fn new(config: Config) -> Self {
        let population = iter::repeat(())
            .map(|()| Genome::new(config.init_len))
            .take(config.pop_size)
            .collect();
        Self {
            population,
            config,
            best: None,
            iteration: 0,
        }
    }

    pub fn update_best(&mut self, champ: &Genome) {
        match self.best {
            Some(ref best) if champ.fitter_than(best) => {
                self.best = Some(champ.clone());
                log::info!("[{}]\tnew champ {:?}", self.iteration, champ)
            }
            None => {
                self.best = Some(champ.clone());
                log::info!("[{}]\tnew champ {:?}", self.iteration, champ)
            }
            _ => {}
        }
    }

    pub fn target_reached(&self, target: Fitness) -> bool {
        self.best
            .as_ref()
            .and_then(|b| b.fitness)
            .map_or(false, |f| f <= target)
    }
}

impl Epochal for Epoch {
    type Observer = observation::Observer;
    type Evaluator = evaluation::Evaluator;

    fn evolve(mut self, observer: &Self::Observer, evaluator: &Self::Evaluator) -> Self {
        let tournament_size = self.config.tournament_size;
        let mut population = std::mem::take(&mut self.population);
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
        self.update_best(&combatants[0]);

        // kill one off for every offspring to be produced
        for _ in 0..self.config.num_offspring {
            let _ = combatants.pop();
        }

        // replace the combatants that will neither breed nor die
        let bystanders = self.config.tournament_size - (self.config.num_offspring + 2);
        for _ in 0..bystanders {
            if let Some(c) = combatants.pop() {
                population.push(c);
            }
        }

        let mother = combatants.pop().unwrap();
        let father = combatants.pop().unwrap();
        let offspring: Vec<Genome> = mother.mate(&father, &self.config);

        // return everyone to the population
        population.push(mother);
        population.push(father);
        for child in offspring.into_iter() {
            population.push(child)
        }

        Self {
            population,
            config: std::mem::take(&mut self.config),
            best: std::mem::take(&mut self.best),
            iteration: self.iteration + 1,
        }
    }
}

pub fn run(config: Config) -> Option<Genome> {
    let mut world = Epoch::new(config);
    let observer = observation::Observer::spawn(&world.config);
    let evaluator = evaluation::Evaluator::spawn(&world.config);

    loop {
        world = world.evolve(&observer, &evaluator);
        if world.target_reached(0) {
            println!("\n***** Success after {} epochs! *****", world.iteration);
            println!("{:#?}", world.best);
            return world.best.clone();
        };
    }
}

mod evaluation {
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::thread::{JoinHandle, spawn};

    use super::*;

    pub struct Evaluator {
        pub handle: JoinHandle<()>,
        tx: Sender<Genome>,
        rx: Receiver<Genome>,
    }

    impl Evaluate for Evaluator {
        type Phenotype = Genome;
        // because this is a GA, not a GP
        type Params = Config;

        fn evaluate(&self, phenome: Self::Phenotype) -> Self::Phenotype {
            self.tx.send(phenome).expect("tx failure");
            self.rx.recv().expect("rx failure")
        }

        fn spawn(params: &Self::Params) -> Self {
            let (tx, our_rx): (Sender<Genome>, Receiver<Genome>) = channel();
            let (our_tx, rx): (Sender<Genome>, Receiver<Genome>) = channel();

            let target = params.target.clone();

            let handle = spawn(move || {
                for mut phenome in our_rx {
                    if phenome.fitness.is_none() {
                        phenome.fitness = Some(distance::damerau_levenshtein(&phenome.genes, &target))
                    }
                    our_tx.send(phenome).expect("Channel failure");
                }
            });

            Self { handle, tx, rx }
        }
    }
}

mod observation {
    use std::sync::mpsc::{channel, Sender, SendError};
    use std::thread::{JoinHandle, spawn};

    use crate::observer::{ObservationWindow, Observe};

    use super::*;

    pub struct Observer {
        pub handle: JoinHandle<()>,
        tx: Sender<Genome>,
    }

    struct Window {
        pub frame: Vec<Option<Genome>>,
        i: usize,
        window_size: usize,
    }

    impl ObservationWindow for Window {
        type Observable = Genome;

        fn insert(&mut self, thing: Self::Observable) {
            self.i = (self.i + 1) % self.window_size;
            self.frame[self.i] = Some(thing);
            if self.i == 0 {
                self.report();
            }
        }

        fn report(&self) {
            let fitnesses: Vec<Fitness> = self
                .frame
                .iter()
                .filter_map(|t| t.as_ref().and_then(|x| x.fitness))
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
        type Observable = Genome;
        type Params = Config;
        type Error = SendError<Self::Observable>;

        fn observe(&self, ob: Self::Observable) {
            self.tx.send(ob).expect("tx failure");
        }

        fn spawn(params: &Self::Params) -> Self {
            let (tx, rx) = channel();

            let window_size = params.observer.window_size;
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
