use std::cmp::Ordering;
use std::iter;
use std::iter::Iterator;
use std::sync::Arc;

use rand::distributions::Alphanumeric;
use rand::prelude::*;
use serde_derive::Deserialize;

use crate::{
    configure::{Configure, ConfigureObserver},
    evaluator::{Evaluate, FitnessFn},
    evolution::*,
    observer::{Observer, ReportFn},
};

pub type Fitness = Vec<f32>;

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

#[derive(Clone, Debug, Default)]
pub struct Genotype {
    pub genes: String,
    fitness: Option<Fitness>,
    tag: u64,
    // used for sorting in heap
    generation: usize,
}

impl PartialEq for Genotype {
    fn eq(&self, other: &Self) -> bool {
        self.tag == other.tag
    }
}

impl Eq for Genotype {}

impl PartialOrd for Genotype {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.tag.partial_cmp(&other.tag)
    }
}

impl Ord for Genotype {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// because this is a GA we identify genome and phenome
impl Phenome for Genotype {
    type Inst = ();
    type Fitness = Fitness;

    fn fitness(&self) -> Option<&Fitness> {
        self.fitness.as_ref()
    }

    fn set_fitness(&mut self, f: Fitness) {
        self.fitness = Some(f);
    }

    fn tag(&self) -> u64 {
        self.tag
    }

    fn set_tag(&mut self, tag: u64) {
        self.tag = tag
    }

    fn problems(&self) -> Option<&Vec<Problem>> {
        unimplemented!("n/a")
    }

    fn store_answers(&mut self, _results: Vec<Problem>) {
        unimplemented!("n/a") // CODE SMELL FIXME!
    }

    fn len(&self) -> usize {
        self.genes.len()
    }
}

impl Genotype {
    pub fn len(&self) -> usize {
        self.genes.len()
    }
}

impl Genome<Config> for Genotype {

    fn random(params: &Config) -> Self {
        let mut rng = thread_rng();
        let len = rng.gen_range(1, params.init_len);
        let s: String = iter::repeat(())
            .map(|()| rng.sample(Alphanumeric))
            .take(len)
            .collect();
        Self {
            genes: s,
            fitness: None,
            tag: rng.gen::<u64>(),
            generation: 0,
        }
    }

    fn crossover(&self, mate: &Self, _params: &Config) -> Vec<Self> {
        let mut rng = thread_rng();
        let split_m: usize = rng.gen::<usize>() % self.len();
        let split_f: usize = rng.gen::<usize>() % mate.len();
        let (m1, m2) = self.genes.split_at(split_m);
        let (f1, f2) = mate.genes.split_at(split_f);
        let generation = self.generation.max(mate.generation) + 1;
        vec![
            Genotype {
                genes: format!("{}{}", m1, f2),
                fitness: None,
                tag: rng.gen::<u64>(),
                generation,
            },
            Genotype {
                genes: format!("{}{}", f1, m2),
                fitness: None,
                tag: rng.gen::<u64>(),
                generation,
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
                    c = bytes[i] ^ (1 << rng.gen_range(1_u8, 7_u8));
                    //c = rng.gen::<u8>();
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

fn report(window: &[Genotype]) {
    let fitnesses: Vec<Fitness> = window.iter().filter_map(|g| g.fitness.clone()).collect();
    let len = fitnesses.len();
    let avg_fit = fitnesses
        .iter()
        .fold(vec![0.0; len], |a: Fitness, b: &Fitness| {
            a.iter()
                .zip(b.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<f32>>()
        })
        .iter()
        .map(|v| v / len as f32)
        .collect::<Vec<f32>>();
    let avg_gen = window.iter().map(|g| g.generation).sum::<usize>() as f32 / window.len() as f32;

    log::info!("AVERAGE FITNESS: {:?}; AVG GEN: {}", avg_fit, avg_gen);
}

use cached::{cached_key, TimedCache};

cached_key! {
    FF_CACHE: TimedCache<String, Vec<f32>> = TimedCache::with_lifespan(2);

    Key = { format!("{}\x00\x00{}", phenome, target ) };

    fn ff_helper(phenome: &str, target: &str) -> Vec<f32> = {
        let l_fitness = distance::damerau_levenshtein(phenome, target);
        let (short, long) = if phenome.len() < target.len() {
            (phenome, target)
        } else {
            (target, phenome)
        };
        let dif = long.len() - short.len();
        let short = short.as_bytes();
        let long = &long.as_bytes()[0..short.len()];
        let h_fitness = hamming::distance(short, long) + (dif * 8) as u64;
        vec![h_fitness as f32, l_fitness as f32]
    }
}

fn fitness_function(mut phenome: Genotype, params: Arc<Config>) -> Genotype {
    if phenome.fitness.is_none() {
        // let l_fitness = distance::damerau_levenshtein(&phenome.genes, &params.target);
        // let (short, long) = if phenome.genes.len() < params.target.len() {
        //     (&phenome.genes, &params.target)
        // } else {
        //     (&params.target, &phenome.genes)
        // };
        // let dif = long.len() - short.len();
        // let short = short.as_bytes();
        // let long = &long.as_bytes()[0..short.len()];
        // let h_fitness = hamming::distance(short, long) + (dif * 8) as u64;

        phenome.set_fitness(ff_helper(&phenome.genes, &params.target));
    };
    phenome
}

impl Epoch<evaluation::Evaluator<Genotype>, Genotype, Config> {
    pub fn new(config: Config) -> Self {
        let config = Arc::new(config);
        let population = iter::repeat(())
            .map(|()| Genotype::random(&config))
            .take(config.population_size())
            .collect();
        let report_fn: ReportFn<_> = Box::new(report);
        let fitness_fn: FitnessFn<_, _> = Box::new(fitness_function);
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

pub fn run(config: Config) -> Option<Genotype> {
    let target_fitness = config.target_fitness as f32;
    let mut world = Epoch::<evaluation::Evaluator<Genotype>, Genotype, Config>::new(config);

    loop {
        world = world.evolve();
        if let Some(true) = world
            .best
            .as_ref()
            .and_then(|b| b.fitness.as_ref())
            .map(|f| f[0] == target_fitness)
        {
            println!("\n***** Success after {} epochs! *****", world.iteration);
            println!("{:#?}", world.best);
            return world.best;
        };
    }
}

mod evaluation {
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use std::thread::{spawn, JoinHandle};

    use crate::evaluator::FitnessFn;

    use super::*;
    use crate::util::count_min_sketch::DecayingSketch;

    pub struct Evaluator<Genotype> {
        pub handle: JoinHandle<()>,
        tx: Sender<Genotype>,
        rx: Receiver<Genotype>,
    }

    impl Evaluate<Genotype> for Evaluator<Genotype> {
        type Params = Config;

        fn evaluate(&self, phenome: Genotype) -> Genotype {
            self.tx.send(phenome).expect("tx failure");
            self.rx.recv().expect("rx failure")
        }

        fn spawn(params: Arc<Self::Params>, fitness_fn: FitnessFn<Genotype, Self::Params>) -> Self {
            let (tx, our_rx): (Sender<Genotype>, Receiver<Genotype>) = channel();
            let (our_tx, rx): (Sender<Genotype>, Receiver<Genotype>) = channel();
            let fitness_fn = Arc::new(fitness_fn);
            let handle = spawn(move || {
                let sketch = Arc::new(Mutex::new(DecayingSketch::default()));
                let mut counter = 0;
                for phenome in our_rx {
                    counter += 1;
                    let sketch = sketch.clone();
                    let our_tx = our_tx.clone();
                    let params = params.clone();
                    let fitness_fn = fitness_fn.clone();
                    let _handle = rayon::spawn(move || {
                        let mut phenome = fitness_fn(phenome, params.clone());
                        let mut sketch = sketch.lock().unwrap();
                        sketch
                            .insert(&phenome.genes, counter)
                            .expect("Failed to insert genes into sketch");
                        let freq = sketch
                            .query(&phenome.genes)
                            .expect("Failed to query sketch");
                        phenome.fitness.as_mut().map(|f| f.push(freq));
                        our_tx.send(phenome).expect("Channel failure");
                    });
                }
            });

            Self { handle, tx, rx }
        }
    }
}

//build_observation_mod!(observation, Genotype, Config);
