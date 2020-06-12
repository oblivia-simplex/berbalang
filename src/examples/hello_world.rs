use std::cmp::Ordering;
use std::iter;
use std::iter::Iterator;
use std::sync::Arc;

use cached::{cached_key, TimedCache};
use rand::distributions::Alphanumeric;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::configure::{Config, Problem};
use crate::evolution::{Genome, Phenome};
use crate::impl_dominance_ord_for_phenome;
use crate::observer::Window;
use crate::util::count_min_sketch::CountMinSketch;
use crate::{evaluator::Evaluate, evolution::tournament::*, observer::Observer};

pub type Fitness = Vec<f64>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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
    type Fitness = Fitness;

    fn fitness(&self) -> Option<&Fitness> {
        self.fitness.as_ref()
    }

    fn scalar_fitness(&self) -> Option<f64> {
        self.fitness.as_ref().map(|v| v[0])
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
}

impl Genotype {
    pub fn len(&self) -> usize {
        self.genes.len()
    }
}

impl Genome for Genotype {
    type Allele = char;

    fn chromosome(&self) -> &[Self::Allele] {
        unimplemented!("rust makes treating strings as &[char] tricky")
    }

    fn chromosome_mut(&mut self) -> &mut [Self::Allele] {
        unimplemented!("rust makes treating strings as &[char] tricky")
    }

    fn random(config: &Config) -> Self {
        let mut rng = thread_rng();
        let len = rng.gen_range(1, config.max_init_len);
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

    fn crossover(mates: &[&Self], _config: &Config) -> Self {
        let mut rng = thread_rng();
        let father = &mates[0];
        let mother = &mates[1];
        let split_m: usize = rng.gen::<usize>() % mother.len();
        let split_f: usize = rng.gen::<usize>() % father.len();
        let (m1, _m2) = mother.genes.split_at(split_m);
        let (_f1, f2) = father.genes.split_at(split_f);
        let generation = mother.generation.max(father.generation) + 1;
        Genotype {
            genes: format!("{}{}", m1, f2),
            fitness: None,
            tag: rng.gen::<u64>(),
            generation,
        }
    }

    fn mutate(&mut self, _config: &Config) {
        let mut rng: ThreadRng = thread_rng();
        let mutation = rng.gen::<u8>() % 4;
        match mutation {
            // replace some character with random character
            0 => unsafe {
                let i: usize = rng.gen::<usize>() % self.len();
                let bytes = self.genes.as_bytes_mut();
                bytes[i] = rng.gen_range(0x20, 0x7e);
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

impl_dominance_ord_for_phenome!(Genotype, Dom);

fn report(window: &Window<Genotype, Dom>, counter: usize, _config: &Config) {
    let frame = &window.frame;
    let fitnesses: Vec<Fitness> = frame.iter().filter_map(|g| g.fitness.clone()).collect();
    let len = fitnesses.len();
    let avg_fit = fitnesses
        .iter()
        .fold(vec![0.0; len], |a: Fitness, b: &Fitness| {
            a.iter()
                .zip(b.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<f64>>()
        })
        .iter()
        .map(|v| v / len as f64)
        .collect::<Vec<f64>>();
    let avg_gen = frame.iter().map(|g| g.generation).sum::<usize>() as f64 / frame.len() as f64;

    log::info!(
        "[{}] AVERAGE FITNESS: {:?}; AVG GEN: {}",
        counter,
        avg_fit,
        avg_gen
    );
}

cached_key! {
    FF_CACHE: TimedCache<String, Vec<f64>> = TimedCache::with_lifespan(2);

    Key = { format!("{}\x00\x00{}", phenome, target ) };

    fn ff_helper(phenome: &str, target: &str) -> Vec<f64> = {
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
        vec![l_fitness as f64, h_fitness as f64]
    }
}

fn fitness_function(
    mut phenome: Genotype,
    sketch: &mut CountMinSketch,
    config: Arc<Config>,
) -> Genotype {
    if phenome.fitness.is_none() {
        phenome.set_fitness(ff_helper(&phenome.genes, &config.hello.target));
        sketch.insert(&phenome.genes);
        let freq = sketch.query(&phenome.genes);
        phenome.fitness.as_mut().map(|f| f.push(freq));
    };
    phenome
}

pub fn run(config: Config) -> Option<Genotype> {
    let target_fitness = config.target_fitness as f64;
    let report_fn = Box::new(report);
    let fitness_fn = Box::new(fitness_function);
    let observer = Observer::spawn(&config, report_fn, Dom);
    let evaluator = evaluation::Evaluator::spawn(&config, fitness_fn);
    let mut world =
        Tournament::<evaluation::Evaluator<Genotype>, Genotype>::new(config, observer, evaluator);

    loop {
        world = world.evolve();
        if let Some(true) = world
            .best
            .as_ref()
            .and_then(|b| b.fitness.as_ref())
            .map(|f| f[0] <= target_fitness && f[1] <= target_fitness)
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
    use crate::util::count_min_sketch::CountMinSketch;

    use super::*;

    pub struct Evaluator<Genotype> {
        pub handle: JoinHandle<()>,
        tx: Sender<Genotype>,
        rx: Receiver<Genotype>,
    }

    impl Evaluate<Genotype, CountMinSketch> for Evaluator<Genotype> {
        fn evaluate(&mut self, phenome: Genotype) -> Genotype {
            self.tx.send(phenome).expect("tx failure");
            self.rx.recv().expect("rx failure")
        }

        fn eval_pipeline<I: Iterator<Item = Genotype>>(&mut self, inbound: I) -> Vec<Genotype> {
            inbound.map(|p| self.evaluate(p)).collect::<Vec<Genotype>>()
        }

        fn spawn(config: &Config, fitness_fn: FitnessFn<Genotype, CountMinSketch, Config>) -> Self {
            let (tx, our_rx): (Sender<Genotype>, Receiver<Genotype>) = channel();
            let (our_tx, rx): (Sender<Genotype>, Receiver<Genotype>) = channel();
            let config = Arc::new(config.clone());
            let fitness_fn = Arc::new(fitness_fn);
            let handle = spawn(move || {
                let sketch = Arc::new(Mutex::new(CountMinSketch::new(&config)));
                for phenome in our_rx {
                    let sketch = sketch.clone();
                    let our_tx = our_tx.clone();
                    let config = config.clone();
                    let fitness_fn = fitness_fn.clone();
                    rayon::spawn(move || {
                        let mut sketch = sketch.lock().unwrap();
                        let phenome = fitness_fn(phenome, &mut sketch, config.clone());

                        our_tx.send(phenome).expect("Channel failure");
                    });
                }
            });

            Self { handle, tx, rx }
        }
    }
}

//build_observation_mod!(observation, Genotype, Config);
