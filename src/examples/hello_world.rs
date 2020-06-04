use std::cmp::Ordering;
use std::iter;
use std::iter::Iterator;
use std::sync::Arc;

use rand::distributions::Alphanumeric;
use rand::prelude::*;

use crate::{evaluator::Evaluate, evolution::tournament::*, observer::Observer};

pub type Fitness = Vec<f64>;

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

    fn len(&self) -> usize {
        self.genes.len()
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

    fn random(params: &Config) -> Self {
        let mut rng = thread_rng();
        let len = rng.gen_range(1, params.max_init_len);
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

    fn crossover(mates: &[&Self], _params: &Config) -> Self {
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

    fn mutate(&mut self, _params: &Config) {
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

fn report(window: &Window<Genotype>, counter: usize, _params: &ObserverConfig) {
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

use crate::configure::{Config, ObserverConfig, Problem};
use crate::evolution::{Genome, Phenome};
use crate::observer::Window;
use cached::{cached_key, TimedCache};

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

        phenome.set_fitness(ff_helper(&phenome.genes, &params.hello.target));

        // let's try to implement genlin's char-dist
        // let length_diff = (phenome.genes.len() as f64 - params.target.len() as f64).abs();
        // let code_diff = phenome.genes.chars().zip(params.target.chars())
        //     .map(|(a,b)| (a as i32 - b as i32).abs() as f64)
        //     .sum::<f64>();
        // phenome.set_fitness(vec![length_diff, code_diff]);
    };
    phenome
}

// impl Tournament<evaluation::Evaluator<Genotype>, Genotype> {
//     pub fn new(config: Config) -> Self {
//         let population = iter::repeat(())
//             .map(|()| Genotype::random(&config))
//             .take(config.population_size())
//             .collect();
//         let report_fn: ReportFn<_> = Box::new(report);
//         let fitness_fn: FitnessFn<_, _> = Box::new(fitness_function);
//         let observer = Observer::spawn(&config, report_fn);
//         let evaluator = evaluation::Evaluator::spawn(&config, fitness_fn);
//         Self {
//             population,
//             config: Arc::new(config),
//             best: None,
//             iteration: 0,
//             observer,
//             evaluator,
//         }
//     }
// }

pub fn run(config: Config) -> Option<Genotype> {
    let target_fitness = config.target_fitness as f64;
    let report_fn = Box::new(report);
    let fitness_fn = Box::new(fitness_function);
    let observer = Observer::spawn(&config, report_fn);
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

        fn eval_pipeline<I: Iterator<Item = Genotype>>(&self, inbound: I) -> Vec<Genotype> {
            inbound.map(|p| self.evaluate(p)).collect::<Vec<Genotype>>()
        }

        fn spawn(params: &Self::Params, fitness_fn: FitnessFn<Genotype, Self::Params>) -> Self {
            let (tx, our_rx): (Sender<Genotype>, Receiver<Genotype>) = channel();
            let (our_tx, rx): (Sender<Genotype>, Receiver<Genotype>) = channel();
            let params = Arc::new(params.clone());
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
                    rayon::spawn(move || {
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
