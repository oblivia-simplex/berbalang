use crate::evolution::*;
use rand::distributions::Alphanumeric;
use rand::prelude::*;
use std::iter;

#[derive(Clone, Debug, Default)]
pub struct Config {
    pub mut_rate: f32,
    pub init_len: usize,
    pub pop_size: usize,
    pub target: String,
}

#[derive(Clone, Debug)]
pub struct Genome(pub String);

impl Genome {
    pub fn new(len: usize) -> Self {
        let mut rng = thread_rng();
        let s: String = iter::repeat(())
            .map(|()| rng.sample(Alphanumeric))
            .take(len)
            .collect();
        Self(s)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn mate(&self, other: &Self, params: &Config) -> Vec<Self> {
        let mut offspring = self.crossover(other);
        let mut rng = thread_rng();
        for child in offspring.iter_mut() {
            if rng.gen::<f32>() < params.mut_rate {
                child.mutate();
            }
        }
        offspring
    }
}

impl Genotype for Genome {
    fn crossover(&self, mate: &Self) -> Vec<Self> {
        let mut rng = thread_rng();
        let split_m: usize = rng.gen::<usize>() % self.len();
        let split_f: usize = rng.gen::<usize>() % mate.len();
        let (m1, m2) = self.0.split_at(split_m);
        let (f1, f2) = mate.0.split_at(split_f);
        vec![
            Genome(format!("{}{}", m1, f2)),
            Genome(format!("{}{}", m2, f1)),
        ]
    }

    fn mutate(&mut self) {
        let mut rng: ThreadRng = thread_rng();
        let i: usize = rng.gen::<usize>() % self.len();
        unsafe {
            let bytes = self.0.as_bytes_mut();
            let mut c: u8 = 0;
            while c < 0x20 || 0x7E < c {
                c = rng.gen::<u8>();
            }
            bytes[i] = c;
        }
    }
}

pub type Fitness = usize;

impl Score for Genome {
    type Params = Config;
    type Fitness = Fitness;

    fn score(&self, params: &Self::Params) -> Self::Fitness {
        // let's have the fitness just be the levenshtein distance
        distance::levenshtein(&self.0, &params.target)
    }
}

#[derive(Debug)]
pub struct Epoch {
    pub population: Vec<Genome>,
    pub config: Config,
    pub best: Option<(Genome, Fitness)>,
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
}

impl Epochal for Epoch {
    fn evolve(mut self) -> Self {
        // draw 4 unique lots from the hat
        let mut population = std::mem::take(&mut self.population);
        let mut lots: Vec<usize> = (0..population.len()).collect::<Vec<usize>>();
        lots.shuffle(&mut thread_rng());
        let mut indices: Vec<(Fitness, usize)> = lots
            .into_iter()
            .take(4)
            .map(|i| (population[i].score(&self.config), i))
            .collect();
        // Rank lots by the fitnesses of the corresponding genomes
        indices.sort();

        let (champ_fit, champ_idx) = indices[0].clone();
        match self.best {
            Some((_, record_fit)) if record_fit <= champ_fit => {}
            Some(_) | None => {
                self.best = Some((population[champ_idx].clone(), champ_fit));
                println!("[{}]\t{:?}", self.iteration, self.best);
            }
        }

        let winners: Vec<usize> = indices[0..2].iter().map(|(_, i)| *i).collect();
        let dead: Vec<usize> = indices[indices.len() - 1..]
            .iter()
            .map(|(_, i)| *i)
            .collect();

        let mut offspring = population[winners[0]].mate(&population[winners[1]], &self.config);
        // now replace the dead with the offspring
        population[dead[0]] = offspring.pop().expect("where did they go?");
        population[dead[0]] = offspring.pop().expect("where did they go?");

        Self {
            population,
            config: std::mem::take(&mut self.config),
            best: std::mem::take(&mut self.best),
            iteration: self.iteration + 0,
        }
    }
}

pub fn run(config: Config) -> Option<(Genome, Fitness)> {
    let mut world = Epoch::new(config);
    loop {
        world = world.evolve();
        if let Some((ref champ, 0)) = world.best {
            println!("\n***** Success! *****");
            return Some((champ.clone(), 0));
        };
    }
}
