use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;

use rand::Rng;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::configure::Config;
use crate::fitness::FitnessScore;
use crate::util;
use crate::util::count_min_sketch::Sketch;
use crate::util::levy_flight::levy_decision;
use crate::util::random::{hash_seed_rng, Prng};

//pub mod lexicase;
pub mod metropolis;
pub mod pareto_roulette;
pub mod population;
pub mod tournament;

pub trait Mutation {
    type Allele;

    fn mutate_point(allele: &mut Self::Allele, config: &Config) -> Self;

    fn mutate(chromosome: &mut [Self::Allele], config: &Config) -> Vec<Option<Self>>
    where
        Self: Sized,
    {
        let mut rng = rand::thread_rng();
        let len = chromosome.len();
        (0..len)
            .map(|i| {
                if !levy_decision(&mut rng, len, config.mutation_exponent) {
                    Some(Self::mutate_point(&mut chromosome[i], &config))
                } else {
                    None
                }
            })
            .collect::<Vec<Option<Self>>>()
    }
}

#[derive(Clone, Hash, Serialize)]
pub struct LinearChromosome<
    A: Debug + Clone + Hash + Serialize + DeserializeOwned,
    M: Debug + Clone + Hash + Serialize + DeserializeOwned + Mutation<Allele = A>,
> {
    pub chromosome: Vec<A>,
    pub mutations: Vec<Option<M>>,
    pub parentage: Vec<usize>,
    pub parent_names: Vec<String>,
    pub name: String,
    pub generation: usize,
}

// TODO: Define a mutation method on the mutation enum type

impl<
        A: Debug + Clone + Hash + Serialize + DeserializeOwned,
        M: Debug + Clone + Hash + Serialize + DeserializeOwned + Mutation<Allele = A>,
    > LinearChromosome<A, M>
{
    pub fn len(&self) -> usize {
        self.chromosome.len()
    }

    pub fn crossover(parents: &[&Self], config: &Config) -> Self {
        let min_mate_len = parents.iter().map(|p| p.len()).min().unwrap();
        let lambda = min_mate_len as f64 / config.crossover_period;
        let distribution =
            rand_distr::Exp::new(lambda).expect("Failed to create random distribution");
        Self::crossover_by_distribution(&distribution, &parents)
    }

    fn crossover_by_distribution<D: rand_distr::Distribution<f64>>(
        distribution: &D,
        parents: &[&Self],
    ) -> Self {
        let mut chromosome = Vec::new();
        let mut parentage = Vec::new();
        let mut rng = hash_seed_rng(&parents[0].chromosome);
        let mut ptrs = vec![0_usize; parents.len()];
        let switch = |rng: &mut Prng| rng.gen_range(0, parents.len());
        let sample = |rng: &mut Prng| distribution.sample(rng).round() as usize + 1;

        loop {
            let src = switch(&mut rng);
            let take_from = ptrs[src];

            if take_from >= parents[src].len() {
                break;
            }
            //let take_to = std::cmp::min(ptrs[src] + sample(&mut rng), parents[src].len());
            let take_to = ptrs[src] + sample(&mut rng);
            let len = parents[src].len();
            for i in take_from..take_to {
                chromosome.push(parents[src].chromosome[i % len].clone())
            }

            for _ in 0..(take_to - take_from) {
                parentage.push(src)
            }

            ptrs[src] = take_to;
            // now slide the other ptrs ahead a random interval
            for i in 0..ptrs.len() {
                if i != src {
                    ptrs[i] += sample(&mut rng);
                }
            }
        }

        let len = chromosome.len();
        let name = util::name::random(4, &chromosome);

        Self {
            chromosome,
            parentage,
            mutations: vec![None; len],
            parent_names: parents
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<String>>(),
            name,
            generation: parents.iter().map(|p| p.generation).max().unwrap_or(0) + 1,
        }
    }

    pub fn mutate(&mut self, config: &Config) {
        // maybe check a uniform mutation rate to see if any pointwise mutations happen at all.
        let mutations = M::mutate(&mut self.chromosome, config);
        self.mutations = mutations;
    }
}

impl<A, M> fmt::Debug for LinearChromosome<A, M>
where
    A: Debug + Clone + Hash + Serialize + DeserializeOwned,
    M: Debug + Clone + Hash + Serialize + DeserializeOwned + Mutation<Allele = A>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Name: {}\nGeneration: {}", self.name, self.generation)?;
        for i in 0..self.chromosome.len() {
            let parent = if self.parent_names.is_empty() {
                "seed"
            } else {
                &self.parent_names[self.parentage[i]]
            };
            let allele = &self.chromosome[i];
            let mutation = &self.mutations[i];
            writeln!(
                f,
                "[{i}][{parent}] {allele:x?}{mutation}",
                i = i,
                parent = parent,
                allele = allele,
                mutation = mutation
                    .as_ref()
                    .map(|m| format!(" {:?}", m))
                    .unwrap_or_else(String::new),
            )?;
        }
        Ok(())
    }
}

pub trait Genome: Hash {
    type Allele: Clone + Debug + PartialEq + Eq + Hash + Serialize + Sized;

    fn chromosome(&self) -> &[Self::Allele];

    fn chromosome_mut(&mut self) -> &mut [Self::Allele];

    fn len(&self) -> usize {
        self.chromosome().len()
    }

    fn random<H: Hash>(config: &Config, salt: H) -> Self
    where
        Self: Sized;

    fn crossover(parents: &[&Self], config: &Config) -> Self
    where
        Self: Sized;

    fn crossover_by_distribution<D: rand_distr::Distribution<f64>>(
        distribution: &D,
        parents: &[&[Self::Allele]],
    ) -> (Vec<Self::Allele>, Vec<usize>) {
        let mut chromosome = Vec::new();
        let mut parentage = Vec::new();
        let mut rng = hash_seed_rng(&parents[0]);
        let mut ptrs = vec![0_usize; parents.len()];
        let switch = |rng: &mut Prng| rng.gen_range(0, parents.len());
        let sample = |rng: &mut Prng| distribution.sample(rng).round() as usize + 1;

        loop {
            let src = switch(&mut rng);
            let take_from = ptrs[src];

            if take_from >= parents[src].len() {
                break;
            }
            //let take_to = std::cmp::min(ptrs[src] + sample(&mut rng), parents[src].len());
            let take_to = ptrs[src] + sample(&mut rng);
            let len = parents[src].len();
            for i in take_from..take_to {
                chromosome.push(parents[src][i % len].clone())
            }

            for _ in 0..(take_to - take_from) {
                parentage.push(src)
            }

            ptrs[src] = take_to;
            // now slide the other ptrs ahead a random interval
            for i in 0..ptrs.len() {
                if i != src {
                    ptrs[i] += sample(&mut rng);
                }
            }
        }

        (chromosome, parentage)
    }

    fn mutate(&mut self, config: &Config);

    fn mate(parents: &[&Self], config: &Config) -> Self
    where
        Self: Sized,
    {
        let mut child = Self::crossover(parents, config);
        // the mutate method should check the mutation rate or exponent and
        // make the mutation decisions internally
        let mut rng = hash_seed_rng(&parents);
        if rng.gen_range(0.0, 1.0) < config.mutation_rate {
            child.mutate(&config);
        }
        child
    }

    fn digrams(&self) -> Box<dyn Iterator<Item = (Self::Allele, Self::Allele)> + '_> {
        if self.chromosome().len() == 1 {
            // FIXME: i don't like this edge case.
            return Box::new(self.chromosome().iter().map(|x| (x.clone(), x.clone())));
        }
        Box::new(
            self.chromosome()
                .iter()
                .zip(self.chromosome().iter().skip(1))
                .map(|(a, b)| (a.clone(), b.clone())),
        )
    }

    fn record_genetic_frequency<S: Sketch>(&self, sketch: &mut S) {
        for digram in self.digrams() {
            sketch.insert(digram)
        }
    }

    fn query_genetic_frequency<S: Sketch>(&self, sketch: &S) -> f64 {
        // The lower the score, the rarer the digrams composing the genome.
        // We divide by the length to avoid penalizing longer genomes.
        // let mut sum = 0_f64;
        // for digram in self.digrams() {
        //     sum += (sketch.query(digram, timestamp)?);
        // };
        // Ok(sum)
        let d = self.len();
        let d = if d <= 1 { 1.0 } else { d as f64 };
        self.digrams()
            .map(|digram| sketch.query(digram))
            .collect::<Vec<_>>()
            .into_iter()
            //.fold(std::f64::MAX, |a, b| a.min(b))
            .sum::<f64>()
            / d
    }

    fn incr_num_offspring(&mut self, _n: usize);
}

pub trait Phenome: Clone + Debug + Send + Serialize + Hash {
    type Fitness: FitnessScore;
    type Problem: Hash;
    // TODO: generalize fitness. should be able to use vecs, etc.

    /// This method should generate a string describing the creature
    /// and attach it to the creature, somehow -- storing it in an
    /// optional field, for example.
    fn generate_description(&mut self) {}

    /// This method is intended for reporting, not measuring, fitness.
    fn fitness(&self) -> Option<&Self::Fitness>;

    fn scalar_fitness(&self) -> Option<f64>;

    fn priority_fitness(&self, _config: &Config) -> Option<f64>;

    fn set_fitness(&mut self, f: Self::Fitness);

    fn name(&self) -> &str {
        "nameless voyager"
    }

    fn tag(&self) -> u64;

    fn set_tag(&mut self, tag: u64);

    fn answers(&self) -> Option<&Vec<Self::Problem>>;

    fn store_answers(&mut self, results: Vec<Self::Problem>);

    /// Return the rank of the Pareto front in which the phenotype most
    /// recently appeared. Unused when not performing Pareto selection.
    fn front(&self) -> Option<usize> {
        unimplemented!("implement as needed")
    }

    fn set_front(&mut self, _rank: usize) {
        unimplemented!("implement as needed")
    }

    fn is_goal_reached(&self, config: &Config) -> bool;

    fn fails(&self, _problem: &Self::Problem) -> bool {
        unimplemented!("implement as needed (for lexicase, e.g.)");
    }

    fn mature(&self) -> bool {
        unimplemented!("implement as needed")
    }
}
