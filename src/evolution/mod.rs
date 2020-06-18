use std::fmt::Debug;
use std::hash::Hash;

use rand::Rng;
use serde::Serialize;

use crate::configure::Config;
use crate::fitness::FitnessScore;
use crate::util::count_min_sketch::Sketch;
use crate::util::random::{hash_seed_rng, Prng};

pub mod lexicase;
pub mod metropolis;
pub mod pareto_roulette;
pub mod population;
pub mod tournament;

pub trait Genome: Debug + Hash {
    type Allele: Clone + Copy + Debug + PartialEq + Eq + Hash + Serialize + Sized;

    fn chromosome(&self) -> &[Self::Allele];

    fn chromosome_mut(&mut self) -> &mut [Self::Allele];

    fn len(&self) -> usize {
        self.chromosome().len()
    }

    fn random<H: Hash>(config: &Config, salt: H) -> Self
    where
        Self: Sized;

    fn crossover(parents: &Vec<&Self>, config: &Config) -> Self
    where
        Self: Sized;

    fn crossover_by_distribution<D: rand_distr::Distribution<f64>>(
        distribution: &D,
        parents: &Vec<&[Self::Allele]>,
    ) -> (Vec<Self::Allele>, Vec<usize>) {
        let mut chromosome = Vec::new();
        let mut parentage = Vec::new();
        let mut rng = hash_seed_rng(parents);
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
                chromosome.push(parents[src][i % len])
            }

            //let slice = &parents[src][take_from..take_to];
            //chromosome.extend_from_slice(slice);
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

    fn mate(parents: &Vec<&Self>, config: &Config) -> Self
    where
        Self: Sized,
    {
        let mut child = Self::crossover(parents, config);
        // the mutate method should check the mutation rate or exponent and
        // make the mutation decisions internally
        child.mutate(&config);
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
                .map(|(a, b)| (*a, *b)),
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

    /// This method is intended for reporting, not measuring, fitness.
    fn fitness(&self) -> Option<&Self::Fitness>;

    fn scalar_fitness(&self) -> Option<f64>;

    fn priority_fitness(&self, _config: &Config) -> Option<f64> {
        unimplemented!("let's break here for now");
        //self.scalar_fitness()
    }

    fn set_fitness(&mut self, f: Self::Fitness);

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
