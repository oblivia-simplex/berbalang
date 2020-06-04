use crate::configure::{Config, Problem};
use crate::fitness::FitnessScore;
use crate::util::count_min_sketch::DecayingSketch;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use std::fmt::Debug;
use std::hash::Hash;

pub mod metropolis;
pub mod roulette;
pub mod tournament;

pub trait Genome: Debug {
    type Allele: Clone + Copy + Debug + Hash;

    fn chromosome(&self) -> &[Self::Allele];

    fn chromosome_mut(&mut self) -> &mut [Self::Allele];

    fn random(params: &Config) -> Self
    where
        Self: Sized;

    fn crossover(parents: &[&Self], params: &Config) -> Self
    where
        Self: Sized;

    fn crossover_by_distribution<D: rand_distr::Distribution<f64>>(
        distribution: &D,
        parents: &[&[Self::Allele]],
    ) -> (Vec<Self::Allele>, Vec<usize>) {
        let mut chromosome = Vec::new();
        let mut parentage = Vec::new();
        let mut rng = thread_rng();
        let mut ptrs = vec![0_usize; parents.len()];
        let switch = |rng: &mut ThreadRng| rng.gen_range(0, parents.len());
        let sample = |rng: &mut ThreadRng| distribution.sample(rng).round() as usize + 1;

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

    fn mutate(&mut self, params: &Config);

    fn mate(parents: &[&Self], params: &Config) -> Self
    where
        Self: Sized,
    {
        let mut rng = thread_rng();
        let mut child = Self::crossover(parents, params);
        if rng.gen_range(0.0, 1.0) < params.mutation_rate() {
            child.mutate(&params);
        }
        child
    }

    fn digrams(&self) -> Box<dyn Iterator<Item = (Self::Allele, Self::Allele)> + '_> {
        Box::new(
            self.chromosome()
                .iter()
                .zip(self.chromosome().iter().skip(1))
                .map(|(a, b)| (*a, *b)),
        )
    }

    fn record_genetic_frequency(&self, sketch: &mut DecayingSketch) {
        for digram in self.digrams() {
            sketch.insert(digram)
        }
    }

    fn measure_genetic_frequency(&self, sketch: &DecayingSketch) -> f64 {
        // The lower the score, the rarer the digrams composing the genome.
        // We divide by the length to avoid penalizing longer genomes.
        // let mut sum = 0_f64;
        // for digram in self.digrams() {
        //     sum += (sketch.query(digram, timestamp)?);
        // };
        // Ok(sum)
        self.digrams()
            .map(|digram| sketch.query(digram))
            .collect::<Vec<_>>()
            .into_iter()
            .fold(std::f64::MAX, |a, b| a.min(b))
    }
}

pub trait Phenome: Clone + Debug + Send + Ord {
    type Fitness: FitnessScore;
    // TODO: generalize fitness. should be able to use vecs, etc.

    /// This method is intended for reporting, not measuring, fitness.
    fn fitness(&self) -> Option<&Self::Fitness>;

    fn scalar_fitness(&self) -> Option<f64>;

    fn set_fitness(&mut self, f: Self::Fitness);

    fn tag(&self) -> u64;

    fn set_tag(&mut self, tag: u64);

    fn problems(&self) -> Option<&Vec<Problem>>;

    fn store_answers(&mut self, results: Vec<Problem>);

    fn len(&self) -> usize;
}
