use std::cmp::{Ordering, PartialOrd};
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;
use std::sync::Arc;

use rand::{thread_rng, Rng};

use crate::configure::{Config, Problem};
use crate::evaluator::Evaluate;
use crate::fitness::FitnessScore;
use crate::observer::Observer;
use crate::util::count_min_sketch;
use crate::util::count_min_sketch::DecayingSketch;
use rand::rngs::ThreadRng;

pub struct Epoch<E: Evaluate<P>, P: Phenome + Debug + Send + Clone + Ord + 'static> {
    pub population: BinaryHeap<P>,
    pub config: Arc<Config>,
    pub best: Option<P>,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
}

impl<E: Evaluate<P>, P: Phenome + Genome> Epoch<E, P> {
    pub fn evolve(self) -> Self {
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
        let mut combatants = iter::repeat(())
            .take(tournament_size)
            .filter_map(|()| population.pop())
            .map(|mut e| {
                e.set_tag(rng.gen::<u64>());
                e
            })
            .map(|e| evaluator.evaluate(e))
            .map(|e| {
                observer.observe(e.clone());
                e
            })
            .collect::<Vec<P>>();

        combatants.sort_by(|a, b| {
            a.fitness()
                .partial_cmp(&b.fitness())
                .unwrap_or(Ordering::Equal)
        });

        //log::debug!("combatants' fitnesses: {:?}", combatants.iter().map(|c| c.fitness()).collect::<Vec<_>>());
        best = Self::update_best(best, &combatants[0]);

        // kill one off for every offspring to be produced
        for _ in 0..config.num_offspring() {
            let _ = combatants.pop();
        }

        // replace the combatants that will neither breed nor die
        let bystanders = config.tournament_size() - (config.num_offspring() + 2);
        for _ in 0..bystanders {
            if let Some(c) = combatants.pop() {
                population.push(c);
            }
        }
        // TODO implement breeder, similar to observer, etc?
        let mother = combatants.pop().unwrap();
        let father = combatants.pop().unwrap();
        let offspring: Vec<P> = mother.mate(&father, &config);

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
            iteration: iteration + 1,
            observer,
            evaluator,
        }
    }

    pub fn update_best(best: Option<P>, champ: &P) -> Option<P> {
        match best {
            Some(ref best) if champ.fitness() < best.fitness() => {
                log::info!("new champ with fitness {:?}:\n{:?}", champ.fitness(), champ);
                Some(champ.clone())
            }
            None => {
                log::info!("new champ with fitness {:?}\n{:?}", champ.fitness(), champ);
                Some(champ.clone())
            }
            _ => best,
        }
    }

    // pub fn target_reached(&self, target: &<P as Phenome>::Fitness) -> bool {
    //     self.best
    //         .as_ref()
    //         .and_then(|b| b.fitness())
    //         .map_or(false, |f| f[0] <= target)
    // }
}

pub trait Genome: Debug {
    type Allele: Clone + Copy + Debug + Hash;

    fn chromosome(&self) -> &[Self::Allele];

    fn chromosome_mut(&mut self) -> &mut [Self::Allele];

    fn random(params: &Config) -> Self
    where
        Self: Sized;

    fn crossover(&self, mate: &Self, params: &Config) -> Vec<Self>
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
            let take_to = std::cmp::min(ptrs[src] + sample(&mut rng), parents[src].len());
            let slice = &parents[src][take_from..take_to];
            chromosome.extend_from_slice(slice);
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

    fn mate(&self, other: &Self, params: &Config) -> Vec<Self>
    where
        Self: Sized,
    {
        let mut rng = thread_rng();
        let mut offspring = self.crossover(other, params);
        for child in offspring.iter_mut() {
            if rng.gen_range(0.0, 1.0) < params.mutation_rate() {
                child.mutate(&params);
            }
        }
        offspring
    }

    fn digrams(&self) -> Box<dyn Iterator<Item = (Self::Allele, Self::Allele)> + '_> {
        Box::new(
            self.chromosome()
                .iter()
                .zip(self.chromosome().iter().skip(1))
                .map(|(a, b)| (*a, *b)),
        )
    }

    fn record_genetic_frequency(
        &self,
        sketch: &mut DecayingSketch,
        timestamp: usize,
    ) -> Result<(), count_min_sketch::Error> {
        for digram in self.digrams() {
            sketch.insert(digram, timestamp)?
        }
        Ok(())
    }

    fn measure_genetic_frequency(
        &self,
        sketch: &DecayingSketch,
    ) -> Result<f64, count_min_sketch::Error> {
        // The lower the score, the rarer the digrams composing the genome.
        // We divide by the length to avoid penalizing longer genomes.
        // let mut sum = 0_f64;
        // for digram in self.digrams() {
        //     sum += (sketch.query(digram, timestamp)?);
        // };
        // Ok(sum)
        self.digrams()
            .map(|digram| sketch.query(digram))
            .collect::<Result<Vec<_>, _>>()
            .map(|v| v.into_iter().fold(std::f64::MAX, |a, b| a.min(b)))
    }
}

pub trait Phenome: Clone + Debug + Send + Ord {
    type Fitness: FitnessScore;
    // TODO: generalize fitness. should be able to use vecs, etc.
    type Inst;

    /// This method is intended for reporting, not measuring, fitness.
    fn fitness(&self) -> Option<&Self::Fitness>;

    fn set_fitness(&mut self, f: Self::Fitness);

    fn tag(&self) -> u64;

    fn set_tag(&mut self, tag: u64);

    fn problems(&self) -> Option<&Vec<Problem>>;

    fn store_answers(&mut self, results: Vec<Problem>);

    fn len(&self) -> usize;
}
