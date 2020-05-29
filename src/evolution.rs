use std::cmp::{Ordering, PartialOrd};
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::iter;
use std::sync::Arc;

use rand::{thread_rng, Rng};
use serde_derive::Deserialize;

use crate::configure::Configure;
use crate::evaluator::Evaluate;
use crate::fitness::FitnessScore;
use crate::observer::Observer;
use rand::rngs::ThreadRng;

#[derive(Debug, Clone, Deserialize, Eq, PartialEq, Hash)]
pub struct Problem {
    pub input: Vec<i32>,
    // TODO make this more generic
    pub output: i32,
    // Ditto
    pub tag: u64,
}

impl PartialOrd for Problem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.tag.partial_cmp(&other.tag)
    }
}

impl Ord for Problem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tag.cmp(&other.tag)
    }
}

pub struct Epoch<E: Evaluate<P>, P: Phenome + Debug + Send + Clone + Ord + 'static, C: Configure> {
    pub population: BinaryHeap<P>,
    pub config: Arc<C>,
    pub best: Option<P>,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
}

impl<E: Evaluate<P>, P: Phenome + Genome<C>, C: Configure> Epoch<E, P, C> {
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

pub trait Genome<C: Configure>: Debug {
    type Allele: Clone + Debug;

    fn chromosome(&self) -> &[Self::Allele];

    fn chromosome_mut(&mut self) -> &mut[Self::Allele];

    fn random(params: &C) -> Self
    where
        Self: Sized;

    fn crossover(&self, mate: &Self, params: &C) -> Vec<Self>
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

    fn mutate(&mut self, params: &C);

    fn mate(&self, other: &Self, params: &C) -> Vec<Self>
    where
        Self: Sized,
    {
        let mut offspring = self.crossover(other, params);
        let mut rng = thread_rng();
        for child in offspring.iter_mut() {
            if rng.gen::<f32>() < params.mutation_rate() {
                child.mutate(&params);
            }
        }
        offspring
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
