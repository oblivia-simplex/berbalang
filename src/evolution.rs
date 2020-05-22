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

#[derive(Debug, Clone, Deserialize, Eq, PartialEq)]
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

impl<E: Evaluate<P>, P: Phenome, C: Configure> Epoch<E, P, C> {
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

        // try using a heap for the population instead, and then
        // follow jackie's suggestion to take a random reservoir
        // sample of combatants.

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

        combatants.sort_by(|a, b| a.fitness().cmp(&b.fitness()));
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
        let offspring: Vec<P> = mother.mate(&father, config.clone());

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

    pub fn target_reached(&self, target: <P as Phenome>::Fitness) -> bool {
        self.best
            .as_ref()
            .and_then(|b| b.fitness())
            .map_or(false, |f| f <= target)
    }
}

pub trait Genome: Debug {
    type Params;

    fn random(params: &Self::Params) -> Self
    where
        Self: Sized;

    fn crossover<C: Configure>(&self, mate: &Self, params: Arc<C>) -> Vec<Self>
    where
        Self: Sized;

    fn mutate(&mut self);

    fn mate<C: Configure>(&self, other: &Self, params: Arc<C>) -> Vec<Self>
    where
        Self: Sized,
    {
        log::debug!("Mating {:?} and {:?}", self, other);
        let mut offspring = self.crossover::<C>(other, params.clone());
        let mut rng = thread_rng();
        for child in offspring.iter_mut() {
            if rng.gen::<f32>() < params.mutation_rate() {
                child.mutate();
            }
        }
        log::debug!("Offspring: {:?}", offspring);
        offspring
    }
}

pub trait Phenome: Clone + Debug + Send + Ord + Genome {
    type Fitness: FitnessScore + From<usize>;
    type Inst;

    /// This method is intended for reporting, not measuring, fitness.
    fn fitness(&self) -> Option<Self::Fitness>;

    fn set_fitness(&mut self, f: Self::Fitness);

    fn tag(&self) -> u64;

    fn set_tag(&mut self, tag: u64);

    fn problems(&self) -> Option<&Vec<Problem>>;

    fn store_answers(&mut self, results: Vec<Problem>);
}
