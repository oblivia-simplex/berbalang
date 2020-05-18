use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::ops::{Add, Div};
use std::sync::Arc;

use serde::Serialize;

use crate::configure::Configure;
use crate::evaluator::Evaluate;
use crate::observer::Observer;

pub trait Epochal {
    type Observer;
    type Evaluator;

    /// The evolve function turns the crank once on the evolutionary
    /// process.
    fn evolve(self) -> Self;
}

pub struct Epoch<E: Evaluate, G: Genome, P: Phenome + Debug + Send + Clone, C: Configure> {
    pub population: Vec<G>,
    pub config: Arc<C>,
    pub best: Option<G>,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
}

pub trait FitnessScoreReq:
PartialEq + Debug + Send + Clone + PartialOrd + Serialize + Add + Div
{}

pub trait FitnessScore:
PartialEq + Debug + Send + Clone + PartialOrd + Serialize + Add + Div
{}

#[derive(Copy, Clone, PartialOrd, PartialEq, Eq, serde_derive::Serialize, Default, Hash, Ord)]
pub struct FitnessScalar<F: FitnessScoreReq>(pub F);

/*
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, serde_derive::Serialize, Hash)]
pub struct FitnessVector<F: FitnessScoreReq>(pub Vec<F>);

impl<T: Add<Output = T> + FitnessScoreReq> Add for FitnessVector<T> {
    type Output = Self;

    // Add the fitness vectors pairwise
    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.0.len(), other.0.len());
        FitnessVector(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| *a + *b)
                .collect::<Vec<_>>(),
        )
    }
}
*/
impl<T: Add<Output=T> + FitnessScoreReq> Add for FitnessScalar<T> {
    type Output = Self;

    // Add the fitness vectors pairwise
    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}
/*
impl<T: Div<Output = T> + FitnessScoreReq> Div for FitnessVector<T> {
    type Output = Self;

    // Div the fitness vectors pairwise
    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.0.len(), rhs.0.len());
        FitnessVector(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(a, b)| *a / *b)
                .collect::<Vec<_>>(),
        )
    }
}
*/
impl<T: Div<Output=T> + FitnessScoreReq> Div for FitnessScalar<T> {
    type Output = Self;

    // Div the fitness vectors pairwise
    fn div(self, rhs: Self) -> Self::Output {
        FitnessScalar(self.0 / rhs.0)
    }
}

macro_rules! impl_scalar_and_vector_fitness_scores {
    ($( $t:ty ), *) => {
       $(
          impl FitnessScoreReq for $t {}
          impl FitnessScore for FitnessScalar<$t> {}
          //impl FitnessScore for FitnessVector<Vec<$t>> {}


          impl std::fmt::Debug for FitnessScalar<$t> {
              fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                  write!(f, "{:?}", self.0)
              }
          }

          impl From<$t> for FitnessScalar<$t> {
              fn from(n: $t) -> Self {
                  Self(n)
              }
          }
/*
          impl From<Vec<$t>> for FitnessVector<$t> {
              fn from(n: Vec<$t>) -> Self {
                  Self(n)
              }
          }
          */

        )*
    }
}

impl_scalar_and_vector_fitness_scores!(usize);

pub trait Genome {
    type Params;

    fn random(params: &Self::Params) -> Self
        where
            Self: Sized;

    fn crossover(&self, mate: &Self) -> Vec<Self>
    where
        Self: Sized;

    fn mutate(&mut self);
}

pub trait Phenome: Clone + Debug + Send {
    type Fitness: FitnessScore;

    /// This method is intended for reporting, not measuring, fitness.
    fn fitness(&self) -> Option<Self::Fitness>;
}
