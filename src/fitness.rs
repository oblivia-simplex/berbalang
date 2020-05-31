use std::fmt::Debug;
use std::ops::{Add, Div, Index, IndexMut};

use serde::Serialize;
use std::cmp::Ordering;

pub trait FitnessScoreReq:
    PartialEq + Debug + Send + Clone + PartialOrd + Serialize + Add + Div
{
}

pub trait FitnessScore:
    Sized + PartialEq + Debug + Send + Clone + PartialOrd + Serialize + PartialOrd + Index<usize>
{
}

// this smells bad
// impl FitnessScore for usize {}
// impl FitnessScore for (usize, usize) {}
// impl FitnessScore for (usize, f32) {}
// impl FitnessScore for (usize, f32, usize) {}
// impl FitnessScore for (usize, f32, f32, usize) {}
// impl FitnessScore for f32 {}
// impl FitnessScore for f64 {}
//impl FitnessScore for Vec<f32> {}
impl FitnessScore for Vec<f64> {}

#[derive(Debug, Clone, Serialize)]
pub struct Pareto(pub Vec<f64>);

impl Pareto {
    pub fn new() -> Self {
        Pareto(Vec::new())
    }

    pub fn push(&mut self, thing: f64) {
        self.0.push(thing)
    }

    pub fn pop(&mut self) -> Option<f64> {
        self.0.pop()
    }
}

impl FitnessScore for Pareto {}

impl PartialOrd for Pareto {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        assert_eq!(
            self.0.len(),
            other.0.len(),
            "vectors must have the same length in order to perform Pareto comparisons"
        );
        if self < other {
            Some(Ordering::Less)
        } else if other < self {
            Some(Ordering::Greater)
        } else {
            None
        }
    }

    fn lt(&self, other: &Self) -> bool {
        assert_eq!(
            self.0.len(),
            other.0.len(),
            "vectors must have the same length in order to perform Pareto comparisons"
        );
        self.0.iter().zip(other.0.iter()).all(|(x, y)| x <= y)
            && self.0.iter().zip(other.0.iter()).any(|(x, y)| x < y)
    }
    // fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    //     let cmps = self.0.iter().zip(other.0.iter())
    //         .map(|(s,o)| s.partial_cmp(o))
    //         .collect::<Vec<Option<Ordering>>>();
    //     if cmps.contains(&Some(Ordering::Less)) && !cmps.contains(&Some(Ordering::Greater)) {
    //         Some(Ordering::Less)
    //     } else if cmps.contains(&Some(Ordering::Greater)) && !cmps.contains(&Some(Ordering::Less)) {
    //         Some(Ordering::Greater)
    //     } else {
    //         None
    //     }
    // }
}

impl PartialEq for Pareto {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(s, o)| s.eq(o))
    }
}

impl From<Vec<f64>> for Pareto {
    fn from(vec: Vec<f64>) -> Self {
        Pareto(vec)
    }
}

impl Index<usize> for Pareto {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl IndexMut<usize> for Pareto {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.0[i]
    }
}

#[macro_export]
macro_rules! pareto {
    ($($v:expr $(,)?)*) => {
        Pareto(vec![$( $v, )*])
    }
}

#[macro_export]
macro_rules! lexical {
    ($($v:expr $(,)?)*) => {
        vec![$( $v, )*]
    }
}

pub type Lexical<T> = Vec<T>;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pareto_ordering() {
        let p1: Pareto = pareto![0.1, 2.0, 3.1];
        let p2: Pareto = pareto![0.1, 1.9, 3.1];
        let mut ps = vec![&p1, &p2];
        ps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        assert_eq!(ps[0], &p2);
    }
}
