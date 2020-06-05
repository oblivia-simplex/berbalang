use std::fmt::Debug;
use std::ops::{Add, Div, Index, IndexMut};

use indexmap::map::IndexMap;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pareto<'a>(#[serde(borrow)] IndexMap<&'a str, f64>);

impl Pareto<'static> {
    pub fn new() -> Self {
        Pareto(IndexMap::new())
    }

    pub fn insert(&mut self, name: &'static str, thing: f64) {
        self.0.insert(name, thing);
    }

    pub fn get(&self, name: &'static str) -> Option<f64> {
        (self.0.get(name)).cloned()
    }

    pub fn values(&self) -> impl Iterator<Item = &f64> {
        self.0.iter().sorted_by_key(|p| p.0).map(|(_k, v)| v)
    }

    pub fn average(frame: &[&Self]) -> Self {
        let mut map = IndexMap::new();
        for p in frame.iter() {
            for (&k, &v) in p.0.iter() {
                *(map.entry(k).or_insert(0.0)) += v;
            }
        }
        let len = frame.len() as f64;
        for (_k, v) in map.iter_mut() {
            *v /= len;
        }
        Self(map)
    }
}

impl FitnessScore for Pareto<'static> {}

impl PartialOrd for Pareto<'static> {
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
        self.values().zip(other.values()).all(|(x, y)| x <= y)
            && self.values().zip(other.values()).any(|(x, y)| x < y)
    }
}

impl PartialEq for Pareto<'static> {
    fn eq(&self, other: &Self) -> bool {
        self.values().zip(other.values()).all(|(s, o)| s.eq(o))
    }
}

static UNNAMED_OBJECTIVES: [&str; 10] = [
    "objective_0",
    "objective_1",
    "objective_2",
    "objective_3",
    "objective_4",
    "objective_5",
    "objective_6",
    "objective_7",
    "objective_8",
    "objective_9",
];

impl From<Vec<f64>> for Pareto<'static> {
    fn from(vec: Vec<f64>) -> Self {
        let mut map = IndexMap::new();
        for (i, v) in vec.iter().enumerate() {
            map.insert(UNNAMED_OBJECTIVES[i], *v);
        }
        Pareto(map)
    }
}

impl From<IndexMap<&'static str, f64>> for Pareto<'static> {
    fn from(map: IndexMap<&'static str, f64>) -> Self {
        Pareto(map)
    }
}

impl Into<Vec<f64>> for Pareto<'static> {
    fn into(self) -> Vec<f64> {
        self.values().cloned().collect::<Vec<f64>>()
    }
}

// impl AsRef<[f64]> for Pareto<'static> {
//     fn as_ref(&self) -> &[f64] {
//         &self.0.values()
//     }
// }

impl Index<&str> for Pareto<'static> {
    type Output = f64;

    fn index(&self, i: &str) -> &Self::Output {
        &self
            .0
            .get(i)
            .unwrap_or_else(|| panic!("Invalid index for Pareto instance: {:?}", i))
    }
}

impl Index<usize> for Pareto<'static> {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        self.values()
            .nth(i)
            .expect("Invalid numeric index for Pareto")
    }
}

impl IndexMut<&str> for Pareto<'static> {
    fn index_mut(&mut self, i: &str) -> &mut Self::Output {
        &mut self.0[i]
    }
}

#[macro_export]
macro_rules! pareto {
    ($($key:expr => $val:expr, $(,)?)*) => {
        Pareto(indexmap!{$( $key => $val, )*})
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
    use indexmap::indexmap;

    #[test]
    fn test_pareto_ordering() {
        let p1: Pareto<'static> = pareto! {"obj_a" => 0.1, "swankiness" => 2.0, "doom" => 3.1, };
        let p2: Pareto<'static> = pareto! {"obj_a" => 0.1, "swankiness" => 1.9, "doom" => 3.1, };
        let mut ps = vec![&p1, &p2];
        ps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        assert_eq!(ps[0], &p2);
    }
}
