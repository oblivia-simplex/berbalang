use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{Add, Div, Index};

use hashbrown::HashMap;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub trait FitnessScoreReq:
    PartialEq + Debug + Send + Clone + PartialOrd + Serialize + Add + Div
{
}

pub trait FitnessScore:
    Sized + PartialEq + Debug + Send + Clone + PartialOrd + Serialize + PartialOrd + Index<usize>
{
}

impl FitnessScore for Vec<f64> {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pareto<'a>(#[serde(borrow)] HashMap<&'a str, f64>);

impl Pareto<'static> {
    pub fn new() -> Self {
        Pareto(HashMap::new())
    }

    pub fn values(&self) -> impl Iterator<Item = &f64> {
        self.inner().iter().sorted_by_key(|p| p.0).map(|(_k, v)| v)
    }
}

impl MapFit for Pareto<'static> {
    fn inner_mut(&mut self) -> &mut HashMap<&'static str, f64> {
        &mut self.0
    }

    fn inner(&self) -> &HashMap<&'static str, f64> {
        &self.0
    }

    fn from_map(map: HashMap<&'static str, f64>) -> Self {
        Self(map)
    }
}

pub trait MapFit {
    fn inner_mut(&mut self) -> &mut HashMap<&'static str, f64>;

    fn inner(&self) -> &HashMap<&'static str, f64>;

    fn from_map(map: HashMap<&'static str, f64>) -> Self
    where
        Self: Sized;

    fn insert(&mut self, name: &'static str, thing: f64) {
        self.inner_mut().insert(name, thing);
    }

    fn get(&self, name: &'static str) -> Option<f64> {
        (self.inner().get(name)).cloned()
    }

    fn average(frame: &[&Self]) -> Self
    where
        Self: Sized,
    {
        let mut map = HashMap::new();
        for p in frame.iter() {
            for (&k, &v) in p.inner().iter() {
                *(map.entry(k).or_insert(0.0)) += v;
            }
        }
        let len = frame.len() as f64;
        for (_k, v) in map.iter_mut() {
            *v /= len;
        }
        Self::from_map(map)
    }
}

impl FitnessScore for Pareto<'static> {}

impl PartialOrd for Pareto<'static> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        debug_assert_eq!(
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
        let mut map = HashMap::new();
        for (i, v) in vec.iter().enumerate() {
            map.insert(UNNAMED_OBJECTIVES[i], *v);
        }
        Pareto(map)
    }
}

impl From<HashMap<&'static str, f64>> for Pareto<'static> {
    fn from(map: HashMap<&'static str, f64>) -> Self {
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

// impl IndexMut<&str> for Pareto<'static> {
//     fn index_mut(&mut self, i: &str) -> &mut Self::Output {
//         &mut self.0[i]
//     }
// }

// Let's define a semilattice over fitness values. This might be more efficient and
// easier to reason over than the "non-dominated sort".

pub type Lexical<T> = Vec<T>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShuffleFit(#[serde(borrow)] HashMap<&'static str, f64>);

impl ShuffleFit {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn values(&self) -> impl Iterator<Item = &f64> {
        self.inner().iter().sorted_by_key(|p| p.0).map(|(_k, v)| v)
    }

    pub fn epoch_key(&self) -> &'static str {
        let epoch = crate::get_epoch_counter();
        let mut hasher = DefaultHasher::new();
        epoch.hash(&mut hasher);
        let h = hasher.finish() as usize;
        let keys = self.0.keys().collect::<Vec<_>>();
        let k = keys[h % keys.len()];
        k
    }
}

impl MapFit for ShuffleFit {
    fn inner_mut(&mut self) -> &mut HashMap<&'static str, f64> {
        &mut self.0
    }

    fn inner(&self) -> &HashMap<&'static str, f64> {
        &self.0
    }

    fn from_map(map: HashMap<&'static str, f64>) -> Self {
        Self(map)
    }
}

impl PartialOrd for ShuffleFit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let k = self.epoch_key();
        self.0[k].partial_cmp(&other.0[k])
    }
}

impl Index<usize> for ShuffleFit {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        self.values()
            .nth(i)
            .expect("Invalid numeric index for Pareto")
    }
}

impl Index<&str> for ShuffleFit {
    type Output = f64;

    fn index(&self, i: &str) -> &Self::Output {
        &self
            .0
            .get(i)
            .unwrap_or_else(|| panic!("Invalid index for Pareto instance: {:?}", i))
    }
}

impl From<HashMap<&'static str, f64>> for ShuffleFit {
    fn from(map: HashMap<&'static str, f64>) -> Self {
        Self::from_map(map)
    }
}

impl FitnessScore for ShuffleFit {}

#[cfg(test)]
mod test {
    use crate::hashmap;
    use crate::pareto;

    use super::*;
    use hashbrown::HashSet;
    use std::iter;

    #[test]
    fn test_pareto_ordering() {
        let p1: Pareto<'static> = pareto! {"obj_a" => 0.1, "swankiness" => 2.0, "doom" => 3.1, };
        let p2: Pareto<'static> = pareto! {"obj_a" => 0.1, "swankiness" => 1.9, "doom" => 3.1, };
        let mut ps = vec![&p1, &p2];
        ps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        assert_eq!(ps[0], &p2);
    }

    #[test]
    fn test_find_minima() {
        fn random_pareto() -> Pareto {
            let mut par = Pareto::new();
            for i in 0..10 {
                par.insert(UNNAMED_OBJECTIVES[i], rand::random::<f64>());
            }
            par
        }

        let sample = iter::repeat(())
            .take(100)
            .map(|()| random_pareto())
            .collect::<Vec<Pareto>>();

        let mut minima = HashSet::new();
        for x in sample {}
    }
}
