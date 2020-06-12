use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::Index;

use fasteval::{Compiler, Slab};
use hashbrown::HashMap;
use itertools::Itertools;
use serde::export::Formatter;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

pub type FitnessMap<'a> = HashMap<&'a str, f64>;

pub trait HasScalar {
    fn scalar(&self) -> f64;
}

impl HasScalar for Vec<f64> {
    fn scalar(&self) -> f64 {
        self.iter().sum()
    }
}

pub trait FitnessScore:
    Sized + PartialEq + Debug + Send + Clone + PartialOrd + Serialize + PartialOrd + HasScalar
{
}

impl FitnessScore for Vec<f64> {}

#[derive(Clone, Serialize, Deserialize)]
pub struct Pareto<'a>(#[serde(borrow)] FitnessMap<'a>);

impl Pareto<'static> {
    pub fn new() -> Self {
        Pareto(HashMap::new())
    }

    pub fn values(&self) -> impl Iterator<Item = &f64> {
        self.inner().iter().sorted_by_key(|p| p.0).map(|(_k, v)| v)
    }
}

impl HasScalar for Pareto<'static> {
    fn scalar(&self) -> f64 {
        self.values().sum()
    }
}

impl MapFit for Pareto<'static> {
    fn inner_mut(&mut self) -> &mut FitnessMap<'static> {
        &mut self.0
    }

    fn inner(&self) -> &FitnessMap<'static> {
        &self.0
    }

    fn from_map(map: FitnessMap<'static>) -> Self {
        Self(map)
    }
}

pub trait MapFit {
    fn inner_mut(&mut self) -> &mut FitnessMap<'static>;

    fn inner(&self) -> &FitnessMap<'static>;

    fn from_map(map: FitnessMap<'static>) -> Self
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

impl From<FitnessMap<'static>> for Pareto<'static> {
    fn from(map: FitnessMap<'static>) -> Self {
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

impl fmt::Debug for Pareto<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pareto [")?;
        self.0
            .iter()
            .sorted_by_key(|p| p.0)
            .map(|(obj, score)| writeln!(f, "\t{} => {},", obj, score))
            .collect::<Result<Vec<()>, _>>()?;
        writeln!(f, "]")
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
pub struct ShuffleFit(#[serde(borrow)] FitnessMap<'static>);

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
        keys[h % keys.len()]
    }
}

impl HasScalar for ShuffleFit {
    fn scalar(&self) -> f64 {
        self.values().sum()
    }
}

impl MapFit for ShuffleFit {
    fn inner_mut(&mut self) -> &mut FitnessMap<'static> {
        &mut self.0
    }

    fn inner(&self) -> &FitnessMap<'static> {
        &self.0
    }

    fn from_map(map: FitnessMap<'static>) -> Self {
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

impl From<FitnessMap<'static>> for ShuffleFit {
    fn from(map: FitnessMap<'static>) -> Self {
        Self::from_map(map)
    }
}

impl FitnessScore for ShuffleFit {}

#[derive(Serialize, Deserialize)]
pub struct Weighted<'a> {
    weights: HashMap<String, String>,
    //   #[serde(skip)]
    //   pub weights: HashMap<String, fasteval::Instruction>,
    //#[serde(skip)]
    //slab: Mutex<Slab>,
    #[serde(borrow)]
    pub scores: HashMap<&'a str, f64>,
    cached_scalar: Mutex<Option<f64>>,
}

impl PartialEq for Weighted<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.scores == other.scores && self.weights == other.weights
    }
}

fn compile_weight_expression(
    expr: &str,
    slab: &mut fasteval::Slab,
    parser: &fasteval::Parser,
) -> fasteval::Instruction {
    // See the `parse` documentation to understand why we use `from` like this:
    parser
        .parse(expr, &mut slab.ps)
        .expect("Failed to parse expression")
        .from(&slab.ps)
        .compile(&slab.ps, &mut slab.cs)
}

impl Clone for Weighted<'_> {
    fn clone(&self) -> Self {
        Self {
            cached_scalar: Mutex::new(None),
            weights: self.weights.clone(),
            scores: self.scores.clone(),
        }
    }
}

// TODO: this could be optimized quite a bit by parsing the fasteval expressions
// // when constructing the Weighted instance.
// fn apply_weighting(
//     weighting: &fasteval::Instruction,
//     score: f64,
//     mut slab: &mut Slab,
// ) -> Result<f64, Error> {
//     let mut map = BTreeMap::new();
//     map.insert("x".to_string(), score);
//     let res = fasteval::eval_compiled_ref!(weighting, &mut slab, &mut map);
//     Ok(res)
// }

fn apply_weighting(weighting: &str, score: f64) -> f64 {
    let mut ns = BTreeMap::new();
    ns.insert("x", score);
    fasteval::ez_eval(weighting, &mut ns).expect("Failed to evaluate weighting expression")
}

fn compile_weight_map(
    weights: &HashMap<String, String>,
    mut slab: &mut Slab,
) -> HashMap<String, fasteval::Instruction> {
    let parser = fasteval::Parser::new();
    let mut weight_map = HashMap::new();
    for (attr, weight) in weights.iter() {
        let compiled = compile_weight_expression(weight, &mut slab, &parser);
        weight_map.insert(attr.clone(), compiled);
    }
    weight_map
}

impl Weighted<'static> {
    pub fn new(weights: HashMap<String, String>) -> Self {
        Self {
            //slab: Mutex::new(slab),
            weights,
            //weights: weight_map,
            scores: FitnessMap::new(),
            cached_scalar: Mutex::new(None),
        }
    }

    pub fn scalar(&self) -> f64 {
        let mut cache = self.cached_scalar.lock().unwrap();
        if let Some(val) = *cache {
            return val;
        }
        let val = self
            .scores
            .iter()
            .sorted_by_key(|p| p.0)
            // FIXME wasteful allocations here.
            .map(|(k, score)| {
                if let Some(weight) = self.weights.get(&(*k).to_string()) {
                    apply_weighting(weight, *score)
                } else {
                    // if no weight is provided, just return the score as-is
                    *score
                }
            })
            .sum::<f64>();
        *cache = Some(val);
        val
    }
}

impl HasScalar for Weighted<'static> {
    fn scalar(&self) -> f64 {
        Weighted::scalar(&self)
    }
}

impl PartialOrd for Weighted<'static> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.scalar().partial_cmp(&other.scalar())
    }
}

impl FitnessScore for Weighted<'static> {}

impl MapFit for Weighted<'static> {
    fn inner_mut(&mut self) -> &mut FitnessMap<'static> {
        &mut self.scores
    }

    fn inner(&self) -> &FitnessMap<'static> {
        &self.scores
    }

    fn from_map(_map: FitnessMap<'static>) -> Self
    where
        Self: Sized,
    {
        unimplemented!("doesn't really make sense for Weighted")
    }

    fn average(frame: &[&Self]) -> Self {
        debug_assert!(!frame.is_empty(), "Don't try to average empty frames");
        let mut map = HashMap::new();
        let mut weights = None;
        for p in frame.iter() {
            if weights.is_none() {
                weights = Some(p.weights.clone());
            }
            for (&k, &v) in p.inner().iter() {
                *(map.entry(k).or_insert(0.0)) += v;
            }
        }
        let len = frame.len() as f64;
        for (_k, v) in map.iter_mut() {
            *v /= len;
        }
        let weights = weights.unwrap();
        Self {
            weights,
            scores: map,
            cached_scalar: Mutex::new(None),
        }
    }
}

impl Index<&str> for Weighted<'static> {
    type Output = f64;

    fn index(&self, index: &str) -> &Self::Output {
        &self.scores[index]
    }
}

impl fmt::Debug for Weighted<'static> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Weighted:")?;
        for (attr, score) in self.scores.iter().sorted_by_key(|p| p.0) {
            if let Some(weight) = self.weights.get(&attr.to_string()) {
                writeln!(f, "    {}: {}, to be weighted by ({})", attr, score, weight)?;
            } else {
                writeln!(f, "    {}: {}, unweighted", attr, score)?;
            }
        }
        writeln!(f, "Scalar: {}", self.scalar())
    }
}

#[cfg(test)]
mod test {
    use crate::hashmap;
    use crate::pareto;

    use super::*;

    #[test]
    fn test_pareto_ordering() {
        let p1: Pareto<'static> = pareto! {"obj_a" => 0.1, "swankiness" => 2.0, "doom" => 3.1, };
        let p2: Pareto<'static> = pareto! {"obj_a" => 0.1, "swankiness" => 1.9, "doom" => 3.1, };
        let mut ps = vec![&p1, &p2];
        ps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        assert_eq!(ps[0], &p2);
    }

    // #[test]
    // fn test_find_minima() {
    //     fn random_pareto() -> Pareto<'static> {
    //         let mut par = Pareto::new();
    //         for i in 0..10 {
    //             par.insert(UNNAMED_OBJECTIVES[i], rand::random::<f64>());
    //         }
    //         par
    //     }
    //
    //     let sample = iter::repeat(())
    //         .take(100)
    //         .map(|()| random_pareto())
    //         .collect::<Vec<Pareto>>();
    //
    //     let mut minima = HashSet::new();
    //     for x in sample {}
    // }
}
