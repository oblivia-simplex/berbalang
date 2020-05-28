use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct CountMinSketch {
    table: Vec<Vec<usize>>,
    num_hash_funcs: usize,
    width: usize,
}

impl Default for CountMinSketch {
    fn default() -> Self {
        let num_hash_funcs = 1<<3;
        let width = 1<<10;
        Self {
            table: vec![vec![0; width]; num_hash_funcs],
            num_hash_funcs,
            width,
        }
    }
}

impl CountMinSketch {
    pub fn new(num_hash_funcs: usize, width: usize) -> Self {
        Self {
            table: vec![vec![0; width]; num_hash_funcs],
            num_hash_funcs,
            width,
        }
    }

    fn hash<T: Hash>(thing: &T, index: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        (thing, index).hash(&mut hasher);
        hasher.finish() as usize
    }

    pub fn insert<T: Hash>(&mut self, thing: T) {
        for i in 0..self.num_hash_funcs {
            let loc = Self::hash(&thing, i) % self.width;
            self.table[i][loc] += 1;
        }
    }

    pub fn query<T: Hash>(&self, thing: T) -> usize {
        (0..self.num_hash_funcs)
            .map(|i| {
                let loc = Self::hash(&thing, i) % self.width;
                self.table[i][loc]
            })
            .fold(std::usize::MAX, std::cmp::min)
    }
}
