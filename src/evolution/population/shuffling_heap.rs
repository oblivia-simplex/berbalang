//! The `ShufflingHeap` data structure stores and `pop()`s its content in random order.
//! Note that, in general, and by design, `heap.push(x); heap.pop(x) != Some(x)`.
//!
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::iter::FromIterator;

use bson::spec::ElementType::Binary;
use rand::Rng;
use rayon::iter::{FromParallelIterator, IntoParallelIterator};

use crate::util::random::{hash_seed, hash_seed_rng, Prng};

type Tag = u64;

struct Cell<P> {
    tag: Tag,
    val: P,
}

impl<P> PartialEq for Cell<P> {
    fn eq(&self, other: &Self) -> bool {
        self.tag.eq(&other.tag)
    }
}

impl<P> Eq for Cell<P> {}

impl<P> PartialOrd for Cell<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.tag.partial_cmp(&other.tag)
    }
}

impl<P> Ord for Cell<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tag.cmp(&other.tag)
    }
}

pub struct ShufflingHeap<P> {
    heap: BinaryHeap<Cell<P>>,
    rng: Prng,
}

impl<P> ShufflingHeap<P> {
    pub fn new<H: Hash>(seed: &H) -> Self {
        let rng = hash_seed_rng(seed);
        let heap = BinaryHeap::new();
        Self { rng, heap }
    }

    pub fn pop(&mut self) -> Option<P> {
        self.heap.pop().map(|Cell { tag, val }| val)
    }

    pub fn push(&mut self, item: P) {
        let cell = Cell {
            val: item,
            tag: self.rng.gen::<Tag>(),
        };
        self.heap.push(cell)
    }
}

impl<P: Hash> FromIterator<P> for ShufflingHeap<P> {
    fn from_iter<I: IntoIterator<Item = P>>(iter: I) -> Self {
        let mut iterator = iter.into_iter();
        if let Some(first) = iterator.next() {
            // use the first entry as a random seed
            let mut heap = ShufflingHeap::new(&first);
            heap.push(first);

            for item in iterator {
                heap.push(item)
            }
            return heap;
        }
        panic!("won't create shuffling heap from empty iterator")
    }
}
