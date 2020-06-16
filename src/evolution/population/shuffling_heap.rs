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
    count: usize,
}

impl<P> ShufflingHeap<P> {
    pub fn new<H: Hash>(seed: &H) -> Self {
        let rng = hash_seed_rng(seed);
        let heap = BinaryHeap::new();
        Self {
            rng,
            heap,
            count: 0,
        }
    }

    pub fn pop(&mut self) -> Option<P> {
        self.count -= 1;
        self.heap.pop().map(|Cell { tag, val }| val)
    }

    pub fn push(&mut self, item: P) {
        self.count += 1;
        let cell = Cell {
            val: item,
            tag: self.rng.gen::<Tag>(),
        };
        self.heap.push(cell)
    }

    pub fn len(&self) -> usize {
        self.count
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

pub struct ShufflingIter<P>(ShufflingHeap<P>);

impl<P: Hash> IntoIterator for ShufflingHeap<P> {
    type Item = P;
    type IntoIter = ShufflingIter<P>;

    fn into_iter(self) -> Self::IntoIter {
        ShufflingIter(self)
    }
}

impl<P: Hash> Iterator for ShufflingIter<P> {
    type Item = P;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

impl<P> Extend<P> for ShufflingHeap<P> {
    // This is a bit simpler with the concrete type signature: we can call
    // extend on anything which can be turned into an Iterator which gives
    // us i32s. Because we need i32s to put into ShufflingHeap.
    fn extend<T: IntoIterator<Item = P>>(&mut self, iter: T) {
        // The implementation is very straightforward: loop through the
        // iterator, and add() each element to ourselves.
        for elem in iter {
            self.push(elem);
        }
    }
}

impl<P> Default for ShufflingHeap<P> {
    fn default() -> Self {
        // FIXME: this introduces unseeded randomness back into things.
        // fine for now, since we never really achieved reproducibility,
        // but if we come back to that project later, this will have to
        // be fixed.
        let seed = rand::random::<u64>();
        ShufflingHeap::new(&seed)
    }
}
