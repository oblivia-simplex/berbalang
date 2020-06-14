use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn hash_seed_rng<H: Hash>(thing: &H) -> impl Rng {
    let seed = hash_seed(thing);
    let rng = StdRng::from_seed(seed);
    rng
}

pub fn hash_seed<H: Hash>(thing: &H) -> [u8; 32] {
    let mut h = DefaultHasher::new();
    thing.hash(&mut h);
    let hash = h.finish();
    let mut seed = [0_u8; 32];
    for i in 0..32 {
        seed[i] = ((hash >> (i * 2) as u64) & 0xFF) as u8;
    }
    seed
}
