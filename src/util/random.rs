use std::hash::{Hash, Hasher};

use rand::SeedableRng;
use rand_xoshiro::Xoroshiro64Star;

pub type Prng = Xoroshiro64Star;

/// Takes a hashable value and returns a PRNG seeded with that
/// value's hash.
pub fn hash_seed_rng<H: Hash>(thing: &H) -> Prng {
    let seed = hash_seed(thing);
    rand_xoshiro::Xoroshiro64Star::from_seed(seed)
}

pub fn hash_seed<H: Hash>(thing: &H) -> [u8; 8] {
    let mut h = fnv::FnvHasher::default();
    thing.hash(&mut h);
    let hash = h.finish();
    let mut seed = [0_u8; 8];
    for i in 0..8 {
        seed[i] = ((hash >> (i * 8) as u64) & 0xFF) as u8;
    }
    seed
}
