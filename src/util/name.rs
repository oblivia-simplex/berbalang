use std::hash::{Hash, Hasher};

use rand::Rng;

use crate::util::five_letter_words::WORDS;
use crate::util::random::hash_seed_rng;

pub fn random_syllables<H: Hash>(syllables: usize, seed: H) -> String {
    let mut rng = hash_seed_rng(&seed);
    let consonants = [
        'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w',
        'x', 'z',
    ];
    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    let mut s = Vec::new();

    for i in 0..syllables {
        s.push(consonants[rng.gen::<usize>() % consonants.len()]);
        s.push(vowels[rng.gen::<usize>() % vowels.len()]);
        s.push(consonants[rng.gen::<usize>() % consonants.len()]);
        if i % 2 == 1 && i < syllables - 1 {
            s.push('-')
        }
    }

    s.iter().collect()
}

pub fn random_words<H: Hash>(words: usize, seed: H) -> String {
    let mut index = hash(seed);
    let mut s = String::new();
    let n = WORDS.len();
    for i in 0..words {
        s.push_str(WORDS[index % n]);
        index = hash(index);
        if i + 1 < words {
            s.push_str("-")
        }
    }
    s
}

pub fn random<H: Hash>(parts: usize, seed: H) -> String {
    random_words(parts, seed)
}

fn hash<H: Hash>(thing: H) -> usize {
    let mut h = fnv::FnvHasher::default();
    thing.hash(&mut h);
    h.finish() as usize
}
