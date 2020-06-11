use crate::util::five_letter_words::WORDS;
use rand::Rng;

pub fn random_syllables(syllables: usize) -> String {
    let mut rng = rand::thread_rng();
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

pub fn random_words(words: usize) -> String {
    let mut rng = rand::thread_rng();
    let mut s = String::new();
    let n = WORDS.len();
    for i in 0..words {
        s.push_str(WORDS[rng.gen_range(0, n)]);
        if i + 1 < words {
            s.push_str("-")
        }
    }
    s
}

pub fn random(parts: usize) -> String {
    random_words(parts)
}
