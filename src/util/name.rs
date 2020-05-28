use rand::Rng;

pub fn random(syllables: usize) -> String {
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
