use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub enum Error {
    InvalidTimestamp { timestamp: usize, elapsed: usize },
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Error::*;
        match self {
            InvalidTimestamp { timestamp, elapsed } => write!(
                f,
                "Timestamp of {} earlier than last seen: {}",
                timestamp, elapsed
            ),
        }
    }
}

fn hash<T: Hash>(thing: &T, index: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    (thing, index).hash(&mut hasher);
    hasher.finish() as usize
}

#[derive(Debug)]
pub struct DecayingSketch {
    freq_table: Vec<Vec<f64>>,
    time_table: Vec<Vec<usize>>,
    num_hash_funcs: usize,
    width: usize,
    half_life: f64,
    counter: usize,
    decay: bool,
} // TODO decay the counter?

impl Default for DecayingSketch {
    fn default() -> Self {
        let num_hash_funcs = 1 << 3;
        let width = 1 << 10;
        let half_life = 100_000_f64; // FIXME tweak
        Self::new(num_hash_funcs, width, half_life)
    }
}

impl DecayingSketch {
    fn new(num_hash_funcs: usize, width: usize, half_life: f64) -> Self {
        assert!(
            half_life > 0.0,
            "half_life for DecayingSketch cannot be less than or equal to 0"
        );
        let time_table = vec![vec![0_usize; width]; num_hash_funcs];
        let freq_table = vec![vec![0_f64; width]; num_hash_funcs];
        Self {
            num_hash_funcs,
            width,
            half_life,
            freq_table,
            time_table,
            counter: 0,
            decay: false, // FIXME
        }
    }

    fn decay_factor(&self, prior_timestamp: usize, current_timestamp: usize) -> f64 {
        if !self.decay {
            return 1.0;
        }
        assert!(current_timestamp > prior_timestamp);

        let age = current_timestamp - prior_timestamp;
        //log::debug!("age = {}", age);
        let d = 2_f64.powf(-(age as f64 / self.half_life));
        //log::debug!("decay factor = {}", d);
        d
    }

    pub fn insert<T: Hash>(&mut self, thing: T) {
        self.counter += 1;
        let current_timestamp = self.counter;

        for i in 0..self.num_hash_funcs {
            let loc = hash(&thing, i) % self.width;
            let s = self.freq_table[i][loc];
            let prior_timestamp = self.time_table[i][loc];
            // log::debug!(
            //     "in insert, about to call decay. prior_timestamp: {}, current_timestamp: {}",
            //     prior_timestamp,
            //     current_timestamp
            // );
            self.freq_table[i][loc] =
                1.0 + s * self.decay_factor(prior_timestamp, current_timestamp);
            self.time_table[i][loc] = current_timestamp;
        }
    }

    pub fn query<T: Hash>(&self, thing: T) -> f64 {
        let current_time = self.counter;
        let (freq, timestamp) = (0..self.num_hash_funcs)
            .map(|i| {
                let loc = hash(&thing, i) % self.width;
                (self.freq_table[i][loc], self.time_table[i][loc])
            })
            .fold(
                (std::f64::MAX, std::usize::MAX),
                |(a_freq, a_time), (b_freq, b_time)| {
                    if a_freq < b_freq {
                        (a_freq, a_time)
                    } else {
                        (b_freq, b_time)
                    }
                },
            );
        //log::debug!("in query, about to call decay");
        let d = self.decay_factor(timestamp, current_time);
        // and we divide the score by the counter because we're interested in
        // *relative* frequency
        let f = d * freq as f64 / self.counter as f64;
        log::debug!("f = {}", f);
        f
    }

    pub fn flush(&mut self) {
        for i in 0..self.num_hash_funcs {
            for j in 0..self.width {
                self.time_table[i][j] = 0;
                self.freq_table[i][j] = 0.0;
            }
        }
    }
}

pub struct CountMinSketch {
    table: Vec<Vec<usize>>,
    num_hash_funcs: usize,
    width: usize,
}

impl Default for CountMinSketch {
    fn default() -> Self {
        let num_hash_funcs = 1 << 3;
        let width = 1 << 10;
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

    pub fn flush(&mut self) {
        for i in 0..self.num_hash_funcs {
            for j in 0..self.width {
                self.table[i][j] = 0
            }
        }
    }

    pub fn insert<T: Hash>(&mut self, thing: T) {
        for i in 0..self.num_hash_funcs {
            let loc = hash(&thing, i) % self.width;
            self.table[i][loc] += 1;
        }
    }

    pub fn query<T: Hash>(&self, thing: T) -> usize {
        (0..self.num_hash_funcs)
            .map(|i| {
                let loc = hash(&thing, i) % self.width;
                self.table[i][loc]
            })
            .fold(std::usize::MAX, std::cmp::min)
    }
}
