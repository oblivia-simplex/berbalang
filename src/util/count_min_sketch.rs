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
    freq_table: Vec<Vec<usize>>,
    time_table: Vec<Vec<usize>>,
    num_hash_funcs: usize,
    width: usize,
    elapsed: usize,
    half_life: f32,
}

impl Default for DecayingSketch {
    fn default() -> Self {
        let num_hash_funcs = 1 << 3;
        let width = 1 << 10;
        let elapsed = 0;
        let half_life = 1024_f32; // FIXME tweak
        Self {
            num_hash_funcs,
            width,
            elapsed,
            half_life,
            freq_table: vec![vec![0_usize; width]; num_hash_funcs],
            time_table: vec![vec![0_usize; width]; num_hash_funcs],
        }
    }
}

impl DecayingSketch {
    fn decay_factor(&self, timestamp: usize, current_time: usize) -> Result<f32, Error> {
        if current_time < timestamp {
            return Err(Error::InvalidTimestamp {
                timestamp,
                elapsed: current_time,
            });
        }
        let age = current_time - timestamp;
        let decay = 2_f32.powf(-(age as f32 / self.half_life));
        log::debug!(
            "decay factor for timestamp {}, current_time {} = {}",
            timestamp,
            current_time,
            decay
        );
        Ok(decay)
    }

    pub fn insert<T: Hash>(&mut self, thing: T, timestamp: usize) -> Result<(), Error> {
        if timestamp < self.elapsed {
            return Err(Error::InvalidTimestamp {
                timestamp,
                elapsed: self.elapsed,
            });
        }
        self.elapsed = timestamp;

        for i in 0..self.num_hash_funcs {
            let loc = hash(&thing, i) % self.width;
            self.freq_table[i][loc] += 1;
            self.time_table[i][loc] = timestamp;
        }

        Ok(())
    }

    pub fn query<T: Hash>(&self, thing: T, current_time: usize) -> Result<f32, Error> {
        let (freq, timestamp) = (0..self.num_hash_funcs)
            .map(|i| {
                let loc = hash(&thing, i) % self.width;
                (self.freq_table[i][loc], self.time_table[i][loc])
            })
            .fold((std::usize::MAX, std::usize::MAX), std::cmp::min);
        self.decay_factor(timestamp, current_time)
            .map(|d| d * freq as f32)
    }

    pub fn flush(&mut self) {
        for i in 0..self.num_hash_funcs {
            for j in 0..self.width {
                self.time_table[i][j] = 0;
                self.freq_table[i][j] = 0;
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
