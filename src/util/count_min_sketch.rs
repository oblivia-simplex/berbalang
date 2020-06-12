use crate::get_epoch_counter;
//use std::collections::hash_map::DefaultHasher;
use crate::configure::Config;
use std::fmt;
use std::hash::{Hash, Hasher};

pub trait Sketch: Send + Sync + Clone {
    fn insert<T: Hash>(&mut self, thing: T);
    fn query<T: Hash>(&self, thing: T) -> f64;
    fn new(config: &Config) -> Self;
}

impl Sketch for DecayingSketch {
    fn insert<T: Hash>(&mut self, thing: T) {
        DecayingSketch::insert(self, thing)
    }

    fn query<T: Hash>(&self, thing: T) -> f64 {
        DecayingSketch::query(self, thing)
    }

    fn new(config: &Config) -> Self {
        DecayingSketch::new(config)
    }
}

impl Sketch for CountMinSketch {
    fn insert<T: Hash>(&mut self, thing: T) {
        CountMinSketch::insert(self, thing)
    }

    fn query<T: Hash>(&self, thing: T) -> f64 {
        CountMinSketch::query(self, thing)
    }

    fn new(config: &Config) -> Self {
        CountMinSketch::new(config)
    }
}

impl Sketch for SeasonalSketch {
    fn insert<T: Hash>(&mut self, thing: T) {
        SeasonalSketch::insert(self, thing)
    }

    fn query<T: Hash>(&self, thing: T) -> f64 {
        SeasonalSketch::query(self, thing)
    }

    fn new(config: &Config) -> Self {
        SeasonalSketch::new(config)
    }
}

#[derive(Debug, Clone)]
pub struct SeasonalSketch {
    period: usize,
    hot: CountMinSketch,
    cold: CountMinSketch,
}

impl SeasonalSketch {
    pub fn new(config: &Config) -> Self {
        let period = config.season_length;
        Self {
            hot: CountMinSketch::new(config),
            cold: CountMinSketch::new(config),
            period,
        }
    }

    fn turn_turn_turn(&mut self) {
        std::mem::swap(&mut self.hot, &mut self.cold);
        self.hot.flush()
    }

    pub fn insert<T: Hash>(&mut self, thing: T) {
        if get_epoch_counter() % self.period == 0 {
            self.turn_turn_turn()
        }
        self.hot.insert(thing)
    }

    pub fn query<T: Hash>(&self, thing: T) -> f64 {
        self.cold.query(thing)
    }
}

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
    //let mut hasher = DefaultHasher::new();
    let mut hasher = fnv::FnvHasher::default();
    (thing, index).hash(&mut hasher);
    hasher.finish() as usize
}

#[derive(Debug, Clone)]
pub struct DecayingSketch {
    freq_table: Vec<Vec<f64>>,
    time_table: Vec<Vec<usize>>,
    depth: usize,
    width: usize,
    half_life: f64,
    counter: usize,
    decay: bool,
} // TODO decay the counter?

impl DecayingSketch {
    /// Decay can be disabled by setting half_life to 0.0.
    pub fn new(config: &Config) -> Self {
        let depth = suggest_depth(config.pop_size);
        let width = suggest_width(config.pop_size);
        let time_table = vec![vec![0_usize; width]; depth];
        let freq_table = vec![vec![0_f64; width]; depth];
        let half_life = config.pop_size as f64; // TODO add config field for this
        Self {
            depth,
            width,
            half_life,
            freq_table,
            time_table,
            counter: 0,
            decay: true,
        }
    }

    fn decay_factor(&self, prior_timestamp: usize, current_timestamp: usize) -> f64 {
        if !self.decay {
            return 1.0;
        }
        assert!(current_timestamp >= prior_timestamp);

        let age = current_timestamp - prior_timestamp;
        2_f64.powf(-(age as f64 / self.half_life))
    }

    pub fn insert<T: Hash>(&mut self, thing: T) {
        self.counter += 1;
        let current_timestamp = get_epoch_counter();

        for i in 0..self.depth {
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
        let current_time = get_epoch_counter();
        let (freq, timestamp) = (0..self.depth)
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
        for i in 0..self.depth {
            for j in 0..self.width {
                self.time_table[i][j] = 0;
                self.freq_table[i][j] = 0.0;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CountMinSketch {
    table: Vec<Vec<usize>>,
    depth: usize,
    width: usize,
    counter: usize,
}

impl Default for CountMinSketch {
    fn default() -> Self {
        let depth = 1 << 3;
        let width = 1 << 10;
        Self {
            table: vec![vec![0; width]; depth],
            depth,
            width,
            counter: 0,
        }
    }
}

impl CountMinSketch {
    pub fn new(config: &Config) -> Self {
        let depth = suggest_depth(config.pop_size);
        let width = suggest_width(config.pop_size);
        Self {
            table: vec![vec![0; width]; depth],
            depth,
            width,
            counter: 0,
        }
    }

    pub fn flush(&mut self) {
        for i in 0..self.depth {
            for j in 0..self.width {
                self.table[i][j] = 0
            }
        }
    }

    pub fn insert<T: Hash>(&mut self, thing: T) {
        self.counter += 1;
        for i in 0..self.depth {
            let loc = hash(&thing, i) % self.width;
            self.table[i][loc] += 1;
        }
    }

    pub fn query<T: Hash>(&self, thing: T) -> f64 {
        (0..self.depth)
            .map(|i| {
                let loc = hash(&thing, i) % self.width;
                self.table[i][loc]
            })
            .fold(std::usize::MAX, std::cmp::min) as f64
            / self.counter as f64
    }
}

pub fn suggest_width(expected_count: usize) -> usize {
    let error = 1.0 / expected_count as f64;
    (std::f64::consts::E / error).ceil() as usize
}

pub fn suggest_depth(expected_count: usize) -> usize {
    (expected_count as f64).ln().ceil() as usize
}

#[cfg(test)]
mod test {
    use std::iter;

    use super::*;
    use crate::increment_epoch_counter;

    #[test]
    fn test_decaying_count_min_sketch() {
        let count = 100;
        let items = iter::repeat(())
            .map(|()| rand::random::<u128>())
            .take(count)
            .collect::<Vec<u128>>();

        let depth = suggest_depth(count) / 2;
        let width = suggest_width(count) / 2;
        let mut d_sketch = DecayingSketch::new(depth, width, 2.0);
        let mut c_sketch = CountMinSketch::new(depth, width);

        increment_epoch_counter();
        increment_epoch_counter();
        // println!("inserting {} random u128 elements", count);
        // for item in &items {
        //     d_sketch.insert(item);
        //     c_sketch.insert(item);
        // }

        println!("querying...");
        let mut show = true;
        let mut d_sum = 0.0;
        let mut c_sum = 0.0;
        for item in &items {
            d_sketch.insert(item);
            c_sketch.insert(item);
            let d_res = d_sketch.query(item);
            let c_res = c_sketch.query(item);
            if show {
                println!("0x{:032x} -> D: {}, C: {}", item, d_res, c_res);
                //show = false;
            }
            d_sum += d_res;
            c_sum += c_res;
        }
        println!("width = {}, depth = {}", width, depth);
        println!("First run:");
        println!(
            "expected sum: 1.0, actual d_sum: {}, actual c_sum: {}",
            d_sum, c_sum
        );

        println!("Second run:");
        show = true;
        println!("inserting {} 1s", count);
        iter::repeat(1).take(count).for_each(|i| {
            d_sketch.insert(i);
            c_sketch.insert(i);
        });
        println!("querying...");
        let mut d_sum = 0.0;
        let mut c_sum = 0.0;
        for item in &items {
            let d_res = d_sketch.query(item);
            let c_res = c_sketch.query(item);
            if show {
                println!("0x{:032x} -> D: {}, C: {}", item, d_res, c_res);
                show = false;
            }
            d_sum += d_res;
            c_sum += c_res;
        }
        let d_res = d_sketch.query(1);
        let c_res = c_sketch.query(1);
        println!("1 -> D: {}, C: {}", d_res, c_res);
        d_sum += d_res;
        c_sum += c_res;
        println!(
            "expected sum: 1.0, actual d_sum: {}, actual c_sum: {}",
            d_sum, c_sum
        );
    }
}
