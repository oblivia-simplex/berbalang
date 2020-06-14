use crate::error::Error;
use rand::prelude::SliceRandom;
use rand::Rng;
use rayon::prelude::{FromParallelIterator, IntoParallelIterator, ParallelIterator};
use std::iter::FromIterator;
use std::ops::Range;

/// For a description and justification of the "trivial geography" algorithm,
/// see Lee Spector & Jon Klein, "Trivial Geography in Genetic Programming"
/// in _Genetic Programming Theory and Practice III_ (ed. Tina Yu, Rick Riolo,
/// Bill Worzel), Springer: 2006.
pub struct TrivialGeography<P> {
    radius: usize,
    deme: Vec<Option<P>>,
    vacancies: Vec<usize>,
}

// impl<P> Default for TrivialGeography<P> {
//     fn default() -> Self {
//         Self {
//             radius: 1_000_000, // effectively no radius at all
//             deme: Vec::new(),
//             vacancies: Vec::new(),
//         }
//     }
// }

impl<P> TrivialGeography<P> {
    pub fn set_radius(&mut self, radius: usize) {
        if self.len() == 0 {
            panic!("Generate the population before setting the radius");
        }
        self.radius = radius.min(self.len())
    }

    pub fn len(&self) -> usize {
        self.deme.len()
    }

    pub fn extract(&mut self, index: usize) -> Option<P> {
        let len = self.deme.len();
        let i = index % len;
        self.vacancies.push(i);
        std::mem::take(&mut self.deme[i])
    }

    pub fn insert(&mut self, creature: P) -> Result<(), Error> {
        if let Some(i) = self.vacancies.pop() {
            debug_assert!(self.deme[i].is_none());
            self.deme[i] = Some(creature);
            Ok(())
        } else {
            Err(Error::NoVacancy)
        }
    }

    pub fn get_range<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
        let len = self.len();
        let base = rng.gen_range(0, len);
        let edge = base + self.radius;

        let mut range: Vec<usize> = if edge < len {
            (base..edge).chain(0..0)
        } else {
            (base..len).chain(0..(edge % len))
        }
        .collect();
        range.dedup();
        range
    }

    fn choose_with_range<R: Rng>(&mut self, range: &Vec<usize>, n: usize, rng: &mut R) -> Vec<P> {
        range
            .choose_multiple(rng, n)
            .filter_map(|i| {
                log::debug!("choosing combatant from index {}", i);
                let c = self.extract(*i);
                debug_assert!(c.is_some());
                c
            })
            .collect()
    }

    pub fn choose_combatants<R: Rng>(&mut self, n: usize, rng: &mut R) -> Vec<P> {
        debug_assert!(
            n < self.radius,
            "don't try to take more creatures than the radius allows"
        );

        let range = self.get_range(rng);
        self.choose_with_range(&range, n, rng)
    }

    pub fn choose_combatants_and_spectators<R: Rng>(
        &mut self,
        n_com: usize,
        n_spec: usize,
        rng: &mut R,
    ) -> (Vec<P>, Vec<P>) {
        debug_assert!(
            n_com < self.radius && n_spec < self.radius,
            "don't try to take more creatures than the radius allows"
        );

        let range = self.get_range(rng);
        let len = self.len();
        let mirror = range
            .iter()
            .map(|n| (*n + len / 2) % len)
            .collect::<Vec<usize>>();

        let combatants = self.choose_with_range(&range, n_com, rng);
        let spectators = self.choose_with_range(&mirror, n_spec, rng);
        (combatants, spectators)
    }
}

impl<P> FromIterator<P> for TrivialGeography<P> {
    fn from_iter<I: IntoIterator<Item = P>>(iter: I) -> Self {
        let deme = iter
            .into_iter()
            .map(Option::Some)
            .collect::<Vec<Option<P>>>();
        Self {
            radius: deme.len(),
            deme: deme,
            vacancies: vec![],
        }
    }
}

impl<P: Send> FromParallelIterator<P> for TrivialGeography<P> {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = P>,
    {
        let deme = par_iter
            .into_par_iter()
            .map(Option::Some)
            .collect::<Vec<Option<P>>>();

        Self {
            radius: deme.len(),
            deme: deme,
            vacancies: vec![],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;
    use rand::prelude::SliceRandom;

    #[test]
    fn test_choose_multiple() {
        for i in 0..100_000 {
            let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            let mut rng = thread_rng();
            let mut choices: Vec<usize> = numbers.choose_multiple(&mut rng, 8).copied().collect();
            let count = choices.len();
            assert_eq!(count, 8);
            choices.dedup();
            let uniq = choices.len();
            assert_eq!(count, uniq);
        }
    }

    #[test]
    fn test_distribution() {
        let size = 256;
        for radius in vec![size, size / 2, size / 4, size / 8].into_iter() {
            let geo = TrivialGeography {
                radius: radius,
                deme: (0..size).map(Option::Some).collect::<Vec<Option<usize>>>(),
                vacancies: vec![],
            };
            let mut rng = thread_rng();

            let mut counts = vec![0; size];

            for i in 0..100_000 {
                let mut range = geo.get_range(&mut rng);
                range.sort();
                //println!("range = {:?}", range);
                for j in range.into_iter() {
                    counts[j] += 1;
                }
            }
            println!("{:?}", counts);
            let std_dev = stats::stddev(counts.into_iter());
            println!("With radius = {}, Standard deviation: {}", radius, std_dev);
            if radius == size {
                assert_eq!(std_dev, 0.0);
            }
            //assert_eq!(std_dev, 0.0);
        }
    }

    #[test]
    fn test_get_range() {
        let geo = TrivialGeography {
            radius: 2048,
            deme: (0..2048).map(Option::Some).collect::<Vec<Option<usize>>>(),
            vacancies: vec![],
        };
        let mut rng = thread_rng();
        for i in 0..100_000 {
            let mut range: Vec<usize> = geo.get_range(&mut rng);

            let n = range.len();
            range.dedup();
            let m = range.len();
            assert_eq!(n, m, "range contained duplicates");

            if range.len() != geo.radius.min(geo.len()) {
                println!(
                    "Range of unexpected size: should be {}, but is {}",
                    geo.radius.min(geo.len()),
                    range.len()
                );
                println!("largest member of range: {}", range.iter().max().unwrap());
                println!("smallest member of range: {}", range.iter().min().unwrap());
                panic!("test failed");
            }

            let mut choices: Vec<usize> = range
                .choose_multiple(&mut rng, 8)
                .copied()
                .collect::<Vec<usize>>();
            let n = choices.len();
            choices.dedup();
            let m = choices.len();
            assert_eq!((i, n), (i, m), "Duplicates returned by choose_multiple!");
        }
    }
}
