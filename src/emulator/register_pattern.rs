use std::convert::TryFrom;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use byteorder::{ByteOrder, LittleEndian};
use serde::{Deserialize, Serialize};
use unicorn::Cpu;

use hashbrown::HashMap;

use crate::emulator::loader;
use crate::emulator::loader::Seg;
use crate::error::Error;
use crate::hashmap;

pub type Register<C> = <C as Cpu<'static>>::Reg;

/// For example, if EAX <- 0xdeadbeef, then EAX holds
/// `RegisterValue { val: 0xdeadbeef, deref: 0 }`.
/// But if EAX <- 0x12345678 <- 0xdeadbeef, then we have
/// `RegisterValue { val: 0xdeadbeef, deref: 1 }`
/// and so on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RegisterValue {
    pub val: u64,
    deref: usize,
}

impl From<u64> for RegisterValue {
    fn from(val: u64) -> Self {
        Self { val, deref: 0 }
    }
}

/// Grammar:
/// ```
/// RegisterValue -> numeric_literal | & RegisterValue
/// ```
impl FromStr for RegisterValue {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        fn parse_rv<I: Iterator<Item = char>>(
            mut chars: I,
            deref: usize,
        ) -> Result<RegisterValue, Error> {
            let ch = chars.next();
            match ch {
                None => Err(Error::Parsing("Invalid register value".into())),
                Some('&') => parse_rv(chars, deref + 1),
                Some(' ') => parse_rv(chars, deref),
                // You could handle short strings here, too.
                // Match on an opening single quote, and pack `word_size` bytes into an
                // integer. TODO
                Some(x) => {
                    let rest = chars.collect::<String>();
                    let numeral = format!("{}{}", x, rest);
                    let val = if numeral.starts_with("0x") {
                        let numeral = numeral.trim_start_matches("0x");
                        u64::from_str_radix(&numeral, 16)?
                    } else {
                        u64::from_str_radix(&numeral, 10)?
                    };
                    Ok(RegisterValue { val, deref })
                }
            }
        }
        parse_rv(s.chars(), 0)
    }
}

// do a numerically thick match along the trunk, and a sparse search of the branches
// register states are trees when spidered

impl<C: Cpu<'static>> From<UnicornRegisterState<C>> for RegisterPattern {
    fn from(registers: UnicornRegisterState<C>) -> Self {
        let mut map = HashMap::new();
        for (k, v) in registers.0.into_iter() {
            map.insert(format!("{:?}", k), v.into()); // FIXME use stable conversion method
        }
        RegisterPattern(map)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterPattern(pub HashMap<String, RegisterValue>);

impl Hash for RegisterPattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.iter().collect::<Vec<_>>().hash(state)
    }
}

impl PartialEq for RegisterPattern {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for RegisterPattern {}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct RegisterPatternConfig(pub HashMap<String, String>);

impl From<&RegisterPatternConfig> for RegisterPattern {
    fn from(rp: &RegisterPatternConfig) -> Self {
        let mut map = HashMap::new();
        for (k, v) in rp.0.iter() {
            map.insert(
                k.to_string(),
                v.parse::<RegisterValue>()
                    .expect("Failed to parse RegisterValue"),
            );
        }
        Self(map)
    }
}

impl From<RegisterPatternConfig> for RegisterPattern {
    fn from(rp: RegisterPatternConfig) -> Self {
        (&rp).into()
    }
}

#[derive(Debug)]
pub struct UnicornRegisterState<C: 'static + Cpu<'static>>(pub HashMap<Register<C>, u64>);

impl<C: 'static + Cpu<'static>> TryFrom<&RegisterPattern> for UnicornRegisterState<C> {
    type Error = Error;

    fn try_from(rp: &RegisterPattern) -> Result<Self, Self::Error> {
        let mut map = HashMap::new();
        for (k, v) in rp.0.iter() {
            let reg = k
                .parse()
                .map_err(|_| Self::Error::Parsing("Failed to parse register string".to_string()))?;
            map.insert(reg, v.val);
        }
        Ok(UnicornRegisterState(map))
    }
}

// TODO: write down your thoughts on the complications of "similarity" or
// "distance" in this framework, where we're concerned, essentially, with
// unseen mappings between distances in the Phenotype graph (where the edit
// moves, or edges, are machine operations) and distances in the Genotype
// graph (where the edits are mutations and crossovers). This is a computationally
// intractable structure. How do we even begin to "approximate" it?

// TODO: write some integration tests for this. there's a LOT of room for error!
impl RegisterPattern {
    pub fn distance(
        &self,
        from_emu: &Self,
        writeable_memory: Option<&[Seg]>,
    ) -> HashMap<&'static str, f64> {
        // assumes identical order!
        // let grain = 2;
        // let self_bytes: Vec<u8> = self.into();
        // let other_bytes: Vec<u8> = other.into();
        // let j = jaccard(&self_bytes, &other_bytes, grain, 10);
        // j
        // NOTE: not supporting checking writeable memory yet
        let spider_map = from_emu.spider(writeable_memory);

        // find the closest ham
        fn word_distance(w1: u64, w2: u64) -> f64 {
            let ham = (w1 ^ w2).count_ones() as f64; // hamming distance
            let num = (w1 as f64 - w2 as f64).abs(); // log-scale numeric distance
                                                     //let num = if num > 0.0 { num.log2() } else { 0.0 };
            ham.min(num)
        }

        // fn reg_distance(reg: &str, target: &str) -> f64 {
        //     if reg == target {
        //         0.0
        //     } else {
        //         1.0
        //     }
        // }

        fn pos_distance(pos: usize, target: usize) -> f64 {
            (pos as i32 - target as i32).abs() as f64
        }

        // let's just try treating each register independently for now
        let summed_dist_for_reg = |reg, r_val: &RegisterValue| {
            log::debug!("want {:x?}", r_val);
            let vals = spider_map.get(reg).expect("missing reg");
            let target_val = r_val.val;
            let target_pos = r_val.deref;
            let least_distance = vals
                .iter()
                .enumerate()
                .map(|(pos, val)| {
                    let pos_dist = pos_distance(pos, target_pos);
                    let word_dist = word_distance(*val, target_val);
                    log::debug!(
                        "{} val 0x{:x}: word_dist {} + pos_dist {} = {}",
                        reg,
                        val,
                        word_dist,
                        pos_dist,
                        word_dist + pos_dist
                    );
                    word_dist + pos_dist
                })
                .fold(std::f64::MAX, |a, b| a.min(b));
            log::debug!("{} least_distance = {}", reg, least_distance);
            least_distance
        };
        const WRONG_REG_PENALTY: f64 = 1.0;
        let summed_dist = self
            .0
            .iter()
            .map(|(reg, r_val): (&String, &RegisterValue)| {
                self.0
                    .keys()
                    .map(|r| {
                        let mut d = summed_dist_for_reg(r, r_val);
                        if r != reg {
                            d += WRONG_REG_PENALTY
                        };
                        d
                    })
                    .fold(std::f64::MAX, |a, b| a.min(b))
            })
            .sum();
        log::debug!("summed_dist = {}", summed_dist);
        //
        // let find_least_ham = |val: &RegisterValue| -> (&str, usize, f64) {
        //     let word = val.val;
        //     spider_map
        //         .iter()
        //         // map (register, path) to (reg name, index of min_ham, min_ham)
        //         .map(|(register, path)| -> (&str, usize, f64) {
        //             let (index, ham): (usize, f64) = path
        //                 .iter()
        //                 .enumerate()
        //                 .map(|(i, w): (usize, &u64)| -> (usize, f64) {
        //                     (i, word_distance(word, *w))
        //                 })
        //                 .fold(
        //                     (0, std::f64::MAX),
        //                     |p, q| {
        //                         if (p.1) <= (q.1) {
        //                             p
        //                         } else {
        //                             q
        //                         }
        //                     },
        //                 );
        //             (register, index, ham)
        //         }) // now iterating over (index, minimal ham for path)
        //         .fold(("DUMMY_REG", 0, std::f64::MAX), |p, q| {
        //             if (p.2) <= (q.2) {
        //                 p
        //             } else {
        //                 q
        //             }
        //         })
        // };
        //
        // let (reg_score, deref_score, ham_score) = self
        //     .0
        //     .iter()
        //     .map(|(reg, val)| {
        //         let (r, deref_steps, least_ham) = find_least_ham(val);
        //         let correct_reg = if r == reg { 0.0 } else { 1.0 };
        //         let deref_error = (val.deref as i32 - deref_steps as i32).abs() as f64;
        //         let hamming_error = least_ham as f64;
        //         (correct_reg, deref_error, hamming_error)
        //     })
        //     .fold((0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

        hashmap! {
            "register_error" => summed_dist,
           // "value_error" => ham_score,
           // "place_error" => reg_score + deref_score,
        }
    }

    pub fn spider(&self, extra_segs: Option<&[Seg]>) -> HashMap<String, Vec<u64>> {
        let mut map = HashMap::new();
        let memory = loader::get_static_memory_image();
        for (k, v) in self.0.iter() {
            let path = memory.deref_chain(v.val, 10, extra_segs);
            map.insert(k.to_string(), path);
        }
        map
    }
}

impl From<&RegisterPattern> for Vec<u8> {
    fn from(rp: &RegisterPattern) -> Vec<u8> {
        const WORD_SIZE: usize = 8; // FIXME
        let len = rp.0.keys().len();
        let mut buf = vec![0_u8; len * WORD_SIZE];
        let mut offset = 0;
        for word in rp.0.values() {
            LittleEndian::write_u64(&mut buf[offset..], word.val);
            offset += WORD_SIZE;
        }
        buf
    }
}

#[cfg(test)]
mod test {
    use serde_derive::Deserialize;

    use crate::hashmap;

    use crate::emulator::loader;
    use crate::emulator::register_pattern::{RegisterPattern, RegisterValue};

    use super::*;

    #[derive(Debug, Clone, Deserialize)]
    struct Conf {
        pub pattern: RegisterPatternConfig,
    }

    #[test]
    fn test_register_value_parser() {
        let rvs = vec![
            (
                "0xdeadbeef",
                RegisterValue {
                    val: 0xdead_beef,
                    deref: 0,
                },
            ),
            (
                "&0xbeef",
                RegisterValue {
                    val: 0xbeef,
                    deref: 1,
                },
            ),
            ("&&0", RegisterValue { val: 0, deref: 2 }),
            (
                "& & & & 1234",
                RegisterValue {
                    val: 1234,
                    deref: 4,
                },
            ),
        ];

        for (s, reg_val) in rvs.into_iter() {
            let rv: RegisterValue = s.parse().expect("Failed to parse");
            println!("{} --> {:?}", s, rv);
            assert_eq!(rv, reg_val);
        }
    }

    #[test]
    fn test_register_pattern_distance() {
        // load a binary
        let _segs = loader::load_from_path(
            "/bin/sh",
            0x1000,
            unicorn::Arch::X86,
            unicorn::Mode::MODE_64,
        )
        .expect("failed to load");
        // set a pattern

        let pat_str = r#"
            [pattern]
            RAX = "0xdeadbeef"
            RCX = "&&0xbeef"
            RDX = "& & & 1234"
        "#;
        println!("pat_str = {}", pat_str);
        let pattern_conf: Conf = toml::from_str(&pat_str).expect("failed to parse");
        let pattern = RegisterPattern::from(&pattern_conf.pattern);
        println!("pattern: {:#x?}", pattern);

        let res_pat = RegisterPattern(hashmap! {
            "RAX".into() => RegisterValue { val: 0xdead_beef, deref: 0 },
            "RCX".into() => RegisterValue { val: 1234, deref: 0 },
            "RDX".into() => RegisterValue { val: 0x40_0000, deref: 0 },
        });

        println!("suppose the resulting state is {:#x?}", res_pat);
        let distance = pattern.distance(&res_pat, None);
        println!("distance = {:?}", distance);
    }
}
