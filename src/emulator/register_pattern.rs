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
use itertools::Itertools;
use std::fmt;

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

fn word_distance(w1: u64, w2: u64) -> f64 {
    let ham = (w1 ^ w2).count_ones() as f64; // hamming distance
    let num = (w1 as f64 - w2 as f64).abs(); // log-scale numeric distance
                                             //let num = if num > 0.0 { num.log2() } else { 0.0 };
    ham.min(num)
}

// TODO: write some integration tests for this. there's a LOT of room for error!
impl RegisterPattern {
    pub fn distance_from_register_state(&self, register_state: &RegisterState) -> f64 {
        const WRONG_REG_PENALTY: f64 = 1.0;

        let summed_dist = self
            .0
            .iter()
            .map(|(reg, r_val): (&String, &RegisterValue)| {
                self.0
                    .keys()
                    .map(|r| {
                        let mut d = register_state
                            .distance_from_register_val(r, r_val)
                            .expect("Failed to get distance from register val");
                        println!("[{}] summed_dist_for_reg({}, {:x?}) = {}", reg, r, r_val, d);
                        if r != reg {
                            d += WRONG_REG_PENALTY
                        };
                        d
                    })
                    .fold(std::f64::MAX, |a, b| a.min(b))
            })
            .sum();
        log::debug!("summed_dist = {}", summed_dist);

        summed_dist
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

/// The `RegisterState` represents the state of the emulator's CPU
/// at the end of execution, seen from the perspective of a subset
/// of registers and their referential chains in memory.
///
/// It can be compared against a `RegisterPattern` specification
/// to produce a distance measure, which can then be used in fitness
/// functions.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegisterState(pub HashMap<String, Vec<u64>>);

impl Hash for RegisterState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (_k, vals) in self.0.iter().sorted_by_key(|p| p.0) {
            for v in vals {
                state.write_u64(*v)
            }
            // And then an unusual sequence to mark the end of a vector
            state.write_u64(0xdeadbeef_01010101);
            state.write_u64(0x01010101_baadf00d);
            state.write_u64(0xc001face_01010101);
        }
    }
}

impl RegisterState {
    pub fn new<C: 'static + Cpu<'static>>(
        registers: &HashMap<Register<C>, u64>,
        extra_segs: Option<&[Seg]>,
    ) -> Self {
        Self(Self::spider::<C>(registers, extra_segs))
    }

    fn spider<C: 'static + Cpu<'static>>(
        registers: &HashMap<Register<C>, u64>,
        extra_segs: Option<&[Seg]>,
    ) -> HashMap<String, Vec<u64>> {
        const MAX_SPIDER_STEPS: usize = 10;
        let mut map = HashMap::new();
        let memory = loader::get_static_memory_image();
        for (k, v) in registers.iter() {
            let path = memory.deref_chain(*v, MAX_SPIDER_STEPS, extra_segs);
            let reg_name = format!("{:?}", k);
            map.insert(reg_name, path);
        }
        map
    }

    fn distance_from_register_val(&self, reg: &str, r_val: &RegisterValue) -> Result<f64, Error> {
        const POS_DIST_SCALE: f64 = 1.0;
        fn pos_distance(pos: usize, target: usize) -> f64 {
            let res = (pos as i32 - target as i32).abs() as f64;
            res
        }
        log::debug!("want {:x?}", r_val);
        if let Some(vals) = self.0.get(reg) {
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
                    word_dist + pos_dist * POS_DIST_SCALE
                })
                .fold(std::f64::MAX, |a, b| a.min(b));
            log::debug!("{} least_distance = {}", reg, least_distance);
            Ok(least_distance)
        } else {
            Err(Error::MissingKey(reg.to_string()))
        }
    }
}

impl fmt::Debug for RegisterState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
            .iter()
            .sorted_by_key(|p| p.0)
            .map(|(r, m)| {
                writeln!(
                    f,
                    "{}: {}",
                    r,
                    m.iter()
                        .map(|v| format!("0x{:x}", v))
                        .collect::<Vec<_>>()
                        .join(" -> ")
                )
            })
            .collect::<Result<Vec<()>, _>>()
            .map(|_| ())
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
        let spider_map: HashMap<String, Vec<u64>> = hashmap! {
            "RAX".to_string() => vec![0xdead, 0xbeef, 0],
            "RBX".to_string() => vec![1, 2, 3],
        };
        let register_state = RegisterState(spider_map);

        let register_pattern = RegisterPattern(hashmap! {
            "RAX".to_string() => RegisterValue {
                val: 0xbeef,
                deref: 1,
            },

            "RBX".to_string() => RegisterValue {
                val: 3,
                deref: 2,
            },
        });

        let res = register_pattern.distance_from_register_state(&register_state);

        assert_eq!(res, 0.0, "nonzero score on match");
    }

    #[test]
    fn test_summed_dist() {
        let spider_map: HashMap<String, Vec<u64>> = hashmap! {
            "RAX".to_string() => vec![0xdead, 0xbeef, 0],
            "RBX".to_string() => vec![1, 2, 3],
        };
        let register_state = RegisterState(spider_map);

        let res = register_state
            .distance_from_register_val(
                "RAX",
                &RegisterValue {
                    val: 0xbeef,
                    deref: 1,
                },
            )
            .unwrap();
        assert_eq!(res, 0.0, "Match failed");

        let res = register_state
            .distance_from_register_val("RBX", &RegisterValue { val: 3, deref: 2 })
            .unwrap();
        assert_eq!(res, 0.0, "Match failed");

        let res = register_state
            .distance_from_register_val(
                "RAX",
                &RegisterValue {
                    val: 0x1000_beef,
                    deref: 1,
                },
            )
            .unwrap();
        assert_eq!(res, 1.0, "Match failed");

        let res = register_state
            .distance_from_register_val(
                "RAX",
                &RegisterValue {
                    val: 0x1000_beef,
                    deref: 0,
                },
            )
            .unwrap();
        assert_eq!(res, 2.0, "Match failed");

        let register_state = RegisterState(hashmap! {
            "RAX".to_string() => vec![1, 7],
        });
        let res = register_state
            .distance_from_register_val("RAX", &RegisterValue { val: 9, deref: 3 })
            .unwrap();
        assert_eq!(res, 1.0 + 3.0);
    }
}
