use std::convert::TryFrom;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use byteorder::{ByteOrder, LittleEndian};
use hashbrown::HashMap;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use unicorn::Cpu;

use crate::emulator::loader;
use crate::emulator::loader::{get_static_memory_image, Seg};
use crate::emulator::profiler::Profile;
use crate::error::Error;
use crate::util;
use crate::util::architecture::Endian;
use crate::util::bitwise;
use crate::util::bitwise::nybble;

pub type Register<C> = <C as Cpu<'static>>::Reg;

// TODO:
// A dereferenced value should only count as "close" to the target if:
// - it contains the head or tail of the target (so that sliding it along may find the target)
// - it points to writeable address. then look at hamming distance.

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
                Some('\'') => {
                    let s = chars.collect::<String>();
                    // FIXME: don't hardcode the endian
                    if let Some(w) = bitwise::try_str_as_word(s, Endian::Little) {
                        Ok(RegisterValue { val: w, deref })
                    } else {
                        Err(Error::Parsing(
                            "can only encode strings of fewer than 8 characters".into(),
                        ))
                    }
                }
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

// This is the algorithm I used in ROPER 1.
fn word_distance2(w1: u64, w2: u64) -> f64 {
    const BOUND: f64 = 2048.0;
    // hamming distance
    let ham = (w1 ^ w2).count_ones() as f64 / 64.0;
    // bounded arithmetic distance
    let diff = (w1 as f64 - w2 as f64).abs().max(BOUND);
    let num = diff / BOUND;
    (ham + num) / 2.0
}

fn word_distance(w1: u64, w2: u64) -> f64 {
    (w1 ^ w2).count_ones() as f64
}

fn max_word_distance() -> f64 {
    get_static_memory_image().word_size as f64 * 8.0
}

// TODO: write some integration tests for this. there's a LOT of room for error!
impl RegisterPattern {
    pub fn distance_from_register_state(&self, register_state: &RegisterState) -> f64 {
        const WRONG_REG_PENALTY: f64 = 5.0;

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
                        log::debug!("[{}] summed_dist_for_reg({}, {:x?}) = {}", reg, r, r_val, d);
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

    /// Returns a vector of register/reference-path pairs *only* for those registers
    /// that do not already perfectly match the target pattern. The intended use of
    /// this method is so that only repeated errors are penalized, and not repeated
    /// successes.
    pub fn incorrect_register_states<'a>(
        &'a self,
        register_state: &'a RegisterState,
    ) -> Vec<(&'a String, &'a Vec<u64>)> {
        self.0
            .iter()
            .filter_map(|(reg, r_val)| {
                let d = register_state
                    .distance_from_register_val(reg, r_val)
                    .expect("Failed to get distance from register val");
                if d > 0.0 {
                    register_state.0.get(reg).map(|v| (reg, v))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn count_writes_of_referenced_values(
        &self,
        profile: &Profile,
        exclude_null: bool,
    ) -> usize {
        self.0
            .values()
            .filter(|v| v.deref > 0 && (!exclude_null || v.val != 0))
            .map(|v| profile.was_this_written(v.val).len())
            .sum()
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

    pub fn features(&self) -> Vec<RegisterFeature> {
        RegisterFeature::decompose_reg_pattern(self)
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

#[derive(Clone, Debug, Hash)]
pub struct RegisterFeature {
    register: String,
    index: usize,
    deref: usize,
    nybble: u8,
}

impl RegisterFeature {
    fn decompose_reg_val(register: &str, reg_val: &RegisterValue, reg_feats: &mut Vec<Self>) {
        let word_size = get_static_memory_image().word_size;

        for i in 0..(word_size * 2) {
            let nybble = nybble(reg_val.val, i);
            let deref = reg_val.deref;
            reg_feats.push(Self {
                register: register.to_string(),
                index: i,
                deref,
                nybble,
            })
        }
    }

    pub fn decompose_reg_pattern(reg_pat: &RegisterPattern) -> Vec<Self> {
        let mut reg_feats = Vec::new();
        for (register, reg_val) in reg_pat.0.iter() {
            Self::decompose_reg_val(register, reg_val, &mut reg_feats);
        }
        reg_feats
    }

    pub fn check_state(&self, state: &RegisterState) -> bool {
        if let Some(deref_chain) = state.0.get(&self.register) {
            if self.deref >= deref_chain.len() {
                false
            } else {
                let val = deref_chain[self.deref];
                let nyb = nybble(val, self.index);
                nyb == self.nybble
            }
        } else {
            false
        }
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

// we can see that identity implies hash identity, so we don't need this warning
#[allow(clippy::derive_hash_xor_eq)]
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
        if let Some(memory) = loader::try_to_get_static_memory_image() {
            for (k, v) in registers.iter() {
                let path = memory.deref_chain(*v, MAX_SPIDER_STEPS, extra_segs);
                let reg_name = format!("{:?}", k);
                map.insert(reg_name, path);
            }
        } else {
            for (k, v) in registers.iter() {
                let reg_name = format!("{:?}", k);
                map.insert(reg_name, vec![*v]);
            }
        }
        map
    }

    fn distance_from_register_val(&self, reg: &str, r_val: &RegisterValue) -> Result<f64, Error> {
        fn pos_distance(pos: usize, target: usize) -> f64 {
            let pos_dist_scale: f64 = 4.0 * get_static_memory_image().word_size as f64;
            let dist = pos as i32 - target as i32;
            dist.abs() as f64 * pos_dist_scale
        }

        fn is_mutable(i: usize, vals: &[u64]) -> bool {
            // a value is mutable iff i == 0 (register) or vals[i-1] points
            // to writeable memory.
            let memory = get_static_memory_image();
            if i == 0 {
                true
            } else {
                memory
                    .perm_of_addr(vals[i - 1])
                    .map(|p| p.intersects(util::architecture::Perms::WRITE))
                    .unwrap_or(false)
            }
        }

        log::debug!("want {:x?}", r_val);
        if let Some(vals) = self.0.get(reg) {
            let distance = if r_val.deref == 0 {
                // Immediate values
                word_distance(vals[0], r_val.val)
            } else {
                // dereferenced values
                vals.iter()
                    .enumerate()
                    .map(|(i, val)| {
                        if is_mutable(i, &vals) {
                            word_distance(*val, r_val.val) + pos_distance(i, r_val.deref)
                        } else if *val == r_val.val {
                            0.0 + pos_distance(i, r_val.deref)
                        } else {
                            2.0 * word_distance(*val, r_val.val) + pos_distance(i, r_val.deref)
                        }
                    })
                    .fold1(|a, b| a.min(b))
                    .unwrap()
            };
            Ok(distance)
        } else {
            Err(Error::MissingKey(reg.to_string()))
        }
    }
}

impl fmt::Debug for RegisterState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let memory = get_static_memory_image();
        let endian = memory.endian;
        let word_size = memory.word_size;
        self.0
            .iter()
            .sorted_by_key(|p| p.0)
            .map(|(r, m)| {
                writeln!(
                    f,
                    "{}: {}",
                    r,
                    m.iter()
                        .map(|v| format!(
                            "0x{:x}{}{}",
                            v,
                            memory
                                .perm_of_addr(*v)
                                .map(|p| format!(" {}", p))
                                .unwrap_or_else(String::new),
                            util::bitwise::try_word_as_string(*v, endian, word_size)
                                .map(|s| format!(" \"{}\"", s))
                                .unwrap_or_else(String::new)
                        ))
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
    use unicorn::{Arch, Mode};

    use crate::configure::RoperConfig;
    use crate::emulator::register_pattern::{RegisterPattern, RegisterValue};
    use crate::hashmap;

    use super::*;

    #[derive(Debug, Clone, Deserialize)]
    struct Conf {
        pub pattern: RegisterPatternConfig,
    }

    fn initialize_mem_image() {
        let config = RoperConfig {
            gadget_file: None,
            output_registers: vec![],
            randomize_registers: false,
            register_pattern: None,
            parsed_register_pattern: None,
            soup: None,
            soup_size: None,
            arch: Arch::X86,
            mode: Mode::MODE_64,
            num_workers: 0,
            num_emulators: 0,
            wait_limit: 0,
            max_emu_steps: None,
            millisecond_timeout: None,
            record_basic_blocks: false,
            record_memory_writes: false,
            emulator_stack_size: 0x1000,
            binary_path: "/bin/sh".to_string(),
            ld_paths: None,
            bad_bytes: None,
            ..Default::default()
        };
        let _ = loader::load_from_path(&config, true);
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
            log::debug!("{} --> {:?}", s, rv);
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

        assert!(res < std::f64::EPSILON, "nonzero score on match");
    }

    #[test]
    fn test_register_features() {
        initialize_mem_image();
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

        let features = register_pattern.features();

        println!(
            "register_pattern: {:#x?}, features: {:#x?}",
            register_pattern, features
        );

        let register_state = RegisterState(hashmap! {
            "RAX".to_string() => vec![0xff, 0xbeef, 0xdead],
            "RBX".to_string() => vec![1, 2, 3],
        });

        let res = features
            .iter()
            .all(|feat| feat.check_state(&register_state));
        assert!(res);

        let register_state = RegisterState(hashmap! {
            "RAX".to_string() => vec![0xff, 0xbee8, 0xdead],
            "RBX".to_string() => vec![1, 2, 4],
        });
        let res = features
            .iter()
            .all(|feat| feat.check_state(&register_state));
        assert!(!res);
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
        assert!(res < std::f64::EPSILON, "Match failed");

        let res = register_state
            .distance_from_register_val("RBX", &RegisterValue { val: 3, deref: 2 })
            .unwrap();
        assert!(res < std::f64::EPSILON, "Match failed");

        let res = register_state
            .distance_from_register_val(
                "RAX",
                &RegisterValue {
                    val: 0x1000_beef,
                    deref: 1,
                },
            )
            .unwrap();
        assert!(res - 1.0 < std::f64::EPSILON, "Match failed");

        let res = register_state
            .distance_from_register_val(
                "RAX",
                &RegisterValue {
                    val: 0x1000_beef,
                    deref: 0,
                },
            )
            .unwrap();
        assert!(res - 2.0 < std::f64::EPSILON, "Match failed");

        // FIXME! There's a problem with the distance measurement being used here.
        // Which might explain the poor performance on this task.
        let register_state = RegisterState(hashmap! {
            "RAX".to_string() => vec![1, 7],
        });
        let res = register_state
            .distance_from_register_val("RAX", &RegisterValue { val: 9, deref: 3 })
            .unwrap();
        println!("res = {}", res);
        assert!(res - (1.0 + 3.0) < std::f64::EPSILON);
    }
}
