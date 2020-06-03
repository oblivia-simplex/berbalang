use crate::emulator::loader;
use crate::error::Error;
use crate::util::distance::jaccard;
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use indexmap::map::IndexMap;
use serde::Deserialize;
use std::convert::TryFrom;
use std::str::FromStr;
use unicorn::Cpu;

pub type Register<C> = <C as Cpu<'static>>::Reg;

/// For example, if EAX <- 0xdeadbeef, then EAX holds
/// `RegisterValue { val: 0xdeadbeef, deref: 0 }`.
/// But if EAX <- 0x12345678 <- 0xdeadbeef, then we have
/// `RegisterValue { val: 0xdeadbeef, deref: 1 }`
/// and so on.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Hash)]
pub struct RegisterValue {
    val: u64,
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
        let mut map = IndexMap::new();
        for (k, v) in registers.0.into_iter() {
            map.insert(format!("{:?}", k), v.into()); // FIXME use stable conversion method
        }
        RegisterPattern(map)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterPattern(pub IndexMap<String, RegisterValue>);

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct RegisterPatternConfig(pub IndexMap<String, String>);

impl From<&RegisterPatternConfig> for RegisterPattern {
    fn from(rp: &RegisterPatternConfig) -> Self {
        let mut map = IndexMap::new();
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
pub struct UnicornRegisterState<C: 'static + Cpu<'static>>(pub IndexMap<Register<C>, u64>);

impl<C: 'static + Cpu<'static>> TryFrom<&RegisterPattern> for UnicornRegisterState<C> {
    type Error = Error;

    fn try_from(rp: &RegisterPattern) -> Result<Self, Self::Error> {
        let mut map = IndexMap::new();
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

impl RegisterPattern {
    pub fn distance(&self, from_emu: &Self) -> Vec<f64> {
        // assumes identical order!
        // let grain = 2;
        // let self_bytes: Vec<u8> = self.into();
        // let other_bytes: Vec<u8> = other.into();
        // let j = jaccard(&self_bytes, &other_bytes, grain, 10);
        // j
        // NOTE: not supporting checking writeable memory yet
        let spider_map = from_emu.spider();

        // find the closest ham
        //log::debug!("Spider map: {:#x?}", spider_map);

        let find_least_ham = |val: &RegisterValue| -> (&str, usize, u32) {
            let word = val.val;
            spider_map
                .iter()
                // map (register, path) to (reg name, index of min_ham, min_ham)
                .map(|(register, path)| -> (&str, usize, u32) {
                    let (index, ham): (usize, u32) = path
                        .iter()
                        .enumerate()
                        .map(|(i, w): (usize, &u64)| -> (usize, u32) {
                            (i, (w ^ word).count_ones())
                        })
                        .fold(
                            (0, std::u32::MAX),
                            |p, q| {
                                if (p.1) <= (q.1) {
                                    p
                                } else {
                                    q
                                }
                            },
                        );
                    (register, index, ham)
                }) // now iterating over (index, minimal ham for path)
                .fold(("DUMMY_REG", 0, std::u32::MAX), |p, q| {
                    if (p.2) <= (q.2) {
                        p
                    } else {
                        q
                    }
                })
        };

        let (reg_score, deref_score, ham_score) = self
            .0
            .iter()
            .map(|(reg, val)| {
                let (r, deref_steps, least_ham) = find_least_ham(val);
                let correct_reg = if r == reg { 0.0 } else { 1.0 };
                let deref_error = (val.deref as i32 - deref_steps as i32).abs() as f64;
                let hamming_error = least_ham as f64 / 64.0;
                (correct_reg, deref_error, hamming_error)
            })
            .fold((0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

        vec![reg_score, deref_score, ham_score]
    }

    // TODO: why not set arch and mode as fields of RegisterPatternConfig?
    fn spider(&self) -> IndexMap<String, Vec<u64>> {
        let mut map = IndexMap::new();
        let memory = loader::get_static_memory_image();
        for (k, v) in self.0.iter() {
            let path = memory.deref_chain(v.val, 10);
            // .iter()
            // .enumerate()
            // .map(|(i, a)| RegisterValue {
            //     val: *a,
            //     deref: i + 1,
            // })
            // .collect::<Vec<_>>();
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
    use super::*;
    use crate::emulator::loader;
    use crate::emulator::register_pattern::{RegisterPattern, RegisterValue};
    use indexmap::indexmap;
    use serde_derive::Deserialize;

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

        let res_pat = RegisterPattern(indexmap! {
            "RAX".into() => RegisterValue { val: 0xdeadbeef, deref: 0 },
            "RCX".into() => RegisterValue { val: 1234, deref: 0 },
            "RDX".into() => RegisterValue { val: 0x40_0000, deref: 0 },
        });

        println!("suppose the resulting state is {:#x?}", res_pat);
        let distance = pattern.distance(&res_pat);
        println!("distance = {:?}", distance);
    }
}
