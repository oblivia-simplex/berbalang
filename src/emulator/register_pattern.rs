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
                Some(x) => {
                    let rest = chars.collect::<String>();
                    let numeral = format!("{}{}", x, rest);
                    let val = numeral.parse::<u64>()?;
                    Ok(RegisterValue { val, deref })
                }
            }
        }
        parse_rv(s.chars(), 0)
    }
}

#[cfg(test)]
mod test {
    use crate::emulator::register_pattern::RegisterValue;

    #[test]
    fn test_register_value_parser() {
        let rvs = vec!["0xdeadbeef", "&0xbeef", "&&0", "& & & & 1234"];

        for s in &rvs {
            let rv: RegisterValue = s.parse().expect("Failed to parse");
            println!("{} --> {:?}", s, rv);
        }
    }
}

// TODO: replace this with a From
pub fn convert_register_map<C: Cpu<'static>>(
    registers: UnicornRegisterState<C>,
) -> RegisterPattern {
    let mut map = IndexMap::new();
    for (k, v) in registers.0.into_iter() {
        map.insert(format!("{:?}", k), v.into()); // FIXME use stable conversion method
    }
    RegisterPattern(map)
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
pub struct RegisterPattern(pub IndexMap<String, RegisterValue>);

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

impl RegisterPattern {
    // FIXME
    fn distance(&self, other: &Self) -> f64 {
        // assumes identical order!
        let grain = 2;
        let self_bytes: Vec<u8> = self.into();
        let other_bytes: Vec<u8> = other.into();
        let j = jaccard(&self_bytes, &other_bytes, grain, 10);
        j
    }

    // TODO: why not set arch and mode as fields of RegisterPatternConfig?
    fn spider(&self) -> IndexMap<String, Vec<u64>> {
        let mut map = IndexMap::new();
        let memory = loader::get_static_memory_image();
        for (k, v) in self.0.iter() {
            let path = memory.deref_chain(v.val, 10);
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
