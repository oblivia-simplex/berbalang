use std::hash::Hash;

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use hashbrown::HashMap;
use rand::Rng;
use unicorn::{Arch, Cpu, Mode};

use crate::emulator::register_pattern::Register;
use crate::util::random::hash_seed_rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Big,
    Little,
}

pub fn read_integer(bytes: &[u8], endian: Endian, word_size: usize) -> Option<u64> {
    use Endian::*;
    if bytes.len() < word_size {
        None
    } else {
        Some(match (endian, word_size) {
            (Little, 8) => LittleEndian::read_u64(bytes) as u64,
            (Big, 8) => BigEndian::read_u64(bytes) as u64,
            (Little, 4) => LittleEndian::read_u32(bytes) as u64,
            (Big, 4) => BigEndian::read_u32(bytes) as u64,
            (Little, 2) => LittleEndian::read_u16(bytes) as u64,
            (Big, 2) => LittleEndian::read_u16(bytes) as u64,
            (_, _) => unreachable!("Invalid word size: {}", word_size),
        })
    }
}

pub fn write_integer(endian: Endian, word_size: usize, word: u64, bytes: &mut [u8]) {
    match (endian, word_size) {
        (Endian::Little, 8) => LittleEndian::write_u64(bytes, word),
        (Endian::Big, 8) => BigEndian::write_u64(bytes, word),
        (Endian::Little, 4) => LittleEndian::write_u32(bytes, word as u32),
        (Endian::Big, 4) => BigEndian::write_u32(bytes, word as u32),
        (Endian::Little, 2) => LittleEndian::write_u16(bytes, word as u16),
        (Endian::Big, 2) => BigEndian::write_u16(bytes, word as u16),
        (_, _) => unimplemented!("Invalid word size: {}", word_size),
    }
}

pub fn random_register_state<H: Hash, C: 'static + Cpu<'static>>(
    registers: &[Register<C>],
    seed: H,
) -> HashMap<Register<C>, u64> {
    let mut map = HashMap::new();
    let mut rng = hash_seed_rng(&seed);
    for reg in registers.iter() {
        map.insert(reg.clone(), rng.gen::<u64>());
    }
    map
}

pub fn endian(arch: Arch, mode: Mode) -> Endian {
    use Arch::*;
    use Endian::*;

    match (arch, mode) {
        (ARM, _) => Big, // this can actually go both ways, check unicorn
        (ARM64, _) => Big,
        (MIPS, _) => Big, // check
        (X86, _) => Little,
        (PPC, _) => Big,
        (SPARC, _) => Big, // check
        (M68K, _) => Big,  // check
    }
}

pub fn word_size_in_bytes(arch: Arch, mode: Mode) -> usize {
    use Arch::*;
    use Mode::*;

    match (arch, mode) {
        (ARM, THUMB) => 2,
        (ARM, _) => 4,
        (ARM64, THUMB) => 2,
        (ARM64, _) => 8,
        (MIPS, _) => 4, // check
        (X86, MODE_16) => 2,
        (X86, MODE_32) => 4,
        (X86, MODE_64) => 8,
        (PPC, MODE_64) => 8,
        (PPC, _) => 4,
        (SPARC, _) => 4, // check
        (M68K, _) => 2,  // check
        (_, _) => unimplemented!("invalid arch/mode combination"),
    }
}
