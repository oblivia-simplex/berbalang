use unicorn::{Arch, Mode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Big,
    Little,
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

pub fn word_size(arch: Arch, mode: Mode) -> usize {
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
