use capstone::{Capstone, Error, NO_EXTRA_MODE};
use unicorn;

pub struct Disassembler(pub Capstone);

impl Disassembler {
    pub fn new(arch: unicorn::Arch, mode: unicorn::Mode) -> Result<Self, Error> {
        let arch = convert_arch(arch);
        let mode = convert_mode(mode);
        Capstone::new_raw(arch, mode, NO_EXTRA_MODE, None).map(Self)
    }

    pub fn disas(&self, code: &[u8], address: u64, count: Option<usize>) -> Result<String, Error> {
        let res = match count {
            Some(count) => self.0.disasm_count(code, address, count),
            None => self.0.disasm_all(code, address),
        };
        res.map(|res| format!("{}", res))
    }
}

fn convert_arch(arch: unicorn::Arch) -> capstone::Arch {
    use capstone::Arch as C;
    use unicorn::Arch as U;

    match arch {
        U::ARM => C::ARM,
        U::ARM64 => C::ARM64,
        U::MIPS => C::MIPS,
        U::X86 => C::X86,
        U::PPC => C::PPC,
        U::SPARC => C::SPARC,
        U::M68K => C::M68K,
    }
}

fn convert_mode(mode: unicorn::Mode) -> capstone::Mode {
    use capstone::Mode as C;
    use unicorn::Mode as U;

    match mode {
        U::MODE_16 => C::Mode16,
        U::MODE_32 => C::Mode32,
        U::MODE_64 => C::Mode64,
        U::LITTLE_ENDIAN => C::Arm, // TODO: fix the unicorn mode system some day
        U::THUMB => C::Thumb,
        _ => unimplemented!("i'll do it later"),
    }
}
