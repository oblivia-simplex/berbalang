use crate::emulator::loader;
use capstone::{Capstone, NO_EXTRA_MODE};
use std::fmt;

pub struct Disassembler(pub Capstone);

impl fmt::Debug for Disassembler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<Capstone disassembler at {:p}>", self as *const Self,)
    }
}
// TODO: go and fix the library code for this.
unsafe impl Send for Disassembler {}
unsafe impl Sync for Disassembler {}

#[derive(Debug, Clone)]
pub enum Error {
    Capstone(capstone::Error),
    BadAddress(u64),
}

impl From<capstone::Error> for Error {
    fn from(e: capstone::Error) -> Self {
        Self::Capstone(e)
    }
}

impl Disassembler {
    pub fn new(arch: unicorn::Arch, mode: unicorn::Mode) -> Result<Self, Error> {
        let arch = convert_arch(arch);
        let mode = convert_mode(mode);
        Capstone::new_raw(arch, mode, NO_EXTRA_MODE, None)
            .map(Self)
            .map_err(Error::from)
    }

    pub fn disas(&self, code: &[u8], address: u64, count: Option<usize>) -> Result<String, Error> {
        let res = match count {
            Some(count) => self.0.disasm_count(code, address, count),
            None => self.0.disasm_all(code, address),
        };
        res.map(|res| format!("{}", res)).map_err(Error::from)
    }

    pub fn disas_from_mem_image(&self, start: u64, num_bytes: usize) -> Result<String, Error> {
        let memory = loader::get_static_memory_image();
        let seg = match memory.containing_seg(start) {
            Some(s) => s,
            None => return Err(Error::BadAddress(start)),
        };
        let offset = seg.offset_of_addr(start);
        let code = &seg.data[(offset as usize)..(offset as usize + num_bytes)];
        self.disas(code, start, None)
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
