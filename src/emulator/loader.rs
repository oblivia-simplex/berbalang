use goblin::{
    elf::{self, Elf},
    Object,
};
use std::fmt;
use unicorn::Protection;

#[derive(Debug)]
pub enum Error {
    ParsingFailure(goblin::error::Error),
    IO(std::io::Error),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::IO(e)
    }
}

impl From<goblin::error::Error> for Error {
    fn from(e: goblin::error::Error) -> Self {
        Error::ParsingFailure(e)
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Seg {
    pub addr: u64,
    pub memsz: usize,
    pub perm: Protection,
    pub segtype: SegType,
    pub data: Vec<u8>,
}
// TODO: document the difference between memsz and data.len()
// I forget what it is, at the moment, but I think there may be one.

impl fmt::Display for Seg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[aligned {:08x} -- {:08x}: {:?}]",
            self.aligned_start(),
            self.aligned_end(),
            self.perm
        )
    }
}

impl Seg {
    pub fn from_phdr(phdr: &elf::ProgramHeader) -> Self {
        let mut uc_perm = unicorn::Protection::NONE;
        if phdr.is_executable() {
            uc_perm |= Protection::EXEC
        };
        if phdr.is_write() {
            uc_perm |= Protection::WRITE
        };
        if phdr.is_read() {
            uc_perm |= Protection::READ
        };
        let addr = phdr.vm_range().start as u64;
        let memsz = (phdr.vm_range().end - phdr.vm_range().start) as usize;
        let size = Self::intern_aligned_size(addr, memsz);
        let data = vec![0x00_u8; size];
        Self {
            addr,
            memsz,
            perm: uc_perm,
            segtype: SegType::new(phdr.p_type),
            data,
        }
    }

    pub fn ensure_data_alignment(&mut self) {
        self.data.resize(self.aligned_size(), 0_u8)
    }

    pub fn is_executable(&self) -> bool {
        self.perm.intersects(Protection::EXEC)
    }

    pub fn is_writeable(&self) -> bool {
        self.perm.intersects(Protection::WRITE)
    }

    pub fn is_readable(&self) -> bool {
        self.perm.intersects(Protection::READ)
    }

    fn intern_aligned_start(addr: u64) -> u64 {
        addr & 0xFFFF_F000
    }

    pub fn aligned_start(&self) -> u64 {
        Self::intern_aligned_start(self.addr)
    }

    fn intern_aligned_end(addr: u64, memsz: usize) -> u64 {
        (addr + memsz as u64 + 0x1000) & 0xFFFF_F000
    }

    pub fn aligned_end(&self) -> u64 {
        Self::intern_aligned_end(self.addr, self.memsz)
    }

    fn intern_aligned_size(addr: u64, memsz: usize) -> usize {
        (Self::intern_aligned_end(addr, memsz) - Self::intern_aligned_start(addr)) as usize
    }

    pub fn aligned_size(&self) -> usize {
        (self.aligned_end() - self.aligned_start()) as usize
    }

    pub fn loadable(&self) -> bool {
        self.segtype.loadable()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SegType {
    Null,
    Load,
    Dynamic,
    Interp,
    Note,
    ShLib,
    PHdr,
    Tls,
    GnuEhFrame,
    GnuStack,
    GnuRelRo,
    Other, /* KLUDGE: a temporary catchall */
}

impl SegType {
    fn new(raw: u32) -> Self {
        match raw {
            0 => SegType::Null,
            1 => SegType::Load,
            2 => SegType::Dynamic,
            3 => SegType::Interp,
            4 => SegType::Note,
            5 => SegType::ShLib,
            6 => SegType::PHdr,
            7 => SegType::Tls,
            0x6474_e550 => SegType::GnuEhFrame,
            0x6474_e551 => SegType::GnuStack,
            0x6474_e552 => SegType::GnuRelRo,
            _ => SegType::Other,
        }
    }
    pub fn loadable(self) -> bool {
        self == SegType::Load
    }
}

fn load_elf(elf: Elf<'_>, code_buffer: &[u8], stack_size: usize) -> Vec<Seg> {
    let mut segs: Vec<Seg> = Vec::new();
    let mut page_one = false;
    let shdrs = &elf.section_headers;
    let phdrs = &elf.program_headers;
    for phdr in phdrs {
        let seg = Seg::from_phdr(&phdr);
        if seg.loadable() {
            let start = seg.aligned_start() as usize;
            if start == 0 {
                page_one = true
            };
            segs.push(seg);
        }
    }
    /* Low memory */
    // NOTE: I can't remember why I put this here.
    // Maybe it's important. Better leave it for now.
    if !page_one {
        segs.push(Seg {
            addr: 0,
            memsz: 0x1000,
            perm: Protection::READ,
            segtype: SegType::Load,
            data: vec![0; 0x1000],
        });
    };

    for shdr in shdrs {
        let (i, j) = (
            shdr.sh_offset as usize,
            (shdr.sh_offset + shdr.sh_size) as usize,
        );
        let aj = usize::min(j, code_buffer.len());
        let sdata = code_buffer[i..aj].to_vec();
        /* find the appropriate segment */

        for seg in segs.iter_mut() {
            if shdr.sh_addr >= seg.aligned_start() && shdr.sh_addr < seg.aligned_end() {
                let mut v_off = (shdr.sh_addr - seg.aligned_start()) as usize;
                for byte in sdata {
                    if v_off >= seg.data.len() {
                        log::warn!(
                            "[x] v_off 0x{:x} > seg.data.len() 0x{:x}",
                            v_off,
                            seg.data.len()
                        );
                        break;
                    };
                    seg.data[v_off] = byte;
                    v_off += 1;
                }
                break;
            }
        }
    }
    /* now allocate the stack */
    let mut bottom = 0;
    for seg in &segs {
        let b = seg.aligned_end();
        if b > bottom {
            bottom = b
        };
    }
    segs.push(Seg {
        addr: bottom,
        perm: Protection::READ | Protection::WRITE,
        segtype: SegType::Load,
        memsz: stack_size,
        data: vec![0; stack_size],
    });
    segs
}

pub fn load(code_buffer: &[u8], stack_size: usize) -> Result<Vec<Seg>, Error> {
    let obj = Object::parse(code_buffer)?;
    let segs = match obj {
        Object::Elf(elf) => load_elf(elf, code_buffer, stack_size),
        _ => unimplemented!("Only ELF binaries are supported at this time."),
    };
    for seg in &segs {
        log::info!("{}, data len: {:x}", seg, seg.data.len());
    }
    Ok(segs)
}

pub fn load_from_path(path: &str, stack_size: usize) -> Result<Vec<Seg>, Error> {
    load(&std::fs::read(path)?, stack_size)
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_loader() {
        let res = load_from_path("/bin/sh", 0x1000)
            .expect("Failed to load /bin/sh");
        for s in res {
            log::info!("Segment: {:?}", s);
        }
    }
}
