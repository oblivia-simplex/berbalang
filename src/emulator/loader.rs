use goblin::{
    elf::{self, Elf},
    Object,
};
use rand::distributions::{Distribution, WeightedIndex};
use rand::{thread_rng, Rng};
use std::fmt;
use std::sync::Once;
use unicorn::Protection;

pub const PAGE_SIZE: u64 = 0x1000;

pub static mut MEM_IMAGE: MemoryImage = MemoryImage(Vec::new());
static INIT_MEM_IMAGE: Once = Once::new();

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

#[derive(Debug, Clone)]
pub struct MemoryImage(pub Vec<Seg>);

impl MemoryImage {
    pub fn first_address(&self) -> u64 {
        self.0[0].aligned_start()
    }

    pub fn containing_seg(&self, addr: u64) -> Option<&Seg> {
        for s in self.0.iter() {
            if s.aligned_start() <= addr && addr < s.aligned_end() {
                return Some(s);
            }
        }
        None
    }

    pub fn perm_of_addr(&self, addr: u64) -> Option<Protection> {
        self.containing_seg(addr).map(|a| a.perm)
    }

    pub fn try_dereference(&self, addr: u64) -> Option<&[u8]> {
        self.containing_seg(addr).map(|s| {
            let bump = (s.addr - s.aligned_start()) as usize;
            let offset = bump + (addr - s.aligned_start()) as usize;
            &s.data[offset..]
        })
    }

    pub fn random_address(&self, permissions: Option<Protection>) -> u64 {
        let mut rng = thread_rng();
        let segments = self
            .segments()
            .iter()
            .filter(|&s| {
                if let Some(perms) = permissions {
                    s.perm.intersects(perms)
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        let weights = segments
            .iter()
            .map(|&s| s.aligned_size())
            .collect::<Vec<_>>();
        let dist =
            WeightedIndex::new(&weights).expect("Failed to generate WeightedIndex over segments");
        let seg = &segments[dist.sample(&mut rng)];
        rng.gen_range(seg.aligned_start(), seg.aligned_end())
    }

    pub fn seek(&self, offset: u64, sequence: &[u8]) -> Option<u64> {
        if let Some(s) = self.containing_seg(offset) {
            let start = (offset - s.aligned_start()) as usize;
            let mut ptr = start;
            for window in s.data[start..].windows(sequence.len()) {
                if window == sequence {
                    return Some(ptr as u64);
                } else {
                    ptr += 1
                }
            }
            None
        } else {
            None
        }
    }

    pub fn seek_from_random_address(&self, sequence: &[u8]) -> Option<u64> {
        self.seek(self.random_address(None), sequence)
    }

    pub fn segments(&self) -> &Vec<Seg> {
        &self.0
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

#[inline]
pub fn align(n: u64) -> u64 {
    (n / PAGE_SIZE) * PAGE_SIZE
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

    #[inline]
    pub fn ensure_data_alignment(&mut self) {
        self.data.resize(self.aligned_size(), 0_u8)
    }

    #[inline]
    pub fn is_executable(&self) -> bool {
        self.perm.intersects(Protection::EXEC)
    }

    #[inline]
    pub fn is_writeable(&self) -> bool {
        self.perm.intersects(Protection::WRITE)
    }

    #[inline]
    pub fn is_readable(&self) -> bool {
        self.perm.intersects(Protection::READ)
    }

    #[inline]
    fn intern_aligned_start(addr: u64) -> u64 {
        align(addr) //addr & 0xFFFF_F000
    }

    #[inline]
    pub fn aligned_start(&self) -> u64 {
        Self::intern_aligned_start(self.addr)
    }

    #[inline]
    fn intern_aligned_end(addr: u64, memsz: usize) -> u64 {
        //(addr + memsz as u64 + 0x1000) & 0xFFFF_F000
        align(addr + memsz as u64)
    }

    #[inline]
    pub fn aligned_end(&self) -> u64 {
        Self::intern_aligned_end(self.addr, self.memsz)
    }

    #[inline]
    fn intern_aligned_size(addr: u64, memsz: usize) -> usize {
        (Self::intern_aligned_end(addr, memsz) - Self::intern_aligned_start(addr)) as usize
    }

    #[inline]
    pub fn aligned_size(&self) -> usize {
        (self.aligned_end() - self.aligned_start()) as usize
    }

    #[inline]
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

    // TODO Reloc tables

    segs.push(Seg {
        addr: bottom,
        perm: Protection::READ | Protection::WRITE,
        segtype: SegType::Load,
        memsz: stack_size,
        data: vec![0; stack_size],
    });
    segs
}

fn initialize_memory_image(segments: &[Seg]) {
    unsafe { MEM_IMAGE = MemoryImage(segments.to_owned()) }
}

pub fn get_static_memory_image() -> &'static MemoryImage {
    if INIT_MEM_IMAGE.is_completed() {
        unsafe { &MEM_IMAGE }
    } else {
        panic!("MEM_IMAGE has not been initialized")
    }
}

pub fn load(code_buffer: &[u8], stack_size: usize) -> Result<Vec<Seg>, Error> {
    if INIT_MEM_IMAGE.is_completed() {
        unsafe { Ok(MEM_IMAGE.segments().clone()) }
    } else {
        let obj = Object::parse(code_buffer)?;
        let mut segs = match obj {
            Object::Elf(elf) => load_elf(elf, code_buffer, stack_size),
            _ => unimplemented!("Only ELF binaries are supported at this time."),
        };
        segs.sort_by_key(|s| s.aligned_start());
        for seg in &segs {
            log::info!("{}, data len: {:x}", seg, seg.data.len());
        }

        // Cache the memory image as a globally accessible static
        INIT_MEM_IMAGE.call_once(|| initialize_memory_image(&segs));

        Ok(segs)
    }
}

pub fn load_from_path(path: &str, stack_size: usize) -> Result<Vec<Seg>, Error> {
    load(&std::fs::read(path)?, stack_size)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_loader() {
        pretty_env_logger::init();
        let res = load_from_path("/bin/sh", 0x1000).expect("Failed to load /bin/sh");
        for s in res {
            log::info!("Segment: {}", s);
        }
    }
}

// TODO:
// - The elf loader is a bit rough, and not entirely exact. I don't handle relocations, e.g.
// - We should factor out the unicorn elf loader into its own method or structure. Right now,
//   it all takes place inside executore::init_emu.
