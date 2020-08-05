use std::fmt;
use std::hash::Hash;
use std::sync::Once;

use capstone::Instructions;
use falcon::il;
use goblin::{
    elf::{self, Elf},
    Object,
};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::configure::RoperConfig;
use crate::disassembler::Disassembler;
use crate::error::Error;
use crate::util::architecture::{endian, read_integer, word_size_in_bytes, Endian, Perms};
use crate::util::random::hash_seed_rng;

pub const PAGE_BITS: u64 = 12;
pub const PAGE_SIZE: u64 = 1 << PAGE_BITS;

// placeholders
pub static mut MEM_IMAGE: MemoryImage = MemoryImage {
    segs: Vec::new(),
    arch: unicorn::Arch::X86,
    mode: unicorn::Mode::MODE_64,
    endian: Endian::Little,
    word_size: 8,
    disasm: None,
    il_program: None,
};
static INIT_MEM_IMAGE: Once = Once::new();

#[derive(Debug)]
pub struct MemoryImage {
    pub segs: Vec<Seg>,
    pub arch: unicorn::Arch,
    pub mode: unicorn::Mode,
    pub endian: Endian,
    pub word_size: usize,
    pub disasm: Option<Disassembler>,
    pub il_program: Option<falcon::il::Program>,
}

impl MemoryImage {
    pub fn disassemble(
        &self,
        addr: u64,
        size: usize,
        count: Option<usize>,
    ) -> Option<Instructions<'_>> {
        self.try_dereference(addr, None)
            .map(|b| &b[..size])
            .and_then(|b| {
                self.disasm
                    .as_ref()
                    .and_then(|dis| dis.disas(b, addr, count).ok())
            })
    }

    pub fn size_of_writeable_memory(&self) -> usize {
        self.size_of_memory_by_perm(Perms::WRITE)
    }

    pub fn size_of_executable_memory(&self) -> usize {
        self.size_of_memory_by_perm(Perms::EXEC)
    }

    pub fn size_of_memory_by_perm(&self, perm: Perms) -> usize {
        self.segs
            .iter()
            .filter(|seg| seg.perm.intersects(perm))
            .map(Seg::aligned_size)
            .sum::<usize>()
    }

    pub fn first_address(&self) -> u64 {
        self.segs[0].aligned_start()
    }

    pub fn containing_seg<'a>(
        &'a self,
        addr: u64,
        extra_segs: Option<&'a [Seg]>,
    ) -> Option<&'a Seg> {
        // check the extra segs first, since they may be shadowed by the mem_image
        if let Some(extra) = extra_segs {
            for s in extra.iter() {
                if s.aligned_start() <= addr && addr < s.aligned_end() {
                    return Some(s);
                }
            }
        }
        for s in self.segs.iter() {
            if s.aligned_start() <= addr && addr < s.aligned_end() {
                return Some(s);
            }
        }

        None
    }

    pub fn perm_of_addr(&self, addr: u64) -> Option<Perms> {
        self.containing_seg(addr, None).map(|a| a.perm)
    }

    pub fn offset_of_addr(&self, addr: u64) -> Option<u64> {
        self.containing_seg(addr, None)
            .map(|a| addr - a.aligned_start())
    }

    pub fn try_dereference<'a>(
        &'a self,
        addr: u64,
        extra_segs: Option<&'a [Seg]>,
    ) -> Option<&'a [u8]> {
        self.containing_seg(addr, extra_segs).and_then(|s| {
            let offset = (addr - s.aligned_start()) as usize;

            if offset > s.data.len() {
                None
            } else {
                Some(&s.data[offset..])
            }
        })
    }

    pub fn random_address<H: Hash>(&self, permissions: Option<Perms>, seed: H) -> u64 {
        let mut rng = hash_seed_rng(&seed);
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

    pub fn seek(&self, offset: u64, sequence: &[u8], extra_segs: Option<&[Seg]>) -> Option<u64> {
        if let Some(s) = self.containing_seg(offset, extra_segs) {
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

    pub fn seek_all_segs(&self, sequence: &[u8], extra_segs: Option<&[Seg]>) -> Option<u64> {
        for seg in self.segments() {
            let offset = seg.aligned_start();
            if let Some(res) = self.seek(offset, sequence, extra_segs) {
                return Some(res);
            }
        }
        None
    }

    pub fn seek_from_random_address<H: Hash>(&self, sequence: &[u8], seed: H) -> Option<u64> {
        self.seek(self.random_address(None, seed), sequence, None)
    }

    pub fn segments(&self) -> &Vec<Seg> {
        &self.segs
    }

    /// Returns a chain of dereferences beginning with the address `start`.
    /// If `start` fails to dereference to any value, the chain will just
    /// be `vec![start]`, so the caller can always assume that the chain
    /// is non-empty.
    pub fn deref_chain(&self, start: u64, steps: usize, extra_segs: Option<&[Seg]>) -> Vec<u64> {
        let word_size = word_size_in_bytes(self.arch, self.mode);
        let endian = endian(self.arch, self.mode);
        let mut chain = vec![start];
        struct Crawl<'s> {
            f: &'s dyn Fn(&Crawl<'_>, u64, usize, &mut Vec<u64>) -> (),
        };
        let crawl = Crawl {
            f: &|crawl, addr, steps, mut chain| {
                if steps > 0 {
                    if let Some(bytes) = self.try_dereference(addr, extra_segs) {
                        if let Some(addr) = read_integer(bytes, endian, word_size) {
                            chain.push(addr);
                            (crawl.f)(crawl, addr, steps - 1, &mut chain);
                        }
                    }
                }
            },
        };
        (crawl.f)(&crawl, start, steps, &mut chain);
        chain
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize)]
pub struct Seg {
    pub addr: u64,
    pub memsz: usize,
    pub perm: Perms,
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
    //n >> PAGE_BITS << PAGE_BITS
    (n + (PAGE_SIZE - 1)) & !(PAGE_SIZE - 1)
}

impl Seg {
    pub fn from_mem_region_and_data(reg: unicorn::MemRegion, data: Vec<u8>) -> Self {
        Self {
            addr: reg.begin,
            memsz: (reg.end - reg.begin) as usize,
            perm: reg.perms.into(),
            segtype: SegType::Load, // FIXME
            data,
        }
    }

    pub fn from_phdr(phdr: &elf::ProgramHeader) -> Self {
        let perm: Perms = phdr.into();

        let addr = phdr.vm_range().start as u64;
        let memsz = (phdr.vm_range().end - phdr.vm_range().start) as usize;
        let size = Self::intern_aligned_size(addr, memsz);
        let data = vec![0x00_u8; size];
        Self {
            addr,
            memsz,
            perm,
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
        self.perm.intersects(Perms::EXEC)
    }

    #[inline]
    pub fn is_writeable(&self) -> bool {
        self.perm.intersects(Perms::WRITE)
    }

    #[inline]
    pub fn is_readable(&self) -> bool {
        self.perm.intersects(Perms::READ)
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

    #[inline]
    pub fn offset_of_addr(&self, addr: u64) -> u64 {
        let start = self.aligned_start();
        debug_assert!(addr > start);
        addr - start
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Deserialize, Serialize)]
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
    //let mut page_one = false;
    let shdrs = &elf.section_headers;
    let phdrs = &elf.program_headers;

    let mut segs = phdrs
        .iter()
        .map(Seg::from_phdr)
        .filter(Seg::loadable)
        .collect::<Vec<Seg>>();
    /* Low memory */
    // I placed this here so that address 0x0 would always resolve, and
    // so that 0 could be used as a stopping address.
    // if !page_one {
    //     segs.push(Seg {
    //         addr: 0,
    //         memsz: 0x1000,
    //         perm: Perms::READ,
    //         segtype: SegType::Load,
    //         data: vec![0; 0x1000],
    //     });
    // };

    for shdr in shdrs {
        let (i, j) = (
            shdr.sh_offset as usize,
            (shdr.sh_offset + shdr.sh_size) as usize,
        );
        let end = usize::min(j, code_buffer.len());
        let sdata = &code_buffer[i..end];
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
                    seg.data[v_off] = *byte;
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
        perm: Perms::READ | Perms::WRITE,
        segtype: SegType::Load,
        memsz: stack_size,
        data: vec![0; stack_size],
    });
    segs
}

fn initialize_memory_image(
    segments: &[Seg],
    arch: unicorn::Arch,
    mode: unicorn::Mode,
    il_program: Option<il::Program>,
) {
    let endian = endian(arch, mode);
    let word_size = word_size_in_bytes(arch, mode);
    unsafe {
        MEM_IMAGE = MemoryImage {
            segs: segments.to_owned(),
            arch,
            mode,
            endian,
            word_size,
            disasm: Some(Disassembler::new(arch, mode).expect("Failed to initialize disassembler")),
            il_program,
        }
    }
}

pub fn try_to_get_static_memory_image() -> Option<&'static MemoryImage> {
    if INIT_MEM_IMAGE.is_completed() {
        unsafe { Some(&MEM_IMAGE) }
    } else {
        None
    }
}

pub fn get_static_memory_image() -> &'static MemoryImage {
    if INIT_MEM_IMAGE.is_completed() {
        unsafe { &MEM_IMAGE }
    } else {
        panic!("MEM_IMAGE has not been initialized")
    }
}

pub fn load(
    code_buffer: &[u8],
    stack_size: usize,
    arch: unicorn::Arch,
    mode: unicorn::Mode,
    init: bool,
) -> Result<Vec<Seg>, Error> {
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
        if init {
            INIT_MEM_IMAGE.call_once(|| initialize_memory_image(&segs, arch, mode, None));
        }

        Ok(segs)
    }
}

pub fn load_from_path(config: &RoperConfig, init: bool) -> Result<Vec<Seg>, Error> {
    let path = &config.binary_path;
    let stack_size = config.emulator_stack_size;
    let arch = config.arch;
    let mode = config.mode;
    load(&std::fs::read(path)?, stack_size, arch, mode, init)
}

pub mod falcon_loader {
    use std::hash::Hasher;
    use std::path::Path;

    use falcon::il;
    use falcon::loader::{ElfLinker, ElfLinkerBuilder, Loader};
    use falcon::memory::MemoryPermissions;
    use fnv::FnvHasher;
    use unicorn::{Arch, Mode};

    use crate::util;
    use crate::util::dump::{ron_dump, ron_undump};

    // A wrapper around falcon's loader.
    use super::*;

    // TODO:
    // add a function that
    // - hashes the binary
    // - checks a directory to see if a lifted serialization of the lifted program
    //   has been stored for that hash
    // - if so, deserializes the program
    // - if not, lifts the binary, then serializes the program and saves it
    //   with a filename matching the hash.

    fn arch_mode_from_linker(linker: &ElfLinker) -> (Arch, Mode) {
        match linker.architecture().name() {
            "amd64" => (Arch::X86, Mode::MODE_64),
            "x86" => (Arch::X86, Mode::MODE_32),
            s => unimplemented!("the falcon-based loader doesn't yet support {}", s),
        }
    }

    pub fn load_from_path(config: &mut RoperConfig, init: bool) -> Result<Vec<Seg>, Error> {
        if INIT_MEM_IMAGE.is_completed() {
            unsafe { Ok(MEM_IMAGE.segments().clone()) }
        } else {
            log::info!("Using falcon loader");
            let path = &config.binary_path;
            if config.ld_paths.is_none() {
                log::warn!(
                    "No ld_paths supplied. Attempting to complete using `ldd {}`.",
                    path
                );
                config.ld_paths = util::ldd::ld_paths(&path).ok();
            }
            println!("ld_paths = {:#?}", config.ld_paths);
            //let elf = Elf::from_file_with_base_address(path, base_address)?;
            let linker = ElfLinkerBuilder::new(path.into())
                .do_relocations(false) // Not yet well-supported
                .ld_paths(config.ld_paths.clone())
                .link()?;

            //let program = elf.program()?;
            let mut memory = linker.memory()?;
            // figure out where the stack should go
            let stack_position = memory
                .sections()
                .iter()
                .map(|(&addr, sec)| addr + sec.data().len() as u64)
                .max()
                .expect("Could not find maximum address");
            // initialize empty memory for the stack
            let stack_data = vec![0; config.emulator_stack_size];
            // insert the stack into memory
            memory.set_memory(
                stack_position,
                stack_data,
                MemoryPermissions::READ | MemoryPermissions::WRITE,
            );

            let mut segs = memory
                .sections()
                .iter()
                .map(|(&addr, sec)| {
                    let memsz = sec.data().len();
                    let data = sec.data().to_vec();
                    let perm = sec.permissions().into();
                    Seg {
                        addr,
                        memsz,
                        perm,
                        segtype: SegType::Load,
                        data,
                    }
                })
                .collect::<Vec<Seg>>();
            for seg in segs.iter_mut() {
                seg.ensure_data_alignment()
            }

            let (arch, mode) = arch_mode_from_linker(&linker);
            config.arch = arch;
            config.mode = mode;

            // Do the lifting, then serialize and save the lifted program
            // but check to see if a previously lifted version already exists.
            let mem_hash = {
                let mut h = FnvHasher::default();
                let mem = linker.memory().expect("Failed to get memory from linker");
                mem.hash(&mut h);
                h.finish()
            };
            let p = format!("./cache/{:x}.ron.gz", mem_hash);
            let cached_path = Path::new(&p);
            // FIXME: temporarily disabled. Problems deserializing RON encoded Program structs.
            // experiment with different formats.
            let program: il::Program = if false && cached_path.exists() {
                ron_undump::<il::Program, &Path>(cached_path).unwrap_or_else(|e| {
                    panic!(
                        "Failed to deserialized cached il::Program at {:?}: {:?}",
                        cached_path, e
                    )
                })
            } else {
                log::info!("Lifting the intermediate representation of the program...");
                let program = linker
                    .program()
                    .expect("Failed to lift il::Program from ElfLinker");
                log::info!("Finished lifting program.");
                ron_dump(&program, cached_path).expect("Failed to dump il::Program");
                program
            };

            if init {
                // TODO: let lift_program be optional, and only activated when using Push
                INIT_MEM_IMAGE
                    .call_once(|| initialize_memory_image(&segs, arch, mode, Some(program)));
            }
            Ok(segs)
        }
    }
}

#[cfg(test)]
mod test {
    use unicorn::{Arch, Mode};

    use super::*;

    #[test]
    fn test_loader() {
        //pretty_env_logger::init();
        let mut config = RoperConfig {
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
            binary_path: "/bin/sh".into(), // "./binaries/X86/MODE_32/sshd".to_string(),
            ld_paths: None,
            bad_bytes: None,
            ..Default::default()
        };
        let res = load_from_path(&config, false).expect("Failed to load /bin/sh");
        println!("With legacy loader");
        for s in res.iter() {
            println!("Segment: {}", s);
        }
        let res2 = falcon_loader::load_from_path(&mut config, false).expect("Failed to load");
        println!("With falcon loader");
        for s in res2.iter() {
            println!("Segment: {}", s);
        }
        // for (s1, s2) in res.iter().zip(res2.iter()) {
        //     assert_eq!(s1, s2);
        // }
    }
}

// TODO:
// - The elf loader is a bit rough, and not entirely exact. I don't handle relocations, e.g.
// - We should factor out the unicorn elf loader into its own method or structure. Right now,
//   it all takes place inside executore::init_emu.
