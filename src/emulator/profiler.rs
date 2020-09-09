use std::cmp::{Ord, PartialOrd};
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use bitflags::_core::ops::Index;
use bson::document::Entry;
use capstone::Instructions;
use crossbeam::queue::SegQueue;
//use indexmap::map::IndexMap;
//use indexmap::set::IndexSet;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
pub use unicorn::unicorn_const::Error as UCError;
use unicorn::Cpu;

use crate::emulator::loader;
use crate::emulator::loader::{get_static_memory_image, try_to_get_static_memory_image, Seg};
use crate::emulator::register_pattern::{Register, RegisterState};
use crate::util::architecture::{write_integer, Endian};

// TODO: why store the size at all, if you're just going to
// throw it away?
#[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize, Hash)]
pub struct Block {
    pub entry: u64,
    pub size: usize,
    //pub code: Vec<u8>,
}

impl Block {
    pub fn get_code(&self) -> &'static [u8] {
        let memory = loader::get_static_memory_image();
        memory
            .try_dereference(self.entry, None)
            .map(|b| &b[..self.size])
            .unwrap()
    }

    pub fn disassemble(&self) -> Instructions<'_> {
        let memory = loader::get_static_memory_image();
        memory
            .disassemble(self.entry, self.size, None)
            .expect("Failed to disassemble basic block")
    }
}

impl fmt::Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[BLOCK 0x{:08x} - 0x{:08x}]",
            self.entry,
            self.entry + self.size as u64
        )
    }
}

pub struct Profiler<C: Cpu<'static>> {
    /// The Arc<RwLock<_>> fields need to be writeable for the unicorn callbacks.
    pub block_log: Arc<SegQueue<Block>>,
    pub gadget_log: Arc<SegQueue<u64>>,
    //Arc<RwLock<Vec<u64>>>,
    /// These fields are written to after the emulation has finished.
    pub written_memory: Vec<Seg>,
    pub write_log: Arc<SegQueue<MemLogEntry>>,
    //Arc<RwLock<Vec<MemLogEntry>>>,
    pub cpu_error: Option<unicorn::Error>,
    pub emulation_time: Duration,
    pub registers: HashMap<Register<C>, u64>,
    registers_to_read: Vec<Register<C>>,
    pub input: HashMap<Register<C>, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Profile {
    pub paths: Vec<Vec<Block>>,
    //PrefixSet<Block>,
    // TODO: cpu_errors should be a vector of Option<usize>
    pub cpu_errors: Vec<Option<UCError>>,
    pub emulation_times: Vec<Duration>,
    pub registers: Vec<RegisterState>,
    pub gadgets_executed: Vec<HashSet<u64>>,
    #[cfg(not(feature = "full_dump"))]
    #[serde(skip)]
    pub memory_writes: Vec<SparseData>,
    pub executable: bool,
}

// FIXME: refactor so that we don't have any code duplication between
// this method and collate. Or just get rid of collate entirely, I guess.

impl<C: 'static + Cpu<'static>> From<Profiler<C>> for Profile {
    fn from(p: Profiler<C>) -> Self {
        let mut paths = Vec::new(); // PrefixSet::new();
        let mut cpu_errors = Vec::new();
        let mut computation_times = Vec::new();
        let mut register_maps = Vec::new();
        let mut gadgets_executed = Vec::new();
        let mut memory_writes = Vec::new();

        let Profiler {
            block_log,
            write_log,
            cpu_error,
            emulation_time,
            registers,
            gadget_log,
            written_memory,
            ..
        } = p;
        paths.push(segqueue_to_vec(block_log));
        let mut executed = HashSet::new();
        while let Ok(g) = gadget_log.pop() {
            executed.insert(g);
        }
        gadgets_executed.push(executed);
        cpu_errors.push(cpu_error);
        computation_times.push(emulation_time);
        register_maps.push(RegisterState::new::<C>(&registers, Some(&written_memory)));

        memory_writes.push(mem_log_to_sparse_data(&segqueue_to_vec(write_log)));

        Self {
            paths,
            cpu_errors,
            emulation_times: computation_times,
            gadgets_executed,
            registers: register_maps,
            memory_writes,
            executable: true,
        }
    }
}

fn segqueue_to_vec<T>(sq: Arc<SegQueue<T>>) -> Vec<T> {
    let mut v = vec![];
    while let Ok(x) = sq.pop() {
        v.push(x)
    }
    //log::debug!("vec of {} blocks", v.len());
    v
}

impl Profile {
    // combine the information in two different profiles by absorbing the second
    // into the first
    pub fn absorb(&mut self, other: Self) {
        let Self {
            paths,
            cpu_errors,
            emulation_times,
            registers,
            gadgets_executed,
            memory_writes,
            executable,
        } = other;

        self.paths.extend(paths.into_iter());
        self.cpu_errors.extend(cpu_errors.into_iter());
        self.emulation_times.extend(emulation_times.into_iter());
        self.registers.extend(registers.into_iter());
        self.gadgets_executed.extend(gadgets_executed.into_iter());
        self.memory_writes.extend(memory_writes.into_iter());
        self.executable &= executable;
    }

    pub fn collate<C: 'static + Cpu<'static>>(profilers: Vec<Profiler<C>>) -> Self {
        //let mut write_trie = Trie::new();
        let mut paths = Vec::new(); // PrefixSet::new();
        let mut cpu_errors = Vec::new();
        let mut computation_times = Vec::new();
        let mut register_maps = Vec::new();
        let mut gadgets_executed = Vec::new();
        let mut memory_writes = Vec::new();

        for Profiler {
            block_log,
            write_log,
            cpu_error,
            emulation_time,
            registers,
            gadget_log,
            written_memory,
            ..
        } in profilers.into_iter()
        {
            paths.push(segqueue_to_vec(block_log));
            let mut executed = HashSet::new();
            while let Ok(g) = gadget_log.pop() {
                executed.insert(g);
            }
            gadgets_executed.push(executed);
            // NOTE: changing gadgets_executed to a vec of hashsets

            cpu_errors.push(cpu_error);
            computation_times.push(emulation_time);
            // FIXME: use a different data type for output states.
            register_maps.push(RegisterState::new::<C>(&registers, Some(&written_memory)));

            memory_writes.push(mem_log_to_sparse_data(&segqueue_to_vec(write_log)));
        }

        Self {
            paths,
            cpu_errors,
            emulation_times: computation_times,
            gadgets_executed,
            registers: register_maps,
            memory_writes,
            executable: true,
        }
    }

    pub fn avg_emulation_micros(&self) -> f64 {
        self.emulation_times.iter().sum::<Duration>().as_micros() as f64
            / self.emulation_times.len() as f64
    }

    pub fn basic_block_path_iterator(&self) -> impl Iterator<Item = &Vec<Block>> + '_ {
        self.paths.iter()
    }

    pub fn disas_paths(&self) -> impl Iterator<Item = String> + '_ {
        self.paths.iter().map(move |path| {
            path.par_iter()
                .map(|b| {
                    let prefix = if self.was_this_executed(b.entry) {
                        "----\n"
                    } else {
                        ""
                    };
                    format!("{}{}\n", prefix, b.disassemble())
                })
                .collect::<String>()
        })
    }

    pub fn was_this_executed(&self, w: u64) -> bool {
        for gads in self.gadgets_executed.iter() {
            if gads.contains(&w) {
                return true;
            }
        }
        false
    }
}

impl<C: 'static + Cpu<'static>> From<Vec<Profiler<C>>> for Profile {
    fn from(v: Vec<Profiler<C>>) -> Self {
        Self::collate(v)
    }
}

impl<C: Cpu<'static>> fmt::Debug for Profiler<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "registers: {:?}; ", self.registers)?;
        write!(f, "cpu_error: {:?}; ", self.cpu_error)?;
        write!(
            f,
            "computation_time: {} μs; ",
            self.emulation_time.as_micros()
        )
    }
}

impl<C: Cpu<'static>> Profiler<C> {
    pub fn new(output_registers: &[Register<C>], input: &HashMap<Register<C>, u64>) -> Self {
        Self {
            registers_to_read: output_registers.to_vec(),
            input: input.clone(),
            ..Default::default()
        }
    }

    pub fn read_registers(&mut self, emu: &mut C) {
        for r in &self.registers_to_read {
            let val = emu.reg_read(*r).expect("Failed to read register!");
            self.registers.insert(*r, val);
        }
    }

    pub fn register(&self, reg: Register<C>) -> Option<u64> {
        self.registers.get(&reg).cloned()
    }

    pub fn set_error(&mut self, error: unicorn::Error) {
        self.cpu_error = Some(error)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize, Hash)]
pub struct MemLogEntry {
    pub program_counter: u64,
    pub address: u64,
    pub num_bytes_written: usize,
    pub value: u64,
}

impl<C: Cpu<'static>> Default for Profiler<C> {
    fn default() -> Self {
        Self {
            write_log: Arc::new(SegQueue::new()), //Arc::new(RwLock::new(Vec::default())),
            input: HashMap::default(),
            registers: HashMap::default(),
            cpu_error: None,
            registers_to_read: Vec::new(),
            emulation_time: Duration::default(),
            block_log: Arc::new(SegQueue::new()),
            gadget_log: Arc::new(SegQueue::new()), //Arc::new(RwLock::new(Vec::new())),
            written_memory: vec![],
        }
    }
}

#[derive(Clone, Debug)]
pub struct SparseData(BTreeMap<u64, u8>);

impl SparseData {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn insert_u8(&mut self, address: u64, byte: u8) -> Option<u8> {
        self.0.insert(address, byte)
    }

    pub fn insert_word(&mut self, address: u64, word: u64, size: usize) {
        let (endian, word_size) = if let Some(memory) = try_to_get_static_memory_image() {
            (memory.endian, memory.word_size)
        } else {
            (Endian::Little, 8)
        };
        let mut buf = vec![0_u8; word_size];
        write_integer(endian, word_size, word, &mut buf);
        for (offset, byte) in buf.into_iter().enumerate() {
            if offset >= size {
                break;
            }
            self.insert_u8(address + offset as u64, byte);
        }
    }

    pub fn find_seq(&self, seq: &[u8]) -> Vec<u64> {
        let mut found_at = Vec::new();
        let mut prev = None;
        let mut i = 0_usize;
        for (addr, byte) in self.0.iter() {
            if i == 0 {
                if byte == &seq[i] {
                    prev = Some(*addr);
                    found_at.push(*addr); // provisionally. we'll pop it if need be
                    i += 1;
                } else {
                    // nothing to do here.
                }
            } else {
                if prev == Some(addr - 1) && *byte == seq[i] {
                    // contiguity check
                    i += 1;
                } else {
                    i = 0;
                    prev = None;
                    found_at.pop();
                }
            }
            if i >= seq.len() {
                i = 0;
                prev = None;
            }
        }
        found_at
    }
}

fn mem_log_to_sparse_data(mem_log: &[MemLogEntry]) -> SparseData {
    let mut sparse = SparseData::new();
    for log in mem_log.iter() {
        debug_assert!(log.num_bytes_written <= 8);
        sparse.insert_word(log.address, log.value, log.num_bytes_written);
    }
    sparse
}

pub trait HasProfile {
    fn profile(&self) -> Option<&Profile>;

    fn add_profile(&mut self, profile: Profile);

    fn set_profile(&mut self, profile: Profile);
}

#[cfg(test)]
mod test {
    use unicorn::CpuX86;

    use super::*;

    macro_rules! segqueue {
        ($($x:expr,)*) => {
            {
                let q = SegQueue::new();
                $(
                   q.push($x);
                )*
                q
            }
        }
    }

    #[test]
    fn test_sparse_data() {
        let mut sparse = SparseData::new();

        sparse.insert_word(0, 0xcafebabe_deadbeef, 8);
        sparse.insert_word(8, 0xbeefface_cafebeef, 8);
        sparse.insert_word(1024, 0xbeefbeef_beefbeef, 8);

        println!("sparse = {:#x?}", sparse);

        let res = sparse.find_seq(&[0xef, 0xbe]);

        println!("res = {:#x?}", res);

        let res = sparse.find_seq(&[0xca, 0xef]);

        println!("res = {:#x?}", res);
    }

    #[test]
    fn test_collate() {
        let profilers: Vec<Profiler<CpuX86<'_>>> = vec![
            Profiler {
                block_log: Arc::new(segqueue![
                    Block { entry: 1, size: 2 },
                    Block { entry: 3, size: 4 },
                ]),
                cpu_error: None,
                emulation_time: Default::default(),
                registers: HashMap::new(),
                ..Default::default()
            },
            Profiler {
                block_log: Arc::new(segqueue![
                    Block { entry: 1, size: 2 },
                    Block { entry: 6, size: 6 },
                ]),
                cpu_error: None,
                emulation_time: Default::default(),
                registers: HashMap::new(),
                ..Default::default()
            },
        ];

        let profile: Profile = profilers.into();

        println!(
            "size of profile in mem: {}",
            std::mem::size_of_val(&profile.paths)
        );
    }
}
