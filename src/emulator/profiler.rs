use std::cmp::{Ord, PartialOrd};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use capstone::Instructions;
use crossbeam::queue::SegQueue;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use subslice::SubsliceExt;
pub use unicorn::unicorn_const::Error as UCError;
use unicorn::Cpu;

use crate::emulator::loader;
use crate::emulator::loader::{get_static_memory_image, try_to_get_static_memory_image, Seg};
use crate::emulator::register_pattern::{Register, RegisterState};
use crate::util::architecture::{write_integer, Endian};

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize, Hash)]
pub struct Block {
    pub entry: u64,
    pub size: usize,
    // pub code: Vec<u8>,
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
            "[CODE 0x{:08x} - 0x{:08x}]",
            self.entry,
            self.entry + self.size as u64
        )
    }
}

pub struct Profiler<C: Cpu<'static>> {
    /// The Arc<RwLock<_>> fields need to be writeable for the unicorn callbacks.
    pub trace_log: Arc<SegQueue<Block>>,
    pub committed_trace_log: Arc<Mutex<Vec<Block>>>,

    pub ret_count: Arc<AtomicUsize>,
    pub call_stack_depth: Arc<AtomicUsize>,
    pub gadget_log: Arc<SegQueue<u64>>,
    //Arc<RwLock<Vec<u64>>>,
    /// These fields are written to after the emulation has finished.
    pub written_memory: Vec<Seg>,
    pub write_log: Arc<SegQueue<MemLogEntry>>,
    pub committed_write_log: Arc<Mutex<SparseDataHelper>>,
    //Arc<RwLock<Vec<MemLogEntry>>>,
    pub cpu_error: Option<unicorn::Error>,
    pub emulation_time: Duration,
    pub registers_at_last_ret: Arc<Mutex<HashMap<Register<C>, u64>>>,
    pub registers_to_read: Vec<Register<C>>,
    pub input: HashMap<Register<C>, u64>,
}

impl<C: Cpu<'static>> Default for Profiler<C> {
    fn default() -> Self {
        Self {
            ret_count: Arc::new(AtomicUsize::new(0)),
            call_stack_depth: Arc::new(AtomicUsize::new(0)),
            write_log: Arc::new(SegQueue::new()), //Arc::new(RwLock::new(Vec::default())),
            input: HashMap::default(),
            registers_at_last_ret: Arc::new(Mutex::new(HashMap::default())),
            cpu_error: None,
            registers_to_read: Vec::new(),
            emulation_time: Duration::default(),
            trace_log: Arc::new(SegQueue::new()),
            gadget_log: Arc::new(SegQueue::new()), //Arc::new(RwLock::new(Vec::new())),
            written_memory: vec![],
            committed_write_log: Default::default(),
            committed_trace_log: Default::default(),
        }
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
        let mut registers = self.registers_at_last_ret.lock().unwrap();
        for r in &self.registers_to_read {
            let val = emu.reg_read(*r).expect("Failed to read register!");
            registers.insert(*r, val);
        }
    }

    pub fn set_error(&mut self, error: unicorn::Error) {
        self.cpu_error = Some(error)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Profile {
    pub paths: Vec<Vec<Block>>,
    pub code_executed: Vec<Vec<u8>>,
    //PrefixSet<Block>,
    // TODO: cpu_errors should be a vector of Option<usize>
    pub cpu_errors: Vec<Option<UCError>>,
    pub emulation_times: Vec<Duration>,
    pub registers: Vec<RegisterState>,
    pub gadgets_executed: Vec<HashMap<u64, usize>>,
    #[cfg(not(feature = "full_dump"))]
    #[serde(skip)]
    pub memory_writes: Vec<SparseData>,
    pub executable: bool,
    pub ret_counts: Vec<usize>,
}

fn fetch_code_executed(path: &Vec<Block>, extra_segs: Option<&[Seg]>) -> Vec<u8> {
    let memory = get_static_memory_image();
    let mut code = vec![];
    for block in path.iter() {
        if let Some(bytes) = memory.try_dereference(block.entry, extra_segs) {
            for b in bytes[0..block.size].iter() {
                code.push(*b);
            }
        }
    }
    code
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
        let mut ret_counts = Vec::new();
        let mut code_paths_executed = Vec::new();

        let Profiler {
            cpu_error,
            emulation_time,
            registers_at_last_ret: registers,
            gadget_log,
            written_memory,
            ret_count,
            committed_write_log,
            committed_trace_log,
            ..
        } = p;
        let path = Arc::try_unwrap(committed_trace_log)
            .ok()
            .unwrap()
            .into_inner()
            .unwrap();
        let code_executed = fetch_code_executed(&path, Some(&written_memory));
        paths.push(path);
        code_paths_executed.push(code_executed);

        let mut executed = HashMap::new();
        while let Ok(g) = gadget_log.pop() {
            (*executed.entry(g).or_insert(0)) += 1;
        }
        gadgets_executed.push(executed);
        cpu_errors.push(cpu_error);
        computation_times.push(emulation_time);
        register_maps.push(RegisterState::new::<C>(
            &registers.lock().unwrap(),
            Some(&written_memory),
        ));

        let log = Arc::try_unwrap(committed_write_log)
            .ok()
            .unwrap()
            .into_inner()
            .unwrap();
        memory_writes.push(log.into());
        // memory_writes.push(segqueue_to_vec(write_log).into());

        ret_counts.push(ret_count.load(std::sync::atomic::Ordering::Relaxed));
        Self {
            paths,
            code_executed: code_paths_executed,
            cpu_errors,
            emulation_times: computation_times,
            gadgets_executed,
            registers: register_maps,
            memory_writes,
            executable: true,
            ret_counts,
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
            code_executed,
            cpu_errors,
            emulation_times,
            registers,
            gadgets_executed,
            memory_writes,
            executable,
            ret_counts,
        } = other;

        self.paths.extend(paths.into_iter());
        self.code_executed.extend(code_executed.into_iter());
        self.cpu_errors.extend(cpu_errors.into_iter());
        self.emulation_times.extend(emulation_times.into_iter());
        self.registers.extend(registers.into_iter());
        self.gadgets_executed.extend(gadgets_executed.into_iter());
        self.memory_writes.extend(memory_writes.into_iter());
        self.ret_counts.extend(ret_counts.into_iter());
        self.executable &= executable;
    }

    pub fn avg_emulation_micros(&self) -> f64 {
        self.emulation_times.iter().sum::<Duration>().as_micros() as f64
            / self.emulation_times.len() as f64
    }

    pub fn execution_trace_iter(&self) -> impl Iterator<Item = &Vec<Block>> + '_ {
        self.paths.iter()
    }

    pub fn disas_paths(&self) -> impl Iterator<Item = String> + '_ {
        self.paths.iter().map(move |path| {
            path.par_iter()
                .map(|b| {
                    let prefix = if self.times_executed(b.entry) > 0 {
                        "----\n"
                    } else {
                        ""
                    };
                    format!("{}{}", prefix, b.disassemble())
                })
                .collect::<String>()
        })
    }

    pub fn times_executed(&self, w: u64) -> usize {
        let mut count = 0;
        for gads in self.gadgets_executed.iter() {
            if let Some(n) = gads.get(&w) {
                count += *n;
            }
        }
        count
    }

    /// The idea here is that we can take the lower bound of
    /// the return count, on the one hand, and the number of unique
    /// addresses that have been executed, to get a rough idea of
    /// how many distinct gadgets have been executed. There are
    /// ways to game this, of course: executing the same ROP-NOP
    /// several times in a row, while executing a number of contiguous
    /// instructions in sequence, each of which belonging to the
    /// chromosome. But it's a good enough place to start.
    pub fn gadgets_executed(&self, index: usize) -> usize {
        if index < self.gadgets_executed.len() {
            0
        } else {
            self.gadgets_executed[index]
                .len()
                .min(self.ret_counts[index])
        }
    }

    pub fn addresses_visited(&self) -> HashSet<u64> {
        let mut set = HashSet::new();
        for path in self.paths.iter() {
            for block in path.iter() {
                for addr in block.entry..(block.entry + block.size as u64) {
                    set.insert(addr);
                }
            }
        }
        set
    }
}

// impl<C: 'static + Cpu<'static>> From<Vec<Profiler<C>>> for Profile {
//     fn from(v: Vec<Profiler<C>>) -> Self {
//         Self::collate(v)
//     }
// }

impl<C: Cpu<'static>> fmt::Debug for Profiler<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "registers: {:?}; ", self.registers_at_last_ret)?;
        write!(f, "cpu_error: {:?}; ", self.cpu_error)?;
        write!(
            f,
            "computation_time: {} μs; ",
            self.emulation_time.as_micros()
        )
    }
}

pub fn read_registers_in_hook<C: Cpu<'static> + 'static>(
    registers_at_last_ret: Arc<Mutex<HashMap<Register<C>, u64>>>,
    registers_to_read: &[Register<C>],
    engine: &unicorn::Unicorn<'_>,
) {
    let mut registers = registers_at_last_ret.lock().unwrap();
    for r in registers_to_read {
        let reg: i32 = (*r).into();
        let val = engine.reg_read(reg).expect("Failed to read register!");
        registers.insert(*r, val);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize, Hash)]
pub struct MemLogEntry {
    pub program_counter: u64,
    pub address: u64,
    pub num_bytes_written: usize,
    pub value: u64,
}

#[derive(Clone, Hash, Default)]
pub struct SparseDataHelper(BTreeMap<u64, u8>);

impl SparseDataHelper {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn any_writes(&self, addr: u64) -> bool {
        self.0.contains_key(&addr)
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

    pub fn absorb_segqueue(&mut self, mem_log: &SegQueue<MemLogEntry>) {
        while let Ok(m) = mem_log.pop() {
            self.insert_word(m.address, m.value, m.num_bytes_written);
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

    pub fn telescope(&self) -> BTreeMap<u64, Vec<u8>> {
        let mut last = 0_u64;
        let mut scoped = BTreeMap::new();
        let mut buf = vec![];
        for (addr, byte) in self.0.iter() {
            if *addr != last + 1 {
                // discontinuity
                let mut new_buf = vec![];
                std::mem::swap(&mut buf, &mut new_buf);
                if !new_buf.is_empty() {
                    let start = 1 + last - new_buf.len() as u64;
                    scoped.insert(start, new_buf);
                }
            }
            buf.push(*byte);

            last = *addr;
        }

        if !buf.is_empty() {
            let start = 1 + last - buf.len() as u64;
            scoped.insert(start, buf);
        }

        scoped
    }
}

#[derive(Clone, Hash)]
pub struct SparseData(BTreeMap<u64, Vec<u8>>);

impl From<SparseDataHelper> for SparseData {
    fn from(helper: SparseDataHelper) -> Self {
        SparseData(helper.telescope())
    }
}

impl SparseData {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn find_seq(&self, seq: &[u8]) -> Vec<u64> {
        let mut found_at = Vec::new();
        for (addr, buf) in self.0.iter() {
            if let Some(offset) = buf.find(seq) {
                found_at.push(addr + offset as u64);
            }
        }
        found_at
    }

    pub fn len(&self) -> usize {
        self.0.values().map(|buf| buf.len()).sum()
    }
}

impl fmt::Debug for SparseData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SparseData {{")?;
        for (addr, buf) in self.0.iter() {
            writeln!(
                f,
                "    0x{:x}..0x{:x} => {}",
                addr,
                addr + buf.len() as u64,
                hex::encode(&buf)
            )?;
        }
        writeln!(f, "}}")
    }
}

impl From<Vec<MemLogEntry>> for SparseData {
    fn from(mem_log: Vec<MemLogEntry>) -> Self {
        let mut sparse = SparseDataHelper::new();
        for log in mem_log.iter() {
            debug_assert!(log.num_bytes_written <= 8);
            sparse.insert_word(log.address, log.value, log.num_bytes_written);
        }
        sparse.into()
    }
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
        let mut sparse = SparseDataHelper::new();

        sparse.insert_word(0, 0xcafebabe_deadbeef, 8);
        sparse.insert_word(8, 0xbeefface_cafebeef, 8);
        sparse.insert_word(1024, 0xbeefbeef_beefbeef, 8);

        let sparse: SparseData = sparse.into();

        println!("sparse = {:#x?}", sparse);

        let res = sparse.find_seq(&[0xef, 0xbe]);

        println!("res = {:#x?}", res);

        let res = sparse.find_seq(&[0xca, 0xef]);

        println!("res = {:#x?}", res);
    }
}
