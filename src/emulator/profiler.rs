use crate::emulator::executor::Register;
use byteorder::{BigEndian, ByteOrder};
use indexmap::map::IndexMap;
use radix_trie::{Trie, TrieKey};
use std::cmp::{Ord, PartialOrd};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use unicorn::Cpu;
use crate::evolution::Problem;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Block {
    pub entry: u64,
    pub size: usize,
    //pub code: Vec<u8>,
}

impl TrieKey for Block {
    fn encode_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        BigEndian::write_u64(&mut buf, self.entry);
        BigEndian::write_u64(&mut buf, self.size as u64);
        buf
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Blocks(pub Vec<Block>);

impl From<Vec<Block>> for Blocks {
    fn from(blocks: Vec<Block>) -> Self {
        Self(blocks)
    }
}

impl TrieKey for Blocks {
    fn encode_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for block in &self.0 {
            let mut b = [0; 8];
            BigEndian::write_u64(&mut b, block.entry);
            buf.extend_from_slice(&b);
            let mut b = [0; 8];
            BigEndian::write_u64(&mut b, block.size as u64);
            buf.extend_from_slice(&b);
        }
        // ensure that 8 bytes were written per field
        assert_eq!(buf.len(), self.0.len() * 16);
        buf
    }
}

pub struct Profiler<C: Cpu<'static>> {
    /// The Arc<Mutex<_>> fields need to be writeable for the unicorn callbacks.
    pub block_log: Arc<Mutex<Vec<Block>>>,
    /// These fields are written to after the emulation has finished.
    pub cpu_error: Option<unicorn::Error>,
    pub computation_time: Duration,
    pub registers: IndexMap<Register<C>, u64>,
    registers_to_read: Vec<Register<C>>,
}

impl<C: Cpu<'static>> Profiler<C> {
    pub fn strong_counts(&self) -> usize {
        Arc::strong_count(&self.block_log)
    }
}

#[derive(Debug)]
pub struct Profile<C: Cpu<'static>> {
    pub block_trie: Trie<Blocks, ()>,
    pub cpu_errors: IndexMap<unicorn::Error, usize>,
    pub computation_times: Vec<Duration>,
    pub registers: Vec<IndexMap<Register<C>, u64>>,
}

impl<C: Cpu<'static>> Profile<C> {
    pub fn collate(profilers: Vec<Profiler<C>>) -> Self {
        let mut block_trie = Trie::new();
        //let mut write_trie = Trie::new();
        let mut cpu_errors = IndexMap::new();
        let mut computation_times = Vec::new();
        let mut register_maps = Vec::new();

        for Profiler {
            block_log,
            //write_log,
            cpu_error,
            computation_time,
            registers,
            ..
        } in profilers.into_iter()
        {
            block_trie.insert((*block_log.lock().unwrap()).clone().into(), ());
            //write_trie.insert((*write_log.lock().unwrap()).clone(), ());
            if let Some(c) = cpu_error {
                *cpu_errors.entry(c).or_insert(0) += 1;
            };
            computation_times.push(computation_time);
            register_maps.push(registers);
        }

        Self {
            block_trie,
            //write_trie,
            cpu_errors,
            computation_times,
            registers: register_maps,
        }
    }
}

impl<C: Cpu<'static>> From<Vec<Profiler<C>>> for Profile<C> {
    fn from(v: Vec<Profiler<C>>) -> Self {
        Self::collate(v)
    }
}

impl<C: Cpu<'static>> fmt::Debug for Profiler<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // write!(
        //     f,
        //     "write_log: {} entries; ",
        //     self.write_log.lock().unwrap().len()
        // )?;
        write!(f, "registers: {:?}; ", self.registers)?;
        write!(f, "cpu_error: {:?}; ", self.cpu_error)?;
        write!(
            f,
            "computation_time: {} μs; ",
            self.computation_time.as_micros()
        )?;
        write!(f, "{} blocks", self.block_log.lock().unwrap().len())
    }
}

impl<C: Cpu<'static>> Profiler<C> {
    pub fn new(output_registers: &[Register<C>]) -> Self {
        Self {
            registers_to_read: output_registers.to_vec(),
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct MemLogEntry {
    pub program_counter: u64,
    pub mem_address: u64,
    pub num_bytes_written: usize,
    pub value: i64,
}

impl<C: Cpu<'static>> Default for Profiler<C> {
    fn default() -> Self {
        Self {
            //write_log: Arc::new(Mutex::new(Vec::default())),
            registers: IndexMap::default(),
            cpu_error: None,
            registers_to_read: Vec::new(),
            computation_time: Duration::default(),
            block_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
}
