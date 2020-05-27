use crate::emulator::executor::{Address, Register};
use indexmap::map::IndexMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use unicorn::Cpu;

#[derive(Debug)]
pub struct Block {
    pub entry: u64,
    pub size: usize,
    pub code: Vec<u8>,
}

pub struct Profiler<C: Cpu<'static>> {
    /// The Arc<Mutex<_>> fields need to be writeable for the unicorn callbacks.
    pub block_log: Arc<Mutex<Vec<Block>>>,
    pub write_log: Arc<Mutex<Vec<MemLogEntry>>>,
    pub ret_log: Arc<Mutex<Vec<Address>>>,
    /// These fields are written to after the emulation has finished.
    pub cpu_error: Option<unicorn::Error>,
    pub computation_time: Duration,
    pub registers: IndexMap<Register<C>, u64>,
    registers_to_read: Arc<Vec<Register<C>>>,
}

impl<C: Cpu<'static>> fmt::Debug for Profiler<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ret_log: {:?}; ", self.ret_log.lock().unwrap())?;
        write!(
            f,
            "write_log: {} entries; ",
            self.write_log.lock().unwrap().len()
        )?;
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
    pub fn new(output_registers: Arc<Vec<Register<C>>>) -> Self {
        Self {
            registers_to_read: output_registers,
            ..Default::default()
        }
    }

    pub fn read_registers(&mut self, emu: &mut C) {
        let registers_to_read = self.registers_to_read.clone();
        registers_to_read.iter().for_each(|r| {
            let val = emu.reg_read(*r).expect("Failed to read register!");
            self.registers.insert(*r, val);
        });
    }

    pub fn register(&self, reg: Register<C>) -> Option<u64> {
        self.registers.get(&reg).cloned()
    }

    pub fn set_error(&mut self, error: unicorn::Error) {
        self.cpu_error = Some(error)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MemLogEntry {
    pub program_counter: u64,
    pub mem_address: u64,
    pub num_bytes_written: usize,
    pub value: i64,
}

impl<C: Cpu<'static>> Default for Profiler<C> {
    fn default() -> Self {
        Self {
            ret_log: Arc::new(Mutex::new(Vec::default())),
            write_log: Arc::new(Mutex::new(Vec::default())),
            registers: IndexMap::default(),
            cpu_error: None,
            registers_to_read: Arc::new(Vec::new()),
            computation_time: Duration::default(),
            block_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
}
