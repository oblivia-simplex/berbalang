use std::cmp::{Ord, PartialOrd};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use indexmap::map::IndexMap;
use indexmap::set::IndexSet;
use prefix_tree::PrefixSet;
use unicorn::Cpu;

use crate::disassembler::Disassembler;
use crate::emulator::loader;
use crate::emulator::register_pattern::{Register, RegisterPattern, UnicornRegisterState};

// TODO: why store the size at all, if you're just going to
// throw it away?
#[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Block {
    pub entry: u64,
    pub size: usize,
    //pub code: Vec<u8>,
}

impl Block {
    pub fn get_code(&self) -> &'static [u8] {
        let memory = loader::get_static_memory_image();
        memory
            .try_dereference(self.entry)
            .map(|b| &b[..self.size])
            .unwrap()
    }

    pub fn disassemble(&self) -> String {
        let memory = loader::get_static_memory_image();
        memory
            .disassemble(self.entry, self.size)
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
    /// The Arc<Mutex<_>> fields need to be writeable for the unicorn callbacks.
    pub block_log: Arc<Mutex<Vec<Block>>>,
    pub gadget_log: Arc<Mutex<Vec<u64>>>,
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

#[derive(Debug, Clone)]
pub struct Profile {
    pub paths: Vec<Vec<Block>>, //PrefixSet<Block>,
    pub cpu_errors: IndexMap<unicorn::Error, usize>,
    pub computation_times: Vec<Duration>,
    pub registers: Vec<RegisterPattern>,
    pub gadgets_executed: IndexSet<u64>,
}

impl Profile {
    pub fn collate<C: 'static + Cpu<'static>>(profilers: Vec<Profiler<C>>) -> Self {
        //let mut write_trie = Trie::new();
        let mut paths = Vec::new(); // PrefixSet::new();
        let mut cpu_errors = IndexMap::new();
        let mut computation_times = Vec::new();
        let mut register_maps = Vec::new();
        let mut gadgets_executed = IndexSet::new();

        for Profiler {
            block_log,
            //write_log,
            cpu_error,
            computation_time,
            registers,
            gadget_log,
            ..
        } in profilers.into_iter()
        {
            // paths.insert::<Vec<Block>>(
            //     (*block_log.lock().unwrap())
            //         .iter()
            //         .map(Clone::clone)
            //         //.map(|b| (b.entry, b.size))
            //         .collect::<Vec<Block>>(),
            // );
            paths.push(block_log.lock().unwrap().to_vec());
            gadget_log.lock().into_iter().for_each(|glog| {
                glog.iter().for_each(|g| {
                    gadgets_executed.insert(*g);
                })
            });
            if let Some(c) = cpu_error {
                *cpu_errors.entry(c).or_insert(0) += 1;
            };
            computation_times.push(computation_time);
            let state: UnicornRegisterState<C> = UnicornRegisterState(registers);
            register_maps.push(state.into());
        }

        Self {
            paths,
            //write_trie,
            cpu_errors,
            computation_times,
            gadgets_executed,
            registers: register_maps,
        }
    }

    pub fn bb_path_iter(&self) -> impl Iterator<Item = &Vec<Block>> + '_ {
        self.paths.iter()
    }

    pub fn disas_paths(&self) -> impl Iterator<Item = String> + '_ {
        let gadgets_executed = self.gadgets_executed.clone();
        self.paths.iter().map(move |path| {
            path.iter()
                .map(|b| {
                    let prefix = if gadgets_executed.contains(&b.entry) {
                        "----\n"
                    } else {
                        ""
                    };
                    format!("{}{}\n", prefix, b.disassemble())
                })
                .collect::<String>()
        })
    }
}

impl<C: 'static + Cpu<'static>> From<Vec<Profiler<C>>> for Profile {
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
            gadget_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[cfg(test)]
mod test {
    use unicorn::CpuX86;

    use super::*;

    #[test]
    fn test_collate() {
        let profilers: Vec<Profiler<CpuX86<'_>>> = vec![
            Profiler {
                block_log: Arc::new(Mutex::new(vec![
                    Block { entry: 1, size: 2 },
                    Block { entry: 3, size: 4 },
                ])),
                cpu_error: None,
                computation_time: Default::default(),
                registers: IndexMap::new(),
                ..Default::default()
            },
            Profiler {
                block_log: Arc::new(Mutex::new(vec![
                    Block { entry: 1, size: 2 },
                    Block { entry: 6, size: 6 },
                ])),
                cpu_error: None,
                computation_time: Default::default(),
                registers: IndexMap::new(),
                ..Default::default()
            },
        ];

        let profile: Profile = profilers.into();

        println!("{:#?}", profile);
        println!(
            "size of profile in mem: {}",
            std::mem::size_of_val(&profile.paths)
        );
    }
}
