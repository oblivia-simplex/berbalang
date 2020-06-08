use std::cmp::{Ord, PartialOrd};
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use capstone::Instructions;
//use indexmap::map::IndexMap;
//use indexmap::set::IndexSet;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use unicorn::Cpu;

use crate::emulator::loader;
use crate::emulator::loader::Seg;
use crate::emulator::register_pattern::{Register, RegisterPattern, UnicornRegisterState};

// TODO: why store the size at all, if you're just going to
// throw it away?
#[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
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

// #[derive(Debug, Clone, Serialize, Deserialize)]
// pub struct Region {
//     pub begin: u64,
//     pub end: u64,
//     pub perm: unicorn::Protection,
//     data: Vec<u8>,
// }
//
// impl Region {
//     pub fn new(mem_reg: unicorn::MemRegion, data: Vec<u8>) -> Self {
//         Self {
//             begin: mem_reg.begin,
//             end: mem_reg.end,
//             perm: mem_reg.perms,
//             data,
//         }
//     }
// }
//
// impl Hash<usize> for Region {
//     type Output = [u8];
//
//     fn index(&self, index: usize) -> &Self::Output {
//         assert!(
//             self.begin <= index as u64,
//             "index cannot be smaller than the first address in the region"
//         );
//         let offset = self.begin - index as u64;
//         assert!(
//             offset < self.end,
//             "index cannot be larger than the last address in the region"
//         );
//         &self.data[offset as usize..]
//     }
// }
// // TODO: fix this. we want something that composes with MemImage.
// // should be in the loader crate.

pub struct Profiler<C: Cpu<'static>> {
    /// The Arc<RwLock<_>> fields need to be writeable for the unicorn callbacks.
    pub block_log: Arc<RwLock<Vec<Block>>>,
    pub gadget_log: Arc<RwLock<Vec<u64>>>,
    /// These fields are written to after the emulation has finished.
    pub written_memory: Vec<Seg>,
    //pub write_log: Arc<Mutex<Vec<MemLogEntry>>>,
    pub cpu_error: Option<unicorn::Error>,
    pub computation_time: Duration,
    pub registers: HashMap<Register<C>, u64>,
    registers_to_read: Vec<Register<C>>,
}

impl<C: Cpu<'static>> Profiler<C> {
    pub fn strong_counts(&self) -> usize {
        Arc::strong_count(&self.block_log)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub paths: Vec<Vec<Block>>, //PrefixSet<Block>,
    pub cpu_errors: HashMap<unicorn::Error, usize>,
    pub computation_times: Vec<Duration>,
    pub registers: Vec<RegisterPattern>,
    pub gadgets_executed: HashSet<u64>,
    pub writeable_memory: Vec<Vec<Seg>>,
}

impl Profile {
    pub fn collate<C: 'static + Cpu<'static>>(profilers: Vec<Profiler<C>>) -> Self {
        //let mut write_trie = Trie::new();
        let mut paths = Vec::new(); // PrefixSet::new();
        let mut cpu_errors = HashMap::new();
        let mut computation_times = Vec::new();
        let mut register_maps = Vec::new();
        let mut gadgets_executed = HashSet::new();
        let mut writeable_memory_regions = Vec::new();

        for Profiler {
            block_log,
            //write_log,
            cpu_error,
            computation_time,
            registers,
            gadget_log,
            written_memory: writeable_memory,
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
            paths.push(block_log.read().unwrap().to_vec());
            gadget_log.read().into_iter().for_each(|glog| {
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
            writeable_memory_regions.push(writeable_memory);
        }

        Self {
            paths,
            //write_trie,
            cpu_errors,
            computation_times,
            gadgets_executed,
            registers: register_maps,
            writeable_memory: writeable_memory_regions,
        }
    }

    pub fn bb_path_iter(&self) -> impl Iterator<Item = &Vec<Block>> + '_ {
        self.paths.iter()
    }

    pub fn disas_paths(&self) -> impl Iterator<Item = String> + '_ {
        let gadgets_executed = self.gadgets_executed.clone();
        self.paths.iter().map(move |path| {
            path.par_iter()
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
        write!(f, "{} blocks", self.block_log.read().unwrap().len())
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct MemLogEntry {
    pub program_counter: u64,
    pub mem_address: u64,
    pub num_bytes_written: usize,
    pub value: i64,
}

impl<C: Cpu<'static>> Default for Profiler<C> {
    fn default() -> Self {
        Self {
            //write_log: Arc::new(RwLock::new(Vec::default())),
            registers: HashMap::default(),
            cpu_error: None,
            registers_to_read: Vec::new(),
            computation_time: Duration::default(),
            block_log: Arc::new(RwLock::new(Vec::new())),
            gadget_log: Arc::new(RwLock::new(Vec::new())),
            written_memory: vec![],
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
                block_log: Arc::new(RwLock::new(vec![
                    Block { entry: 1, size: 2 },
                    Block { entry: 3, size: 4 },
                ])),
                cpu_error: None,
                computation_time: Default::default(),
                registers: HashMap::new(),
                ..Default::default()
            },
            Profiler {
                block_log: Arc::new(RwLock::new(vec![
                    Block { entry: 1, size: 2 },
                    Block { entry: 6, size: 6 },
                ])),
                cpu_error: None,
                computation_time: Default::default(),
                registers: HashMap::new(),
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
