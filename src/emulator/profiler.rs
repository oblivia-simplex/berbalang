use crate::emulator::executor::Register;
use indexmap::map::IndexMap;
use prefix_tree::PrefixSet;
use std::cmp::{Ord, PartialOrd};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use unicorn::Cpu;

// TODO: why store the size at all, if you're just going to
// throw it away?
#[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Block {
    pub entry: u64,
    pub size: usize,
    //pub code: Vec<u8>,
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

fn convert_register_map<C: Cpu<'static>>(
    registers: IndexMap<Register<C>, u64>,
) -> IndexMap<String, u64> {
    let mut map = IndexMap::new();
    for (k, v) in registers.into_iter() {
        map.insert(format!("{:?}", k), v); // FIXME use stable conversion method
    }
    map
}

#[derive(Debug)]
pub struct Profile {
    pub paths: PrefixSet<Block>,
    pub cpu_errors: IndexMap<unicorn::Error, usize>,
    pub computation_times: Vec<Duration>,
    pub registers: Vec<IndexMap<String, u64>>,
}

impl Profile {
    pub fn collate<C: Cpu<'static>>(profilers: Vec<Profiler<C>>) -> Self {
        //let mut write_trie = Trie::new();
        let mut paths = PrefixSet::new();
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
            paths.insert::<Vec<Block>>(
                (*block_log.lock().unwrap())
                    .iter()
                    .map(Clone::clone)
                    //.map(|b| (b.entry, b.size))
                    .collect::<Vec<Block>>(),
            );
            //write_trie.insert((*write_log.lock().unwrap()).clone(), ());
            if let Some(c) = cpu_error {
                *cpu_errors.entry(c).or_insert(0) += 1;
            };
            computation_times.push(computation_time);
            register_maps.push(convert_register_map::<C>(registers));
        }

        Self {
            paths,
            //write_trie,
            cpu_errors,
            computation_times,
            registers: register_maps,
        }
    }
}

impl<C: Cpu<'static>> From<Vec<Profiler<C>>> for Profile {
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

#[cfg(test)]
mod test {

    use super::*;
    use unicorn::CpuX86;

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
                registers_to_read: vec![],
            },
            Profiler {
                block_log: Arc::new(Mutex::new(vec![
                    Block { entry: 1, size: 2 },
                    Block { entry: 6, size: 6 },
                ])),
                cpu_error: None,
                computation_time: Default::default(),
                registers: IndexMap::new(),
                registers_to_read: vec![],
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
