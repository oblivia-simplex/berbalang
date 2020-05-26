use std::fmt;
use std::sync::mpsc::{sync_channel, Receiver, RecvError, SendError};
use std::sync::{Arc, Mutex};
use std::thread::spawn;
use std::time::{Duration, Instant};

use crate::emulator::loader;
use crate::emulator::loader::Seg;
use indexmap::map::IndexMap;
use object_pool::Pool;
use serde_derive::Deserialize;
use std::pin::Pin;
use threadpool::ThreadPool;
use unicorn::{Arch, Cpu, Mode};

type Code = Vec<u8>;
type Register<C> = <C as Cpu<'static>>::Reg;
type Address = u64;
type EmuPrepFn<C> = Box<
    dyn Fn(&mut C, &HatcheryParams, &[u8], &Profiler<C>) -> Result<Address, Error>
        + 'static
        + Send
        + Sync,
>;

pub struct Hatchery<C: Cpu<'static> + Send> {
    emu_pool: Arc<Pool<C>>,
    thread_pool: Arc<Mutex<ThreadPool>>,
    params: Arc<HatcheryParams>,
    memory: Arc<Option<Pin<Vec<Seg>>>>,
}

impl<C: Cpu<'static> + Send> Drop for Hatchery<C> {
    fn drop(&mut self) {
        // unmap the unwriteable memory in the emu pool's emus
        log::debug!("Dropping Hatchery");
        if let Some(segments) = self.memory.as_ref() {
            // Once a shared, mapped region is unmapped from one emulator, it's unmapped
            // from them all. Attempting to unmap it again will trigger a NOMEM error.
            // And I think that attempting to access that unmapped segment *may* trigger a
            // use-after-free bug.
            if let Some(mut emu) = self.emu_pool.try_pull() {
                segments.iter()
                    .filter(|&s| !s.is_writeable())
                    .for_each(|s| {
                        log::debug!("Unmapping region 0x{:x} - 0x{:x} [{:?}]", s.aligned_start(), s.aligned_end(), s.perm);
                        //log::debug!("Unmapping segment at 0x{:x}", s.aligned_start());
                        emu.mem_unmap(s.aligned_start(), s.aligned_size())
                            .unwrap_or_else(|e| log::error!("Failed to unmap segment: {:?}", e));
                    });

            }
        }
    }
}

const fn default_num_workers() -> usize {
    8
}
const fn default_wait_limit() -> u64 {
    200
}

const fn default_stack_size() -> usize {
    0x1000
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct HatcheryParams {
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,
    #[serde(default = "default_wait_limit")]
    pub wait_limit: u64,
    pub arch: Arch,
    pub mode: Mode,
    pub millisecond_timeout: Option<u64>,
    #[serde(default = "Default::default")]
    pub record_basic_blocks: bool,
    #[serde(default = "Default::default")]
    pub record_memory_writes: bool,
    #[serde(default = "default_stack_size")]
    pub stack_size: usize,
    pub binary_path: Option<String>,
}

#[derive(Debug)]
pub enum Error {
    Unicorn(unicorn::Error),
    Channel(String),
    Loader(loader::Error),
}

impl From<loader::Error> for Error {
    fn from(e: loader::Error) -> Error {
        Error::Loader(e)
    }
}

impl From<unicorn::Error> for Error {
    fn from(e: unicorn::Error) -> Error {
        Error::Unicorn(e)
    }
}

impl<T> From<SendError<T>> for Error {
    fn from(e: SendError<T>) -> Error {
        Error::Channel(format!("SendError: {:?}", e))
    }
}

impl From<RecvError> for Error {
    fn from(e: RecvError) -> Error {
        Error::Channel(format!("RecvError: {:?}", e))
    }
}

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
        write!(f, "write_log: {} entries; ", self.write_log.lock().unwrap().len())?;
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
    fn new(output_registers: Arc<Vec<Register<C>>>) -> Self {
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

// macro_rules! notice {
//     ($e:expr) => {
//         ($e).map_err(|e| {
//             log::error!("Notice! {:?}", e);
//             e
//         })
//     };
// }

// TODO: Here is where you'll do the ELF loading.
fn init_emu<C: Cpu<'static>>(
    params: &HatcheryParams,
    memory: &Option<Pin<Vec<Seg>>>,
) -> Result<C, Error> {
    let mut emu = C::new(params.mode)?;
    if let Some(segments) = memory {
        //notice!(emu.mem_map(0x1000, 0x4000, unicorn::Protection::ALL))?;
        let mut results = Vec::new();
        // First, map the non-writeable segments to memory. These can be shared.
        segments.iter().for_each(|s| {
            log::info!("Mapping segment 0x{:x} - 0x{:x} [{:?}]", s.aligned_start(), s.aligned_end(), s.perm);
            if !s.is_writeable() {
                // This is a bit risky, but we want our many emulator instances to share common regions
                // of non-writeable memory.
                unsafe {
                    let res = emu.mem_map_const_ptr(
                        s.aligned_start(),
                        s.aligned_size(),
                        s.perm,
                        s.data.as_ptr(),
                    );
                    results.push(res);
                }
            } else {
                // Next, map the writeable segments
                let res = emu.mem_map(s.aligned_start(), s.aligned_size(), s.perm);
                results.push(res);
            }
        });
        // Return an error if there's been an error.
        let _ = results
            .into_iter()
            .collect::<Result<Vec<_>, unicorn::Error>>()?;
    };
    emu.mem_regions()?.iter().for_each(|rgn| {
       log::info!("Mapped region: 0x{:x} - 0x{:x} [{:?}]", rgn.begin, rgn.end, rgn.perms);
    });
    Ok(emu)
}

fn wait_for_emu<C: Cpu<'static>>(
    pool: &Pool<C>,
    wait_limit: u64,
    mode: Mode,
) -> object_pool::Reusable<'_, C> {
    let mut wait_time = 0;
    let wait_unit = 1;
    loop {
        if let Some(c) = pool.try_pull() {
            if wait_time > 0 {
                log::warn!("Waited {} milliseconds for CPU", wait_time);
            }
            return c;
        } else if wait_time > wait_limit {
            log::warn!(
                "Waited {} milliseconds for CPU, creating new one",
                wait_time
            );
            return pool.pull(|| C::new(mode).expect("Failed to spawn replacement CPU"));
        }
        {
            std::thread::sleep(Duration::from_millis(wait_unit));
            wait_time += wait_unit;
        }
    }
}

impl<C: 'static + Cpu<'static> + Send> Hatchery<C> {
    pub fn new(params: Arc<HatcheryParams>) -> Self {
        log::debug!("Creating hatchery with {} workers", params.num_workers);
        let memory = if let Some(path) = params.binary_path.as_ref() {
            Arc::new(Some(Pin::new(
                loader::load_from_path(path, params.stack_size)
                    .expect("Failed to load binary from path"),
            )))
        } else {
            Default::default()
        };
        let emu_pool = Arc::new(Pool::new(params.num_workers, || {
            init_emu(&params, &memory).expect("failed to initialize emulator")
        }));
        let thread_pool = Arc::new(Mutex::new(ThreadPool::new(params.num_workers)));
        log::debug!("Pool created");
        Self {
            emu_pool,
            thread_pool,
            params,
            memory,
        }
    }

    /// Example.
    /// Adapting this method for a ROP executor would involve a few changes.
    pub fn pipeline<I: 'static + Iterator<Item = Code> + Send>(
        &self,
        inbound: I,
        output_registers: Arc<Vec<Register<C>>>,
        // TODO: we might want to set some callbacks with this function.
        emu_prep_fn: EmuPrepFn<C>,
    ) -> Receiver<(Code, Profiler<C>)> {
        let (tx, rx) = sync_channel::<(Code, Profiler<C>)>(self.params.num_workers);
        let thread_pool = self.thread_pool.clone();
        let mode = self.params.mode;
        let pool = self.emu_pool.clone();
        let wait_limit = self.params.wait_limit;
        let emu_prep_fn = Arc::new(emu_prep_fn);
        let params = self.params.clone();
        let millisecond_timeout = self.params.millisecond_timeout.unwrap_or(0);
        let memory = self.memory.clone();
        let _pipe_handler = spawn(move || {
            for payload in inbound {
                let emu_prep_fn = emu_prep_fn.clone();
                let params = params.clone();
                let tx = tx.clone();
                let output_registers = output_registers.clone();
                let thread_pool = thread_pool
                    .lock()
                    .expect("Failed to unlock thread_pool mutex");
                let pool = pool.clone();
                let memory = memory.clone();
                thread_pool.execute(move || {
                    // Acquire an emulator from the pool.
                    let mut emu = wait_for_emu(&pool, wait_limit, mode);
                    // Initialize the profiler
                    let mut profiler = Profiler::new(output_registers.clone());
                    // Save the register context
                    let context = emu.context_save().expect("Failed to save context");

                    if params.record_basic_blocks {
                        let _hooks = util::install_basic_block_hook(&mut (*emu), &profiler)
                            .expect("Failed to install basic_block_hook");
                    }

                    if params.record_memory_writes {
                        let _hooks = util::install_mem_write_hook(&mut (*emu), &profiler)
                            .expect("Failed to install mem_write_hook");
                    }
                    // Prepare the emulator with the user-supplied preparation function.
                    // This function will generally be used to load the payload and install
                    // callbacks, which should be able to write to the Profiler instance.
                    let start_addr = emu_prep_fn(&mut emu, &params, &payload, &profiler)
                        .expect("Failure in the emulator preparation function.");
                    // If the preparation was successful, launch the emulator and execute
                    // the payload. We want to hang onto the exit code of this task.
                    let start_time = Instant::now();
                    let result = emu.emu_start(
                        start_addr,
                        0,
                        millisecond_timeout * unicorn::MILLISECOND_SCALE,
                        0,
                    );
                    profiler.computation_time = start_time.elapsed();
                    if let Err(error_code) = result {
                        profiler.set_error(error_code)
                    };
                    profiler.read_registers(&mut emu);

                    // cleanup
                    emu.remove_all_hooks().expect("Failed to clean up hooks");
                    emu.context_restore(&context)
                        .expect("Failed to restore context");
                    // clean up writeable memory
                    if let Some(memory) = memory.as_ref() {
                        memory.iter().filter(|s| s.is_writeable()).for_each(|seg| {
                            emu.mem_write(seg.aligned_start(),
                                &seg.data
                            ).unwrap_or_else(|e| {
                                log::error!("Failed to refresh writeable memory at 0x{:x} - 0x{:x}",
                                    seg.aligned_start(), seg.aligned_end()
                                )
                            });
                        });
                    }
                    // Now send the code back, along with its profile information.
                    // (The genotype, along with its phenotype.)
                    tx.send((payload, profiler))
                        .map_err(Error::from)
                        .expect("TX Failure in pipeline");
                });
            }
        });
        rx
    }
}
// TODO: try to reduce the number of mutexes needed in this setup. it seems like a code smell.

mod util {
    use unicorn::{CodeHookType, MemHookType, MemRegion, MemType, Protection, Unicorn};

    use super::*;

    pub fn install_basic_block_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        profiler: &Profiler<C>,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
        let block_log = profiler.block_log.clone();
        let bb_callback = move |engine: &unicorn::Unicorn<'_>, entry: u64, size: u32| {
            let size = size as usize;
            let code = match engine.mem_read_as_vec(entry, size) {
                Ok(xs) => xs,
                Err(e) => {
                    log::error!(
                        "Failed to read basic block from bb_callback at 0x{:x}: {:?}",
                        entry,
                        e
                    );
                    return;
                }
            };
            let block = Block { entry, size, code };
            block_log
                .lock()
                .expect("Poisoned mutex in bb_callback")
                .push(block);
        };

        let hooks = code_hook_all(emu, CodeHookType::BLOCK, bb_callback)?;

        Ok(hooks)
    }

    pub fn install_mem_write_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        profiler: &Profiler<C>,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
        let pc: i32 = emu.program_counter().into();
        let write_log = profiler.write_log.clone();
        let mem_write_callback =
            move |engine: &Unicorn<'_>, mem_type, address, num_bytes_written, value| {
                //log::trace!("Inside memory hook!");
                if let MemType::WRITE = mem_type {
                    let program_counter = engine.reg_read(pc).expect("Failed to read PC register");
                    let entry = MemLogEntry {
                        program_counter,
                        mem_address: address,
                        num_bytes_written,
                        value,
                    };
                    let mut write_log = write_log.lock().expect("Poisoned mutex in callback");
                    write_log.push(entry);
                    true // NOTE: I'm not really sure what this return value means, here.
                } else {
                    false
                }
            };

        let hooks = util::mem_hook_by_prot(
            emu,
            MemHookType::MEM_WRITE,
            Protection::WRITE,
            mem_write_callback,
        )?;

        Ok(hooks)
    }

    /// Returns the uppermost readable/writeable memory region, in the emulator's
    /// memory map.
    pub fn find_stack<C: 'static + Cpu<'static>>(emu: &C) -> Result<MemRegion, Error> {
        let regions = emu.mem_regions()?;
        let mut bottom = 0;
        let mut stack = None;
        for region in regions.iter() {
            if region.writeable() && region.readable() && region.begin >= bottom {
                bottom = region.begin;
                stack = Some(region)
            };
        }
        match stack {
            Some(m) => Ok(m.clone()),
            None => unimplemented!("do this later"),
        }
    }

    // Reads all memory that carries a Protection::WRITE permission.
    // This can be used, e.g., to check to see what a specimen has written
    // to memory.
    pub fn read_writeable_memory<C: 'static + Cpu<'static>>(
        emu: &C,
    ) -> Result<Vec<Vec<u8>>, Error> {
        emu.mem_regions()?
            .into_iter()
            .filter(MemRegion::writeable)
            .map(|m| emu.mem_read_as_vec(m.begin, m.size()))
            .collect::<Result<Vec<Vec<u8>>, unicorn::Error>>()
            .map_err(Error::from)
    }

    /// Add a memory hook wherever the specified protections are satisfied.
    ///
    /// The callback takes four arguments:
    /// - a pointer to the unicorn engine
    /// - the memory address accessed
    /// - the number of bytes written or read
    /// - the value
    pub fn mem_hook_by_prot<F, C: 'static + Cpu<'static>>(
        emu: &mut C,
        hook_type: MemHookType,
        protections: Protection,
        callback: F,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error>
    where
        F: 'static
            + FnMut(&'static unicorn::Unicorn<'static>, MemType, u64, usize, i64) -> bool
            + Clone,
    {
        let mut hooks = Vec::new();
        for region in emu.mem_regions()? {
            if region.perms.intersects(protections) {
                let hook =
                    emu.add_mem_hook(hook_type, region.begin, region.end, callback.clone())?;
                hooks.push(hook);
            }
        }
        Ok(hooks)
    }

    /// Add a code hook on every executable region of memory.
    ///
    /// The callback takes three arguments:
    /// - a pointer to the unicorn engine
    /// - the address of the current instruction or block
    /// - the size of the current instruction (or block?)
    pub fn code_hook_all<F, C: 'static + Cpu<'static>>(
        emu: &mut C,
        hook_type: CodeHookType,
        callback: F,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error>
    where
        F: 'static + FnMut(&'static unicorn::Unicorn<'static>, u64, u32) -> () + Clone,
    {
        let mut hooks = Vec::new();
        for region in emu.mem_regions()? {
            if region.executable() {
                let hook =
                    emu.add_code_hook(hook_type, region.begin, region.end, callback.clone())?;
                hooks.push(hook)
            }
        }
        Ok(hooks)
    }

    pub fn word_size(arch: Arch, mode: Mode) -> usize {
        use Arch::*;
        use Mode::*;

        match (arch, mode) {
            (ARM, THUMB) => 2,
            (ARM, _) => 4,
            (ARM64, THUMB) => 2,
            (ARM64, _) => 8,
            (MIPS, _) => 4, // check
            (X86, MODE_16) => 2,
            (X86, MODE_32) => 4,
            (X86, MODE_64) => 8,
            (PPC, MODE_64) => 8,
            (PPC, _) => 4,
            (SPARC, _) => 4, // check
            (M68K, _) => 2, // check
            (_, _) => unimplemented!("invalid arch/mode combination"),
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Endian {
        Big,
        Little,
    }

    pub fn endian(arch: Arch, mode: Mode) -> Endian {
        use Arch::*;
        use Mode::*;
        use Endian::*;

        match (arch, mode) {
            (ARM, _) => Big,     // this can actually go both ways, check unicorn
            (ARM64, _) => Big,
            (MIPS, _) => Big, // check
            (X86, _) => Little,
            (PPC, _) => Big,
            (SPARC, _) => Big, // check
            (M68K, _) => Big, // check
            (_, _) => unimplemented!("invalid arch/mode combination"),
        }
    }
}

#[cfg(test)]
mod test {
    use std::iter;

    use unicorn::{CpuX86, MemHookType, MemType, RegisterX86};

    use example::*;

    use byteorder::{BigEndian, LittleEndian, ByteOrder};
    use super::*;
    use rand::{thread_rng, Rng};

    mod example {
        use super::*;
        use unicorn::{Protection, Unicorn};
        use byteorder::{BigEndian, LittleEndian, ByteOrder};
        use crate::emulator::executor::util::Endian;

        pub fn simple_emu_prep_fn<C: 'static + Cpu<'static>>(
            emu: &mut C,
            _params: &HatcheryParams,
            code: &[u8],
            profiler: &Profiler<C>,
        ) -> Result<Address, Error> {
            let address = 0x41_b000;
            // let's try adding some hooks

            // now write the payload
            let stack = util::find_stack(emu)?;
            let sp = stack.begin + (stack.end - stack.begin)/2;
            emu.mem_write(sp, code)?;
            // set the stack pointer to the middle of the stack
            // now "pop" the stack into the program counter
            let word_size = util::word_size(emu.arch(), emu.mode());
            let a_bytes = emu.mem_read_as_vec(sp, word_size)?;
            let address =
                match util::endian(emu.arch(), emu.mode()) {
                Endian::Big => {
                    BigEndian::read_u64(&a_bytes)
                }
                Endian::Little => {
                    LittleEndian::read_u64(&a_bytes)
                }
            };
            emu.write_stack_pointer(sp + word_size as u64)?;


            Ok(address)
        }
    }

    #[test]
    fn test_params() {
        let config = r#"
            num_workers = 8
            wait_limit = 150
            mode = "MODE_64"
            arch = "X86"
            record_basic_blocks = true
            record_memory_writes = true
        "#;

        let params: HatcheryParams = toml::from_str(config).unwrap();
        assert_eq!(
            params,
            HatcheryParams {
                num_workers: 8,
                wait_limit: 150,
                mode: unicorn::Mode::MODE_64,
                arch: unicorn::Arch::X86,
                millisecond_timeout: None,
                record_basic_blocks: true,
                record_memory_writes: true,
                stack_size: 0x1000, // default
                binary_path: None,
            }
        );
    }

    #[test]
    fn test_hatchery() {
        pretty_env_logger::env_logger::init();
        let params = HatcheryParams {
            num_workers: 128,
            wait_limit: 150,
            mode: unicorn::Mode::MODE_64,
            arch: unicorn::Arch::X86,
            millisecond_timeout: Some(100),
            record_basic_blocks: true,
            record_memory_writes: true,
            stack_size: 0x1000,
            binary_path: Some("/bin/sh".to_string()),
        };
        fn random_rop() -> Vec<u8> {
            let mut rng = thread_rng();
            let addresses: Vec<u64> = iter::repeat(())
                .take(100)
                .map(|()| rng.gen_range(0x41_b000_u64, 0x4a_5fff_u64))
                .collect();
            let mut packed = vec![0_u8; addresses.len() * 8];
            LittleEndian::write_u64_into(&addresses, &mut packed);
            packed
        }

        let hatchery: Hatchery<CpuX86<'_>> = Hatchery::new(Arc::new(params));

        let expected_num = 0x1000;
        let mut counter = 0;
        let code_iterator = iter::repeat(()).take(expected_num).map(|()| random_rop());

        use RegisterX86::*;
        let output_registers = Arc::new(vec![RAX, RSP, RIP, RBP, RBX, RCX, EFLAGS]);
        hatchery
            .pipeline(
                code_iterator,
                output_registers,
                Box::new(simple_emu_prep_fn),
            )
            .iter()
            .for_each(|(_code, profile)| {
                counter += 1;
                log::info!("[{}] Output: {:x?}", counter, profile);
            });
        assert_eq!(counter, expected_num);
    }
}
