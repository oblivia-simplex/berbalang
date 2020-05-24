use std::fmt;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Receiver, RecvError, SendError, sync_channel};
use std::thread::spawn;
use std::time::Duration;

use indexmap::map::IndexMap;
use object_pool::Pool;
use serde_derive::Deserialize;
use threadpool::ThreadPool;
use unicorn::{Arch, Cpu, MemHookType, MemType, Mode};

type Code = Vec<u8>;
type Register<C> = <C as Cpu<'static>>::Reg;
type Address = u64;
type EmuPrepFn<C> = Box<
    dyn Fn(&mut C, &HatcheryParams, &[u8], &Profiler<C>) -> Result<Address, unicorn::Error>
        + 'static
        + Send
        + Sync,
>;

pub struct Hatchery<C: Cpu<'static> + Send> {
    emu_pool: Arc<Pool<C>>,
    thread_pool: Arc<Mutex<ThreadPool>>,
    params: Arc<HatcheryParams>,
}

const fn default_num_workers() -> usize {
    8
}
const fn default_wait_limit() -> u64 {
    200
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
}

#[derive(Debug)]
pub enum Error {
    Unicorn(String),
    Channel(String),
}

impl From<unicorn::Error> for Error {
    fn from(e: unicorn::Error) -> Error {
        Error::Unicorn(format!("{:?}", e))
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

pub struct Profiler<C: Cpu<'static>> {
    pub ret_log: Arc<Mutex<Vec<Address>>>,
    pub write_log: Arc<Mutex<Vec<WriteLogEntry>>>,
    pub registers: Arc<Mutex<IndexMap<Register<C>, u64>>>,
    registers_to_read: Arc<Vec<Register<C>>>,
    pub cpu_error: Arc<Mutex<Option<unicorn::Error>>>,
}

impl<C: Cpu<'static>> fmt::Debug for Profiler<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ret_log: {:?}; ", self.ret_log.lock().unwrap())?;
        write!(f, "write_log: {:?}; ", self.write_log.lock().unwrap())?;
        write!(f, "registers: {:?}; ", self.registers.lock().unwrap())?;
        write!(f, "cpu_error: {:?}", self.cpu_error.lock().unwrap())
    }
}

impl<C: Cpu<'static>> Profiler<C> {
    fn new(output_registers: Arc<Vec<Register<C>>>) -> Self {
        Self {
            registers_to_read: output_registers,
            ..Default::default()
        }
    }

    pub fn read_registers(&self, emu: &mut C) {
        let mut registers = self.registers.lock().expect("Poison!");
        self.registers_to_read.iter().for_each(|r| {
            let val = emu.reg_read(*r).expect("Failed to read register!");
            registers.insert(*r, val);
        });
    }

    pub fn register(&self, reg: Register<C>) -> Option<u64> {
        self.registers
            .lock()
            .expect("panic getting mutex on Profiler::registers")
            .get(&reg)
            .cloned()
    }

    pub fn set_error(&self, error: unicorn::Error) {
        *(self.cpu_error.lock().unwrap()) = Some(error)
    }
}

#[derive(Debug)]
pub struct WriteLogEntry {
    pub address: u64,
    pub num_bytes_written: usize,
    // might store the actual bytes written here too
}

impl<C: Cpu<'static>> Default for Profiler<C> {
    fn default() -> Self {
        Self {
            ret_log: Arc::new(Mutex::new(Vec::default())),
            write_log: Arc::new(Mutex::new(Vec::default())),
            registers: Arc::new(Mutex::new(IndexMap::default())),
            cpu_error: Arc::new(Mutex::new(None)),
            registers_to_read: Arc::new(Vec::new()),
        }
    }
}

macro_rules! notice {
    ($e:expr) => {
        ($e).map_err(|e| {
            log::error!("Notice! {:?}", e);
            e
        })
    };
}

// TODO: Here is where you'll do the ELF loading.
fn init_emu<C: Cpu<'static>>(mode: unicorn::Mode) -> Result<C, unicorn::Error> {
    let mut emu = notice!(C::new(mode))?;
    notice!(emu.mem_map(0x1000, 0x4000, unicorn::Protection::ALL))?;
    Ok(emu)
}

fn wait_for_emu<C: Cpu<'static>>(
    pool: &Pool<C>,
    wait_limit: u64,
    mode: Mode,
) -> object_pool::Reusable<C> {
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
    pub fn new(params: HatcheryParams) -> Self {
        log::debug!("Creating hatchery with {} workers", params.num_workers);
        let emu_pool = Arc::new(Pool::new(params.num_workers, || {
            init_emu(params.mode).expect("failed to initialize emulator")
        }));
        let thread_pool = Arc::new(Mutex::new(ThreadPool::new(params.num_workers)));
        log::debug!("Pool created");
        Self {
            emu_pool,
            thread_pool,
            params: Arc::new(params),
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
        let _pipe_handler = spawn(move || {
            for code in inbound {
                let emu_prep_fn = emu_prep_fn.clone();
                let params = params.clone();
                let tx = tx.clone();
                let output_registers = output_registers.clone();
                let thread_pool = thread_pool
                    .lock()
                    .expect("Failed to unlock thread_pool mutex");
                let pool = pool.clone();
                thread_pool.execute(move || {
                    // Acquire an emulator from the pool.
                    let mut emu = wait_for_emu(&pool, wait_limit, mode);
                    // Initialize the profiler
                    let profiler = Profiler::new(output_registers.clone());
                    // Save the register context
                    let context = emu.context_save().expect("Failed to save context");

                    // Prepare the emulator with the user-supplied preparation function.
                    // This function will generally be used to load the payload and install
                    // callbacks, which should be able to write to the Profiler instance.
                    let start_addr = emu_prep_fn(&mut emu, &params, &code, &profiler)
                        .expect("Failure in the emulator preparation function.");
                    // If the preparation was successful, launch the emulator and execute
                    // the payload. We want to hang onto the exit code of this task.
                    let result = emu.emu_start(
                        start_addr,
                        0,
                        millisecond_timeout * unicorn::MILLISECOND_SCALE,
                        0,
                    );
                    if let Err(error_code) = result {
                        profiler.set_error(error_code)
                    };
                    profiler.read_registers(&mut emu);

                    // cleanup
                    emu.remove_all_hooks().expect("Failed to clean up hooks");
                    emu.context_restore(&context)
                        .expect("Failed to restore context");

                    // Now send the code back, along with its profile information.
                    // (The genotype, along with its phenotype.)
                    tx.send((code, profiler))
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
    use unicorn::MemRegion;

    use super::*;

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
            None => Err(Error::Unicorn("Couldn't find the stack.".into())),
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
}

#[cfg(test)]
mod test {
    use std::iter;

    use log;
    use unicorn::{CpuX86, RegisterX86};

    pub use example::*;

    use super::*;

    mod example {
        use super::*;

        pub fn simple_emu_prep_fn<C: 'static + Cpu<'static>>(
            emu: &mut C,
            _params: &HatcheryParams,
            code: &[u8],
            profiler: &Profiler<C>,
        ) -> Result<Address, unicorn::Error> {
            let address = 0x1000_u64;
            // let's try adding some hooks
            let write_log = profiler.write_log.clone();
            let callback = move |_engine, mem_type, address, num_bytes_written, _idk| {
                log::error!("Inside memory hook!");
                if let MemType::WRITE = mem_type {
                    let mut write_log = write_log.lock().expect("Poisoned mutex in callback");
                    let entry = WriteLogEntry {
                        address,
                        num_bytes_written,
                    };
                    write_log.push(entry);
                    true // NOTE: I'm not really sure what this return value means, here.
                } else {
                    false
                }
            };
            for region in emu.mem_regions()? {
                if region.writeable() {
                    let _hook = emu.add_mem_hook(
                        MemHookType::MEM_WRITE,
                        region.begin,
                        region.end,
                        callback.clone(),
                    )?;
                }
            }

            // now write the payload
            emu.mem_write(address, code).map(|()| 0x1000_u64)?; // TODO in the ROP case, this will be the address of the first gadget
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
            }
        );
    }

    #[test]
    fn test_hatchery() {
        pretty_env_logger::env_logger::init();
        let params = HatcheryParams {
            num_workers: 32,
            wait_limit: 150,
            mode: unicorn::Mode::MODE_64,
            arch: unicorn::Arch::X86,
            millisecond_timeout: Some(100),
        };
        fn random_code() -> Vec<u8> {
            iter::repeat(())
                .take(1024)
                .map(|()| rand::random::<u8>())
                .collect()
        }

        let hatchery: Hatchery<CpuX86> = Hatchery::new(params);

        let expected_num = 0x100000;
        let mut counter = 0;
        let code_iterator = iter::repeat(()).take(expected_num).map(|()| random_code());

        use RegisterX86::*;
        let output_registers = Arc::new(vec![EAX, ESP, EIP, EBP, EBX, ECX, EFLAGS]);
        hatchery
            .pipeline(
                code_iterator,
                output_registers,
                Box::new(simple_emu_prep_fn),
            )
            .iter()
            .for_each(|(code, profile)| {
                counter += 1;
                log::info!("[{}] Output: {:x?}", counter, profile);
            });
        assert_eq!(counter, expected_num);
    }
}
