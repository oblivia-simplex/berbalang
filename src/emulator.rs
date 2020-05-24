use std::sync::mpsc::{sync_channel, Receiver, RecvError, SendError};
use std::sync::{Arc, Mutex};
use std::thread::spawn;
use std::time::Duration;

use indexmap::map::IndexMap;
use object_pool::Pool;
use serde_derive::Deserialize;
use std::cell::RefCell;
use std::rc::Rc;
use threadpool::ThreadPool;
use unicorn::{Arch, Cpu, MemHookType, MemType, Mode};

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
    pub registers: Arc<Mutex<IndexMap<Register<'static, C>, u64>>>,
}

impl<C: Cpu<'static>> fmt::Debug for Profiler<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ret_log: {:?}; ", self.ret_log.lock().unwrap())?;
        write!(f, "write_log: {:?}; ", self.write_log.lock().unwrap())?;
        write!(f, "registers: {:?}", self.registers.lock().unwrap())
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
        }
    }
}

type Code = Vec<u8>;
type Register<'a, C> = <C as Cpu<'a>>::Reg;
type Address = u64;
type EmuPrepFn<C> = Box<
    dyn Fn(&mut C, &HatcheryParams, &[u8], Arc<Profiler<C>>) -> Result<Address, unicorn::Error>
        + 'static
        + Send
        + Sync,
>;

mod example {
    use super::*;

    pub fn simple_emu_prep_fn<C: 'static + Cpu<'static>>(
        emu: &mut C,
        _params: &HatcheryParams,
        code: &[u8],
        profiler: Arc<Profiler<C>>,
    ) -> Result<Address, unicorn::Error> {
        let address = 0x1000_u64;
        // let's try adding some hooks
        let profiler = profiler.clone();
        let callback = move |engine, mem_type, address, num_bytes_written, idk| {
            log::error!("Inside memory hook!");
            if let MemType::WRITE = mem_type {
                let mut write_log = profiler.write_log.lock()
                    .expect("Poisoned mutex in callback");
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
pub use example::*;
use std::fmt;

macro_rules! notice {
    ($e:expr) => {
        ($e).map_err(|e| {
            log::error!("Notice! {:?}", e);
            e
        })
    };
}

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
        output_registers: Arc<Vec<Register<'static, C>>>,
        // TODO: we might want to set some callbacks with this function.
        emu_prep_fn: EmuPrepFn<C>,
    ) -> Receiver<Result<(Code, Arc<Profiler<C>>), Error>> {
        let (tx, rx) = sync_channel::<Result<(Code, Arc<Profiler<C>>), Error>>(self.params.num_workers);
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
                    let mut emu = wait_for_emu(&pool, wait_limit, mode);
                    let mut profiler = Arc::new(Profiler::default());
                    log::trace!("executing code {:02x?}", code);
                    // TODO: port the hatchery and memory map code over from ROPER 2
                    // so that this harness will run ROP chains. But make it nice and
                    // generic. How can we parameterize this?
                    // We could have a few functional fields of Hatchery, maybe.
                    // `prepare_emu`, etc.
                    let context = emu.context_save()
                        .expect("Failed to save context");
                    let res = emu_prep_fn(&mut emu, &params, &code, profiler.clone())
                        .and_then(|start_addr| {
                            emu.emu_start(
                                start_addr,
                                0,
                                millisecond_timeout * unicorn::MILLISECOND_SCALE,
                                0,
                            )
                        })
                        .and_then(|()| {
                            {
                                let mut registers = profiler.registers.lock().expect("Poison!");
                                output_registers
                                    .iter()
                                    .for_each(|r| {
                                        let val = emu.reg_read(*r)
                                            .expect("Failed to read register!");
                                        registers.insert(*r, val);
                                    });
                            }
                            Ok(profiler)
                        })
                        .map(|reg| (code, reg))
                        .map_err(Error::from);

                    // cleanup
                    emu.remove_all_hooks().expect("Failed to clean up hooks");
                    emu.context_restore(&context).expect("Failed to restore context");

                    tx.send(res)
                        .map_err(Error::from)
                        .expect("TX Failure in pipeline");
                });
            }
        });
        rx
    }
}


mod util {
    use super::*;
    use unicorn::MemRegion;

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

    use super::*;
    use unicorn::{CpuX86, RegisterX86};

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
            num_workers: 8,
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

        let output_registers = Arc::new(vec![RegisterX86::EAX]);
        hatchery
            .pipeline(
                code_iterator,
                output_registers,
                Box::new(simple_emu_prep_fn),
            )
            .iter()
            .for_each(|out| {
                counter += 1;
                match out {
                    Ok((ref _code, ref regs)) => log::info!("[{}] Output: {:x?}", counter, regs),
                    Err(ref e) => log::info!("[{}] Crash: {:?}", counter, e),
                }
                drop(out);
            });
        assert_eq!(counter, expected_num);
    }
}
