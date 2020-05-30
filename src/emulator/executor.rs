use std::sync::mpsc::{sync_channel, Receiver, RecvError, SendError};
use std::sync::{Arc, Mutex};
use std::thread::spawn;
use std::time::{Duration, Instant};
//use std::sync::atomic::{AtomicUsize, Ordering};

use crate::emulator::loader;
use crate::emulator::loader::Seg;
use crate::emulator::profiler::{Profile, Profiler};

use object_pool::Pool;
use serde_derive::Deserialize;
use std::pin::Pin;
use threadpool::ThreadPool;
//use rayon::{ThreadPoolBuilder, };
use indexmap::map::IndexMap;
use rayon::prelude::*;
use unicorn::{Arch, Cpu, Mode};

type Code = Vec<u8>;
pub type Register<C> = <C as Cpu<'static>>::Reg;
pub type Address = u64;
pub type EmuPrepFn<C> = Box<
    dyn Fn(&mut C, &HatcheryParams, &[u8], &Profiler<C>) -> Result<Address, Error>
        + 'static
        + Send
        + Sync,
>;

// static TRACING_THREAD: AtomicUsize = AtomicUsize::new(0);
//
// fn trace_enabled() -> bool {
//     let thread: usize = std::thread::current().id().as_u64() as usize;
//     TRACING_THREAD.compare_and_swap(0, thread, Ordering::Relaxed) == thread
// }

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
                segments
                    .iter()
                    .filter(|&s| !s.is_writeable())
                    .for_each(|s| {
                        log::debug!(
                            "Unmapping region 0x{:x} - 0x{:x} [{:?}]",
                            s.aligned_start(),
                            s.aligned_end(),
                            s.perm
                        );
                        //log::debug!("Unmapping segment at 0x{:x}", s.aligned_start());
                        emu.mem_unmap(s.aligned_start(), s.aligned_size())
                            .unwrap_or_else(|e| log::error!("Failed to unmap segment: {:?}", e));
                    });
            }
        }
    }
}

pub use crate::configure::RoperConfig as HatcheryParams;
use crate::emulator::pack::Pack;

// #[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
// pub struct HatcheryParams {
//     #[serde(default = "default_num_workers")]
//     pub num_workers: usize,
//     #[serde(default = "default_num_workers")]
//     pub num_emulators: usize,
//     #[serde(default = "default_wait_limit")]
//     pub wait_limit: u64,
//     pub arch: Arch,
//     pub mode: Mode,
//     pub max_emu_steps: Option<usize>,
//     pub millisecond_timeout: Option<u64>,
//     #[serde(default = "Default::default")]
//     pub record_basic_blocks: bool,
//     #[serde(default = "Default::default")]
//     pub record_memory_writes: bool,
//     #[serde(default = "default_stack_size")]
//     pub emulator_stack_size: usize,
//     pub binary_path: Option<String>,
// }

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
            log::info!(
                "Mapping segment 0x{:x} - 0x{:x} [{:?}]",
                s.aligned_start(),
                s.aligned_end(),
                s.perm
            );
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
        log::info!(
            "Mapped region: 0x{:x} - 0x{:x} [{:?}]",
            rgn.begin,
            rgn.end,
            rgn.perms
        );
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

// pub trait Hatch {
//     fn new(params: Arc<HatcheryParams>) -> Self
//     where
//         Self: Sized;
//
//     fn pipeline<C: 'static + Cpu<'static>, I: 'static + Iterator<Item = Code> + Send>(
//         &self,
//         inbound: I,
//         inputs: Arc<Vec<IndexMap<Register<C>, u64>>>,
//         output_registers: Arc<Vec<Register<C>>>,
//         // TODO: we might want to set some callbacks with this function.
//         emu_prep_fn: EmuPrepFn<C>,
//     ) -> Receiver<(Code, Profile<C>)>;
// }

impl<C: 'static + Cpu<'static> + Send> Hatchery<C> {
    pub fn new(params: Arc<HatcheryParams>) -> Self {
        log::debug!("Creating hatchery with {} workers", params.num_workers);
        let memory = if let Some(path) = params.binary_path.as_ref() {
            Arc::new(Some(Pin::new(
                loader::load_from_path(path, params.emulator_stack_size)
                    .expect("Failed to load binary from path"),
            )))
        } else {
            Default::default()
        };
        let emu_pool = Arc::new(Pool::new(params.num_workers, || {
            init_emu(&params, &memory).expect("failed to initialize emulator")
        }));
        let thread_pool = Arc::new(Mutex::new(ThreadPool::new(params.num_workers)));
        // let _thread_pool = ThreadPoolBuilder::new()
        //     .num_threads(params.num_workers)
        //     //.stack_size(0x1000_000)
        //     .build_global();
        // .map(Mutex::new)
        // .map(Arc::new)
        // .expect("Failed to build thread pool");
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
    pub fn pipeline<X: 'static + Pack + Send + Sync, I: 'static + Iterator<Item = X> + Send>(
        &self,
        inbound: I,
        inputs: Arc<Vec<IndexMap<Register<C>, u64>>>,
        output_registers: Arc<Vec<Register<C>>>,
        // TODO: we might want to set some callbacks with this function.
        emu_prep_fn: Option<EmuPrepFn<C>>,
    ) -> Receiver<(X, Profile)> {
        let (tx, rx) = sync_channel::<(X, Profile)>(self.params.num_workers);
        //let thread_pool = self.thread_pool.clone();
        let emu_prep_fn = emu_prep_fn.unwrap_or_else(|| Box::new(util::emu_prep_fn));
        let mode = self.params.mode;
        let pool = self.emu_pool.clone();
        let wait_limit = self.params.wait_limit;
        let emu_prep_fn = Arc::new(emu_prep_fn);
        let params = self.params.clone();
        let millisecond_timeout = self.params.millisecond_timeout.unwrap_or(0);
        let max_emu_steps = self.params.max_emu_steps.unwrap_or(0);
        let memory = self.memory.clone();
        let thread_pool = self.thread_pool.clone();
        let word_size = crate::util::architecture::word_size(params.arch, params.mode);
        let endian = crate::util::architecture::endian(params.arch, params.mode);
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
                let inputs = inputs.clone();
                thread_pool.execute(move || {
                    let profile = inputs.par_iter().map(|input| {
                        // Acquire an emulator from the pool.
                        let mut emu = wait_for_emu(&pool, wait_limit, mode);
                        // Initialize the profiler
                        let mut profiler = Profiler::new(&output_registers);
                        // Save the register context
                        let context = emu.context_save().expect("Failed to save context");

                        if params.record_basic_blocks {
                            let _hooks = util::install_basic_block_hook(&mut (*emu), &profiler)
                                .expect("Failed to install basic_block_hook");
                        }

                        // if params.record_memory_writes {
                        //     let _hooks = util::install_mem_write_hook(&mut (*emu), &profiler)
                        //         .expect("Failed to install mem_write_hook");
                        // }
                        // Prepare the emulator with the user-supplied preparation function.
                        // This function will generally be used to load the payload and install
                        // callbacks, which should be able to write to the Profiler instance.
                        let code = payload.pack(word_size, endian);
                        let start_addr = emu_prep_fn(&mut emu, &params, &code, &profiler)
                            .expect("Failure in the emulator preparation function.");
                        // load the inputs
                        // TODO refactor into separate method
                        for (reg,val) in input.iter() {
                            emu.reg_write(*reg, *val)
                                .expect("Failed to load registers");
                        }
                        // If the preparation was successful, launch the emulator and execute
                        // the payload. We want to hang onto the exit code of this task.
                        let start_time = Instant::now();
                        let result = emu.emu_start(
                            start_addr,
                            0,
                            millisecond_timeout * unicorn::MILLISECOND_SCALE,
                            max_emu_steps,
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
                                    log::error!("Failed to refresh writeable memory at 0x{:x} - 0x{:x}: {:?}",
                                    seg.aligned_start(), seg.aligned_end(), e
                                )
                                });
                            });
                        }
                        profiler

                    }).collect::<Vec<Profiler<C>>>().into();
                    // Now send the code back, along with its profile information.
                    // (The genotype, along with its phenotype.)
                    tx.send((payload, profile))
                        .map_err(Error::from)
                        .expect("TX Failure in pipeline");
                });
            }
        });
        // println!("joining pipe_handler");
        // pipe_handler.join().unwrap();
        // println!("joined pipe_handler");
        rx
    }
}
// TODO: try to reduce the number of mutexes needed in this setup. it seems like a code smell.

pub mod util {
    use unicorn::{CodeHookType, MemHookType, MemRegion, MemType, Protection};

    use super::*;
    use crate::emulator::profiler::Block;
    use crate::util::architecture::{endian, word_size, Endian};
    use byteorder::{BigEndian, ByteOrder, LittleEndian};

    pub fn emu_prep_fn<C: 'static + Cpu<'static>>(
        emu: &mut C,
        _params: &HatcheryParams,
        code: &[u8],
        _profiler: &Profiler<C>,
    ) -> Result<u64, Error> {
        // now write the payload
        let stack = find_stack(emu)?;
        let sp = stack.begin + (stack.end - stack.begin) / 2;
        emu.mem_write(sp, code)?;
        // set the stack pointer to the middle of the stack
        // now "pop" the stack into the program counter
        let word_size = word_size(emu.arch(), emu.mode());
        let a_bytes = emu.mem_read_as_vec(sp, word_size)?;
        let address = match endian(emu.arch(), emu.mode()) {
            Endian::Big => BigEndian::read_u64(&a_bytes),
            Endian::Little => LittleEndian::read_u64(&a_bytes),
        };
        emu.write_stack_pointer(sp + word_size as u64)?;

        Ok(address)
    }

    pub fn install_basic_block_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        profiler: &Profiler<C>,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
        let block_log = profiler.block_log.clone();
        //let ret_log = profiler.ret_log.clone();
        let bb_callback = move |_engine: &unicorn::Unicorn<'_>, entry: u64, size: u32| {
            let size = size as usize;

            // If the code ends with a return, log it in the ret log.
            // but how to make this platform-generic? We could define a
            // return_insn method on the trait, like we did for the special
            // registers. TODO: low priority
            let block = Block { entry, size };
            block_log
                .lock()
                .expect("Poisoned mutex in bb_callback")
                .push(block);
        };

        let hooks = code_hook_all(emu, CodeHookType::BLOCK, bb_callback)?;

        Ok(hooks)
    }

    // pub fn install_mem_write_hook<C: 'static + Cpu<'static>>(
    //     emu: &mut C,
    //     profiler: &Profiler<C>,
    // ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
    //     let pc: i32 = emu.program_counter().into();
    //     let write_log = profiler.write_log.clone();
    //     let mem_write_callback =
    //         move |engine: &Unicorn<'_>, mem_type, address, num_bytes_written, value| {
    //             //log::trace!("Inside memory hook!");
    //             if let MemType::WRITE = mem_type {
    //                 let program_counter = engine.reg_read(pc).expect("Failed to read PC register");
    //                 let entry = MemLogEntry {
    //                     program_counter,
    //                     mem_address: address,
    //                     num_bytes_written,
    //                     value,
    //                 };
    //                 let mut write_log = write_log.lock().expect("Poisoned mutex in callback");
    //                 write_log.push(entry);
    //                 true // NOTE: I'm not really sure what this return value means, here.
    //             } else {
    //                 false
    //             }
    //         };
    //
    //     let hooks = util::mem_hook_by_prot(
    //         emu,
    //         MemHookType::MEM_WRITE,
    //         Protection::WRITE,
    //         mem_write_callback,
    //     )?;
    //
    //     Ok(hooks)
    // }

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
}

#[cfg(test)]
mod test {
    use std::iter;

    use unicorn::{CpuX86, RegisterX86};

    use example::*;

    use super::*;
    use crate::util::architecture::{endian, word_size, Endian};
    use byteorder::{ByteOrder, LittleEndian};
    use indexmap::indexmap;
    use rand::{thread_rng, Rng};

    mod example {
        use super::*;
        use byteorder::{BigEndian, LittleEndian};

        pub fn simple_emu_prep_fn<C: 'static + Cpu<'static>>(
            emu: &mut C,
            _params: &HatcheryParams,
            code: &[u8],
            _profiler: &Profiler<C>,
        ) -> Result<Address, Error> {
            // now write the payload
            let stack = util::find_stack(emu)?;
            let sp = stack.begin + (stack.end - stack.begin) / 2;
            emu.mem_write(sp, code)?;
            // set the stack pointer to the middle of the stack
            // now "pop" the stack into the program counter
            let word_size = word_size(emu.arch(), emu.mode());
            let a_bytes = emu.mem_read_as_vec(sp, word_size)?;
            let address = match endian(emu.arch(), emu.mode()) {
                Endian::Big => BigEndian::read_u64(&a_bytes),
                Endian::Little => LittleEndian::read_u64(&a_bytes),
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
                gadget_file: None,
                num_workers: 8,
                num_emulators: 8,
                wait_limit: 150,
                mode: unicorn::Mode::MODE_64,
                arch: unicorn::Arch::X86,
                max_emu_steps: None,
                millisecond_timeout: None,
                record_basic_blocks: true,
                record_memory_writes: true,
                emulator_stack_size: 0x1000, // default
                binary_path: None,
                soup: vec![]
            }
        );
    }

    #[test]
    fn test_spawn_hatchery() {
        let params = HatcheryParams {
            gadget_file: None,
            num_workers: 32,
            num_emulators: 32,
            wait_limit: 150,
            mode: unicorn::Mode::MODE_64,
            arch: unicorn::Arch::X86,
            max_emu_steps: None,
            millisecond_timeout: Some(100),
            record_basic_blocks: true,
            record_memory_writes: true,
            emulator_stack_size: 0x1000,
            binary_path: Some("/bin/sh".to_string()),
            soup: vec![],
        };

        let _: Hatchery<CpuX86<'_>> = Hatchery::new(Arc::new(params));
    }

    #[test]
    fn test_hatchery() {
        pretty_env_logger::env_logger::init();
        let params = HatcheryParams {
            gadget_file: None,
            num_workers: 500,
            num_emulators: 510,
            wait_limit: 50,
            mode: unicorn::Mode::MODE_64,
            arch: unicorn::Arch::X86,
            max_emu_steps: Some(0x10000),
            millisecond_timeout: Some(100),
            record_basic_blocks: true,
            record_memory_writes: false,
            emulator_stack_size: 0x1000,
            binary_path: Some("/bin/sh".to_string()),
            soup: vec![],
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

        let expected_num = 0x10000;
        let mut counter = 0;
        let code_iterator = iter::repeat(()).take(expected_num).map(|()| random_rop());

        use RegisterX86::*;
        let output_registers = Arc::new(vec![RAX, RSP, RIP, RBP, RBX, RCX, EFLAGS]);
        let mut inputs = vec![indexmap! { RCX => 0xdead_beef, RDX => 0xcafe_babe }];
        for _ in 0..100 {
            inputs.push(indexmap! { RCX => rand::random(), RAX => rand::random() });
        }
        for (_code, _profile) in hatchery
            .pipeline(
                code_iterator,
                Arc::new(inputs),
                output_registers,
                Some(Box::new(simple_emu_prep_fn)),
            )
            .iter()
        {
            counter += 1;
            log::info!("[{}] Output: {:#?}", counter, _profile.paths);
            //log::info!("{} processed", counter);
            drop(_profile);
        }
        assert_eq!(counter, expected_num);
        log::info!("FINISHED");
    }
}
