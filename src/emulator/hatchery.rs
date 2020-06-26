use std::fmt::Debug;
use std::pin::Pin;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};
use std::time::{Duration, Instant};

//use rayon::{ThreadPoolBuilder, };
//use indexmap::map::IndexMap;
use hashbrown::HashMap;
use object_pool::{Pool, Reusable};
use rayon::prelude::*;
use threadpool::ThreadPool;
use unicorn::{Context, Cpu, Mode};

pub use crate::configure::RoperConfig;
use crate::disassembler::Disassembler;
use crate::emulator::loader;
use crate::emulator::loader::Seg;
use crate::emulator::pack::Pack;
use crate::emulator::profiler::{Profile, Profiler};
use crate::emulator::register_pattern::Register;
use crate::error::Error;

//use std::sync::atomic::{AtomicUsize, Ordering};

type Code = Vec<u8>;
pub type Address = u64;
pub type EmuPrepFn<C> = Box<
    dyn Fn(&mut C, &RoperConfig, &[u8], &Profiler<C>) -> Result<Address, Error>
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

pub struct Hatchery<C: Cpu<'static> + Send, X: Pack + Sync + Send + 'static> {
    emu_pool: Arc<Pool<C>>,
    thread_pool: Arc<Mutex<ThreadPool>>,
    config: Arc<RoperConfig>,
    memory: Arc<Option<Pin<Vec<Seg>>>>,
    tx: SyncSender<X>,
    rx: Receiver<(X, Profile)>,
    handle: JoinHandle<()>,
    disassembler: Arc<Disassembler>,
}

impl<C: Cpu<'static> + Send, X: Pack + Sync + Send + 'static> Drop for Hatchery<C, X> {
    fn drop(&mut self) {
        // unmap the unwriteable memory in the emu pool's emus
        log::debug!("Dropping Hatchery");
        let Self {
            emu_pool,
            thread_pool: _thread_pool,
            config: _config,
            memory,
            tx: _tx,
            rx: _rx,
            handle: _handle,
            disassembler: _disassembler,
        } = self;
        // handle.join().expect("Failed to join handle in hatchery");
        if let Some(segments) = memory.as_ref() {
            // Once a shared, mapped region is unmapped from one emulator, it's unmapped
            // from them all. Attempting to unmap it again will trigger a NOMEM error.
            // And I think that attempting to access that unmapped segment *may* trigger a
            // use-after-free bug.
            if let Some(mut emu) = emu_pool.try_pull() {
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

fn init_emu<C: Cpu<'static>>(
    config: &RoperConfig,
    memory: &Option<Pin<Vec<Seg>>>,
) -> Result<C, Error> {
    let mut emu = C::new(config.mode)?;
    if let Some(segments) = memory {
        //notice!(emu.mem_map(0x1000, 0x4000, unicorn::Protection::ALL))?;
        let mut results = Vec::new();
        // First, map the non-writeable segments to memory. These can be shared.
        segments.iter().for_each(|s| {
            log::debug!(
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
                        s.perm.into(),
                        s.data.as_ptr(),
                    );
                    results.push(res);
                }
            } else {
                // Next, map the writeable segments
                let res = emu.mem_map(s.aligned_start(), s.aligned_size(), s.perm.into());
                results.push(res);
            }
        });
        // Return an error if there's been an error.
        let _ = results
            .into_iter()
            .collect::<Result<Vec<_>, unicorn::Error>>()?;
    };
    emu.mem_regions()?.iter().for_each(|rgn| {
        log::debug!(
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

type InboundChannel<T> = (SyncSender<T>, Receiver<T>);
type OutboundChannel<T> = (SyncSender<(T, Profile)>, Receiver<(T, Profile)>);

impl<C: 'static + Cpu<'static> + Send, X: Pack + Send + Sync + Debug + 'static> Hatchery<C, X> {
    pub fn new(
        config: Arc<RoperConfig>,
        inputs: Arc<Vec<HashMap<Register<C>, u64>>>,
        output_registers: Arc<Vec<Register<C>>>,
        // TODO: we might want to set some callbacks with this function.
        emu_prep_fn: Option<EmuPrepFn<C>>,
    ) -> Self {
        let disassembler = Arc::new(
            Disassembler::new(config.arch, config.mode).expect("Failed to build disassembler"),
        );
        let (tx, our_rx): InboundChannel<X> = sync_channel(config.num_workers);
        let (our_tx, rx): OutboundChannel<X> = sync_channel(config.num_workers);

        let static_memory = loader::get_static_memory_image();

        let memory = Some(Pin::new(static_memory.segments().clone()));

        let emu_pool = Arc::new(Pool::new(config.num_workers, || {
            init_emu(&config, &memory).expect("failed to initialize emulator")
        }));
        let thread_pool = Arc::new(Mutex::new(ThreadPool::new(config.num_workers)));

        let emu_prep_fn = Arc::new(emu_prep_fn.unwrap_or_else(|| Box::new(hooking::emu_prep_fn)));
        let wait_limit = config.wait_limit;
        let mode = config.mode;
        let millisecond_timeout = config.millisecond_timeout.unwrap_or(0);
        let max_emu_steps = config.max_emu_steps.unwrap_or(0);
        let word_size = crate::util::architecture::word_size_in_bytes(config.arch, config.mode);
        let endian = crate::util::architecture::endian(config.arch, config.mode);

        let e_pool = emu_pool.clone();
        let t_pool = thread_pool.clone();
        let parameters = config.clone();
        let mem = memory.clone();
        let disas = disassembler.clone();
        let handle = spawn(move || {
            for payload in our_rx.iter() {
                let emu_prep_fn = emu_prep_fn.clone();
                let config = parameters.clone();
                let our_tx = our_tx.clone();
                let output_registers = output_registers.clone();
                let thread_pool = t_pool.lock().expect("Failed to unlock thread_pool mutex");
                let emulator_pool = e_pool.clone();
                let memory = mem.clone();
                let inputs = inputs.clone();
                let disas = disas.clone();
                thread_pool.execute(move || {
                    let profile = inputs.par_iter().map(|input| {
                        // Acquire an emulator from the pool.
                        let mut emu: Reusable<'_, C> = wait_for_emu(&emulator_pool, wait_limit, mode);
                        // Initialize the profiler
                        let mut profiler = Profiler::new(&output_registers, &input);
                        // Save the register context
                        let context: Context = (*emu).context_save().expect("Failed to save context");

                        if config.record_basic_blocks {
                            let _hook = hooking::install_basic_block_hook(&mut (*emu), &mut profiler, &payload.as_code_addrs(word_size, endian)).expect("Failed to install basic_block_hook");
                        }

                        if cfg!(feature = "disassemble_trace") {
                            // install the disassembler hook
                            let _hook = hooking::install_disas_tracer_hook(&mut (*emu), disas.clone(), output_registers.clone()).expect("Failed to install tracer hook");
                        }

                        let _hook = hooking::install_syscall_hook(&mut (*emu), config.arch, config.mode);
                        if config.record_memory_writes {
                            let _hooks = hooking::install_mem_write_hook(&mut (*emu), &profiler).expect("Failed to install mem_write_hook");
                        }
                        let code = payload.pack(word_size, endian);
                        // load the inputs
                        for (reg, val) in input.iter() {
                            emu.reg_write(*reg, *val).expect("Failed to load registers");
                        }
                        // Prepare the emulator with the user-supplied preparation function.
                        // This function will generally be used to load the payload and install
                        // callbacks, which should be able to write to the Profiler instance.
                        let start_addr = emu_prep_fn(&mut emu, &config, &code, &profiler).expect("Failure in the emulator preparation function.");
                        // If the preparation was successful, launch the emulator and execute
                        // the payload. We want to hang onto the exit code of this task.
                        let start_time = Instant::now();
                        /*******************************************************************/
                        let result = emu.emu_start(
                            start_addr,
                            0,
                            millisecond_timeout * unicorn::MILLISECOND_SCALE,
                            max_emu_steps,
                        );
                        /*******************************************************************/
                        profiler.emulation_time = start_time.elapsed();
                        if let Err(error_code) = result {
                            profiler.set_error(error_code)
                        };
                        profiler.read_registers(&mut emu);

                        // TODO : add this to the profiler
                        let written_memory = tools::read_writeable_memory(&(*emu)).expect("Failed to read writeable memory").into_par_iter().filter(|seg| {
                            // bit of a space/time tradeoff here. see how it goes.
                            let stat = static_memory.try_dereference(seg.addr, None).unwrap();
                            debug_assert_eq!(stat.len(), seg.data.len());
                            stat != seg.data.as_slice()
                        }).collect::<Vec<Seg>>();
                        // could some sort of COW structure help here?
                        // TODO consider defining a data structure that looks, from the
                        // outside, like a vector, but which is actually composed of an
                        // indexmap of written-to areas. use the mem-write log to learn
                        // which areas of the writeable memory need to be read, and read
                        // only those. then fallback to MEM_IMAGE, which should also 
                        // contain writeable areas besides the stack.


                        profiler.written_memory = written_memory;

                        // cleanup
                        emu.remove_all_hooks().expect("Failed to clean up hooks");
                        emu.context_restore(&context).expect("Failed to restore context");
                        // clean up writeable memory
                        // there will never be *too* many segments, so iterating over them is cheap.
                        if let Some(memory) = memory.as_ref() {
                            memory.iter().filter(|s| s.is_writeable()).for_each(|seg| {
                                emu.mem_write(seg.aligned_start(),
                                              &seg.data,
                                ).unwrap_or_else(|e| {
                                    log::error!("Failed to refresh writeable memory at 0x{:x} - 0x{:x}: {:?}",
                                    seg.aligned_start(), seg.aligned_end(), e
                                )
                                });
                            });
                        }
                        profiler
                    }).collect::<Vec<Profiler<C>>>().into(); // into Profile
                    // Now send the code back, along with its profile information.
                    // (The genotype, along with its phenotype.)
                    our_tx.send((payload, profile)).map_err(Error::from).expect("TX Failure in pipeline");
                });
            }
        });
        Self {
            emu_pool,
            thread_pool,
            config,
            memory: Arc::new(memory),
            tx,
            rx,
            handle,
            disassembler,
        }
    }

    pub fn execute(&self, payload: X) -> Result<(X, Profile), Error> {
        self.tx.send(payload)?;
        self.rx.recv().map_err(Error::from)
    }

    pub fn execute_batch<I: Iterator<Item = X>>(
        &self,
        payloads: I,
    ) -> Result<Vec<(X, Profile)>, Error> {
        let mut count = 0;
        for x in payloads {
            self.tx.send(x)?;
            count += 1;
        }
        let mut res = Vec::new();
        for _ in 0..count {
            res.push(self.rx.recv()?)
        }
        Ok(res)
    }
}
// TODO: try to reduce the number of mutexes needed in this setup. it seems like a code smell.

pub mod tools {
    use unicorn::MemRegion;

    use super::*;

    // Reads all memory that carries a Protection::WRITE permission.
    // This can be used, e.g., to check to see what a specimen has written
    // to memory.
    pub fn read_writeable_memory<C: 'static + Cpu<'static>>(emu: &C) -> Result<Vec<Seg>, Error> {
        emu.mem_regions()?
            .into_iter()
            .filter(MemRegion::writeable)
            .map(|mem_reg| {
                emu.mem_read_as_vec(mem_reg.begin, mem_reg.size())
                    .map(|data| Seg::from_mem_region_and_data(mem_reg, data))
            })
            .collect::<Result<Vec<Seg>, unicorn::Error>>()
            .map_err(Error::from)
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
}

pub mod hooking {
    use capstone::Insn;
    use hashbrown::HashSet;
    use unicorn::{CodeHookType, MemHookType, MemType, Protection};

    use crate::emulator::profiler::{Block, MemLogEntry};
    use crate::util::architecture::{endian, read_integer, word_size_in_bytes};

    use super::*;

    fn is_syscall(arch: unicorn::Arch, mode: unicorn::Mode, inst: &Insn<'_>) -> bool {
        use unicorn::Arch::*;
        use unicorn::Mode::*;

        match (arch, mode, inst.mnemonic()) {
            (X86, MODE_64, Some("syscall")) | (X86, MODE_64, Some("sysenter")) => true,
            (X86, _, Some("int")) if inst.op_str() == Some("0x80") => true,
            (X86, _, _) => false,
            _ => unimplemented!("TODO later"),
        }
    }

    /// We want the emulator to halt on a syscall.
    pub fn install_syscall_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        arch: unicorn::Arch,
        mode: unicorn::Mode,
    ) -> Result<unicorn::uc_hook, unicorn::Error> {
        let pc: i32 = emu.program_counter().into();

        let callback = move |engine: &unicorn::Unicorn<'_>, a| {
            // TODO log the errors
            if let Ok(address) = engine.reg_read(pc) {
                log::trace!("Interrupt at address 0x{:x}. a = {:?}", pc, a);
                let memory = loader::get_static_memory_image();
                if let Some(insts) = memory.disassemble(address, 64, Some(1)) {
                    if let Some(inst) = insts.iter().next() {
                        if is_syscall(arch, mode, &inst) {
                            engine.emu_stop().expect("Failed to stop engine!");
                        }
                    }
                }
            }
        };
        emu.add_intr_hook(callback)
    }

    pub fn install_disas_tracer_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        disassembler: Arc<Disassembler>,
        output_registers: Arc<Vec<Register<C>>>,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
        let callback = move |engine: &unicorn::Unicorn<'_>, address: u64, block_length: u32| {
            let disas = disassembler
                .disas_from_mem_image(address, block_length as usize)
                .expect("Failed to disassemble block");

            let registers: String = output_registers
                .iter()
                .map(|reg| {
                    let reg_i = (*reg).into();
                    let val = engine.reg_read(reg_i).expect("Failed to read register");
                    format!("{:?}: 0x{:x}", reg, val)
                })
                .collect::<Vec<String>>()
                .join("\n");

            log::trace!("\n{}\n{}", registers, disas);
        };
        // TODO: print registers
        code_hook_all(emu, CodeHookType::BLOCK, callback)
    }

    pub fn install_address_tracking_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        profiler: &Profiler<C>,
        addresses: &[u64],
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
        let gadget_log = profiler.gadget_log.clone();

        let callback = move |_engine, address, _insn_len| gadget_log.push(address);

        addresses
            .iter()
            .map(|addr| emu.add_code_hook(CodeHookType::CODE, *addr, addr + 1, callback.clone()))
            .collect::<Result<Vec<unicorn::uc_hook>, _>>()
    }

    pub fn emu_prep_fn<C: 'static + Cpu<'static>>(
        emu: &mut C,
        _config: &RoperConfig,
        code: &[u8],
        _profiler: &Profiler<C>,
    ) -> Result<u64, Error> {
        // now write the payload
        let stack = tools::find_stack(emu)?;
        let sp = stack.begin + (stack.end - stack.begin) / 2;
        emu.mem_write(sp, code)?;
        // set the stack pointer to the middle of the stack
        // now "pop" the stack into the program counter
        let word_size = word_size_in_bytes(emu.arch(), emu.mode());
        let a_bytes = emu.mem_read_as_vec(sp, word_size)?;
        let endian = endian(emu.arch(), emu.mode());
        if let Some(address) = read_integer(&a_bytes, endian, word_size) {
            emu.write_stack_pointer(sp + word_size as u64)?;

            Ok(address)
        } else {
            Err(Error::Misc("Failed to initialize stack pointer".into()))
        }
    }

    pub fn install_basic_block_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        profiler: &mut Profiler<C>,
        gadget_addrs: &Vec<u64>,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
        let gadget_addrs: Arc<HashSet<u64>> = Arc::new(gadget_addrs.iter().cloned().collect());
        let block_log = profiler.block_log.clone();
        let gadget_log = profiler.gadget_log.clone();
        //let ret_log = profiler.ret_log.clone();
        let bb_callback = move |_engine: &unicorn::Unicorn<'_>, entry: u64, size: u32| {
            let size = size as usize;

            if gadget_addrs.contains(&entry) {
                gadget_log.push(entry);
            }

            // If the code ends with a return, log it in the ret log.
            // but how to make this platform-generic? We could define a
            // return_insn method on the trait, like we did for the special
            // registers. TODO: low priority
            let block = Block { entry, size };
            block_log.push(block);
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
            // TODO: we might want to track the # of unique addresses written to instead.
            move |engine: &unicorn::Unicorn<'_>, mem_type, address, num_bytes_written, value| {
                //log::trace!("Inside memory hook!");
                if let MemType::WRITE = mem_type {
                    let program_counter = engine.reg_read(pc).expect("Failed to read PC register");
                    let entry = MemLogEntry {
                        program_counter,
                        address,
                        num_bytes_written,
                        value: value as u64,
                    };
                    write_log.push(entry);
                    true // NOTE: I'm not really sure what this return value means, here.
                } else {
                    false
                }
            };

        let hooks = mem_hook_by_prot(
            emu,
            MemHookType::MEM_WRITE,
            Protection::WRITE,
            mem_write_callback,
        )?;

        Ok(hooks)
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

    use byteorder::{ByteOrder, LittleEndian};
    use rand::{thread_rng, Rng};
    use unicorn::{CpuX86, RegisterX86};

    use crate::hashmap;
    use crate::util::architecture::{endian, word_size_in_bytes};

    use super::*;

    mod example {
        use crate::util::architecture::read_integer;

        use super::*;

        pub fn simple_emu_prep_fn<C: 'static + Cpu<'static>>(
            emu: &mut C,
            _config: &RoperConfig,
            code: &[u8],
            _profiler: &Profiler<C>,
        ) -> Result<Address, Error> {
            // now write the payload
            let stack = tools::find_stack(emu)?;
            let sp = stack.begin + (stack.end - stack.begin) / 2;
            emu.mem_write(sp, code)?;
            // set the stack pointer to the middle of the stack
            // now "pop" the stack into the program counter
            let word_size = word_size_in_bytes(emu.arch(), emu.mode());
            let a_bytes = emu.mem_read_as_vec(sp, word_size)?;
            let endian = endian(emu.arch(), emu.mode());
            if let Some(address) = read_integer(&a_bytes, endian, word_size) {
                emu.write_stack_pointer(sp + word_size as u64)?;
                Ok(address)
            } else {
                Err(Error::Misc("Failed to initialize stack pointer".into()))
            }
        }
    }

    #[test]
    fn test_config() {
        let config = r#"
            num_workers = 8
            wait_limit = 150
            use_registers = [ "EAX" ]
            mode = "MODE_64"
            arch = "X86"
            record_basic_blocks = true
            record_memory_writes = true
            binary_path = "/bin/sh"
        "#;

        let config: RoperConfig = toml::from_str(config).unwrap();
        assert_eq!(
            config,
            RoperConfig {
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
                binary_path: "/bin/sh".to_string(),
                soup: None,
                soup_size: None,
                ..Default::default()
            }
        );
    }

    // FIXME - currently broken for want for full Pack impl for Vec<u8> #[test]
    fn test_hatchery() {
        env_logger::init();
        let config = RoperConfig {
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
            binary_path: "/bin/sh".to_string(),
            soup: None,
            soup_size: None,
            ..Default::default()
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

        let expected_num = 0x100;
        let mut counter = 0;
        let code_iterator = iter::repeat(()).take(expected_num).map(|()| random_rop());

        use RegisterX86::*;
        let output_registers = Arc::new(vec![RAX, RSP, RIP, RBP, RBX, RCX, EFLAGS]);
        let mut inputs = vec![hashmap! { RCX => 0xdead_beef, RDX => 0xcafe_babe }];
        for _ in 0..100 {
            inputs.push(hashmap! { RCX => rand::random(), RAX => rand::random() });
        }
        let hatchery: Hatchery<CpuX86<'_>, Code> =
            Hatchery::new(Arc::new(config), Arc::new(inputs), output_registers, None);

        let results = hatchery.execute_batch(code_iterator);
        for (_code, profile) in results.expect("boo") {
            log::info!("[{}] Output: {:#?}", counter, profile.paths);
            counter += 1;
        }
        // for code in code_iterator {
        //     counter += 1;
        //     let code_in = code.clone();
        //     let (code_out, profile) = hatchery.execute(code).expect("execution failure");
        //     assert_eq!(code_in, code_out); // should fail sometimes
        //     log::info!("[{}] Output: {:#?}", counter, profile.paths);
        // }
        assert_eq!(counter, expected_num);
        log::info!("FINISHED");
    }
}
