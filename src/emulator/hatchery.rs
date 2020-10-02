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
use crate::emulator::hatchery::hooking::emu_prep_fn;
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

struct EmuPool<C: Cpu<'static>> {
    pub pool: Pool<C>,
    init_context: Context,
    mode: Mode,
    wait_limit: u64,
    memory: Arc<Option<Pin<Vec<Seg>>>>,
}

impl<C: Cpu<'static>> EmuPool<C> {
    pub fn new(config: &RoperConfig) -> Self {
        let static_memory = loader::get_static_memory_image();

        let memory = Some(Pin::new(static_memory.segments().clone()));

        let pool: Pool<C> = Pool::new(config.num_workers, || {
            Self::init_emu(&config, &memory).expect("failed to initialize emulator")
        });
        let init_context = {
            let emu = Self::wait_for_emu(&pool, config.wait_limit, config.mode);
            let ctx = (*emu).context_save().expect("Failed to save context");
            ctx
        };

        Self {
            pool,
            init_context,
            mode: config.mode,
            wait_limit: config.wait_limit,
            memory: Arc::new(memory),
        }
    }

    /// Returns a reusable pointer to an emulator, which will be returned to the pool when it's
    /// dropped.
    pub fn pull(&self) -> object_pool::Reusable<'_, C> {
        let mut emu = Self::wait_for_emu(&self.pool, self.wait_limit, self.mode);
        emu.context_restore(&self.init_context)
            .expect("Failed to restore context");
        emu
    }

    fn init_emu(config: &RoperConfig, memory: &Option<Pin<Vec<Seg>>>) -> Result<C, Error> {
        let mut emu = C::new(config.mode)?;
        if let Some(segments) = memory {
            //notice!(emu.mem_map(0x1000, 0x4000, unicorn::Protection::ALL))?;
            let mut results = Vec::new();
            // First, map the non-writeable segments to memory. These can be shared.
            segments.iter().for_each(|s| {
                log::info!(
                    "Mapping segment 0x{:x} - 0x{:x} {:?} [{:?}]",
                    s.aligned_start(),
                    s.aligned_end(),
                    s.segtype,
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
            log::info!(
                "Mapped region: 0x{:x} - 0x{:x} [{:?}]",
                rgn.begin,
                rgn.end,
                rgn.perms
            );
        });
        Ok(emu)
    }

    fn wait_for_emu(pool: &Pool<C>, wait_limit: u64, mode: Mode) -> object_pool::Reusable<'_, C> {
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
}

type InboundTx<T, C> = SyncSender<(T, Option<HashMap<Register<C>, u64>>)>;
type InboundRx<T, C> = Receiver<(T, Option<HashMap<Register<C>, u64>>)>;
type OutboundTx = SyncSender<Profile>;
type OutboundRx = Receiver<Profile>;
type InboundChannel<T, C> = (InboundTx<T, C>, InboundRx<T, C>);
type OutboundChannel = (OutboundTx, OutboundRx);

pub struct Hatchery<C: Cpu<'static> + Send> {
    emu_pool: Arc<EmuPool<C>>,
    thread_pool: Arc<Mutex<ThreadPool>>,
    config: Arc<RoperConfig>,
    memory: Arc<Option<Pin<Vec<Seg>>>>,
    tx: InboundTx<Vec<u64>, C>,
    rx: OutboundRx,
    handle: JoinHandle<()>,
    disassembler: Arc<Disassembler>,
}

impl<C: Cpu<'static> + Send> Drop for Hatchery<C> {
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
            if let Some(mut emu) = emu_pool.pool.try_pull() {
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

impl<C: 'static + Cpu<'static> + Send> Hatchery<C> {
    pub fn new(
        config: Arc<RoperConfig>,
        initial_register_state: Arc<HashMap<Register<C>, u64>>,
        output_registers: Arc<Vec<Register<C>>>,
    ) -> Self {
        let disassembler = Arc::new(
            Disassembler::new(config.arch, config.mode).expect("Failed to build disassembler"),
        );
        let (tx, our_rx): InboundChannel<Vec<u64>, C> = sync_channel(config.num_workers);
        let (our_tx, rx): OutboundChannel = sync_channel(config.num_workers);

        let static_memory = loader::get_static_memory_image();

        let memory = Some(Pin::new(static_memory.segments().clone()));

        let emu_pool = Arc::new(EmuPool::new(&config));
        let thread_pool = Arc::new(Mutex::new(ThreadPool::new(config.num_workers)));

        let millisecond_timeout = config.millisecond_timeout.unwrap_or(0);
        let max_emu_steps = config.max_emu_steps.unwrap_or(0);
        let word_size = crate::util::architecture::word_size_in_bytes(config.arch, config.mode);
        let endian = crate::util::architecture::endian(config.arch, config.mode);

        let e_pool = emu_pool.clone();
        let t_pool = thread_pool.clone();
        let parameters = config.clone();
        let mem = memory.clone();
        let disas = disassembler.clone();
        let bad_bytes: Arc<Option<HashMap<u8, u8>>> =
            Arc::new(config.bad_bytes.as_ref().map(|table| {
                table
                    .iter()
                    // FIXME do this in a less dirty, shotgunny way
                    .map(|(k, v)| (u8::from_str_radix(k, 16).unwrap(), *v))
                    .collect::<HashMap<u8, u8>>()
            }));
        let handle = spawn(move || {
            for (payload, args) in our_rx.iter() {
                let config = parameters.clone();
                let bad_bytes = bad_bytes.clone();
                let our_tx = our_tx.clone();
                let output_registers = output_registers.clone();
                let thread_pool = t_pool.lock().expect("Failed to unlock thread_pool mutex");
                let emulator_pool = e_pool.clone();
                let memory = mem.clone();
                let initial_register_state = if let Some(args) = args {
                    Arc::new(args)
                } else {
                    initial_register_state.clone()
                };
                let disas = disas.clone();
                // let's get a clean context to use here.
                thread_pool.execute(move || {
                    // Acquire an emulator from the pool.
                    let mut emu: Reusable<'_, C> = emulator_pool.pull();
                    // Initialize the profiler
                    let mut profiler = Profiler::new(&output_registers, &initial_register_state);
                    // load the inputs
                    for (reg, val) in initial_register_state.iter() {
                        emu.reg_write(*reg, *val).expect("Failed to load registers");
                    }
                    profiler.registers_at_last_ret = Arc::new(Mutex::new((*initial_register_state).clone()));

                    // Pedantically check to make sure the registers are initialized
                    if true || cfg!(debug_assertions) {
                        for (r, expected) in initial_register_state.iter() {
                            // TODO: figure out why the context restore isn't taking care of this
                            emu.reg_write(*r, *expected).expect("Failed to write regsiter");
                            // let val = emu.reg_read(*r).expect("Failed to read register!");
                            // assert_eq!(val, *expected, "register has not been initialized");
                        }
                    }

                    let code = payload.pack(word_size, endian, (*bad_bytes).as_ref());
                    let initial_pc = emu_prep_fn(&mut (*emu), &config, &code, &profiler).expect("Failure in the emulator preparation function.");

                    if config.record_basic_blocks {
                        let _hook = hooking::install_code_logging_hook(&mut (*emu), &profiler, &payload.as_code_addrs(word_size, endian), config.break_on_calls).expect("Failed to install code_logging_hook");
                    }

                    // WONTFIX: It turns out that Unicorn never implemented a fetch hook. It's an unused enum in the C code. Balls.
                    // let _hook = hooking::install_gadget_fetching_hook(&mut (*emu), &profiler).expect("Failed to install gadget_fetching_hook");

                    if cfg!(feature = "disassemble_trace") {
                        // install the disassembler hook
                        let _hook = hooking::install_disas_tracer_hook(&mut (*emu), disas.clone(), output_registers.clone()).expect("Failed to install tracer hook");
                    }

                    // let _hook = hooking::install_syscall_hook(&mut (*emu), config.arch, config.mode);
                    if config.record_memory_writes {
                        let _hooks = hooking::install_mem_write_hook(&mut (*emu), &profiler, config.monitor_stack_writes).expect("Failed to install mem_write_hook");
                    }

                    ;
                    // If the preparation was successful, launch the emulator and execute
                    // the payload. We want to hang onto the exit code of this task.
                    let start_time = Instant::now();
                    /*******************************************************************/
                    let result = emu.emu_start(
                        initial_pc,
                        0,
                        millisecond_timeout * unicorn::MILLISECOND_SCALE,
                        max_emu_steps,
                    );
                    /*******************************************************************/
                    profiler.emulation_time = start_time.elapsed();
                    if let Err(error_code) = result {
                        profiler.set_error(error_code)
                    };

                    let written_memory = tools::read_writeable_memory(&(*emu)).expect("Failed to read writeable memory").into_par_iter().filter(|seg| {
                        let stat = static_memory.try_dereference(seg.addr, None).unwrap();
                        debug_assert_eq!(stat.len(), seg.data.len());
                        stat != seg.data.as_slice()
                    }).collect::<Vec<Seg>>();

                    profiler.written_memory = written_memory;

                    // cleanup
                    emu.remove_all_hooks().expect("Failed to clean up hooks");


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
                    let profile = profiler.into();
                    // Now send the code back, along with its profile information.
                    // (The genotype, along with its phenotype.)
                    our_tx.send(profile).map_err(Error::from).expect("TX Failure in pipeline");
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

    pub fn execute(
        &self,
        payload: Vec<u64>,
        args: Option<HashMap<Register<C>, u64>>,
    ) -> Result<Profile, Error> {
        self.tx.send((payload, args))?;
        self.rx.recv().map_err(Error::from)
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
    pub fn find_stack<C: 'static + Cpu<'static>>(emu: &C) -> Option<MemRegion> {
        if let Ok(regions) = emu.mem_regions() {
            let mut bottom = 0;
            let mut stack = None;
            for region in regions.iter() {
                if region.writeable() && region.readable() && region.begin >= bottom {
                    bottom = region.begin;
                    stack = Some(region)
                };
            }
            stack.cloned()
        } else {
            None
        }
    }
}

pub mod hooking {
    use core::sync::atomic;

    use capstone::Insn;
    use hashbrown::HashSet;
    use unicorn::{CodeHookType, MemHookType, MemType, Protection};

    use crate::emulator::hatchery::tools::find_stack;
    use crate::emulator::loader::get_static_memory_image;
    use crate::emulator::profiler::{read_registers_in_hook, Block, MemLogEntry};
    use crate::util::architecture::{endian, read_integer, word_size_in_bytes, Perms};

    use super::*;

    fn is_syscall(arch: unicorn::Arch, mode: unicorn::Mode, inst_bytes: &[u8]) -> bool {
        use unicorn::Arch::*;
        use unicorn::Mode::*;

        let memory = get_static_memory_image();
        if let Some(inst) = memory.disassemble_bytes(inst_bytes) {
            if let Some(inst) = inst.iter().next() {
                match (arch, mode, inst.mnemonic()) {
                    (X86, MODE_64, Some("syscall")) | (X86, MODE_64, Some("sysenter")) => true,
                    (X86, _, Some("int")) if inst.op_str() == Some("0x80") => true,
                    (X86, _, _) => false,
                    _ => unimplemented!("TODO later"),
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    fn is_call(arch: unicorn::Arch, mode: unicorn::Mode, inst: &Insn<'_>) -> bool {
        // use unicorn::Arch::*;
        // use unicorn::Mode::*;

        match (arch, mode, inst.mnemonic()) {
            (_, _, Some("call")) => true,
            _ => false, // TODO: implement other arches later if needed
        }
    }

    fn is_ret(arch: unicorn::Arch, _mode: unicorn::Mode, inst_bytes: &[u8]) -> bool {
        if (inst_bytes[0] == 0xC3 || inst_bytes[0] == 0xC2) && arch == unicorn::Arch::X86 {
            return true;
        }
        false
    }

    /// We want the emulator to halt on a syscall.
    /// NOTE: this only really works on x86 architectures. TODO: generalize somehow
    // pub fn install_syscall_hook<C: 'static + Cpu<'static>>(
    //     emu: &mut C,
    //     arch: unicorn::Arch,
    //     mode: unicorn::Mode,
    // ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
    //     let pc: i32 = emu.program_counter().into();
    //
    //     let int_callback = move |engine: &unicorn::Unicorn<'_>, a| {
    //         // TODO log the errors
    //         if let Ok(address) = engine.reg_read(pc) {
    //             log::trace!("Interrupt at address 0x{:x}. a = {:?}", pc, a);
    //             let memory = loader::get_static_memory_image();
    //             if let Some(insts) = memory.disassemble(address, 64, Some(1)) {
    //                 if let Some(inst) = insts.iter().next() {
    //                     if is_syscall(arch, mode, &inst) {
    //                         engine.emu_stop().expect("Failed to stop engine!");
    //                     }
    //                 }
    //             }
    //         }
    //     };
    //     // for that, we need a HOOK_INSN
    //     let intr_hook = emu.add_intr_hook(int_callback)?;
    //
    //     let syscall_callback = move |engine: &unicorn::Unicorn<'_>| {
    //         log::trace!("Syscall or Sysenter. Halting.");
    //         engine.emu_stop().expect("Failed to stop engine!");
    //     };
    //     // VERIFY THAT BEGIN > END means range check always passes
    //     let syscall_hook = emu.add_insn_sys_hook(InsnSysX86::SYSCALL, 1, 0, syscall_callback)?;
    //     let sysenter_hook = emu.add_insn_sys_hook(InsnSysX86::SYSENTER, 1, 0, syscall_callback)?;
    //
    //     Ok(vec![intr_hook, syscall_hook, sysenter_hook])
    // }

    pub fn install_disas_tracer_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        disassembler: Arc<Disassembler>,
        output_registers: Arc<Vec<Register<C>>>,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
        let callback = move |engine: &unicorn::Unicorn<'_>, address: u64, block_length: u32| {
            let registers: String = output_registers
                .iter()
                .map(|reg| {
                    let reg_i = (*reg).into();
                    let val = engine.reg_read(reg_i).expect("Failed to read register");
                    format!("{:?}: 0x{:x}", reg, val)
                })
                .collect::<Vec<String>>()
                .join(", ");

            if let Ok(disas) = disassembler.disas_from_mem_image(address, block_length as usize) {
                log::trace!("\n{}\n{}", registers, disas);
            } else {
                log::trace!(
                    "\n{}\nUnable to disassemble 0x{:x} bytes at address 0x{:x}",
                    registers,
                    block_length,
                    address
                );
            }
        };

        // TODO: print registers
        code_hook_all(emu, CodeHookType::CODE, callback)
    }

    pub fn install_gadget_fetching_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        profiler: &Profiler<C>,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error> {
        let gadget_log = profiler.gadget_log.clone();

        let callback = move |_engine, _mem_type, address, _size, value| {
            log::warn!(
                "mem: {:?}, address = 0x{:x}, size = 0x{:x}, value = 0x{:x}",
                _mem_type,
                address,
                _size,
                value
            );
            gadget_log.push(address);
            false
        };

        mem_hook_by_prot(emu, MemHookType::MEM_FETCH, Protection::ALL, callback, true)
    }

    pub fn emu_prep_fn<C: 'static + Cpu<'static>>(
        emu: &mut C,
        _config: &RoperConfig,
        code: &[u8],
        _profiler: &Profiler<C>,
    ) -> Result<u64, Error> {
        // now write the payload
        let stack = tools::find_stack(emu).expect("Can't find stack");
        let pad = 0x100;
        let sp = stack.begin + pad;
        let room = (stack.end - (stack.begin + pad)) as usize;
        let end = room.min(code.len());
        let payload = &code[0..end];
        emu.mem_write(sp, payload)?;
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

    pub fn install_code_logging_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        profiler: &Profiler<C>,
        gadget_addrs: &[u64],
        break_on_calls: bool,
    ) -> Result<unicorn::uc_hook, unicorn::Error> {
        let memory = get_static_memory_image();
        // let stack_region: MemRegion = find_stack(emu).expect("Could not find stack");
        // let stack_begin = stack_region.begin;
        // let stack_end = stack_region.end;
        let word_size = memory.word_size;
        let endian = memory.endian;
        let arch = memory.arch;
        let mode = memory.mode;
        let gadget_addrs: Arc<HashSet<u64>> = Arc::new(gadget_addrs.iter().cloned().collect());
        let block_log = profiler.trace_log.clone();
        let gadget_log = profiler.gadget_log.clone();
        let ret_count = profiler.ret_count.clone();
        let call_stack_depth = profiler.call_stack_depth.clone();
        let register_state = profiler.registers_at_last_ret.clone();
        let registers_to_read = Arc::new(profiler.registers_to_read.clone());
        let committed_write_log = profiler.committed_write_log.clone();
        let committed_trace_log = profiler.committed_trace_log.clone();
        let write_log = profiler.write_log.clone();
        let sp: i32 = emu.stack_pointer().into();

        macro_rules! commit_logs {
            ($engine: expr, $registers_to_read: expr => $register_state: expr, $write_log: expr => $committed_write_log: expr, $trace_log: expr => $committed_trace_log: expr) => {
                read_registers_in_hook::<C>(
                    ($register_state).clone(),
                    &($registers_to_read),
                    &($engine),
                );
                ($committed_write_log)
                    .lock()
                    .unwrap()
                    .absorb_segqueue(&($write_log));
                if let Ok(mut log) = $committed_trace_log.lock() {
                    while let Ok(b) = $trace_log.pop() {
                        log.push(b)
                    }
                }
            };
        }

        let bb_callback = move |engine: &unicorn::Unicorn<'_>, entry: u64, size: u32| {
            let size = size as usize;
            let code = engine
                .mem_read_as_vec(entry, size as usize)
                .unwrap_or_default();
            let memory = get_static_memory_image();

            let block = Block { entry, size, code };
            block_log.push(block);
            if gadget_addrs.contains(&entry) {
                gadget_log.push(entry);
            }

            if let Ok(inst) = engine.mem_read_as_vec(entry, size) {
                // Once we have reached a `ret` instruction, we have reached a point at which our
                // gadget chain is composable with additional gadgets. This is where we want to
                // commit our various trace logs.
                if is_ret(arch, mode, &inst) {
                    // it would actually be interesting to see if the stack pointer is ever pointing to the heap
                    // could we evolve a stack pivot?
                    // TODO: we can actually check, on each ret, to see if the stack pointer
                    // is pointing to the payload region! Fuck, why didn't I think of this
                    // sooner.
                    let stack_pointer = engine.reg_read(sp).expect("Failed to read stack pointer");

                    if let Some(addr) = engine
                        .mem_read_as_vec(stack_pointer, word_size)
                        .ok()
                        .and_then(|v| read_integer(&v, endian, word_size))
                    {
                        // We check to see if the return address is 0, too, because this is what we expect at
                        // the end of a healthy rop chain execution.
                        // TODO: use a magic word instead of 0? like 0xBAAD_F00D?
                        // we could use a weaker restriction here, and just ensure that the addr is executable
                        // but no, then we wouldn't really be safeguarding composability
                        // more loosely, the ret marks a composable joint if the stack pointer points to writeable memory
                        if addr == 0
                            || (Some(true)
                                == memory
                                    .perm_of_addr(stack_pointer)
                                    .map(|p| p.intersects(Perms::WRITE)))
                        {
                            // should be okay if it points beyond the payload, on last gadget
                            // it's composable!
                            let registers_to_read = registers_to_read.clone();
                            if !break_on_calls
                                && call_stack_depth.load(atomic::Ordering::Relaxed) > 0
                            {
                                call_stack_depth.fetch_sub(1, atomic::Ordering::Relaxed);
                            } else {
                                ret_count.fetch_add(1, atomic::Ordering::Relaxed);
                                commit_logs!(engine, registers_to_read => register_state, write_log => committed_write_log, block_log => committed_trace_log);
                            }
                            // Quietly stop the emulator if there's an attempt to return to 0
                            if addr == 0 {
                                engine.emu_stop().expect("Failed to stop emulator");
                            }
                        }
                    }
                // it would be cool if we could save the context at each ret, so that we can rewind
                // bad gadgets.
                } else if is_syscall(arch, mode, &inst) {
                    commit_logs!(engine, registers_to_read => register_state, write_log => committed_write_log, block_log => committed_trace_log);
                    engine.emu_stop().expect("Failed to stop emulator");
                } else {
                    // if not a RETURN
                    // EXPERIMENTAL: halt execution at calls. see what happens.
                    if let Some(insts) = memory.disassemble(entry, size, Some(1)) {
                        for inst in insts.iter() {
                            if is_call(arch, mode, &inst) {
                                call_stack_depth.fetch_add(1, atomic::Ordering::Relaxed);
                                if break_on_calls {
                                    engine.emu_stop().expect("Failed to stop emulator");
                                }
                            }
                            break;
                        }
                    }
                }
            }
        };

        emu.add_code_hook(CodeHookType::CODE, 1, 0, bb_callback) //code_hook_all(emu, CodeHookType::CODE, bb_callback)?;
    }

    pub fn install_mem_write_hook<C: 'static + Cpu<'static>>(
        emu: &mut C,
        profiler: &Profiler<C>,
        monitor_stack_writes: bool,
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
                    false // means "handled"
                } else {
                    false
                }
            };

        let hooks = mem_hook_by_prot(
            emu,
            MemHookType::MEM_WRITE,
            Protection::WRITE,
            mem_write_callback,
            monitor_stack_writes,
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
    /// NOTE: we want to ignore the stack here.
    pub fn mem_hook_by_prot<F, C: 'static + Cpu<'static>>(
        emu: &mut C,
        hook_type: MemHookType,
        protections: Protection,
        callback: F,
        monitor_stack_writes: bool,
    ) -> Result<Vec<unicorn::uc_hook>, unicorn::Error>
    where
        F: 'static
            + FnMut(&'static unicorn::Unicorn<'static>, MemType, u64, usize, i64) -> bool
            + Clone,
    {
        let mut hooks = Vec::new();
        let stack_region = find_stack(emu).expect("Could not find stack");
        for region in emu
            .mem_regions()?
            .into_iter()
            .filter(|reg| monitor_stack_writes || reg.begin != stack_region.begin)
        {
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
        F: 'static + FnMut(&'static unicorn::Unicorn<'static>, u64, u32) -> (),
    {
        let mut hooks = Vec::new();
        hooks.push(emu.add_code_hook(hook_type, 1, 0, callback)?);
        // for region in emu.mem_regions()? {
        //     if region.executable() {
        //         let hook =
        //             emu.add_code_hook(hook_type, region.begin, region.end, callback.clone())?;
        //         hooks.push(hook)
        //     }
        // }
        Ok(hooks)
    }
}

#[cfg(test)]
mod test {
    use rand::{thread_rng, Rng};

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
            let stack = tools::find_stack(emu).expect("no stack");
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
    // fn test_hatchery() {
    //     env_logger::init();
    //     let config = RoperConfig {
    //         gadget_file: None,
    //         num_workers: 500,
    //         num_emulators: 510,
    //         wait_limit: 50,
    //         mode: unicorn::Mode::MODE_64,
    //         arch: unicorn::Arch::X86,
    //         max_emu_steps: Some(0x10000),
    //         millisecond_timeout: Some(100),
    //         record_basic_blocks: true,
    //         record_memory_writes: false,
    //         emulator_stack_size: 0x1000,
    //         binary_path: "/bin/sh".to_string(),
    //         soup: None,
    //         soup_size: None,
    //         ..Default::default()
    //     };
    //     fn random_rop() -> Vec<u8> {
    //         let mut rng = thread_rng();
    //         let addresses: Vec<u64> = iter::repeat(())
    //             .take(100)
    //             .map(|()| rng.gen_range(0x41_b000_u64, 0x4a_5fff_u64))
    //             .collect();
    //         let mut packed = vec![0_u8; addresses.len() * 8];
    //         LittleEndian::write_u64_into(&addresses, &mut packed);
    //         packed
    //     }
    //
    //     let expected_num = 0x100;
    //     let mut counter = 0;
    //     let code_iterator = iter::repeat(()).take(expected_num).map(|()| random_rop());
    //
    //     use RegisterX86::*;
    //     let output_registers = Arc::new(vec![RAX, RSP, RIP, RBP, RBX, RCX, EFLAGS]);
    //     let mut inputs = vec![hashmap! { RCX => 0xdead_beef, RDX => 0xcafe_babe }];
    //     for _ in 0..100 {
    //         inputs.push(hashmap! { RCX => rand::random(), RAX => rand::random() });
    //     }
    //     let hatchery: Hatchery<CpuX86<'_>, Code> =
    //         Hatchery::new(Arc::new(config), Arc::new(inputs), output_registers, None);
    //
    //     let results = hatchery.execute_batch(code_iterator);
    //     for (_code, profile) in results.expect("boo") {
    //         log::info!("[{}] Output: {:#?}", counter, profile.paths);
    //         counter += 1;
    //     }
    //     // for code in code_iterator {
    //     //     counter += 1;
    //     //     let code_in = code.clone();
    //     //     let (code_out, profile) = hatchery.execute(code).expect("execution failure");
    //     //     assert_eq!(code_in, code_out); // should fail sometimes
    //     //     log::info!("[{}] Output: {:#?}", counter, profile.paths);
    //     // }
    //     assert_eq!(counter, expected_num);
    //     log::info!("FINISHED");
    // }
}
