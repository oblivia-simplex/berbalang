use std::process::exit;
use std::sync::Arc;

use serde_json::from_str;
use unicorn::Cpu;

use berbalib::configure::Config;
use berbalib::emulator::hatchery::Hatchery;
use berbalib::emulator::loader::falcon_loader::load_from_path;
use berbalib::emulator::register_pattern::Register;
use berbalib::error::Error;
use berbalib::logger;
use berbalib::util::architecture::{constant_register_state, random_register_state};

/// This purpose of this tool is to:
/// - load a binary into unicorn emulator memory
/// - parse a json document containing a ROP payload
/// - execute that binary with the payload loaded
/// - report the CPU status

pub fn main() {
    let argv = std::env::args().collect::<Vec<String>>();
    if argv.len() != 3 {
        println!("Usage: {} <binary> <payload>", argv[0]);
        exit(1);
    }
    let binary = &argv[1];
    let payload = &argv[2];
    logger::init("no population");
    let mut config = generate_config(binary);
    load_from_path(&mut config, true).expect("Failed to load binary");
    log::info!("Binary {} loaded. Config: {:#?}", binary, config);
    set_significant_registers(&mut config);
    let chain = parse_payload(payload).expect("Failed to parse payload");
    log::info!("About to emulate payload: {:#x?}", chain);
    use unicorn::Arch::*;
    match config.roper.arch {
        X86 => emulate::<unicorn::CpuX86<'_>>(config, chain),
        ARM => emulate::<unicorn::CpuARM<'_>>(config, chain),
        ARM64 => emulate::<unicorn::CpuARM64<'_>>(config, chain),
        MIPS => emulate::<unicorn::CpuMIPS<'_>>(config, chain),
        SPARC => emulate::<unicorn::CpuSPARC<'_>>(config, chain),
        M68K => emulate::<unicorn::CpuM68K<'_>>(config, chain),
        _ => unimplemented!("architecture unimplemented"),
    }
}

fn generate_config(binary: &str) -> Config {
    let mut config = Config::default();
    config.roper.binary_path = binary.to_string();
    config.roper.max_emu_steps = Some(0xFFFF_FFFF);
    config.roper.num_emulators = 1;
    config.roper.num_workers = 1;
    config.roper.randomize_registers = false;
    config.roper.record_basic_blocks = true;
    config.roper.record_memory_writes = true;
    config
}

fn parse_payload(payload: &str) -> Result<Vec<u64>, Error> {
    let data = std::fs::read_to_string(payload)?;
    // TODO:
    // try parsing the payload as a full creature. if you fail,
    // try parsing it as a Vec<u64>. Can't do this until we get
    // Deserialize working for LinearChromosome
    from_str::<Vec<u64>>(&data).map_err(Error::from)
}

macro_rules! strings {
    ($($v:expr $(,)?)*) => {
        vec![$( $v, )*].into_iter().map(|s| s.to_string()).collect::<Vec<String>>()
    }
}

fn set_significant_registers(config: &mut Config) {
    use unicorn::{Arch::*, Mode::*};
    let arch = config.roper.arch;
    let mode = config.roper.mode;
    let regs = match (arch, mode) {
        (X86, MODE_64) => {
            strings!["RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "R9", "RBP", "RSP", "RIP", "EFLAGS",]
        }
        (X86, MODE_32) => {
            strings!["EAX", "EBX", "ECX", "EDX", "ESI", "EDI", "EBP", "ESP", "EIP", "EFLAGS",]
        }
        _ => unimplemented!("this mode/arch combination is not yet implemented"),
    };
    config.roper.output_registers = regs.clone();
    config.roper.input_registers = regs;
}

fn emulate<C: 'static + Cpu<'static>>(config: Config, chain: Vec<u64>) {
    let output_registers: Vec<Register<C>> = {
        config
            .roper
            .registers_to_check()
            .into_iter()
            // running error through ok() because it can't be formatted with Debug
            .map(|r| r.parse().ok().expect("Failed to parse register name"))
            .collect::<Vec<_>>()
    };
    let initial_register_state = if config.roper.randomize_registers {
        random_register_state::<u64, C>(&output_registers, config.random_seed)
    } else {
        constant_register_state::<C>(&output_registers, 1_u64)
    };
    let hatchery: Hatchery<C> = Hatchery::new(
        Arc::new(config.roper),
        Arc::new(initial_register_state),
        Arc::new(output_registers),
    );
    let profile = hatchery.execute(chain, None).expect("Emulation failed!");
    log::info!("Execution complete.");
    println!("{:#x?}", profile);
}
