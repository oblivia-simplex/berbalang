use std::sync::Arc;
use std::sync::mpsc::{channel, Receiver, RecvError, SendError};
use std::thread::{spawn};
use std::time::Duration;

use object_pool::Pool;
use unicorn::{Cpu, Mode};
use rayon::prelude::*;

pub struct Hatchery<C: Cpu<'static> + Send> {
    pool: Arc<Pool<C>>,
    mode: Mode,
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

type Code = Vec<u8>;

type Register<C> = <C as Cpu<'static>>::Reg;


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

fn wait_for_emu<'a, C: Cpu<'static>>(pool: &'a Pool<C>, wait_limit: u64, mode: Mode) -> object_pool::Reusable<'a, C> {
    let mut wait_time = 0;
    let wait_unit = 1;
    loop {
        if let Some(c) = pool.try_pull() {
            if wait_time > 0 {
                log::warn!("Waited {} milliseconds for CPU", wait_time);
            }
            return c;
        } else if wait_time > wait_limit {
            log::warn!("Waited {} milliseconds for CPU, creating new one", wait_time);
            return pool.pull(|| {
                C::new(mode).expect("Failed to spawn replacement CPU")
            });
        }{
            std::thread::sleep(Duration::from_millis(wait_unit));
            wait_time += wait_unit;
        }

    }
}

impl<C: 'static + Cpu<'static> + Send> Hatchery<C> {
    pub fn new(mode: unicorn::Mode, num_workers: usize) -> Self {
        log::debug!("Creating pool with {} workers", num_workers);
        let pool = Arc::new(Pool::new(num_workers, || init_emu(mode).unwrap()));
        log::debug!("Pool created");
        Self {
            pool,
            mode,
        }
    }

    /// Example.
    /// Adapting this method for a ROP executor would involve a few changes.
    pub fn pipeline<I: Iterator<Item=Code> + IntoParallelIterator>(&self, inbound: I, output_registers: Arc<Vec<Register<C>>>) -> Receiver<Result<(Code, Vec<i32>), Error>>
    {
        let (tx, rx) = channel::<Result<(Code, Vec<i32>), Error>>();
        let mode = self.mode;
        for code in inbound {
            let pool = self.pool.clone();
            let tx = tx.clone();
            let output_registers = output_registers.clone();
            let _handle = spawn(move || -> Result<(), Error> {
                let mut emu = wait_for_emu(&pool, 2000, mode);
                log::trace!("executing code {:02x?} after nap", code);
                let res = emu.mem_write(0x1000, &code)
                    .and_then(|()| {
                        emu.emu_start(
                            0x1000,
                            (0x1000 + code.len()) as u64,
                            10 * unicorn::SECOND_SCALE,
                            1024,
                        )
                    })
                    .and_then(|()| {
                        output_registers
                            .iter()
                            .map(|r| emu.reg_read_i32(*r))
                            .collect::<Result<Vec<i32>, unicorn::Error>>()
                    })
                    .map(|reg| (code, reg))
                    .map_err(Error::from);

                tx.send(res)
                    .map_err(Error::from)
            });
        }
        rx
    }
}


#[cfg(test)]
mod test {
    use std::iter;

    use log;
    use rand::Rng;

    use super::*;
    use unicorn::{RegisterX86, CpuX86};

    #[test]
    fn test_hatchery() {
        pretty_env_logger::env_logger::init();

        fn random_code() -> Vec<u8> {
            iter::repeat(())
                .take(1024)
                .map(|()| rand::random::<u8>())
                .collect()
        }

        let hatchery: Hatchery<CpuX86> = Hatchery::new(unicorn::Mode::MODE_32, 6);

        let expected_num = 1024;
        let mut counter = 0;
        let code_iterator = iter::repeat(())
            .take(expected_num)
            .map(|()| random_code());

        let output_registers = Arc::new(vec![RegisterX86::EAX]);
        hatchery.pipeline(code_iterator, output_registers).iter()
            .for_each(|out| {
                counter += 1;
                match out {
                    Ok((_code, regs)) => log::info!("Output: {:x?}", regs),
                    Err(e) => log::info!("Crash: {:?}", e),
                }
            });
        assert_eq!(counter, expected_num);

    }
}