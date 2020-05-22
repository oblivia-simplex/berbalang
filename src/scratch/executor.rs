use std::sync::mpsc::{channel, Receiver, RecvError, SendError, Sender};
use std::sync::Arc;
use std::thread::{sleep, spawn, JoinHandle};

use rand::{thread_rng, Rng};
use std::time::Duration;
use threadpool::ThreadPool;
use unicorn::{uc_handle, Cpu, CpuX86};
use object_pool::Pool;


struct Hatchery<C: Cpu<'static>> {
    pool: Arc<Pool<C>>,
    mode: unicorn::Mode,
}

#[derive(Debug)]
enum Error {
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

fn exec<C: Cpu<'static>>(emu: &mut C, code: &Code) -> Result<i32, unicorn::Error> {
    // NOTE: we're being lazy here. We should be zeroing out the registers
    // and the writeable memory, but the point here is just to figure out
    // a viable parallelization method.
    // NOTE: we're adding a sleep here to see how things perform.
    log::trace!("executing code {:02x?} after nap", code);
    //sleep(Duration::from_millis(1000));
    emu.mem_write(0x1000, &code)?;
    emu.emu_start(
        0x1000,
        (0x1000 + code.len()) as u64,
        10 * unicorn::SECOND_SCALE,
        1024,
    )?;
    let reg: <C as Cpu<'_>>::Reg = 0.into();
    emu.reg_read_i32(reg)
    // FIXME: needs to be made genetic // emu.reg_read_i32(unicorn::RegisterX86::EAX)
}

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

impl<C: Cpu<'static>> Hatchery<C> {
    // TODO: it would be nice if the unicorn::Mode were just determined
    // by the Cpu type, though this wouldn't work nicely for ARM/Thumb
    pub fn new(mode: unicorn::Mode, num_workers: usize) -> Self {
        let pool = Arc::new(Pool::new(num_workers, || init_emu(mode).unwrap()));
        Self {
            pool,
            mode,
        }
    }

    pub fn pipeline<'a, I: IntoIterator<Item=&'a Code>>(&self, inbound: I) -> Receiver<Result<i32, Error>> {
        let (tx, rx) = channel::<Result<i32, Error>>();
        for code in inbound {
            let pool = self.pool.clone();
            let handle = spawn(move || -> Result<(), Error> {
                let mut emu = pool.pull(|| {
                    log::warn!("Pool empty, creating new emulator");
                    init_emu::<C>(self.mode).expect("Failed to create new emulator")
                });
                let res = exec(&mut emu, &code)
                    .map(|r| (code, r))
                    .map_err(Error::from);
                tx.res
            });
        }
        rx.iter()
    }
}

    /* pub fn spawn(mode: unicorn::Mode, num_workers: usize) -> Self {
        let (hatchery_egress, hatchery_ingress) = channel::<Result<(Code, i32), Error>>();

        // TODO: we don't really need to double up on the parallelism. remove the nest carousel
        // TODO: try something with fewer channels
        // Each inner thread has permanent ownership of a unicorn emulator.
        let mut nests = Vec::new();
        for _ in 0..num_workers {
            let hatchery_egress = hatchery_egress.clone();
            let (nest_ingress, nest_egress) = channel::<Code>();
            let handle = spawn(move || -> Result<(), Error> {
                let pool = Arc::new(object_pool::Pool::new(num_workers, || init_emu(mode).unwrap()));
                for code in nest_egress.iter() {
                    let hatchery_egress = hatchery_egress.clone();
                    let pool = pool.clone();
                    let _h = spawn(move || -> Result<(), Error>  {
                        let mut emu = pool.pull(|| {
                            log::warn!("Pool empty, creating new emulator");
                            init_emu(mode).unwrap()
                        });

                        let res = exec(&mut emu, &code)
                            .map(|r| (code, r))
                            .map_err(Error::from);
                        notice!(hatchery_egress.send(res))?;
                        Ok(())
                    });
                }
                Ok(())
            });
            nests.push(Nest {
                tx: nest_ingress,
                handle,
            });
        }

        Self {
            nests,
            rx: hatchery_ingress,
        }
    }

     */






#[cfg(test)]
mod test {
    use rand::Rng;

    use super::*;
    use std::iter;

    #[test]
    fn test_hatchery() {
        pretty_env_logger::env_logger::init();
        let hatchery = Hatchery::spawn(unicorn::Mode::MODE_32, 8);
        let mut rng = thread_rng();
        let futs = iter::repeat(())
            .take(1000)
            .map(|()| {
                let code = iter::repeat(()).take(rng.gen_range(10,1000))
                    .map(|()| rng.gen::<u8>())
                    .collect();
                hatchery.execute(code)
            })
            .collect::<Vec<_>>();
        let results = futures::executor::block_on(futures::future::join_all(futs));
        for res in results {
            match res {
                Ok((ret_code, res)) => {
                    println!("Code = {:02x?}", ret_code);
                    println!("EAX = 0x{:0x}", res);
                }
                Err(e) => println!("Hatchery Error: {:?}", e),
            }
        }
        /*
        let codes = iter::repeat(()).map(|()| rand::thread_rng().gen::<[u8;32]>().to_vec()).take(10_000).collect::<Vec<Code>>();
        let results = hatchery.execute_batch(codes);
        for res in results {
            match res {
                Ok((ret_code, res)) => {
                    println!("EAX = 0x{:08x}", res);
                }
                Err(e) => println!("Hatchery Error: {:?}", e),
            }
        }

         */
    }
}
