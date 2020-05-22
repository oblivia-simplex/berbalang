use std::sync::mpsc::{channel, Receiver, RecvError, SendError, Sender};
use std::sync::Arc;
use std::thread::{sleep, spawn, JoinHandle};

use rand::{thread_rng, Rng};
use std::time::Duration;
use threadpool::ThreadPool;
use unicorn::{uc_handle, Cpu, CpuX86};
use object_pool::Pool;


pub struct Hatchery<C: Cpu<'static> + Send> {
    pool: Arc<Pool<C>>,
    mode: unicorn::Mode,
}

/*
trait Hatch<C: Cpu<'static>> {
    fn exec(emu: &mut C, code: &Code) -> Result<i32, unicorn::Error>;
}

impl Hatch<CpuX86<'static>> for Hatchery<CpuX86<'static>> {
    fn exec(emu: &mut CpuX86, code: &Code) -> Result<i32, unicorn::Error> {
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
        emu.reg_read_i32(unicorn::RegisterX86::EAX)
    }
}
*/

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

impl<C: 'static + Cpu<'static> + Send> Hatchery<C> {
    // TODO: it would be nice if the unicorn::Mode were just determined
    // by the Cpu type, though this wouldn't work nicely for ARM/Thumb
    pub fn new(mode: unicorn::Mode, num_workers: usize) -> Self {
        let pool = Arc::new(Pool::new(num_workers, || init_emu(mode).unwrap()));
        Self {
            pool,
            mode,
        }
    }

    pub fn pipeline<'a, I: Iterator<Item=Code>>(&self, inbound: I) -> Receiver<Result<Vec<u8>, Error>> {
        let (tx, rx) = channel::<Result<Vec<u8>, Error>>();
        let mode = self.mode;
        for code in inbound {
            let pool = self.pool.clone();
            let tx = tx.clone();
            let handle = spawn(move || -> Result<(), Error> {
                let mut emu = pool.pull(|| {
                    log::warn!("Pool empty, creating new emulator");
                    init_emu::<C>(mode).expect("Failed to create new emulator")
                });
                log::trace!("executing code {:02x?} after nap", code);
                //sleep(Duration::from_millis(1000));
                emu.mem_write(0x1000, &code)?;
                emu.emu_start(
                    0x1000,
                    (0x1000 + code.len()) as u64,
                    10 * unicorn::SECOND_SCALE,
                    1024,
                )?;
                let res: Result<Vec<u8>, Error> = emu.mem_read_as_vec(0x3000, 0x1000)
                    .map_err(Error::from);
                tx.send(res)
                    .map_err(Error::from)
            });
        }
        rx
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



fn iter_stuff() {
    let hatchery: Hatchery<CpuX86> = Hatchery::new(unicorn::Mode::MODE_32, 16);

    let code_iterator = iter::repeat(())
        .take(1024)
        .map(|()| random_code);

    let rx: Receiver<_> = hatchery.pipeline(code_iterator);
    for x in rx.iter() {
        println!("Returned {:x?}", x);
    }
}


#[cfg(test)]
mod test {
    use rand::Rng;

    use super::*;
    use std::iter;
    use log;

    #[test]
    fn test_hatchery() {
        pretty_env_logger::env_logger::init();

        fn random_code() -> Vec<u8> {
            iter::repeat(())
                .take(1024)
                .map(|()| rand::random::<u8>())
                .collect()
        }

        let hatchery: Hatchery<CpuX86> = Hatchery::new(unicorn::Mode::MODE_32, 16);

        let code_iterator = iter::repeat(())
            .take(1024)
            .map(|()| random_code);

        let rx: Receiver<_> = hatchery.pipeline(code_iterator);
        for x in rx.iter() {
            println!("Returned {:x?}", x);
        }
        //hatchery.pipeline(it).iter()
        //    .for_each(|c| {
        //        println!("Returned: {:x?}", c);
        //    });

    }
}