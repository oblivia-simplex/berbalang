use std::sync::Arc;
use std::sync::mpsc::{channel, Receiver, RecvError, Sender, SendError};
use std::thread::{JoinHandle, spawn};

use rand::{Rng, thread_rng};
use threadpool::ThreadPool;
use unicorn::{Cpu, CpuX86, uc_handle};

struct Nest {
    handle: JoinHandle<()>,
    tx: Sender<Code>,
}

struct Hatchery {
    nests: Vec<Nest>,
    rx: Receiver<Result<(Code, i32), Error>>,
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

fn exec(emu: &mut CpuX86, code: &Code) -> Result<i32, unicorn::Error> {
    // NOTE: we're being lazy here. We should be zeroing out the registers
    // and the writeable memory, but the point here is just to figure out
    // a viable parallelization method.
    emu.mem_write(0x1000, &code)?;
    emu.emu_start(
        0x1000,
        (0x1000 + code.len()) as u64,
        10 * unicorn::SECOND_SCALE,
        1024,
    )?;
    emu.reg_read_i32(unicorn::RegisterX86::EAX)
}


// So far, there's not many advantages to this model over just holding the cpu in a field
// of the struct. All we get, essentially, is a form of interior mutability with an immutable
// shell.
// But what if we created a pool of inner threads, each with a unicorn, and cloned
// the `our_tx` sender? Then each would send work back to the same receiver, which
// connects us back to the outside.
impl Hatchery {
    pub fn spawn(mode: unicorn::Mode, num_workers: usize) -> Self {
        let (hatchery_egress, hatchery_ingress) = channel::<Result<(Code, i32), Error>>();

        // Each inner thread has permanent ownership of a unicorn emulator.
        let mut nests = Vec::new();
        for _ in 0..num_workers {
            let hatchery_egress = hatchery_egress.clone();
            let (nest_ingress, nest_egress) = channel::<Code>();
            let handle = spawn(move || {
                let mut emu = CpuX86::new(mode).expect("failed to instantiate emu");
                emu.mem_map(0x1000, 0x4000, unicorn::Protection::ALL)
                    .expect("mem_map failure");

                for code in nest_egress.iter() {
                    let res = exec(&mut emu, &code).map(|r| (code, r))
                        .map_err(Error::from);
                    hatchery_egress.send(res).expect("our_tx failure");
                }
            });
            nests.push(Nest { tx: nest_ingress, handle });
        }

        Self { nests, rx: hatchery_ingress }
    }

    // There's no guarantee that the code returned == the code sent!
    // But eventually all code sent will be returned.
    pub fn submit(&self, code: Code) -> Result<(Code, i32), Error> {
        // TODO: improve this nest selection method. For now, random.
        let mut rng = thread_rng();
        let n = rng.gen_range(0, self.nests.len());
        self.nests[n].tx.send(code)?;
        self.rx.recv().map_err(Error::from)?
    }
}

#[cfg(test)]
mod test {
    use rand::Rng;

    use super::*;

    #[test]
    fn test_hatchery() {
        let hatchery = Hatchery::spawn(unicorn::Mode::MODE_32, 16);
        for _ in 0..100 {
            let code = rand::thread_rng().gen::<[u8; 32]>().to_vec();
            println!("Code: {:02x?}", code);
            match hatchery.submit(code.clone()) {
                Ok((ret_code, res)) => {
                    println!("EAX = 0x{:0x}", res);
                    assert_eq!(ret_code, code);
                }
                Err(e) => println!("Hatchery Error: {:?}", e),
            }
        }
    }
}
