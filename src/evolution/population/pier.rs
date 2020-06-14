use std::sync::mpsc::{channel, Receiver, SendError, Sender};
use std::thread::spawn;

use crossbeam_deque::{self as deque, Steal, Worker};

// for island shit
use crate::configure::Config;
use crate::evolution::Phenome;

#[derive(Clone)]
pub struct Pier<P: Phenome> {
    dock: deque::Stealer<P>,
    inbound: Sender<P>,
    //handle: JoinHandle<()>,
}

impl<P: Phenome + 'static> Pier<P> {
    pub fn spawn(_config: &Config) -> Self {
        let (tx, rx): (Sender<P>, Receiver<P>) = channel();
        let ship: Worker<P> = Worker::new_fifo();
        let dock = ship.stealer();

        let _handle = spawn(move || {
            for incoming in rx.into_iter() {
                log::info!("incoming");
                ship.push(incoming)
            }
        });

        Self {
            dock,
            inbound: tx,
            //handle,
        }
    }

    pub fn embark(&self, traveller: P) -> Result<(), SendError<P>> {
        self.inbound.send(traveller)
    }

    pub fn disembark(&self) -> Option<P> {
        if let Steal::Success(p) = self.dock.steal() {
            Some(p)
        } else {
            None
        }
    }
}
