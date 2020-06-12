// for island shit
use crate::configure::Config;
use crate::evolution::Phenome;
use crossbeam_deque::{self as deque, Stealer, Worker};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{spawn, JoinHandle};

#[derive(Clone)]
pub struct Pier<P: Phenome> {
    dock: deque::Stealer<P>,
    pub inbound: Sender<P>,
    //handle: JoinHandle<()>,
}

impl<P: Phenome + 'static> Pier<P> {
    pub fn spawn(config: &Config) -> Self {
        let (tx, rx): (Sender<P>, Receiver<P>) = channel();
        let ship: Worker<P> = Worker::new_fifo();
        let dock = ship.stealer();

        let handle = spawn(move || {
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

    pub fn steal(&self) -> Option<P> {
        if let Steal::Data(p) = self.dock.steal() {
            Some(p)
        } else {
            None
        }
    }
}
