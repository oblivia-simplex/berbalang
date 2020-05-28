// A Logger needs to asynchronously gather and periodically
// record information on the evolutionary process.

use std::sync::Arc;
// a hack to make the imports more meaningful
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{spawn, JoinHandle};

use crate::configure::Configure;
use crate::evolution::Phenome;

pub struct Observer<O: Send> {
    pub handle: JoinHandle<()>,
    tx: Sender<O>,
    // TODO: add a reporter struct field
}

pub type ReportFn<T> = Box<dyn Fn(&[T]) -> () + Sync + Send + 'static>;

pub struct Window<O> {
    pub frame: Vec<O>,
    i: usize,
    window_size: usize,
    report_fn: ReportFn<O>,
}

impl<O> Window<O> {
    fn new(window_size: usize, report_fn: ReportFn<O>) -> Self {
        assert!(window_size > 0);
        Self {
            frame: Vec::new(),
            i: 0,
            window_size,
            report_fn,
        }
    }

    fn insert(&mut self, thing: O) {
        self.i = (self.i + 1) % self.window_size;
        if self.frame.len() < self.window_size {
            self.frame.push(thing)
        } else {
            self.frame[self.i] = thing;
        }
        if self.i == 0 {
            self.report();
        }
    }

    fn report(&self) {
        (self.report_fn)(&self.frame);
    }
}

impl<O: 'static + Send> Observer<O> {
    /// The observe method should take a clone of the observable
    /// and store in something like a sliding observation window.
    pub fn observe(&self, ob: O) {
        self.tx.send(ob).expect("tx failure");
    }

    pub fn spawn<C: Configure>(params: Arc<C>, report_fn: ReportFn<O>) -> Observer<O> {
        let (tx, rx): (Sender<O>, Receiver<O>) = channel();

        let window_size: usize = params.observer_window_size();

        let handle: JoinHandle<()> = spawn(move || {
            let mut window: Window<O> = Window::new(window_size, report_fn);
            for observable in rx {
                window.insert(observable);
            }
        });

        Observer { handle, tx }
    }
}
