// A Logger needs to asynchronously gather and periodically
// record information on the evolutionary process.

use std::sync::Arc;
// a hack to make the imports more meaningful
use serde::Deserialize;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{spawn, JoinHandle};

use crate::configure::Configure;

pub struct Observer<O: Send> {
    pub handle: JoinHandle<()>,
    tx: Sender<O>,
    // TODO: add a reporter struct field
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObserverConfig {
    pub window_size: usize,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            window_size: 0x1000,
        }
    }
}

pub type ReportFn<T> = Box<dyn Fn(&[T], usize, &ObserverConfig) -> () + Sync + Send + 'static>;

pub struct Window<'a, O> {
    pub frame: Vec<O>,
    pub params: &'a ObserverConfig,
    counter: usize,
    i: usize,
    window_size: usize,
    report_fn: ReportFn<O>,
}

impl<'a, O> Window<'a, O> {
    fn new(report_fn: ReportFn<O>, params: &'a ObserverConfig) -> Self {
        let window_size = params.window_size;
        assert!(window_size > 0);
        Self {
            frame: Vec::new(),
            params,
            counter: 0,
            i: 0,
            window_size,
            report_fn,
        }
    }

    fn insert(&mut self, thing: O) {
        self.counter += 1;
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
        (self.report_fn)(&self.frame, self.counter, &self.params);
    }
}

impl<O: 'static + Send> Observer<O> {
    /// The observe method should take a clone of the observable
    /// and store in something like a sliding observation window.
    pub fn observe(&self, ob: O) {
        self.tx.send(ob).expect("tx failure");
    }

    pub fn spawn<C: 'static + Configure>(params: Arc<C>, report_fn: ReportFn<O>) -> Observer<O> {
        let (tx, rx): (Sender<O>, Receiver<O>) = channel();

        let handle: JoinHandle<()> = spawn(move || {
            let observer_config = params.observer_config();
            let mut window: Window<'_, O> = Window::new(report_fn, &observer_config);
            for observable in rx {
                window.insert(observable);
            }
        });

        Observer { handle, tx }
    }
}
