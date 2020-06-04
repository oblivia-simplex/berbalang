// A Logger needs to asynchronously gather and periodically
// record information on the evolutionary process.

use std::sync::Arc;
// a hack to make the imports more meaningful
use crate::configure::{Config, ObserverConfig};
use crate::evolution::{Genome, Phenome};
use crate::util::count_min_sketch::DecayingSketch;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{spawn, JoinHandle};

// TODO: we need to maintain a Pareto archive in the observation window.
// setting the best to be the lowest scalar fitness is wrong.

pub struct Observer<O: Send> {
    pub handle: JoinHandle<()>,
    tx: Sender<O>,
    // TODO: add a reporter struct field
}

pub type ReportFn<T> =
    Box<dyn Fn(&Window<T>, usize, &ObserverConfig) -> () + Sync + Send + 'static>;

pub fn default_report_fn<T: Phenome + Genome>(
    window: &Window<T>,
    counter: usize,
    _params: &ObserverConfig,
)
// where
//     <<T as Phenome>::Fitness as Index<usize>>::Output: Sized,
{
    let frame = &window.frame;
    log::info!("default report function");
    let avg_len = frame.iter().map(|c| c.len()).sum::<usize>() as f64 / frame.len() as f64;
    let mut sketch = DecayingSketch::default();
    for g in frame {
        g.record_genetic_frequency(&mut sketch, 1000).unwrap();
    }
    let avg_freq = frame
        .iter()
        .map(|g| g.measure_genetic_frequency(&sketch).unwrap())
        .sum::<f64>()
        / frame.len() as f64;
    let avg_fit: f64 =
        frame.iter().filter_map(|g| g.scalar_fitness()).sum::<f64>() / frame.len() as f64;
    log::info!(
        "[{}] Average length: {}, average genetic frequency: {}, avg scalar fit: {}",
        counter,
        avg_len,
        avg_freq,
        avg_fit,
    );
    log::info!("[{}] Reigning champion: {:#?}", counter, window.best);
}

pub struct Window<O: Phenome + 'static + Send> {
    pub frame: Vec<O>,
    pub params: ObserverConfig,
    counter: usize,
    report_every: usize,
    i: usize,
    window_size: usize,
    report_fn: ReportFn<O>,
    pub best: Option<O>,
}

impl<O: Phenome + Genome + 'static> Default for Window<O> {
    fn default() -> Self
// where
    //     <<O as Phenome>::Fitness as Index<usize>>::Output: Sized,
    {
        let params = ObserverConfig::default();
        let window_size = params.window_size;
        let frame = Vec::with_capacity(window_size);
        let counter = 0;
        let report_every = params.report_every;
        let i = 0;
        let report_fn = Box::new(default_report_fn);

        Self {
            params,
            window_size,
            frame,
            counter,
            report_every,
            i,
            report_fn,
            best: None,
        }
    }
}

impl<O: Phenome + 'static + Send> Window<O> {
    fn new(report_fn: ReportFn<O>, params: ObserverConfig) -> Self {
        assert!(params.window_size > 0, "window_size must be > 0");
        assert!(params.report_every > 0, "report_every must be > 0");
        let window_size = params.window_size;
        let report_every = params.report_every;
        Self {
            frame: Vec::with_capacity(params.window_size),
            params,
            counter: 0,
            i: 0,
            window_size,
            report_every,
            report_fn,
            best: None,
        }
    }

    fn insert(&mut self, thing: O) {
        match &self.best {
            None => self.best = Some(thing.clone()),
            Some(champ) => {
                if let (Some(champ_fit), Some(thing_fit)) =
                    (champ.scalar_fitness(), thing.scalar_fitness())
                {
                    if thing_fit < champ_fit {
                        self.best = Some(thing.clone())
                    }
                }
            }
        }
        self.counter += 1;
        self.i = (self.i + 1) % self.window_size;
        if self.frame.len() < self.window_size {
            self.frame.push(thing)
        } else {
            self.frame[self.i] = thing;
        }
        if self.counter % self.report_every == 0 {
            self.report();
        }
    }

    fn report(&self) {
        (self.report_fn)(&self, self.counter, &self.params);
    }
}

impl<O: 'static + Send + Phenome> Observer<O> {
    /// The observe method should take a clone of the observable
    /// and store in something like a sliding observation window.
    pub fn observe(&self, ob: O) {
        self.tx.send(ob).expect("tx failure");
    }

    pub fn spawn(params: &Config, report_fn: ReportFn<O>) -> Observer<O> {
        let (tx, rx): (Sender<O>, Receiver<O>) = channel();
        let params = Arc::new(params.clone());
        let handle: JoinHandle<()> = spawn(move || {
            let observer_config = params.observer_config();
            let mut window: Window<O> = Window::new(report_fn, observer_config);
            for observable in rx {
                window.insert(observable);
            }
        });

        Observer { handle, tx }
    }
}
