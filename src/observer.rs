// A Logger needs to asynchronously gather and periodically
// record information on the evolutionary process.

use std::fs;
use std::io::Write;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};

use deflate::write::GzEncoder;
use deflate::Compression;
use hashbrown::HashSet;
use rand::seq::SliceRandom;
use serde::Serialize;

use non_dominated_sort::{non_dominated_sort, DominanceOrd};

// a hack to make the imports more meaningful
use crate::configure::Config;
use crate::evolution::{Genome, Phenome};
use crate::util::count_min_sketch::DecayingSketch;
use rand::thread_rng;

// TODO: we need to maintain a Pareto archive in the observation window.
// setting the best to be the lowest scalar fitness is wrong.
fn stat_writer(params: &Config) -> csv::Writer<fs::File> {
    let s = format!("{}/statistics.tsv", params.data_directory());
    let path = std::path::Path::new(&s);
    let add_headers = !path.exists();
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| log::error!("Error opening statistics file at {:?}: {:?}", path, e))
        .expect("Failed to open statistics file");
    csv::WriterBuilder::new()
        .delimiter(b'\t')
        .terminator(csv::Terminator::Any(b'\n'))
        .has_headers(add_headers)
        .from_writer(file)
}

pub struct Observer<O: Send> {
    pub handle: JoinHandle<()>,
    tx: Sender<O>,
    stop_signal_rx: Receiver<bool>,
}

pub type ReportFn<T, D> = Box<dyn Fn(&Window<T, D>, usize, &Config) -> () + Sync + Send + 'static>;

#[allow(dead_code)]
pub fn default_report_fn<P: Phenome + Genome, D: DominanceOrd<P>>(
    window: &Window<P, D>,
    counter: usize,
    _params: &Config,
) {
    let frame = &window.frame;
    log::info!("default report function");
    let avg_len = frame.iter().map(|c| c.len()).sum::<usize>() as f64 / frame.len() as f64;
    let mut sketch = DecayingSketch::default();
    for g in frame {
        g.record_genetic_frequency(&mut sketch);
    }
    let avg_freq = frame
        .iter()
        .map(|g| g.measure_genetic_frequency(&sketch))
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

pub struct Window<O: Phenome + 'static, D: DominanceOrd<O>> {
    pub frame: Vec<O>,
    pub params: Arc<Config>,
    counter: usize,
    report_every: usize,
    i: usize,
    window_size: usize,
    report_fn: ReportFn<O, D>,
    pub best: Option<O>,
    pub archive: Vec<O>,
    dominance_order: D,
    stat_writer: Arc<Mutex<csv::Writer<fs::File>>>,
    stop_signal_tx: Sender<bool>,
}

impl<O: Genome + Phenome + 'static, D: DominanceOrd<O>> Window<O, D> {
    fn new(
        report_fn: ReportFn<O, D>,
        params: Arc<Config>,
        dominance_order: D,
        stop_signal_tx: Sender<bool>,
    ) -> Self {
        let window_size = params.observer.window_size;
        let report_every = params.observer.report_every;
        assert!(window_size > 0, "window_size must be > 0");
        assert!(report_every > 0, "report_every must be > 0");
        let stat_writer = Arc::new(Mutex::new(stat_writer(&params)));
        Self {
            frame: Vec::with_capacity(window_size),
            params,
            counter: 0,
            i: 0,
            window_size,
            report_every,
            report_fn,
            best: None,
            archive: vec![],
            stat_writer,
            dominance_order,
            stop_signal_tx,
        }
    }

    pub fn stop_evolution(&self) {
        log::info!("Target condition reached. Halting the evolution...");
        if let Err(e) = self.stop_signal_tx.send(true) {
            log::error!("Error sending stop signal: {:?}", e);
        }
    }

    fn insert(&mut self, thing: O) {
        // Update the "best" seen so far, using scalar fitness measures
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
        // insert the incoming thing into the observation window
        self.i = (self.i + 1) % self.window_size;
        if self.frame.len() < self.window_size {
            self.frame.push(thing)
        } else {
            self.frame[self.i] = thing;
        }

        // Perform various periodic tasks
        self.counter += 1;
        if self.counter % self.params.observer.dump_every == 0 {
            self.dump_soup();
            self.dump_population();
        }
        if self.counter % self.params.pop_size == 0 {
            self.update_archive();
        }
        if self.counter % self.report_every == 0 {
            self.report();
        }

        if self.params.num_epochs != 0 && self.params.num_epochs <= crate::get_epoch_counter() {
            self.stop_evolution();
        }
    }

    fn update_archive(&mut self) {
        // TODO: optimize this. it's quite bad.

        let arena = self
            .archive
            .iter()
            .chain(self.frame.iter().filter(|g| g.front() == Some(0)))
            .cloned() // trouble non_dom sorting refs
            .collect::<Vec<O>>();

        // arena.sort_by(|a, b| {
        //     a.fitness()
        //         .partial_cmp(&b.fitness())
        //         .unwrap_or(Ordering::Equal)
        // });

        let front = non_dominated_sort(&arena, &self.dominance_order);
        let sample = front
            .current_front_indices()
            .choose_multiple(&mut thread_rng(), self.params.pop_size);

        self.archive = sample
            .into_iter()
            .map(|i| arena[*i].clone())
            .collect::<Vec<O>>();
    }

    fn report(&self) {
        (self.report_fn)(&self, self.counter, &self.params);
    }

    pub fn log_record<S: Serialize>(&self, record: S) {
        self.stat_writer
            .lock()
            .expect("poisoned lock on window's logger")
            .serialize(record)
            .map_err(|e| log::error!("Error logging record: {:?}", e))
            .expect("Failed to log record!");
        self.stat_writer
            .lock()
            .expect("poisoned lock on window's logger")
            .flush()
            .expect("Failed to flush");
    }

    pub fn dump_population(&self) {
        if !self.params.observer.dump_population {
            log::debug!("Not dumping population");
            return;
        }
        let path = format!(
            "{}/population/population_{}.json.gz",
            self.params.data_directory(),
            self.counter,
        );
        if let Ok(mut population_file) = fs::File::create(&path)
            .map_err(|e| log::error!("Failed to create population file: {:?}", e))
        {
            let mut gz = GzEncoder::new(Vec::new(), Compression::Default);
            let _ = serde_json::to_writer(&mut gz, &self.frame)
                .map_err(|e| log::error!("Failed to compress serialized population: {:?}", e));
            let compressed_population_data = gz.finish().expect("Failed to finish Gzip encoding");
            let _ = population_file
                .write_all(&compressed_population_data)
                .map_err(|e| log::error!("Failed to write population file: {:?}", e));
            log::info!("Population dumped to {}", path);
        };
    }

    pub fn dump_soup(&self) {
        if !self.params.observer.dump_soup {
            log::debug!("Not dumping soup");
            return;
        }
        let mut soup = self
            .frame
            .iter()
            .map(|g| g.chromosome())
            .flatten()
            .cloned()
            .collect::<HashSet<_>>();
        let path = format!(
            "{}/soup/soup_{}.json",
            self.params.data_directory(),
            self.counter,
        );
        if let Ok(mut soup_file) =
            fs::File::create(&path).map_err(|e| log::error!("Failed to create soup file: {:?}", e))
        {
            let soup_vec = soup.drain().collect::<Vec<_>>();
            serde_json::to_writer(&mut soup_file, &soup_vec).expect("Failed to dump soup!");
            log::info!("Soup dumped to {}", path);
        }
    }
}

impl<O: 'static + Phenome + Genome> Observer<O> {
    /// The observe method should take a clone of the observable
    /// and store in something like a sliding observation window.
    pub fn observe(&self, ob: O) {
        self.tx.send(ob).expect("tx failure");
    }

    pub fn spawn<D: 'static + DominanceOrd<O> + Send>(
        params: &Config,
        report_fn: ReportFn<O, D>,
        dominance_order: D,
    ) -> Observer<O> {
        let (tx, rx): (Sender<O>, Receiver<O>) = channel();
        let (window_tx, stop_signal_rx): (Sender<bool>, Receiver<bool>) = channel();

        let params = Arc::new(params.clone());
        let handle: JoinHandle<()> = spawn(move || {
            let mut window: Window<O, D> =
                Window::new(report_fn, params.clone(), dominance_order, window_tx);
            for observable in rx {
                window.insert(observable);
            }
        });

        Observer {
            handle,
            tx,
            stop_signal_rx,
        }
    }

    pub fn keep_going(&self) -> bool {
        if let Ok(true) = self.stop_signal_rx.try_recv() {
            false
        } else {
            true
        }
    }
}
