// A Logger needs to asynchronously gather and periodically
// record information on the evolutionary process.

use std::fmt::Debug;
use std::fs;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{self, AtomicUsize};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};

use hashbrown::HashMap;

use crate::configure::Config;
use crate::evolution::{Genome, Phenome};
use crate::util::count_min_sketch::CountMinSketch;
use crate::util::dump::dump;

// TODO: fix the stat writer so that it uses the header() and row() functions.

// TODO: we need to maintain a Pareto archive in the observation window.
// setting the best to be the lowest scalar fitness is wrong.
fn get_log_filename(name: &str, config: &Config) -> String {
    format!("{}/{}_statistics.csv", config.data_directory(), name)
}

// fn stat_writer(config: &Config, name: &str) -> csv::Writer<fs::File> {
//     let s = get_log_filename(name, config);
//     let path = std::path::Path::new(&s);
//     let add_headers = !path.exists();
//     let file = fs::OpenOptions::new()
//         .create(true)
//         .append(true)
//         .open(path)
//         .map_err(|e| log::error!("Error opening statistics file at {:?}: {:?}", path, e))
//         .expect("Failed to open statistics file");
//     csv::WriterBuilder::new()
//         .delimiter(b',')
//         .terminator(csv::Terminator::Any(b'\n'))
//         .has_headers(add_headers)
//         .from_writer(file)
// }

pub struct Observer<O: Send> {
    pub handle: JoinHandle<()>,
    tx: Sender<O>,
}

pub type ReportFn<T> = Box<dyn Fn(&Window<T>, usize, &Config) -> () + Sync + Send + 'static>;

#[allow(dead_code)]
pub fn default_report_fn<P: Phenome + Genome>(window: &Window<P>, counter: usize, config: &Config) {
    let frame = &window.frame;
    log::info!("default report function");
    let avg_len = frame.iter().map(|c| c.len()).sum::<usize>() as f64 / frame.len() as f64;
    let mut sketch = CountMinSketch::new(config);
    for g in frame {
        g.record_genetic_frequency(&mut sketch);
    }
    let avg_freq = frame
        .iter()
        .map(|g| g.query_genetic_frequency(&sketch))
        .sum::<f64>()
        / frame.len() as f64;
    let avg_fit: f64 = frame
        .iter()
        .filter_map(|g| g.scalar_fitness(&window.config.fitness.weighting))
        .sum::<f64>()
        / frame.len() as f64;
    log::info!(
        "[{}] Average length: {}, average genetic frequency: {}, avg scalar fit: {}",
        counter,
        avg_len,
        avg_freq,
        avg_fit,
    );
    log::info!("[{}] Reigning champion: {:#?}", counter, window.best);
}

fn epoch_length(config: &Config) -> usize {
    // TODO: have this switch on selection type, if you go back to
    // experimenting with Roulette, etc. Right now, a Tournament is assumed
    config.pop_size / config.tournament.num_offspring
}

pub struct Window<O: Phenome + 'static> {
    pub frame: Vec<O>,
    window_size: usize,
    pub config: Arc<Config>,
    counter: usize,
    i: usize,
    report_fn: ReportFn<O>,
    pub best: Option<O>,
    pub champion: Option<O>,
    // priority fitness best
    pub archive: Vec<O>,
    pub local_epoch: AtomicUsize,
    // stat_writers: HashMap<&'static str, Arc<Mutex<csv::Writer<fs::File>>>>,
}

impl<O: Genome + Phenome + 'static> Window<O> {
    fn new(report_fn: ReportFn<O>, config: Arc<Config>) -> Self {
        let window_size = epoch_length(&config);
        Self {
            frame: Vec::with_capacity(window_size),
            window_size,
            config,
            counter: 0,
            i: 0,
            report_fn,
            best: None,
            champion: None,
            archive: vec![],
            local_epoch: AtomicUsize::new(0),
        }
    }

    pub fn is_halting_condition_reached(&self) {
        // This is how you check for an unweighted fitness value.
        // This shows the advantage of having Weighted fitness as
        // as distinct type, which returns its scalar value through
        // a method -- the components are easily retrievable in their
        // raw state.

        let epoch_limit_reached =
            self.config.num_epochs != 0 && self.config.num_epochs <= crate::get_epoch_counter();
        if epoch_limit_reached {
            log::debug!("epoch limit reached");
            self.report();
            self.dump_soup();
            self.dump_population();
            crate::stop_everything(self.config.island_id, false);
        }

        if let Some(ref champion) = self.champion {
            if champion.is_goal_reached(&self.config) {
                let path = format!("{}/winning_champion.json.gz", self.config.data_directory());
                log::info!("dumping winning champion to {}", path);
                dump(champion, &path).expect("failed to dump champion");
                self.report();
                crate::stop_everything(self.config.island_id, true);
            }
        }
    }

    pub fn get_local_epoch(&self) -> usize {
        self.local_epoch.load(atomic::Ordering::Relaxed)
    }

    fn maybe_increment_epoch(&self) -> bool {
        // A generation should be considered to have elapsed once
        // `pop_size` offspring have been spawned.
        // For now, only Island 0 can increment the epoch. We can weigh the
        // pros and cons of letting each island have its own epoch, later.
        if self.counter > 0 && self.counter % self.window_size == 0 {
            let epoch = self.local_epoch.fetch_add(1, atomic::Ordering::Relaxed) + 1;
            log::info!(
                "Epoch {} on island {} ({} specimens seen)",
                epoch,
                self.config.island_id,
                self.counter
            );

            if self.config.island_id == 0 {
                let global_epoch = crate::increment_epoch_counter();
                log::info!("New global epoch: {}", global_epoch);
            }
            true
        } else {
            false
        }
    }

    fn insert(&mut self, thing: O) {
        self.update_best(&thing);
        self.update_champion(&thing);

        // insert the incoming thing into the observation window
        self.i = (self.i + 1) % self.window_size;
        if self.frame.len() < self.window_size {
            self.frame.push(thing)
        } else {
            self.frame[self.i] = thing;
        }

        // Perform various periodic tasks
        self.counter += 1;

        let epoch_has_incremented = self.maybe_increment_epoch();

        if epoch_has_incremented {
            self.dump_soup();
            self.dump_population();
            self.report();
        }

        self.is_halting_condition_reached();
    }

    fn update_best(&mut self, specimen: &O) {
        let mut updated = false;
        if let Some(specimen_fitness) = specimen.scalar_fitness(&self.config.fitness.weighting) {
            match self.best.as_ref() {
                None => {
                    updated = true;
                    self.best = Some(specimen.clone())
                }
                Some(champ) => {
                    if specimen_fitness
                        < champ
                            .scalar_fitness(&self.config.fitness.weighting)
                            .expect("There should be a fitness score here")
                    {
                        updated = true;
                        self.best = Some(specimen.clone())
                    }
                }
            }
        }

        if updated {
            log::info!(
                "Island {}: new best:\n{:#?}",
                self.config.island_id,
                self.best.as_ref().expect("Updated, but no best?")
            );
        }
    }

    fn update_champion(&mut self, specimen: &O) {
        let mut updated = false;
        if let Some(specimen_fitness) = specimen.scalar_fitness(&self.config.fitness.priority()) {
            match self.champion.as_ref() {
                None => {
                    updated = true;
                    self.champion = Some(specimen.clone())
                }
                Some(champ) => {
                    if specimen_fitness
                        < champ
                            .scalar_fitness(&self.config.fitness.priority())
                            .expect("there should be a fitness score here")
                    {
                        updated = true;
                        self.champion = Some(specimen.clone())
                    }
                }
            }
        }

        if updated {
            if let Some(ref mut champion) = self.champion {
                champion.generate_description();
                log::info!(
                    "Island {}: new champion:\n{:#?}",
                    self.config.island_id,
                    champion
                );
                // dump the champion
                let path = format!(
                    "{}/champions/champion_{}.json.gz",
                    self.config.data_directory(),
                    self.counter,
                );
                log::info!("Dumping new champion to {}", path);
                dump(champion, &path).expect("Failed to dump champion");
                let latest = format!(
                    "{}/champions/latest_champion.json.gz",
                    self.config.data_directory()
                );
                let latest = Path::new(&latest);
                if latest.exists() {
                    fs::remove_file(latest).expect("failed to remove old symlink");
                }
                std::os::unix::fs::symlink(path, latest).expect("Failed to make symlink");
            }
        }
    }

    fn report(&self) {
        (self.report_fn)(&self, self.counter, &self.config);
    }

    pub fn log_record<S: LogRecord + Debug>(&self, record: S, name: &str) {
        log::debug!(
            "Island {}, logging to {}: {:#?}",
            self.config.island_id,
            name,
            record
        );
        let filename = get_log_filename(name, &self.config);
        // check to see if file exists yet
        let msg = if !Path::exists((&filename).as_ref()) {
            log::debug!("Creating header for {}", filename);
            format!("{}\n{}\n", record.header(), record.row())
        } else {
            format!("{}\n", record.row())
        };
        let fd = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&filename)
            .expect("Failed to open log file");
        let mut w = BufWriter::new(fd);
        write!(w, "{}", msg).expect("Failed to log row");

        // self.stat_writers[name]
        //     .lock()
        //     .expect("poisoned lock on window's logger")
        //     .serialize(record)
        //     .map_err(|e| log::error!("Error logging record: {:?}", e))
        //     .expect("Failed to log record!");
        // self.stat_writers[name]
        //     .lock()
        //     .expect("poisoned lock on window's logger")
        //     .flush()
        //     .expect("Failed to flush");
    }

    pub fn dump_population(&self) {
        if !self.config.observer.dump_population {
            log::debug!("Not dumping population");
            return;
        }
        let path = format!(
            "{}/population/population_{}.json.gz",
            self.config.data_directory(),
            self.get_local_epoch(),
        );
        dump(&self.frame, &path).expect("Failed to dump population");
    }

    pub fn soup(&self) -> HashMap<<O as Genome>::Allele, usize> {
        let mut map: HashMap<<O as Genome>::Allele, usize> = HashMap::new();
        self.frame
            .iter()
            .map(|g| g.chromosome())
            .flatten()
            .cloned()
            .for_each(|a| {
                *map.entry(a).or_insert(0) += 1;
            });
        map
    }

    pub fn dump_soup(&self) {
        if !self.config.observer.dump_soup {
            log::debug!("Not dumping soup");
            return;
        }
        let mut soup = self.soup();
        log::debug!(
            "Island {} soup size: {} alleles",
            self.config.island_id,
            soup.len()
        );
        let path = format!(
            "{}/soup/soup_at_epoch_{}.json",
            self.config.data_directory(),
            self.get_local_epoch(),
        );
        if let Ok(mut soup_file) =
            fs::File::create(&path).map_err(|e| log::error!("Failed to create soup file: {:?}", e))
        {
            let soup_vec = soup.drain().collect::<Vec<_>>();
            serde_json::to_writer(&mut soup_file, &soup_vec).expect("Failed to dump soup!");
            log::debug!("Soup dumped to {}", path);
        }
    }
}

impl<O: 'static + Phenome + Genome> Observer<O> {
    /// The observe method should take a clone of the observable
    /// and store in something like a sliding observation window.
    pub fn observe(&self, ob: O) {
        self.tx.send(ob).expect("tx failure");
    }

    pub fn spawn(config: &Config, report_fn: ReportFn<O>) -> Observer<O> {
        let (tx, rx): (Sender<O>, Receiver<O>) = channel();

        let config = Arc::new(config.clone());
        let handle: JoinHandle<()> = spawn(move || {
            let mut window: Window<O> = Window::new(report_fn, config.clone());
            for observable in rx {
                window.insert(observable);
            }
        });

        Observer { handle, tx }
    }

    // pub fn stop_evolution(&mut self) {
    //     self.stop_flag = true
    // }
}

pub trait LogRecord {
    fn header(&self) -> String;
    fn row(&self) -> String;
}
