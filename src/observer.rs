// A Logger needs to asynchronously gather and periodically
// record information on the evolutionary process.

use std::fs;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};

use hashbrown::HashMap;
use rand::seq::SliceRandom;
use serde::Serialize;

use non_dominated_sort::{non_dominated_sort, DominanceOrd};

use crate::configure::Config;
use crate::evolution::{Genome, Phenome};
use crate::hashmap;
use crate::util::count_min_sketch::CountMinSketch;
use crate::util::dump::dump;
use crate::util::random::hash_seed_rng;
use std::path::Path;

// TODO: we need to maintain a Pareto archive in the observation window.
// setting the best to be the lowest scalar fitness is wrong.
fn stat_writer(config: &Config, name: &str) -> csv::Writer<fs::File> {
    let s = format!("{}/{}_statistics.tsv", config.data_directory(), name);
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
    stop_flag: bool,
}

pub type ReportFn<T, D> = Box<dyn Fn(&Window<T, D>, usize, &Config) -> () + Sync + Send + 'static>;

#[allow(dead_code)]
pub fn default_report_fn<P: Phenome + Genome, D: DominanceOrd<P>>(
    window: &Window<P, D>,
    counter: usize,
    config: &Config,
) {
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
    pub config: Arc<Config>,
    counter: usize,
    report_every: usize,
    i: usize,
    window_size: usize,
    report_fn: ReportFn<O, D>,
    pub best: Option<O>,
    pub champion: Option<O>,
    // priority fitness best
    pub archive: Vec<O>,
    #[allow(dead_code)] // TODO: re-establish pareto archive as optional
    dominance_order: D,
    stat_writers: HashMap<&'static str, Arc<Mutex<csv::Writer<fs::File>>>>,
}

impl<O: Genome + Phenome + 'static, D: DominanceOrd<O>> Window<O, D> {
    fn new(report_fn: ReportFn<O, D>, config: Arc<Config>, dominance_order: D) -> Self {
        let window_size = config.observer.window_size;
        let report_every = config.observer.report_every;
        assert!(window_size > 0, "window_size must be > 0");
        assert!(report_every > 0, "report_every must be > 0");
        let mean_stat_writer = Arc::new(Mutex::new(stat_writer(&config, "mean")));
        let best_stat_writer = Arc::new(Mutex::new(stat_writer(&config, "best")));
        let champion_stat_writer = Arc::new(Mutex::new(stat_writer(&config, "champion")));
        let stat_writers = hashmap! {
            "mean" => mean_stat_writer,
            "best" => best_stat_writer,
            "champion" => champion_stat_writer
        };
        Self {
            frame: Vec::with_capacity(window_size),
            config,
            counter: 0,
            i: 0,
            window_size,
            report_every,
            report_fn,
            best: None,
            champion: None,
            archive: vec![],
            stat_writers,
            dominance_order,
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
            crate::stop_everything(self.config.island_identifier, false);
        }

        if let Some(ref champion) = self.champion {
            if champion.is_goal_reached(&self.config) {
                let path = format!("{}/winning_champion.json.gz", self.config.data_directory());
                log::info!("dumping winning champion to {}", path);
                dump(champion, &path).expect("failed to dump champion");
                crate::stop_everything(self.config.island_identifier, true);
            }
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
        if self.counter % self.config.observer.dump_every == 0 {
            self.dump_soup();
            self.dump_population();
        }
        if self.counter % self.config.pop_size == 0 {
            // UNCOMMENT FOR PARETO FIXME // self.update_archive();
            self.update_best();
            self.update_champion();
        }
        if self.counter % self.report_every == 0 {
            self.report();
        }

        self.is_halting_condition_reached();
    }

    fn update_best(&mut self) {
        let mut updated = false;
        for specimen in self.frame.iter() {
            if let Some(f) = specimen.scalar_fitness() {
                match self.best.as_ref() {
                    None => {
                        updated = true;
                        self.best = Some(specimen.clone())
                    }
                    Some(champ) => {
                        if f < champ.scalar_fitness().unwrap() {
                            updated = true;
                            self.best = Some(specimen.clone())
                        }
                    }
                }
            }
        }

        if updated {
            log::info!(
                "Island {}: new best:\n{:#?}",
                self.config.island_identifier,
                self.best.as_ref().unwrap()
            );
        }
    }

    fn update_champion(&mut self) {
        let mut updated = false;
        for specimen in self.frame.iter() {
            if let Some(f) = specimen.priority_fitness(&self.config) {
                match self.champion.as_ref() {
                    None => {
                        updated = true;
                        self.champion = Some(specimen.clone())
                    }
                    Some(champ) => {
                        if f < champ.priority_fitness(&self.config).unwrap() {
                            updated = true;
                            self.champion = Some(specimen.clone())
                        }
                    }
                }
            }
        }

        if updated {
            if let Some(ref mut champion) = self.champion {
                champion.generate_description();
                log::info!(
                    "Island {}: new champion:\n{:#?}",
                    self.config.island_identifier,
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

    #[allow(dead_code)] // TODO re-establish as optional
    fn update_archive(&mut self) {
        // TODO: optimize this. it's quite bad.

        let arena = self
            .archive
            .iter()
            //.chain(self.frame.iter().filter(|g| g.front() == Some(0)))
            .chain(self.frame.iter())
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
            .choose_multiple(&mut hash_seed_rng(&arena), self.config.pop_size);

        self.archive = sample.map(|i| arena[*i].clone()).collect::<Vec<O>>();
    }

    fn report(&self) {
        (self.report_fn)(&self, self.counter, &self.config);
    }

    pub fn log_record<S: Serialize>(&self, record: S, name: &str) {
        self.stat_writers[name]
            .lock()
            .expect("poisoned lock on window's logger")
            .serialize(record)
            .map_err(|e| log::error!("Error logging record: {:?}", e))
            .expect("Failed to log record!");
        self.stat_writers[name]
            .lock()
            .expect("poisoned lock on window's logger")
            .flush()
            .expect("Failed to flush");
    }

    pub fn dump_population(&self) {
        if !self.config.observer.dump_population {
            log::debug!("Not dumping population");
            return;
        }
        let path = format!(
            "{}/population/population_{}.json.gz",
            self.config.data_directory(),
            self.counter,
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
        let path = format!(
            "{}/soup/soup_{}.json",
            self.config.data_directory(),
            self.counter,
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

    pub fn spawn<D: 'static + DominanceOrd<O> + Send>(
        config: &Config,
        report_fn: ReportFn<O, D>,
        dominance_order: D,
    ) -> Observer<O> {
        let (tx, rx): (Sender<O>, Receiver<O>) = channel();

        let config = Arc::new(config.clone());
        let handle: JoinHandle<()> = spawn(move || {
            let mut window: Window<O, D> = Window::new(report_fn, config.clone(), dominance_order);
            for observable in rx {
                window.insert(observable);
            }
        });

        Observer {
            handle,
            tx,
            stop_flag: false,
        }
    }

    pub fn stop_evolution(&mut self) {
        self.stop_flag = true
    }
}
