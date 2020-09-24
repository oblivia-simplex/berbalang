use hashbrown::HashSet;
use itertools::Itertools;
use serde::Serialize;

use crate::configure::Config;
use crate::emulator::loader::get_static_memory_image;
use crate::emulator::profiler::{HasProfile, Profile};
use crate::evolution::{Genome, Phenome};
use crate::fitness::{average_weighted, stddev_weighted, Weighted};
use crate::get_epoch_counter;
use crate::observer::{LogRecord, Window};

#[derive(Serialize, Clone, Debug)]
pub struct StatRecord {
    pub counter: usize,
    pub epoch: usize,
    pub generation: f64,
    pub ratio_visited: f64,
    pub soup_len: usize,
    pub length: f64,
    pub emulation_time: f64,
    pub fitness: Weighted<'static>,
    pub stdev_fitness: Option<Weighted<'static>>,
}

impl LogRecord for StatRecord {
    fn header(&self) -> String {
        let mut s =
            format!("epoch,generation,length,emulation_time,ratio_visited,soup_len,fitness");
        if self.stdev_fitness.is_some() {
            s.push_str(",stdev_fitness");
        }
        for factor in self.fitness.scores.keys().sorted() {
            s.push_str(",");
            s.push_str(factor);
            if self.stdev_fitness.is_some() {
                s.push_str(&format!(",stdev_{}", factor));
            }
        }
        s
    }

    fn row(&self) -> String {
        let mut s = format!(
            "{epoch},{generation},{length},{emulation_time},{ratio_visited},{soup_len},{fitness}",
            epoch = self.epoch,
            generation = self.generation,
            length = self.length,
            emulation_time = self.emulation_time,
            ratio_visited = self.ratio_visited,
            soup_len = self.soup_len,
            fitness = self.fitness.scalar(),
        );
        if let Some(ref stdev_fitness) = self.stdev_fitness {
            s.push_str(&format!(",{}", stdev_fitness.scalar()));
        }
        let f_values = self.fitness.values().collect::<Vec<_>>();
        let stdev_values = self
            .stdev_fitness
            .as_ref()
            .map(|s| s.values().collect::<Vec<_>>());

        for i in 0..(f_values.len()) {
            // sorted by key
            s.push_str(&format!(",{}", f_values[i]));
            if let Some(ref stdev_values) = stdev_values {
                s.push_str(&format!(",{}", stdev_values[i]));
            }
        }
        s
    }
}

impl StatRecord {
    fn for_specimen<C>(specimen: &C, counter: usize, epoch: usize) -> Self
    where
        C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized,
    {
        let specimen_len = specimen.chromosome().len() as f64;
        let specimen_emulation_time = specimen
            .profile()
            .as_ref()
            .map(|p| p.avg_emulation_micros())
            .unwrap_or(0.0);

        let addresses_visited = specimen
            .profile()
            .map(|p| p.addresses_visited().len())
            .unwrap_or(0) as f64;
        let code_size = get_static_memory_image().size_of_executable_memory();
        let ratio_visited = addresses_visited / code_size as f64;

        Self {
            counter,
            epoch,
            generation: specimen.generation() as f64,
            soup_len: 0,
            length: specimen_len,
            emulation_time: specimen_emulation_time,
            ratio_visited,
            fitness: specimen
                .fitness()
                .expect("Missing fitness in specimen")
                .clone(),
            stdev_fitness: None,
        }
    }

    fn mean_from_window<C>(window: &Window<C>, counter: usize) -> Self
    where
        C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized,
    {
        let frame = &window
            .frame
            .iter()
            .filter(|c| {
                if let Some(&Profile {
                    executable: true, ..
                }) = c.profile()
                {
                    true
                } else {
                    false
                }
            })
            .collect::<Vec<&C>>();

        let mut addresses_visited = HashSet::new();

        for addresses in frame
            .iter()
            .filter_map(|c| c.profile())
            .map(|p| p.addresses_visited())
        {
            addresses_visited.extend(addresses)
        }

        let soup_len = window.soup().len();

        let code_size = get_static_memory_image().size_of_executable_memory();
        let ratio_visited = addresses_visited.len() as f64 / code_size as f64;
        //
        // let ratio_eligible = frame.len() as f64 / window.frame.len() as f64;
        // log::info!("Ratio eligible = {}", ratio_eligible);

        let length =
            frame.iter().map(|c| c.chromosome().len()).sum::<usize>() as f64 / frame.len() as f64;

        let fitnesses = frame
            .iter()
            .filter_map(|g| g.fitness())
            .cloned()
            .collect::<Vec<_>>();
        let mean_fitness = average_weighted(&fitnesses);
        let stdev_fitness = stddev_weighted(&fitnesses, &mean_fitness);

        let emulation_time = frame
            .iter()
            .filter_map(|g| g.profile().as_ref().map(|p| p.avg_emulation_micros()))
            .sum::<f64>()
            / frame.len() as f64;

        let generation =
            frame.iter().map(|g| g.generation()).sum::<usize>() as f64 / frame.len() as f64;

        StatRecord {
            counter,
            epoch: get_epoch_counter(),
            ratio_visited,
            soup_len,
            generation,
            length,
            emulation_time,
            fitness: mean_fitness,
            stdev_fitness: Some(stdev_fitness),
        }
    }
}

pub fn report_fn<C>(window: &Window<C>, counter: usize, config: &Config)
where
    C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized,
{
    let epoch = get_epoch_counter();

    let record = StatRecord::mean_from_window(window, counter);
    log::debug!(
        "Island #{island} {record:#?}",
        island = config.island_id,
        record = record,
    );
    window.log_record(record, "mean");

    if let Some(ref champion) = window.champion {
        let champion_record = StatRecord::for_specimen(champion, counter, epoch);
        window.log_record(champion_record, "champion");
    }

    if let Some(ref best) = window.best {
        let best_record = StatRecord::for_specimen(best, counter, epoch);
        window.log_record(best_record, "best");
    }

    log::debug!(
        "Island #{island} Champion: {champion:#?}\nIsland #{island} Best: {best:#?}\nIsland #{island}, Epoch {epoch}",
        island = config.island_id,
        best = window.best,
        champion = window.champion,
        epoch = epoch,
    );

    if let Ok(stat) = procinfo::pid::statm_self() {
        log::debug!("Memory status: {:#x?}", stat);
    }
}
//
// pub mod lexicase {
//     use super::*;
//
//     pub fn report_fn(
//         window: &Window<Creature, super::CreatureDominanceOrd>,
//         _counter: usize,
//         _config: &Config,
//     ) {
//         let soup = window.soup();
//         log::info!("soup size: {}", soup.len());
//     }
// }
