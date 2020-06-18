use serde::Serialize;

use crate::configure::Config;
use crate::evolution::{Genome, Phenome};
use crate::fitness::MapFit;
use crate::get_epoch_counter;
use crate::observer::Window;
use crate::roper::creature::Creature;
use crate::util::count_min_sketch::CountMinSketch;

use super::CreatureDominanceOrd;

#[derive(Serialize, Clone, Debug, Default)]
pub struct StatRecord {
    pub counter: usize,
    pub epoch: usize,
    // #[serde(flatten)]
    // Fitness scores // TODO find a way to make this more flexible
    // TODO: how to report on fitness vectors?
    // now the best's attributes
    // pub avg_crash_count: f64,
    // pub avg_place_error: f64,
    // pub avg_value_error: f64,
    //pub avg_genetic_freq_by_gen: f64,
    pub avg_exec_ratio: f64,
    pub avg_genetic_freq_by_window: f64,
    pub avg_len: f64,
    pub avg_register_error: f64,
    pub avg_scalar_fitness: f64,
    pub avg_uniq_exec_count: f64,
    pub avg_ratio_written: f64,
    pub avg_emulation_time: f64,

    pub best_exec_ratio: f64,
    pub best_genetic_freq_by_window: f64,
    pub best_len: f64,
    pub best_register_error: f64,
    pub best_scalar_fitness: f64,
    pub best_uniq_exec_count: f64,
    pub best_ratio_written: f64,

    pub immigrant_ratio: f64,
    pub soup_len: f64,
    // NOTE duplication here. maybe define a struct that is used for both
    // average and best, and a func that derives it
}

impl StatRecord {
    fn from_window<'a>(
        window: &Window<Creature<u64>, CreatureDominanceOrd>,
        counter: usize,
    ) -> Self {
        let frame = &window.frame;
        let best = window.best.as_ref().unwrap();
        let avg_len = frame.iter().map(|c| c.len()).sum::<usize>() as f64 / frame.len() as f64;
        let mut sketch = CountMinSketch::new(&window.config);
        for g in frame.iter() {
            g.record_genetic_frequency(&mut sketch);
        }
        best.record_genetic_frequency(&mut sketch);
        let avg_genetic_freq = frame
            .iter()
            .map(|g| g.query_genetic_frequency(&sketch))
            .sum::<f64>()
            / frame.len() as f64;
        let avg_scalar_fitness: f64 =
            frame.iter().filter_map(|g| g.scalar_fitness()).sum::<f64>() / frame.len() as f64;
        let fitnesses = frame.iter().filter_map(|g| g.fitness()).collect::<Vec<_>>();
        let fit_vec = MapFit::average(&fitnesses);

        let avg_exec_count: f64 = frame
            .iter()
            .map(|g| g.num_uniq_alleles_executed() as f64)
            .sum::<f64>()
            / frame.len() as f64;
        let avg_exec_ratio: f64 =
            frame.iter().map(|g| g.execution_ratio()).sum::<f64>() / frame.len() as f64;

        let avg_emulation_time = frame
            .iter()
            .filter_map(|g| g.profile.as_ref().map(|p| p.avg_emulation_millis()))
            .sum::<f64>()
            / frame.len() as f64;

        let immigrant_ratio = frame
            .iter()
            .filter(|g| g.native_island != window.config.island_identifier)
            .count() as f64
            / frame.len() as f64;

        let best_genetic_freq_by_window = best.query_genetic_frequency(&sketch);
        let best_exec_ratio = best.execution_ratio();
        let best_scalar_fitness = best.scalar_fitness().unwrap();
        let best_register_error = best
            .fitness
            .as_ref()
            .unwrap()
            .scores
            .get("register_error")
            .cloned()
            .unwrap_or_default();
        let best_len = best.len() as f64;
        let best_uniq_exec_count = best.num_uniq_alleles_executed() as f64;
        let best_ratio_written = best
            .fitness
            .as_ref()
            .unwrap()
            .scores
            .get("mem_write_ratio")
            .cloned()
            .unwrap_or_default();

        let soup = window.soup();
        let soup_len = soup.len() as f64;

        StatRecord {
            counter,
            epoch: get_epoch_counter(),
            avg_len,
            avg_genetic_freq_by_window: avg_genetic_freq,
            avg_scalar_fitness,
            avg_uniq_exec_count: avg_exec_count,
            avg_exec_ratio,
            // fitness scores
            // TODO: it would be nice if this were less hard-coded
            // avg_place_error: fit_vec["place_error"],
            // avg_value_error: fit_vec["value_error"],
            // avg_crash_count: fit_vec["crash_count"],
            avg_register_error: fit_vec.get("register_error").unwrap_or_default(),
            avg_ratio_written: fit_vec.get("mem_write_ratio").unwrap_or_default(),
            avg_emulation_time,
            //avg_genetic_freq_by_gen: fit_vec["genetic_frequency"],
            best_genetic_freq_by_window,
            best_scalar_fitness,
            best_exec_ratio,
            best_register_error,
            best_len,
            best_uniq_exec_count,
            best_ratio_written,

            immigrant_ratio,

            soup_len,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_stat_record_serialization() {
        let record = StatRecord::default();
        let file = std::io::stderr();
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .terminator(csv::Terminator::Any(b'\n'))
            .has_headers(false)
            .from_writer(file);

        writer
            .serialize(&record)
            .expect("Failed to serialize record!");
        writer.flush().expect("Failed to flush");
    }
}

pub fn report_fn<'a>(
    window: &Window<Creature<u64>, super::CreatureDominanceOrd>,
    counter: usize,
    config: &Config,
) {
    let record = StatRecord::from_window(window, counter);

    log::info!(
        "Island #{island}. Current best: {best:#x?}\n{pop_name} island #{island} {record:#?}",
        island = config.island_identifier,
        best = window.best,
        record = record,
        pop_name = config.observer.population_name,
    );

    window.log_record(record);

    // let total = window.archive.len();
    // log::info!("Current Pareto front contains {} specimens:", total);
    // for (i, specimen) in window.archive.iter().enumerate() {
    //     log::info!("Specimen #{}: {:#?}", i, specimen);
    //     break;
    // }

    if let Ok(stat) = procinfo::pid::statm_self() {
        log::debug!("Memory status: {:#x?}", stat);
    }
}

pub mod lexicase {
    use super::*;

    pub fn report_fn<'a>(
        window: &Window<Creature<u64>, super::CreatureDominanceOrd>,
        _counter: usize,
        _config: &Config,
    ) {
        let soup = window.soup();
        log::info!("soup size: {}", soup.len());
    }
}
