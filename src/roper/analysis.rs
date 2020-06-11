use serde::Serialize;

use crate::configure::Config;
use crate::evolution::{Genome, Phenome};
use crate::fitness::MapFit;
use crate::observer::Window;
use crate::roper::creature::Creature;
use crate::util::count_min_sketch::{suggest_depth, suggest_width, DecayingSketch};

use super::CreatureDominanceOrd;

#[derive(Serialize, Clone, Debug, Default)]
pub struct StatRecord {
    pub counter: usize,
    pub avg_len: f64,
    pub avg_genetic_freq_by_window: f64,
    //pub avg_genetic_freq_by_gen: f64,
    pub avg_scalar_fitness: f64,
    // TODO: how to report on fitness vectors?
    // #[serde(flatten)]
    pub avg_uniq_exec_count: f64,
    pub avg_exec_ratio: f64,
    // Fitness scores // TODO find a way to make this more flexible
    // pub avg_place_error: f64,
    // pub avg_value_error: f64,
    // pub avg_crash_count: f64,
    pub avg_register_error: f64,
}

impl StatRecord {
    fn from_window(window: &Window<Creature, CreatureDominanceOrd>, counter: usize) -> Self {
        let frame = &window.frame;
        let avg_len = frame.iter().map(|c| c.len()).sum::<usize>() as f64 / frame.len() as f64;
        let mut sketch =
            DecayingSketch::new(suggest_depth(frame.len()), suggest_width(frame.len()), 0.0);
        for g in frame.iter() {
            g.record_genetic_frequency(&mut sketch);
        }
        let avg_genetic_freq = frame
            .iter()
            .map(|g| g.measure_genetic_frequency(&sketch))
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

        StatRecord {
            counter,
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
            avg_register_error: fit_vec["register_error"],
            //avg_genetic_freq_by_gen: fit_vec["genetic_frequency"],
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

pub fn report_fn(
    window: &Window<Creature, super::CreatureDominanceOrd>,
    counter: usize,
    params: &Config,
) {
    let record = StatRecord::from_window(window, counter);

    log::info!("{:#?}", record);

    window.log_record(record);

    let total = window.archive.len();
    // log::info!("Current Pareto front contains {} specimens:", total);
    // for (i, specimen) in window.archive.iter().enumerate() {
    //     log::info!("Specimen #{}: {:#?}", i, specimen);
    //     break;
    // }
    log::info!("Current best: {:#x?}", window.best);

    if let Ok(stat) = procinfo::pid::statm_self() {
        log::debug!("Memory status: {:#x?}", stat);
    }
    if halting_condition_reached(window, params) {
        window.stop_evolution();
    }
}

fn halting_condition_reached(
    _window: &Window<Creature, super::CreatureDominanceOrd>,
    _params: &Config,
) -> bool {
    false
}
