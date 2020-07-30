use serde::Serialize;

use crate::configure::Config;
use crate::evolution::{Genome, Phenome};
use crate::fitness::MapFit;
use crate::get_epoch_counter;
use crate::observer::Window;
use crate::roper::bare::Creature;
use crate::roper::CreatureDominanceOrd;

#[derive(Serialize, Clone, Debug, Default)]
pub struct StatRecord {
    pub counter: usize,
    pub epoch: usize,

    pub exec_ratio: f64,
    pub length: f64,
    pub register_error: f64,
    pub scalar_fitness: f64,
    pub priority_fitness: f64,
    pub uniq_exec_count: f64,
    pub ratio_written: f64,
    pub emulation_time: f64,
}

impl StatRecord {
    fn for_specimen(
        window: &Window<Creature, CreatureDominanceOrd>,
        specimen: &Creature,
        counter: usize,
        epoch: usize,
    ) -> Self {
        let specimen_exec_ratio = specimen.execution_ratio();
        let specimen_scalar_fitness = specimen.scalar_fitness().unwrap();
        let specimen_priority_fitness = specimen.priority_fitness(&window.config).unwrap();
        let specimen_register_error = specimen
            .fitness
            .as_ref()
            .unwrap()
            .scores
            .get("register_error")
            .cloned()
            .unwrap_or_default();
        let specimen_len = specimen.len() as f64;
        let specimen_uniq_exec_count = specimen.num_uniq_alleles_executed() as f64;
        let specimen_ratio_written = specimen
            .fitness
            .as_ref()
            .unwrap()
            .scores
            .get("mem_write_ratio")
            .cloned()
            .unwrap_or_default();
        let specimen_emulation_time = specimen
            .profile
            .as_ref()
            .map(|p| p.avg_emulation_micros())
            .unwrap_or(0.0);

        Self {
            counter,
            epoch,
            exec_ratio: specimen_exec_ratio,
            length: specimen_len,
            register_error: specimen_register_error,
            scalar_fitness: specimen_scalar_fitness,
            priority_fitness: specimen_priority_fitness,
            uniq_exec_count: specimen_uniq_exec_count,
            ratio_written: specimen_ratio_written,
            emulation_time: specimen_emulation_time,
        }
    }

    fn mean_from_window(window: &Window<Creature, CreatureDominanceOrd>, counter: usize) -> Self {
        let frame = &window.frame;

        let length = frame.iter().map(|c| c.len()).sum::<usize>() as f64 / frame.len() as f64;

        let scalar_fitness: f64 =
            frame.iter().filter_map(|g| g.scalar_fitness()).sum::<f64>() / frame.len() as f64;
        let priority_fitness: f64 = frame
            .iter()
            .filter_map(|g| g.priority_fitness(&window.config))
            .sum::<f64>()
            / frame.len() as f64;
        let fitnesses = frame.iter().filter_map(|g| g.fitness()).collect::<Vec<_>>();
        let fit_vec = MapFit::average(&fitnesses);

        let avg_exec_count: f64 = frame
            .iter()
            .map(|g| g.num_uniq_alleles_executed() as f64)
            .sum::<f64>()
            / frame.len() as f64;
        let exec_ratio: f64 =
            frame.iter().map(|g| g.execution_ratio()).sum::<f64>() / frame.len() as f64;

        let emulation_time = frame
            .iter()
            .filter_map(|g| g.profile.as_ref().map(|p| p.avg_emulation_micros()))
            .sum::<f64>()
            / frame.len() as f64;

        StatRecord {
            counter,
            epoch: get_epoch_counter(),
            length,
            scalar_fitness,
            priority_fitness,
            uniq_exec_count: avg_exec_count,
            exec_ratio,
            register_error: fit_vec.get("register_error").unwrap_or_default(),
            ratio_written: fit_vec.get("mem_write_ratio").unwrap_or_default(),
            emulation_time,
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

pub fn report_fn(window: &Window<Creature, CreatureDominanceOrd>, counter: usize, config: &Config) {
    let epoch = get_epoch_counter();

    let record = StatRecord::mean_from_window(window, counter);
    log::info!(
        "Island #{island} {record:#?}",
        island = config.island_identifier,
        record = record,
    );
    window.log_record(record, "mean");

    if let Some(ref best) = window.best {
        let best_record = StatRecord::for_specimen(&window, best, counter, epoch);
        window.log_record(best_record, "best");
    }

    if let Some(ref champion) = window.champion {
        let champion_record = StatRecord::for_specimen(&window, champion, counter, epoch);
        window.log_record(champion_record, "champion");
    }

    log::info!(
        "Island #{island} Best: {best:#?}\nIsland #{island} Champion: {champion:#?}",
        island = config.island_identifier,
        best = window.best,
        champion = window.champion,
    );

    if let Ok(stat) = procinfo::pid::statm_self() {
        log::debug!("Memory status: {:#x?}", stat);
    }
}

pub mod lexicase {
    use super::*;

    pub fn report_fn(
        window: &Window<Creature, super::CreatureDominanceOrd>,
        _counter: usize,
        _config: &Config,
    ) {
        let soup = window.soup();
        log::info!("soup size: {}", soup.len());
    }
}
