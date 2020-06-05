use std::io::Write;

use deflate::write::GzEncoder;
use deflate::Compression;
use serde::Serialize;

use indexmap::{indexmap, IndexMap, IndexSet};

use crate::configure::ObserverConfig;
use crate::evolution::{Genome, Phenome};
use crate::fitness::Pareto;
use crate::observer::Window;
use crate::roper::creature::Creature;
use crate::util::count_min_sketch::DecayingSketch;

#[derive(Serialize, Clone, Debug)]
pub struct StatRecord {
    pub avg_len: f64,
    pub avg_genetic_freq: f64,
    pub avg_scalar_fitness: f64,
    // TODO: how to report on fitness vectors?
    pub avg_vectoral_fitness: Pareto<'static>,
    pub avg_exec_count: f64,
    pub avg_exec_ratio: f64,
    pub counter: usize,
}

fn write_files<F>(files: &IndexMap<&'static str, Vec<u8>>, params: &ObserverConfig) {}

pub fn report_fn(window: &Window<Creature>, counter: usize, _params: &ObserverConfig) {
    let frame = &window.frame;
    log::info!("default report function");
    let frame_len = frame.len() as f64;
    let avg_len = frame.iter().map(|c| c.len()).sum::<usize>() as f64 / frame.len() as f64;
    let mut sketch = DecayingSketch::default();
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
    let avg_vectoral_fitness = Pareto::average(&fitnesses);
    // let avg_vectoral_fitness: Vec<f64> = frame
    //     .iter()
    //     .filter_map(|g| g.fitness().map(|f| f.as_ref()))
    //     .fold(vec![0.0; 20], |a, b: &[f64]| {
    //         a.iter()
    //             .zip(b.iter())
    //             .map(|(aa, bb)| aa + bb)
    //             .collect::<Vec<f64>>()
    //     })
    //     .iter()
    //     .map(|n| n / frame_len)
    //     .collect::<Vec<f64>>();
    let avg_exec_count: f64 = frame
        .iter()
        .map(|g| g.num_alleles_executed() as f64)
        .sum::<f64>()
        / frame.len() as f64;
    let avg_exec_ratio: f64 =
        frame.iter().map(|g| g.execution_ratio()).sum::<f64>() / frame.len() as f64;

    let soup = frame
        .iter()
        .map(|g| g.chromosome())
        .flatten()
        .cloned()
        .collect::<IndexSet<u64>>();

    log::info!(
            "[{}] Average length: {}, average genetic frequency: {}, avg scalar fit: {}, avg vectoral fit: {:?}, avg # alleles exec'd: {}, avg exec ratio: {}, soup size: {}",
            counter,
            avg_len,
            avg_genetic_freq,
            avg_scalar_fitness,
            avg_vectoral_fitness,
            avg_exec_count,
            avg_exec_ratio,
            soup.len(),
        );
    log::info!("[{}] Reigning champion: {:#?}", counter, window.best);
    // let serialized_creatures: Vec<_> = frame
    //     .iter()
    //     .map(|g| bson::to_bson(g).unwrap())
    //     .collect::<Vec<_>>();

    log::info!("Serializing population...");
    if let Ok(mut population_file) = std::fs::File::create("./population.json.gz")
        .map_err(|e| log::error!("Failed to create population file: {:?}", e))
    {
        let mut gz = GzEncoder::new(Vec::new(), Compression::Default);
        let _ = serde_json::to_writer(&mut gz, &frame)
            .map_err(|e| log::error!("Failed to compress serialized population: {:?}", e));
        let compressed_population_data = gz.finish().expect("Failed to finish Gzip encoding");
        let _ = population_file
            .write_all(&compressed_population_data)
            .map_err(|e| log::error!("Failed to write population file: {:?}", e));
        log::info!("Population dumped!");
    };

    let record = StatRecord {
        avg_len,
        avg_genetic_freq,
        avg_scalar_fitness,
        avg_vectoral_fitness,
        avg_exec_count,
        avg_exec_ratio,
        counter,
    };
}
