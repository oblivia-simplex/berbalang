use std::sync::Arc;

use hashbrown::HashSet;

use crate::configure::Config;
use crate::emulator::loader::get_static_memory_image;
use crate::emulator::profiler::HasProfile;
use crate::evolution::{Genome, Phenome};
use crate::fitness::Weighted;
use crate::ontogenesis::FitnessFn;
use crate::roper::Sketches;
use crate::util::entropy::Entropy;

pub fn just_novelty_ff<C>(mut creature: C, sketch: &mut Sketches, config: Arc<Config>) -> C
where
    C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized,
{
    if let Some(ref profile) = creature.profile() {
        let mut scores = vec![];
        for reg_state in &profile.registers {
            for (reg, vals) in reg_state.0.iter() {
                sketch.register_error.insert((reg, vals));
                scores.push(sketch.register_error.query((reg, vals)));
            }
        }
        let register_freq = stats::mean(scores.into_iter());
        let mut fitness = Weighted::new(&config.fitness.weighting);
        fitness.insert("register_freq", register_freq);
        let gadgets_executed = profile.gadgets_executed.len();
        fitness.insert("gadgets_executed", gadgets_executed as f64);
        creature.set_fitness(fitness);
    }

    creature
}

// TODO: I'm in the middle of the somewhat tedious process of refactoring
// the code so that it handles batches of problems, and not single problems.
// As it stands, I think the code is in an inconsistent state. First thing on
// Monday morning, we'll get this sorted out.
pub fn register_pattern_ff<C>(mut creature: C, sketch: &mut Sketches, config: Arc<Config>) -> C
where
    C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized,
{
    // measure fitness
    // for now, let's just handle the register pattern task
    if let Some(ref profile) = creature.profile() {
        //sketch.insert(&profile.registers);
        //let reg_freq = sketch.query(&profile.registers);
        //if let Some(pattern) = config.roper.register_patterns() {
        let number_of_cases = profile.registers.len();
        let mut fitness = Weighted::new(&config.fitness.weighting);
        // If the specimen doesn't report the right number of register states, then
        // something must have gone wrong in execution. Mark that specimen as a total
        // failure, and exit the function.
        if number_of_cases != config.roper.register_patterns().len() {
            log::error!(
                "Creature has only {} register states! Expecting {}!",
                number_of_cases,
                config.roper.register_patterns().len()
            );
            creature.set_fitness(fitness);
            return creature;
        }
        for (idx, pattern) in config.roper.register_patterns().iter().enumerate() {
            let register_error = pattern.distance_from_register_state(&profile.registers[idx]);
            let mut weighted_fitness = Weighted::new(&config.fitness.weighting);
            weighted_fitness.insert_or_add("register_error", register_error);

            // Calculate the novelty of register state errors
            let register_freq = stats::mean(
                pattern
                    .incorrect_register_states(&profile.registers[idx])
                    .iter()
                    .map(|goof| {
                        sketch.register_error.insert(goof);
                        sketch.register_error.query(goof)
                    }),
            );

            weighted_fitness.insert_or_add("register_freq", register_freq);

            // how many times did it crash?
            let crashes = profile.cpu_errors.iter().filter_map(|x| *x).count();
            weighted_fitness.insert_or_add("crash_count", crashes as f64);

            let ret_count = profile.ret_counts[idx];
            weighted_fitness.insert_or_add("ret_count", ret_count as f64);

            // FIXME: not sure how sound this frequency gauging scheme is.
            creature.record_genetic_frequency(sketch);
            let gen_freq = creature.query_genetic_frequency(sketch);
            weighted_fitness.scores.insert("genetic_freq", gen_freq);

            log::debug!("adding {:?} to {:?}", weighted_fitness, fitness);
            // NB: There's a quirk in the fitness addition implementation that makes it
            // non-commutative, in the general case. FIXME
            fitness = weighted_fitness + fitness;
            log::debug!("Result is {:?}", fitness);
        }
        fitness.scale_by(number_of_cases as f64);
        // Now add a constancy penalty if appropriate
        let mut regs = profile.registers.clone();
        regs.dedup();
        fitness.insert(
            "constancy_penalty",
            (profile.registers.len() - regs.len()) as f64,
        );
        log::debug!("Setting creature fitness to {:#?}", fitness);
        creature.set_fitness(fitness);
    }
    creature
}

pub fn register_entropy_ff<C>(mut creature: C, sketch: &mut Sketches, config: Arc<Config>) -> C
where
    C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized,
{
    if let Some(ref profile) = creature.profile() {
        if let Some(registers) = profile.registers.last() {
            let just_regs = registers.0.values().map(|v| v[0]).collect::<Vec<u64>>();
            let entropy = just_regs.entropy();
            let mut weighted_fitness = Weighted::new(&config.fitness.weighting);
            weighted_fitness.insert("register_entropy", entropy);
            log::debug!("registers = {:x?}\n1/entropy = {}", just_regs, entropy);

            sketch.register_error.insert(&just_regs);
            let reg_freq = sketch.register_error.query(&just_regs);
            weighted_fitness.insert("register_freq", reg_freq);

            weighted_fitness.insert("gadgets_executed", profile.gadgets_executed.len() as f64);

            creature.set_fitness(weighted_fitness);
        }
    }
    creature
}

// FIXME: This needs to be brought in line with the new profile format
// in particular, we need to iterate through the runs in the profile.
pub fn register_conjunction_ff<C>(mut creature: C, sketch: &mut Sketches, config: Arc<Config>) -> C
where
    C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized,
{
    if let Some(ref profile) = creature.profile() {
        if let Some(registers) = profile.registers.last() {
            let word_size = get_static_memory_image().word_size * 8;
            let mut conj = registers.0.values().fold(!0_u64, |a, b| a & b[0]);
            let mask = match word_size {
                64 => 0x0000_0000_0000_0000,
                32 => 0xFFFF_FFFF_0000_0000,
                16 => 0xFFFF_FFFF_FFFF_0000,
                _ => unreachable!("not a size"),
            };
            conj |= mask;
            let score = conj.count_zeros() as f64;
            // ignore bits outside of the register's word size
            debug_assert!(score <= word_size as f64);
            let mut weighted_fitness = Weighted::new(&config.fitness.weighting);
            weighted_fitness.insert("zeroes", score);
            weighted_fitness.insert("gadgets_executed", profile.gadgets_executed.len() as f64);

            sketch.register_error.insert(registers);
            let reg_freq = sketch.register_error.query(registers);
            weighted_fitness.insert("register_freq", reg_freq);

            creature.set_fitness(weighted_fitness);
        }
    }
    creature
}

pub fn memory_pattern_ff<C>(mut creature: C, sketch: &mut Sketches, config: Arc<Config>) -> C
where
    C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized,
{
    static SUBPATTERN_LABELS: [&str; 5] = [
        "subpattern_1",
        "subpattern_2",
        "subpattern_3",
        "subpattern_4",
        "subpattern_5",
    ];
    if let Some(profile) = creature.profile() {
        let mut fitness = Weighted::new(&config.fitness.weighting);

        let pattern = config
            .roper
            .memory_pattern
            .as_ref()
            .expect("No memory pattern provided");

        for i in 1..pattern.len() + 1 {
            let sub_pattern = &pattern[0..i];
            let occurrences = profile
                .memory_writes
                .iter()
                .map(|data| data.find_seq(sub_pattern).len())
                .sum::<usize>() as f64;
            fitness.insert_or_add(SUBPATTERN_LABELS[i - 1], occurrences);
        }

        sketch.memory_writes.insert(&profile.memory_writes);
        let memory_freq = sketch.memory_writes.query(&profile.memory_writes);

        fitness.insert_or_add("memory_freq", memory_freq);

        let num_mem_writes = profile
            .memory_writes
            .iter()
            .map(|data| data.len())
            .sum::<usize>() as f64;

        // let gadgets_executed = profile
        //     .gadgets_executed
        //     .iter()
        //     .map(|h| h.len())
        //     .sum::<usize>() as f64;

        let gadgets_executed = profile.gadgets_executed(0) as f64;
        let ret_count = profile.ret_counts[0] as f64;

        creature.record_genetic_frequency(&mut sketch.genetic);
        let genetic_freq = creature.query_genetic_frequency(&sketch.genetic);

        fitness.insert_or_add("num_writes", num_mem_writes);
        fitness.insert_or_add("gadgets_executed", gadgets_executed);
        fitness.insert_or_add("ret_count", ret_count);
        // fitness.insert_or_add("ret_count", ret_count);
        fitness.insert_or_add("genetic_freq", genetic_freq);

        creature.set_fitness(fitness);
        // actually, we probably want to average normalized scores
        // for the time being, not a huge concern. we'll start with a single case.
    }
    creature
}

pub fn code_coverage_ff<C>(mut creature: C, sketch: &mut Sketches, config: Arc<Config>) -> C
where
    C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized,
{
    // if creature.fitness().is_some() {
    //     return creature;
    // }

    if let Some(ref profile) = creature.profile() {
        let mut addresses_visited = HashSet::new();
        // TODO: optimize this, maybe parallelize
        profile.execution_trace_iter().for_each(|path| {
            for block in path {
                for addr in block.entry..(block.entry + block.size as u64) {
                    addresses_visited.insert(addr);
                }
            }
        });
        let mut freq_score = 0.0;
        for addr in addresses_visited.iter() {
            sketch.addresses_visited.insert(*addr);
            freq_score += sketch.addresses_visited.query(*addr);
        }
        let num_addr_visit = addresses_visited.len() as f64;
        let avg_freq = if num_addr_visit < 1.0 {
            1.0
        } else {
            freq_score / num_addr_visit
        };
        // might be worth memoizing this call, but it's pretty cheap
        let code_size = get_static_memory_image().size_of_executable_memory();
        let code_coverage = num_addr_visit / code_size as f64;

        let mut fitness = Weighted::new(&config.fitness.weighting);
        fitness.insert("code_coverage", code_coverage);
        fitness.insert("code_frequency", avg_freq);

        let gadgets_executed = profile.ret_counts.iter().sum::<usize>();
        fitness.insert("ret_count", gadgets_executed as f64);

        creature.set_fitness(fitness);
    }
    // creature.set_profile(Profile::default()); // experimental. see if we can free some mem here.

    creature
}

pub fn get_fitness_function<C>(name: &str) -> FitnessFn<C, Sketches, Config>
where
    C: HasProfile + Genome + Phenome<Fitness = Weighted<'static>> + Sized + 'static,
{
    match name {
        "register_pattern" => Box::new(register_pattern_ff),
        "register_conjunction" => Box::new(register_conjunction_ff),
        "register_entropy" => Box::new(register_entropy_ff),
        "code_coverage" => Box::new(code_coverage_ff),
        "memory_pattern" => Box::new(memory_pattern_ff),
        "just_novelty" => Box::new(just_novelty_ff),
        s => unimplemented!("No such fitness function as {}", s),
    }
}
