use std::convert::TryInto;
use std::sync::Arc;

use hashbrown::HashSet;
use unicorn::Cpu;

use crate::emulator::loader::get_static_memory_image;
use crate::emulator::register_pattern::{Register, UnicornRegisterState};
use crate::fitness::Weighted;
use crate::ontogenesis::FitnessFn;
use crate::{
    configure::Config, emulator::hatchery::Hatchery, evolution::Phenome, ontogenesis::Develop,
    util, util::count_min_sketch::CountMinSketch,
};

use super::Creature;

pub fn code_coverage_ff<'a>(
    mut creature: Creature<u64>,
    sketch: &mut Sketches,
    config: Arc<Config>,
) -> Creature<u64> {
    if let Some(ref profile) = creature.profile {
        let mut addresses_visited = HashSet::new();
        // TODO: optimize this, maybe parallelize
        profile.basic_block_path_iterator().for_each(|path| {
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
        let code_coverage = 1.0 - num_addr_visit / code_size as f64;

        let mut fitness = Weighted::new(config.fitness.weights.clone());
        fitness.insert("code_coverage", code_coverage);
        fitness.insert("code_frequency", avg_freq);

        let gadgets_executed = profile.gadgets_executed.len();
        fitness.insert("gadgets_executed", gadgets_executed as f64);

        let mem_write_ratio = 1.0 - profile.mem_write_ratio();
        fitness.insert("mem_write_ratio", mem_write_ratio);

        creature.set_fitness(fitness);

        // TODO: look into how unicorn tracks bbs. might be surprising in the context of ROP
    }

    creature
}

pub fn register_pattern_ff<'a>(
    mut creature: Creature<u64>,
    sketch: &mut Sketches,
    config: Arc<Config>,
) -> Creature<u64> {
    // measure fitness
    // for now, let's just handle the register pattern task
    if let Some(ref profile) = creature.profile {
        //sketch.insert(&profile.registers);
        //let reg_freq = sketch.query(&profile.registers);
        if let Some(pattern) = config.roper.register_pattern() {
            // assuming that when the register pattern task is activated, there's only one register state
            // to worry about. this may need to be adjusted in the future. bit sloppy now.
            let register_error = pattern.distance_from_register_state(&profile.registers[0]);
            let mut weighted_fitness = Weighted::new(config.fitness.weights.clone());
            weighted_fitness
                .scores
                .insert("register_error", register_error);

            // Calculate the novelty of register state errors
            let iter = profile
                .registers
                .iter()
                .map(|r| pattern.incorrect_register_states(r))
                .flatten()
                .map(|p| {
                    sketch.register_error.insert(p);
                    sketch.register_error.query(p)
                });

            let register_novelty = stats::mean(iter);
            weighted_fitness.insert("register_novelty", register_novelty);

            // Measure write novelty
            let mem_scores = profile
                .write_logs
                .iter()
                .flatten()
                .map(|m| {
                    sketch.memory_writes.insert(m);
                    sketch.memory_writes.query(m)
                })
                .collect::<Vec<f64>>();
            let mem_write_novelty = if mem_scores.is_empty() {
                1.0
            } else {
                stats::mean(mem_scores.into_iter())
            };
            weighted_fitness.insert("mem_write_novelty", mem_write_novelty);

            // // get the number of times a referenced value, aside from 0, has been written
            // // to memory
            // let writes_of_referenced_values =
            //     pattern.count_writes_of_referenced_values(&profile, true);
            // weighted_fitness.insert("important_writes", writes_of_referenced_values as f64);

            // how many times did it crash?
            let crashes = profile.cpu_errors.values().sum::<usize>() as f64;
            weighted_fitness.insert("crash_count", crashes);

            let gadgets_executed = profile.gadgets_executed.len();
            weighted_fitness.insert("gadgets_executed", gadgets_executed as f64);

            // FIXME: not sure how sound this frequency gauging scheme is.
            //let gen_freq = creature.query_genetic_frequency(sketch);
            //creature.record_genetic_frequency(sketch);

            //weighted_fitness.scores.insert("genetic_freq", gen_freq);

            creature.set_fitness(weighted_fitness);
        } else {
            log::error!("No register pattern?");
        }
    }
    creature
}

pub fn register_conjunction_ff<'a>(
    mut creature: Creature<u64>,
    _sketch: &mut Sketches,
    config: Arc<Config>,
) -> Creature<u64> {
    if let Some(ref profile) = creature.profile {
        if let Some(registers) = profile.registers.last() {
            let word_size = get_static_memory_image().word_size * 8;
            let mask = 2 ^ word_size - 1;
            let conj = registers.0.values().fold(mask as u64, |a, b| a & b[0]);
            let score = conj.count_zeros() as usize;
            // ignore bits outside of the register's word size
            let ignore_bits = 64 - word_size;
            let score = (score - ignore_bits) as f64;
            let mut weighted_fitness = Weighted::new(config.fitness.weights.clone());
            weighted_fitness.scores.insert("zeroes", score);
            weighted_fitness
                .scores
                .insert("gadgets_executed", profile.gadgets_executed.len() as f64);

            let mem_write_ratio = profile.mem_write_ratio();
            weighted_fitness
                .scores
                .insert("mem_write_ratio", mem_write_ratio);
            creature.set_fitness(weighted_fitness);
        }
    }
    creature
}

pub struct Sketches {
    pub register_error: CountMinSketch,
    pub memory_writes: CountMinSketch,
    pub addresses_visited: CountMinSketch,
}

impl Sketches {
    pub fn new(config: &Config) -> Self {
        Self {
            register_error: CountMinSketch::new(config),
            memory_writes: CountMinSketch::new(config),
            addresses_visited: CountMinSketch::new(config),
        }
    }
}

pub struct Evaluator<C: 'static + Cpu<'static>> {
    config: Arc<Config>,
    hatchery: Hatchery<C, Creature<u64>>,
    sketches: Sketches,
    fitness_fn: Box<FitnessFn<Creature<u64>, Sketches, Config>>,
}

impl<C: 'static + Cpu<'static>> Evaluator<C> {
    pub fn spawn(config: &Config, fitness_fn: FitnessFn<Creature<u64>, Sketches, Config>) -> Self {
        let mut config = config.clone();
        config.roper.parse_register_pattern();
        let hatch_config = Arc::new(config.roper.clone());
        let register_pattern = config.roper.register_pattern();
        let output_registers: Vec<Register<C>> = {
            let mut out_reg: Vec<Register<C>> = config
                .roper
                .output_registers
                .iter()
                .map(|s| s.parse().ok().expect("Failed to parse output register"))
                .collect::<Vec<_>>();
            if let Some(pat) = register_pattern {
                let arch_specific_pat: UnicornRegisterState<C> =
                    pat.try_into().expect("Failed to parse register pattern");
                let regs_in_pat = arch_specific_pat.0.keys().cloned().collect::<Vec<_>>();
                out_reg.extend_from_slice(&regs_in_pat);
                out_reg.dedup();
                out_reg
            } else {
                out_reg
                //todo!("implement a conversion method from problem sets to register maps");
            }
        };
        let inputs = vec![util::architecture::random_register_state::<u64, C>(
            &output_registers,
            config.random_seed,
        )];
        let hatchery: Hatchery<C, Creature<u64>> = Hatchery::new(
            hatch_config,
            Arc::new(inputs),
            Arc::new(output_registers),
            None,
        );

        let sketches = Sketches::new(&config);
        Self {
            config: Arc::new(config),
            hatchery,
            sketches,
            fitness_fn: Box::new(fitness_fn),
        }
    }
}

impl<'a, C: 'static + Cpu<'static>> Develop<Creature<u64>> for Evaluator<C> {
    fn develop(&self, creature: Creature<u64>) -> Creature<u64> {
        if creature.profile.is_none() {
            let (mut creature, profile) = self
                .hatchery
                .execute(creature)
                .expect("Failed to evaluate creature");
            creature.profile = Some(profile);
            creature
        } else {
            creature
        }
    }

    fn apply_fitness_function(&mut self, creature: Creature<u64>) -> Creature<u64> {
        (self.fitness_fn)(creature, &mut self.sketches, self.config.clone())
    }

    fn development_pipeline<'b, I: 'static + Iterator<Item = Creature<u64>> + Send>(
        &self,
        inbound: I,
    ) -> Vec<Creature<u64>> {
        // we need to have the entire sample pass through the count-min sketch
        // before we can use it to measure the frequency of any individual
        let (old_meat, fresh_meat): (Vec<Creature<u64>>, _) =
            inbound.partition(Creature::<u64>::mature);
        let batch = self
            .hatchery
            .execute_batch(fresh_meat.into_iter())
            .expect("execute batch failure")
            .into_iter()
            .map(|(mut creature, profile)| {
                creature.profile = Some(profile);
                creature
            })
            .chain(old_meat)
            // NOTE: in progress of decoupling fitness function from development
            // .map(|creature| (self.fitness_fn)(creature, &mut self.sketch, self.config.clone()))
            .collect::<Vec<_>>();
        batch
    }
}

pub mod lexi {
    use crate::emulator::register_pattern::RegisterFeature;
    use crate::roper::creature::Creature;

    #[derive(Clone, Debug, Hash)]
    pub enum Task {
        Reg(RegisterFeature),
        UniqExec(usize),
    }

    impl Task {
        pub fn check_creature(&self, creature: &Creature<u64>) -> bool {
            match self {
                Task::Reg(rf) => {
                    if let Some(ref profile) = creature.profile {
                        profile.registers.iter().all(|state| rf.check_state(state))
                    } else {
                        false
                    }
                }
                Task::UniqExec(n) => {
                    if let Some(ref profile) = creature.profile {
                        profile.gadgets_executed.len() >= *n
                    } else {
                        false
                    }
                }
            }
        }
    }
}
