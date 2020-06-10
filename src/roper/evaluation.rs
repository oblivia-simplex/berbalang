use std::convert::TryInto;
use std::sync::Arc;

use unicorn::Cpu;

use crate::emulator::register_pattern::{Register, UnicornRegisterState};
use crate::evaluator::FitnessFn;
use crate::util::count_min_sketch;
use crate::{
    configure::Config,
    emulator::hatchery::Hatchery,
    evaluator::Evaluate,
    evolution::{Genome, Phenome},
    util,
    util::count_min_sketch::DecayingSketch,
};

use super::Creature;
use crate::fitness::Weighted;

pub fn register_pattern_fitness_fn(
    mut creature: Creature,
    sketch: &mut DecayingSketch,
    params: Arc<Config>,
) -> Creature {
    // measure fitness
    // for now, let's just handle the register pattern task
    if let Some(ref profile) = creature.profile {
        sketch.insert(&profile.registers);
        //let reg_freq = sketch.query(&profile.registers);
        if let Some(pattern) = params.roper.register_pattern() {
            // assuming that when the register pattern task is activated, there's only one register state
            // to worry about. this may need to be adjusted in the future. bit sloppy now.
            let writeable_memory = Some(&profile.writeable_memory[0][..]);
            let mut fitness_vector = pattern.distance(&profile.registers[0], writeable_memory);
            let mut weighted_fitness = Weighted {
                scores: fitness_vector,
                weights: params.fitness_weights.clone(),
            };
            // FIXME broken // fitness_vector.push(reg_freq);

            // how many times did it crash?
            // let crashes = profile.cpu_errors.values().sum::<usize>() as f64;
            // fitness_vector.insert("crash_count", crashes);

            let gadgets_executed = profile.gadgets_executed.len();
            weighted_fitness
                .scores
                .insert("gadgets_executed", gadgets_executed as f64);

            //let gen_freq = creature.measure_genetic_frequency(sketch);
            //fitness_vector.insert("genetic_frequency", gen_freq);
            // let longest_path = profile
            //     .bb_path_iter()
            //     .map(|v: &Vec<_>| {
            //         let mut s = v.clone();
            //         s.dedup();
            //         s.len()
            //     })
            //     .max()
            //     .unwrap_or(0) as f64;
            // fitness_vector.insert("longest_path", -longest_path); // let's see what happens when we use negative val
            creature.set_fitness(weighted_fitness);
        //log::debug!("fitness: {:?}", creature.fitness());
        } else {
            log::error!("No register pattern?");
        }
    }
    creature
}
pub struct Evaluator<C: 'static + Cpu<'static>> {
    params: Arc<Config>,
    hatchery: Hatchery<C, Creature>,
    sketch: DecayingSketch,
    fitness_fn: Box<FitnessFn<Creature, DecayingSketch, Config>>,
}

impl<C: 'static + Cpu<'static>> Evaluate<Creature> for Evaluator<C> {
    type Params = Config;
    type State = DecayingSketch;

    fn evaluate(&mut self, creature: Creature) -> Creature {
        let (mut creature, profile) = self
            .hatchery
            .execute(creature)
            .expect("Failed to evaluate creature");

        creature.profile = Some(profile);
        creature.tag = rand::random::<u64>(); // TODO: rethink this tag thing
        (self.fitness_fn)(creature, &mut self.sketch, self.params.clone())
    }

    fn eval_pipeline<I: 'static + Iterator<Item = Creature> + Send>(
        &mut self,
        inbound: I,
    ) -> Vec<Creature> {
        // we need to have the entire sample pass through the count-min sketch
        // before we can use it to measure the frequency of any individual

        let mut batch = self
            .hatchery
            .execute_batch(inbound)
            .expect("execute batch failure")
            .into_iter()
            .map(|(mut creature, profile)| {
                creature.profile = Some(profile);
                (self.fitness_fn)(creature, &mut self.sketch, self.params.clone())
            })
            .collect::<Vec<_>>();
        batch
            .iter_mut()
            .for_each(|c| c.record_genetic_frequency(&mut self.sketch));
        batch
    }

    fn spawn(
        params: &Self::Params,
        fitness_fn: FitnessFn<Creature, Self::State, Self::Params>,
    ) -> Self {
        let mut params = params.clone();
        params.roper.parse_register_pattern();
        let hatch_params = Arc::new(params.roper.clone());
        // fn random_register_state<C: 'static + Cpu<'static>>(
        //     registers: &[Register<C>],
        // ) -> HashMap<Register<C>, u64> {
        //     let mut map = HashMap::new();
        //     let mut rng = thread_rng();
        //     for reg in registers.iter() {
        //         map.insert(reg.clone(), rng.gen::<u64>());
        //     }
        //     map
        // }
        let register_pattern = params.roper.register_pattern();
        let output_registers: Vec<Register<C>> = {
            let mut out_reg: Vec<Register<C>> = params
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
                // sort alphabetically (lame)
                // out_reg.sort_by_key(|r| format!("{:?}", r));
                out_reg
            } else {
                todo!("implement a conversion method from problem sets to register maps");
                //out_reg
            }
        };
        let inputs = vec![util::architecture::random_register_state::<C>(
            &output_registers,
        )];
        let hatchery: Hatchery<C, Creature> = Hatchery::new(
            hatch_params,
            Arc::new(inputs),
            Arc::new(output_registers),
            None,
        );

        let sketch = DecayingSketch::new(
            count_min_sketch::suggest_depth(params.pop_size),
            count_min_sketch::suggest_width(params.pop_size),
            params.pop_size as f64 * 2.0,
        );
        Self {
            params: Arc::new(params),
            hatchery,
            sketch,
            fitness_fn: Box::new(fitness_fn),
        }
    }
}
