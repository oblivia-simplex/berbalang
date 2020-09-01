use std::convert::TryInto;
use std::sync::Arc;

use unicorn::Cpu;

use crate::emulator::register_pattern::{Register, UnicornRegisterState};
use crate::ontogenesis::FitnessFn;
use crate::roper::Sketches;
use crate::{
    configure::Config, emulator::hatchery::Hatchery, evolution::Phenome, ontogenesis::Develop, util,
};

use super::*;

pub struct Evaluator<C: 'static + Cpu<'static>> {
    config: Arc<Config>,
    hatchery: Hatchery<C, Creature>,
    sketches: Sketches,
    fitness_fn: Box<FitnessFn<Creature, Sketches, Config>>,
}

impl<C: 'static + Cpu<'static>> Evaluator<C> {
    pub fn spawn(config: &Config, fitness_fn: FitnessFn<Creature, Sketches, Config>) -> Self {
        let mut config = config.clone();
        config.roper.parse_register_patterns();
        let hatch_config = Arc::new(config.roper.clone());
        let output_registers: Vec<Register<C>> = {
            config
                .roper
                .registers_to_check()
                .into_iter()
                // running error through ok() because it can't be formatted with Debug
                .map(|r| r.parse().ok().expect("Failed to parse register name"))
                .collect::<Vec<_>>()
        };
        let initial_register_states = if config.roper.randomize_registers {
            vec![util::architecture::random_register_state::<u64, C>(
                &output_registers,
                config.random_seed,
            )]
        } else {
            vec![util::architecture::constant_register_state::<C>(
                &output_registers,
                1_u64,
            )]
        };
        let hatchery: Hatchery<C, Creature> = Hatchery::new(
            hatch_config,
            Arc::new(initial_register_states),
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

// And refactor the modules a bit.
impl<'a, C: 'static + Cpu<'static>> Develop<Creature> for Evaluator<C> {
    fn develop(&self, creature: Creature) -> Creature {
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

    fn apply_fitness_function(&mut self, creature: Creature) -> Creature {
        (self.fitness_fn)(creature, &mut self.sketches, self.config.clone())
    }

    fn development_pipeline<'b, I: 'static + Iterator<Item = Creature> + Send>(
        &self,
        inbound: I,
    ) -> Vec<Creature> {
        // we need to have the entire sample pass through the count-min sketch
        // before we can use it to measure the frequency of any individual
        let (old_meat, fresh_meat): (Vec<Creature>, _) = inbound.partition(Creature::mature);
        self.hatchery
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
            .collect::<Vec<_>>()
    }
}
