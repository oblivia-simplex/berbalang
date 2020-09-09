use std::convert::TryInto;
use std::sync::Arc;

use unicorn::Cpu;

use crate::configure::ClassificationProblem;
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
        let config = config.clone();
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
        let initial_register_state = if config.roper.randomize_registers {
            util::architecture::random_register_state::<u64, C>(
                &output_registers,
                config.random_seed,
            )
        } else {
            util::architecture::constant_register_state::<C>(&output_registers, 1_u64)
        };
        let hatchery: Hatchery<C, Creature> = Hatchery::new(
            hatch_config,
            Arc::new(initial_register_state),
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

// TODO: refactor classification problems substantially.
// TODO: Cache this function. Memoize.
fn classification_problem_to_register_map<C: 'static + Cpu<'static>>(
    problem: &ClassificationProblem,
    input_registers: &Vec<String>,
) -> HashMap<Register<C>, u64> {
    let mut reg_map = HashMap::new();
    for (reg, val) in input_registers.iter().zip(problem.input.iter()) {
        let reg: Register<C> = reg.parse().ok().expect("Failed to parse register name");
        let val: u64 = *val as u64;
        reg_map.insert(reg, val);
    }
    reg_map
}

// And refactor the modules a bit.
impl<'a, C: 'static + Cpu<'static>> Develop<Creature> for Evaluator<C> {
    fn develop(&self, mut creature: Creature) -> Creature {
        if creature.profile.is_some() {
            return creature;
        }
        // TODO: implement classification task here.
        if let Some(ref problems) = self.config.problems {
            for problem in problems {
                let reg_map: HashMap<Register<C>, u64> = classification_problem_to_register_map::<C>(
                    problem,
                    &self.config.roper.input_registers,
                );
                let (mut executed_creature, profile) = self
                    .hatchery
                    .execute(creature, Some(reg_map))
                    .expect("Failed to evaluate creature");
                executed_creature.add_profile(profile);
                creature = executed_creature
            }
            return creature;
        }
        // We could detach the chromosome here (Vec<u64>) and send it
        // as the payload, instead of the entire creature, but the truth
        // is that it doesn't really matter. Only the ownership of the
        // creature is passed, so very little data is actually copied.
        // Cloning the chromosome, or snipping it off and retaching it,
        // is probably no less expensive, all things considered.
        // However, if we start appending arguments to the payload, then
        // we might want to do this differently.
        let (mut creature, profile) = self
            .hatchery
            .execute(creature, None)
            .expect("Failed to evaluate creature");
        creature.add_profile(profile);
        creature
    }

    fn apply_fitness_function(&mut self, creature: Creature) -> Creature {
        (self.fitness_fn)(creature, &mut self.sketches, self.config.clone())
    }

    fn development_pipeline<I: 'static + Iterator<Item = Creature> + Send>(
        &self,
        inbound: I,
    ) -> Vec<Creature> {
        inbound
            .into_iter()
            .map(|c| self.develop(c))
            .collect::<Vec<Creature>>()
    }
}
