use std::convert::TryInto;
use std::sync::Arc;

use unicorn::Cpu;

use crate::configure::Config;
use crate::emulator::hatchery::Hatchery;
use crate::emulator::register_pattern::{Register, UnicornRegisterState};
use crate::evolution::Genome;
use crate::ontogenesis::{Develop, FitnessFn};
use crate::roper::push;
use crate::roper::push::{Creature, MachineState};
use crate::roper::Sketches;
use crate::util;

pub struct Evaluator<C: Cpu<'static> + 'static> {
    config: Arc<Config>,
    hatchery: Hatchery<C, push::Creature>,
    sketches: Sketches,
    fitness_fn: Box<FitnessFn<push::Creature, Sketches, Config>>,
}

impl<C: 'static + Cpu<'static>> Evaluator<C> {
    pub fn spawn(config: &Config, fitness_fn: FitnessFn<Creature, Sketches, Config>) -> Self {
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
        let inputs = if config.roper.randomize_registers {
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

// TODO: impl Develop<push::Creature> for Evaluator.
// It will look a lot like this, except that it will first generate the payload by executing the
// pushvm code.
impl<C: 'static + Cpu<'static>> Develop<push::Creature> for Evaluator<C> {
    fn develop(&self, mut creature: push::Creature) -> push::Creature {
        let args = vec![];

        let mut machine = MachineState::default();
        if creature.payload.is_none() {
            let payload = machine.exec(creature.chromosome(), &args, self.config.push_vm.max_steps);
            creature.payload = Some(payload);
        }
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

    fn apply_fitness_function(&mut self, creature: push::Creature) -> push::Creature {
        (self.fitness_fn)(creature, &mut self.sketches, self.config.clone())
    }

    fn development_pipeline<I: 'static + Iterator<Item = push::Creature> + Send>(
        &self,
        inbound: I,
    ) -> Vec<push::Creature> {
        let (old_meat, fresh_meat): (Vec<Creature>, _) = inbound.partition(|c| c.profile.is_some());
        self.hatchery
            .execute_batch(fresh_meat.into_iter().map(|mut c| {
                if c.payload.is_none() {
                    let args = vec![];
                    let payload = MachineState::default().exec(
                        c.chromosome(),
                        &args,
                        self.config.push_vm.max_steps,
                    );
                    c.payload = Some(payload)
                }
                c
            }))
            .expect("execute batch failure")
            .into_iter()
            .map(|(mut creature, profile)| {
                creature.profile = Some(profile);
                creature
            })
            .chain(old_meat)
            .collect::<Vec<_>>()
    }
}
