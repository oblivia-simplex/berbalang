use std::convert::TryInto;
use std::sync::Arc;

use unicorn::Cpu;

use crate::configure::Config;
use crate::emulator::hatchery::Hatchery;
use crate::emulator::profiler::{HasProfile, Profile};
use crate::emulator::register_pattern::{Register, RegisterPattern, UnicornRegisterState};
use crate::evolution::{Genome, Phenome};
use crate::fitness::Weighted;
use crate::ontogenesis::{Develop, FitnessFn};
use crate::roper::push;
use crate::roper::push::{register_pattern_to_push_args, Creature, MachineState};
use crate::roper::Sketches;
use crate::util;

pub struct Evaluator<C: Cpu<'static> + 'static> {
    config: Arc<Config>,
    hatchery: Hatchery<C>,
    sketches: Sketches,
    fitness_fn: Box<FitnessFn<push::Creature, Sketches, Config>>,
}

impl<C: 'static + Cpu<'static>> Evaluator<C> {
    pub fn spawn(config: &Config, fitness_fn: FitnessFn<Creature, Sketches, Config>) -> Self {
        let config = config.clone();
        let hatch_config = Arc::new(config.roper.clone());
        let register_patterns = config.roper.register_patterns();
        let output_registers: Vec<Register<C>> = {
            let mut out_reg: Vec<Register<C>> = config
                .roper
                .output_registers
                .iter()
                .map(|s| s.parse().ok().expect("Failed to parse output register"))
                .collect::<Vec<_>>();
            // This is a bit of a hack, where we assume that all the register patterns
            // use the same registers. It can be FIXME'd later, so that we gather ALL the
            // registers used. But let's keep things simple for now.
            if let Some(pat) = register_patterns.last() {
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
        let initial_register_state = if config.roper.randomize_registers {
            util::architecture::random_register_state::<u64, C>(
                &output_registers,
                config.random_seed,
            )
        } else {
            util::architecture::constant_register_state::<C>(&output_registers, 0_u64)
        };
        let hatchery: Hatchery<C> = Hatchery::new(
            hatch_config,
            Arc::new(initial_register_state),
            Arc::new(output_registers),
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

pub fn problem_to_payload(
    creature: &push::Creature,
    problem: &RegisterPattern,
    steps: usize,
) -> Vec<u64> {
    let args = register_pattern_to_push_args(&problem);
    let mut machine = MachineState::default();
    machine.exec(creature.chromosome(), &args, steps)
}

impl<C: 'static + Cpu<'static>> Develop<push::Creature> for Evaluator<C> {
    fn develop(&self, mut creature: push::Creature) -> push::Creature {
        // TODO: make this a bit more generic, so we don't assume we're doing a register pattern task
        // for now, this doesn't matter -- we haven't defined any other kinds of tasks
        if creature.fitness.is_none() {
            let mut payloads = Vec::new();
            // TODO: Refactor and generalize to other problem types.
            for register_pattern in self.config.roper.register_patterns() {
                let payload =
                    problem_to_payload(&creature, register_pattern, self.config.push_vm.max_steps);
                payloads.push(payload);
            }

            // TODO refactor bare roper in a similar fashion. just send the payload,
            // not the whole creature.
            let mut used_payloads = Vec::new();
            for payload in payloads.into_iter() {
                if !payload.is_empty() {
                    let profile = self
                        .hatchery
                        .execute(payload.clone(), None)
                        .expect("Failed to evaluate creature");
                    creature.add_profile(profile);
                    used_payloads.push(payload);
                } else {
                    // this will mark the profile as non-executable
                    let profile = Profile::default();
                    debug_assert!(!profile.executable);
                    used_payloads.push(payload);
                    creature.payloads = used_payloads;
                    creature.add_profile(profile);
                    return creature;
                }
            }
            creature.payloads = used_payloads;
            log::debug!(
                "Finished developing creature. profile: {:#x?}",
                creature.profile
            );
        }
        creature
    }

    fn apply_fitness_function(&mut self, mut creature: push::Creature) -> push::Creature {
        let profile = creature
            .profile()
            .expect("Attempted to apply fitness function to undeveloped creature");
        if !profile.executable {
            let mut fitness = Weighted::new(&self.config.fitness.weighting);
            fitness.declare_failure();
            creature.set_fitness(fitness);
            creature
        } else {
            (self.fitness_fn)(creature, &mut self.sketches, self.config.clone())
        }
    }

    fn development_pipeline<I: 'static + Iterator<Item = push::Creature> + Send>(
        &self,
        inbound: I,
    ) -> Vec<push::Creature> {
        inbound
            .into_iter()
            .map(|c| self.develop(c))
            .collect::<Vec<push::Creature>>()
    }
}
