use rand::{thread_rng, Rng};

use crate::configure::Config;
use crate::evaluator::Evaluate;
use crate::evolution::{Genome, Phenome};
use crate::observer::Observer;
use crate::util::count_min_sketch::{SeasonalSketch, Sketch};
use crate::EPOCH_COUNTER;
use std::sync::atomic::Ordering;

pub struct Metropolis<E: Evaluate<P, SeasonalSketch>, P: Phenome + Genome + 'static> {
    pub specimen: P,
    pub config: Config,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
    pub best: Option<P>,
}

impl<E: Evaluate<P, SeasonalSketch>, P: Phenome + Genome + 'static> Metropolis<E, P> {
    pub fn new(config: Config, observer: Observer<P>, evaluator: E) -> Self {
        let specimen = P::random(&config);

        Self {
            specimen,
            config,
            iteration: 0,
            observer,
            evaluator,
            best: None,
        }
    }
}

impl<E: Evaluate<P, SeasonalSketch>, P: Phenome + Genome> Metropolis<E, P> {
    pub fn evolve(self) -> Self {
        let Self {
            specimen,
            config,
            iteration,
            observer,
            mut evaluator,
            best,
        } = self;

        EPOCH_COUNTER.fetch_add(1, Ordering::Relaxed);

        let mut specimen = if specimen.fitness().is_none() {
            evaluator.evaluate(specimen)
        } else {
            specimen
        };
        let variation = Genome::mate(&[&specimen, &specimen], &config);
        let variation = evaluator.evaluate(variation);

        let mut rng = thread_rng();
        let vari_fit = variation.scalar_fitness().unwrap();
        let spec_fit = specimen.scalar_fitness().unwrap();
        let delta = if (vari_fit - spec_fit).abs() < std::f64::EPSILON {
            0.0
        } else {
            1.0 / (vari_fit - spec_fit)
        };

        // if the variation is fitter, replace.
        // otherwise, let there be a chance of replacement inversely proportionate to the
        // difference in fitness.
        if delta < 0.0 || ((-delta).exp()) < rng.gen_range(0.0, 1.0) {
            //if delta < 0.0 { // pure hillclimbing
            specimen = variation;
            log::info!(
                "[{}] best: {:?}. specimen: {}, variation: {} (delta {}), switching",
                iteration,
                best.as_ref().and_then(Phenome::scalar_fitness),
                spec_fit,
                vari_fit,
                delta
            );
        };
        observer.observe(specimen.clone());

        let mut updated_best = false;
        let best = match best {
            Some(b) if specimen.scalar_fitness().unwrap() < b.scalar_fitness().unwrap() => {
                updated_best = true;
                Some(specimen.clone())
            }
            None => {
                updated_best = true;
                Some(specimen.clone())
            }
            _ => best,
        };
        if updated_best {
            log::info!("new best: {:?}", best.as_ref().unwrap());
        }
        Self {
            specimen,
            config,
            iteration: iteration + 1,
            observer,
            evaluator,
            best,
        }
    }
}
