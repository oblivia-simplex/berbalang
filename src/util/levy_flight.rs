use rand::Rng;
use rand_distr::Distribution;

/// See "A New Mutation Paradigm for Genetic Programming", by
/// Christian Darabos et al, in _Genetic Programming: Theory and
/// Practice, Vol X_.
pub fn levy_flight_rate(length: usize, exponent: f64) -> f64 {
    let c = 1.0
        / (1..(length + 1))
            .map(|l| (l as f64).powf(exponent))
            .sum::<f64>();
    c * (length as f64).powf(exponent)
}

pub fn levy_decision<R: Rng>(rng: &mut R, length: usize, exponent: f64) -> bool {
    debug_assert!(length > 0);
    let thresh = 1.0 - (1.0 / length as f64);
    rand_distr::Exp::new(exponent)
        .expect("Bad exponent for Exp distribution")
        .sample(rng)
        >= thresh
}

#[cfg(test)]
mod test {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_levy_flight_rate() {
        let rate = levy_flight_rate(10, 2.0);
        println!("rate = {}", rate);

        let len = 2;
        let mut rng = thread_rng();
        for _ in 0..len {
            println!(
                "[{}]",
                if levy_decision(&mut rng, len, 2.9) {
                    '*'
                } else {
                    ' '
                }
            );
        }
    }
}
