/// Calculates the Shannon entropy of a byte slice.
#[inline]
fn protected_log(n: f64) -> f64 {
    if n <= 0.0 {
        0.0
    } else {
        n.log(2.0)
    }
}

#[inline]
fn shannon_entropy(bytes: &[u8]) -> f64 {
    let mut counts = [0.0; 256];

    for &b in bytes {
        counts[b as usize] += 1.0;
    }

    let s: f64 = counts.iter().sum();
    let l: f64 = counts.iter().copied().map(|c| c * protected_log(c)).sum();
    s.log(2.0) - l / s
}

pub fn metric_entropy(bytes: &[u8]) -> f64 {
    shannon_entropy(bytes) / bytes.len() as f64
}

pub trait Entropy {
    fn entropy(&self) -> f64;
    fn metric_entropy(&self) -> f64;
}

impl Entropy for [u8] {
    fn entropy(&self) -> f64 {
        assert!(!self.is_empty());
        shannon_entropy(self)
    }

    fn metric_entropy(&self) -> f64 {
        metric_entropy(self)
    }
}

impl Entropy for [u64] {
    /// Shannon Entropy for `[u64]` is implemented on a bytewise basis.
    fn entropy(&self) -> f64 {
        let mut counts = [0; 256];

        for &w in self {
            for i in 0..8 {
                let b = (w >> (i * 8)) & 0xFF;
                counts[b as usize] += 1;
            }
        }

        let len = 8.0 * self.len() as f64;
        counts
            .iter()
            .filter(|&&n| n > 0)
            .map(|&n| n as f64 / len)
            .map(|p| -(p * p.log(2.0)))
            .sum()
    }

    fn metric_entropy(&self) -> f64 {
        self.entropy() / (self.len() as f64 * 8.0)
    }
}

#[cfg(test)]
mod test {
    use std::iter;

    use crate::assert_close_f64;

    use super::*;

    #[test]
    fn test_entropy_a() {
        let h = b"a".entropy();
        assert!(h <= std::f64::EPSILON);
    }

    #[test]
    fn test_entropy_aaaaaaaa() {
        let h = b"AAAAAAAA".entropy();
        assert!(h <= std::f64::EPSILON);
        let bytes: [u8; 8] = [0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41];
        let h = bytes.entropy();
        assert!(h <= std::f64::EPSILON);
    }

    #[test]
    fn test_entropy_0x41414141_41414141() {
        let h = [0x41414141_41414141_u64].entropy();
        assert!(h <= std::f64::EPSILON);
    }

    #[test]
    fn test_entropy_ab() {
        let h = b"ab".entropy();
        assert!(h - 1.0 <= std::f64::EPSILON);
    }

    #[test]
    fn test_entropy_aab() {
        let h = b"aab".entropy();
        assert!(h - 0.9182958340544896 <= std::f64::EPSILON);
    }

    #[test]
    fn test_entropy_equal_distribution() {
        let mut bytes = [0u8; 256];
        for i in 0..256 {
            bytes[i] = i as u8;
        }
        let h = bytes.entropy();
        assert_close_f64!(h, 8.0);
    }

    #[test]
    fn test_entropy_equal_distribution_2() {
        let mut bytes = [0u8; 256 * 2];
        for i in 0..256 * 2 {
            bytes[i] = (i % 256) as u8;
        }
        let h = bytes.entropy();
        assert_close_f64!(h, 8.0);
    }

    #[test]
    fn test_entropy_helloworld() {
        let h = b"hello, world".entropy();
        assert_close_f64!(h, 3.0220552088742005);
    }

    #[test]
    fn test_entropy_of_random_u64s() {
        fn rnd_vec(len: usize) -> Vec<u64> {
            iter::repeat(())
                .map(|()| rand::random::<u64>())
                .take(len)
                .collect::<Vec<u64>>()
        }

        for _ in 0..100 {
            let v = rnd_vec(4);
            let h = v.entropy();
            println!("{:016x?}: {}", v, h);
        }

        let num = 1000;
        let it = iter::repeat(())
            .take(1000)
            .map(|()| rnd_vec(4))
            .map(|v| v.entropy());
        let mean = stats::mean(it.clone());
        let std_dev = stats::stddev(it);
        println!(
            "mean entropy of {} [u64; 4]: {}, std_dev: {}",
            num, mean, std_dev
        );
    }
}
