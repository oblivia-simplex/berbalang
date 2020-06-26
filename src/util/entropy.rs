/// Calculates the Shannon entropy of a byte string.
/// (Code borrowed from the shannon_entropy crate.
pub fn shannon_entropy(bytes: &[u8]) -> f64 {
    let mut counts = [0; 256];

    for &b in bytes {
        counts[b as usize] += 1;
    }

    let len = bytes.len() as f64;

    counts
        .iter()
        .filter(|&&n| n > 0)
        .map(|&n| n as f64 / len)
        .map(|p| -(p * p.log(2.0)))
        .sum()
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

    use super::*;

    #[test]
    fn test_entropy_empty() {
        let h = b"".entropy();
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_entropy_a() {
        let h = shannon_entropy(b"a");
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_entropy_aaaaaaaa() {
        let h = shannon_entropy(b"AAAAAAAA");
        assert_eq!(h, 0.0);
        let bytes: [u8; 8] = [0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41];
        let h = shannon_entropy(&bytes);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_entropy_0x41414141_41414141() {
        let h = [0x41414141_41414141_u64].entropy();
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_entropy_ab() {
        let h = shannon_entropy(b"ab");
        assert_eq!(h, 1.0);
    }

    #[test]
    fn test_entropy_aab() {
        let h = shannon_entropy(b"aab");
        assert_eq!(h, 0.9182958340544896);
    }

    #[test]
    fn test_entropy_equal_distribution() {
        let mut bytes = [0u8; 256];
        for i in 0..256 {
            bytes[i] = i as u8;
        }
        let h = shannon_entropy(&bytes);
        assert_eq!(h, 8.0);
    }

    #[test]
    fn test_entropy_equal_distribution_2() {
        let mut bytes = [0u8; 256 * 2];
        for i in 0..256 * 2 {
            bytes[i] = (i % 256) as u8;
        }
        let h = shannon_entropy(&bytes);
        assert_eq!(h, 8.0);
    }

    #[test]
    fn test_entropy_helloworld() {
        let h = shannon_entropy(b"hello, world");
        assert_eq!(h, 3.0220552088742005);
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
