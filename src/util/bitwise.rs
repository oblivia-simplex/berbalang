pub fn bit(n: u64, bit: usize) -> bool {
    (n >> (bit as u64 % 64)) & 1 == 1
}

/// We can measure the similarity between two words as the number of bit-flips
/// necessary to transform one into the other, divided by the size of the word.
/// This is the same as counting the ones in the first `xor` the second.
/// For example:
/// ```
/// 0101 ^ 0110 = 0011 -> 2 ones -> 50% similar
/// ```
#[inline]
pub fn ham_rat(a: u64, b: u64) -> f64 {
    (a ^ b).count_ones() as f64 / 64.0
}
