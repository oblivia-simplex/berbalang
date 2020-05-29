pub fn bit(n: u64, bit: usize) -> bool {
    (n >> (bit as u64 % 64)) & 1 == 1
}
