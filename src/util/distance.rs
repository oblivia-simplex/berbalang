use seahash::hash_seeded;
use std::convert::Into;
use std::iter::Iterator;

/// See https://en.wikipedia.org/wiki/MinHash for discussion of algorithm
pub fn jaccard(one: &[u8], two: &[u8], grain: usize, num_hashes: u64) -> f64 {
    // the keys in the profile's maps and the pattern's map
    // should be in an identical order, just because nothing should
    // have disturbed them. But it would be better to verify this.
    let self_byte_pos = byte_positions(&one, grain);
    let other_byte_pos = byte_positions(&two, grain);

    (0_u64..num_hashes)
        .filter(|seed| {
            let s = self_byte_pos
                .iter()
                .map(|b| hash_seeded(b, *seed, 0, 0, 0))
                .min();
            let o = other_byte_pos
                .iter()
                .map(|b| hash_seeded(b, *seed, 0, 0, 0))
                .min();
            s == o
        })
        .count() as f64
        / num_hashes as f64
}

fn byte_positions(bytes: &[u8], grain: usize) -> Vec<[u8; 4]> {
    let len = bytes.len();
    debug_assert!(grain < len);
    let chunk = len / grain;
    bytes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let (byte, pos) = (*b, i / chunk);
            let mut buf = [0_u8; 4];
            buf[0] = byte;
            buf[1] = (pos & 0xFF) as u8;
            buf[2] = ((pos >> 8) & 0xFF) as u8;
            buf[3] = ((pos >> 16) & 0xFF) as u8;
            buf
        })
        .collect::<Vec<[u8; 4]>>()
}
