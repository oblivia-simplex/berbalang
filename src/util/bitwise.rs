use byteorder::{BigEndian, ByteOrder, LittleEndian};

use crate::util::architecture::Endian;

pub fn bit(n: u64, bit: usize) -> bool {
    (n >> (bit as u64 % 64)) & 1 == 1
}

pub fn nybble(n: u64, index: usize) -> u8 {
    ((n >> (4 * index as u64)) & 0x0F) as u8
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

pub fn try_word_as_string(w: u64, endian: Endian, word_size: usize) -> Option<String> {
    debug_assert!(word_size == 4 || word_size == 8);
    let mut buf = [0_u8; 8];
    let mut s = String::new();
    let mut max_ch_count = 0;
    let mut ch_count = 0;
    match endian {
        Endian::Little => {
            LittleEndian::write_u64(&mut buf, w);
        }
        Endian::Big => {
            BigEndian::write_u64(&mut buf, w);
        }
    }

    let buf = match endian {
        Endian::Little => &buf[..word_size],
        Endian::Big => &buf[(8 - word_size)..],
    };

    for byte in buf.iter() {
        if 0x20 <= *byte && *byte < 0x7f {
            let ch = *byte as char;
            s.push(ch);
            ch_count += 1;
        } else {
            if ch_count > max_ch_count {
                max_ch_count = ch_count;
            }
            ch_count = 0;
            s.push('·');
        }
    }
    if ch_count > max_ch_count {
        max_ch_count = ch_count;
    }
    if max_ch_count >= 3 {
        Some(s)
    } else {
        None
    }
}

pub fn try_str_as_word(mut s: String, endian: Endian) -> Option<u64> {
    if s.len() > 8 {
        return None;
    }
    while s.len() < 8 {
        s.push('\x00')
    }
    match endian {
        Endian::Little => Some(LittleEndian::read_u64(s.as_bytes())),
        Endian::Big => Some(BigEndian::read_u64(s.as_bytes())),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_try_word_as_str() {
        let w = 0x41424344_45464748;
        let s = try_word_as_string(w, Endian::Little, 4);
        println!("0x{:x} -> {:?}", w, s);
        assert_eq!(s, Some("HGFE".to_string()));
        let s = try_word_as_string(w, Endian::Big, 4);
        println!("0x{:x} -> {:?}", w, s);
        assert_eq!(s, Some("EFGH".to_string()));
        let s = try_word_as_string(w, Endian::Big, 8);
        println!("0x{:x} -> {:?}", w, s);
        assert_eq!(s, Some("ABCDEFGH".to_string()));
        let s = try_word_as_string(w, Endian::Little, 8);
        println!("0x{:x} -> {:?}", w, s);
        assert_eq!(s, Some("HGFEDCBA".to_string()));
    }
}
