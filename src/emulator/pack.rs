use byteorder::{BigEndian, ByteOrder, LittleEndian};
use hashbrown::HashMap;

use crate::emulator::loader;
use crate::util::architecture::{Endian, Perms};

pub trait Pack {
    fn pack(
        &self,
        word_size: usize,
        endian: Endian,
        byte_filter: Option<&HashMap<u8, u8>>,
    ) -> Vec<u8>;
    fn as_code_addrs(&self, word_size: usize, endian: Endian) -> Vec<u64>;
}

impl Pack for Vec<u8> {
    fn pack(&self, _w: usize, _e: Endian, _byte_filter: Option<&HashMap<u8, u8>>) -> Vec<u8> {
        self.clone()
    }
    fn as_code_addrs(&self, _word_size: usize, _endian: Endian) -> Vec<u64> {
        unimplemented!("not implemented")
    }
}

impl Pack for Vec<u64> {
    fn pack(
        &self,
        word_size: usize,
        endian: Endian,
        byte_filter: Option<&HashMap<u8, u8>>,
    ) -> Vec<u8> {
        let packer = |&word, mut bytes: &mut [u8]| match (endian, word_size) {
            (Endian::Little, 8) => LittleEndian::write_u64(&mut bytes, word),
            (Endian::Big, 8) => BigEndian::write_u64(&mut bytes, word),
            (Endian::Little, 4) => LittleEndian::write_u32(&mut bytes, word as u32),
            (Endian::Big, 4) => BigEndian::write_u32(&mut bytes, word as u32),
            (Endian::Little, 2) => LittleEndian::write_u16(&mut bytes, word as u16),
            (Endian::Big, 2) => BigEndian::write_u16(&mut bytes, word as u16),
            (_, _) => unimplemented!("I think we've covered the bases"),
        };
        let mut ptr = 0;
        let mut buffer = vec![0_u8; self.len() * word_size];
        for word in self {
            packer(word, &mut buffer[ptr..]);
            ptr += word_size;
        }

        if let Some(byte_filter) = byte_filter {
            buffer
                .into_iter()
                .map(|b| {
                    if let Some(x) = byte_filter.get(&b) {
                        *x
                    } else {
                        b
                    }
                })
                .collect::<Vec<u8>>()
        } else {
            buffer
        }
    }

    fn as_code_addrs(&self, _word_size: usize, _endian: Endian) -> Vec<u64> {
        let memory = loader::get_static_memory_image();
        self.iter()
            .filter(|a| {
                memory
                    .perm_of_addr(**a)
                    .map(|p| p.intersects(Perms::EXEC))
                    .unwrap_or(false)
            })
            .cloned()
            .collect::<Vec<_>>()
    }
}
