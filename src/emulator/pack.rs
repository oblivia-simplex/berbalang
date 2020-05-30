use crate::util::architecture::Endian;

pub trait Pack {
    fn pack(&self, word_size: usize, endian: Endian) -> Vec<u8>;
}

impl Pack for Vec<u8> {
    fn pack(&self, _w: usize, _e: Endian) -> Vec<u8> {
        self.clone()
    }
}
