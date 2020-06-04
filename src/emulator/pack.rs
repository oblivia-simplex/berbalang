use crate::util::architecture::Endian;

pub trait Pack {
    fn pack(&self, word_size: usize, endian: Endian) -> Vec<u8>;
    fn as_addrs(&self, word_size: usize, endian: Endian) -> &[u64];
}

impl Pack for Vec<u8> {
    fn pack(&self, _w: usize, _e: Endian) -> Vec<u8> {
        self.clone()
    }
    fn as_addrs(&self, _word_size: usize, _endian: Endian) -> &[u64] {
        unimplemented!("not implemented")
    }
}
