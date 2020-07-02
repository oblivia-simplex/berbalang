use crate::util::architecture::Endian;
use hashbrown::HashMap;

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
