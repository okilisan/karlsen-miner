#![allow(clippy::unreadable_literal)]
use crate::Hash;
use blake2b_simd::State as Blake2bState;

const BLOCK_HASH_DOMAIN: &[u8] = b"BlockHash";

#[derive(Clone)]
pub struct HeaderHasher(Blake2bState);

impl HeaderHasher {
    #[inline(always)]
    pub fn new() -> Self {
        Self(blake2b_simd::Params::new().hash_length(32).key(BLOCK_HASH_DOMAIN).to_state())
    }

    pub fn write<A: AsRef<[u8]>>(&mut self, data: A) {
        self.0.update(data.as_ref());
    }

    #[inline(always)]
    pub fn finalize(self) -> Hash {
        Hash::from_le_bytes(self.0.finalize().as_bytes().try_into().expect("this is 32 bytes"))
    }
}

pub trait Hasher {
    fn update<A: AsRef<[u8]>>(&mut self, data: A) -> &mut Self;
}

impl Hasher for HeaderHasher {
    fn update<A: AsRef<[u8]>>(&mut self, data: A) -> &mut Self {
        self.write(data);
        self
    }
}
