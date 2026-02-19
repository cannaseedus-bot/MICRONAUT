/// 40-byte .s7l file header.
pub struct Header {
    pub magic:     [u8; 4],
    pub version:   u16,
    pub class:     u8,
    pub flags:     u8,
    pub root_hash: [u8; 32],
}

impl Header {
    pub fn parse(data: &[u8]) -> Self {
        let mut root = [0u8; 32];
        root.copy_from_slice(&data[8..40]);
        Self {
            magic:     data[0..4].try_into().unwrap(),
            version:   u16::from_be_bytes(data[4..6].try_into().unwrap()),
            class:     data[6],
            flags:     data[7],
            root_hash: root,
        }
    }

    /// Serialise header to bytes (used by the packer binary).
    pub fn to_bytes(&self) -> [u8; 40] {
        let mut out = [0u8; 40];
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version.to_be_bytes());
        out[6] = self.class;
        out[7] = self.flags;
        out[8..40].copy_from_slice(&self.root_hash);
        out
    }
}
