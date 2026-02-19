/// FIELD lane (id=2) deserialiser for the S7-LLM-MOE weight tensors.
///
/// Each tensor record in the FIELD lane payload:
///   u16  name_len
///   [u8] name (UTF-8, name_len bytes)
///   u8   rank
///   u32  dim[0..rank]  (big-endian)
///   f32  scale         (IEEE 754, big-endian)
///   [i8] data          (numel bytes, AVX2-padded to 32-byte boundary)
use super::Int8Tensor;
use super::int8::avx2_pad;

pub struct TensorRecord {
    pub tensor: Int8Tensor,
}

pub struct FieldLaneReader<'a> {
    data:   &'a [u8],
    offset: usize,
}

impl<'a> FieldLaneReader<'a> {
    pub fn new(payload: &'a [u8]) -> Self {
        Self { data: payload, offset: 0 }
    }

    pub fn has_more(&self) -> bool {
        self.offset < self.data.len()
    }

    /// Read the next tensor record. Returns None when the payload is exhausted.
    pub fn next_tensor(&mut self) -> Option<TensorRecord> {
        if !self.has_more() {
            return None;
        }

        // name
        let name_len = self.read_u16()? as usize;
        let name = std::str::from_utf8(self.read_bytes(name_len)?)
            .unwrap_or("<invalid-utf8>")
            .to_string();

        // dims
        let rank = self.read_u8()? as usize;
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(self.read_u32()? as usize);
        }

        // scale
        let scale = f32::from_be_bytes(self.read_bytes(4)?.try_into().ok()?);

        // data (AVX2-padded)
        let numel: usize = dims.iter().product();
        let padded = avx2_pad(numel);
        let raw = self.read_bytes(padded)?;
        let data: Vec<i8> = raw.iter().map(|&b| b as i8).collect();

        Some(TensorRecord {
            tensor: Int8Tensor { name, dims, scale, data },
        })
    }

    fn read_u8(&mut self) -> Option<u8> {
        if self.offset >= self.data.len() { return None; }
        let v = self.data[self.offset];
        self.offset += 1;
        Some(v)
    }

    fn read_u16(&mut self) -> Option<u16> {
        let b = self.read_bytes(2)?;
        Some(u16::from_be_bytes(b.try_into().ok()?))
    }

    fn read_u32(&mut self) -> Option<u32> {
        let b = self.read_bytes(4)?;
        Some(u32::from_be_bytes(b.try_into().ok()?))
    }

    fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.offset + n > self.data.len() { return None; }
        let slice = &self.data[self.offset..self.offset + n];
        self.offset += n;
        Some(slice)
    }
}

/// Write a tensor record to a byte buffer (used by the packer binary).
pub fn write_tensor_record(tensor: &Int8Tensor, buf: &mut Vec<u8>) {
    let name_bytes = tensor.name.as_bytes();
    let name_len = name_bytes.len() as u16;
    buf.extend_from_slice(&name_len.to_be_bytes());
    buf.extend_from_slice(name_bytes);
    buf.push(tensor.dims.len() as u8);
    for &d in &tensor.dims {
        buf.extend_from_slice(&(d as u32).to_be_bytes());
    }
    buf.extend_from_slice(&tensor.scale.to_be_bytes());
    // Write padded data (AVX2 alignment, pad with 0x00).
    let numel = tensor.numel();
    let padded = avx2_pad(numel);
    let raw: Vec<u8> = tensor.data.iter().map(|&b| b as u8).collect();
    buf.extend_from_slice(&raw[..padded.min(raw.len())]);
    for _ in raw.len()..padded {
        buf.push(0u8);
    }
}
