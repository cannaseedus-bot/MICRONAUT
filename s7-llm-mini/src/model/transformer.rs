use super::linear::Linear;

pub struct S7Mini {
    pub embedding: Linear,
    pub lm_head: Linear,
}

impl S7Mini {
    /// Construct from a parsed .s7l file.
    /// FIELD lane (id=2) carries packed INT8 weight tensors.
    pub fn from_s7(_file: &crate::s7l::S7File) -> Self {
        // Minimal mock wiring — replace with lane deserialisation once
        // a real weight file is sealed into the FIELD lane.
        unimplemented!(
            "Wire S7Mini from .s7l FIELD lane (id=2). \
             Seal weights with: felc --lane FIELD model/weights.bin → model/mini.s7l"
        )
    }

    /// Forward pass: embed tokens → project to logits.
    pub fn forward(&self, tokens: &[i8]) -> Vec<i8> {
        let hidden = self.embedding.forward(tokens);
        self.lm_head.forward(&hidden)
    }
}
