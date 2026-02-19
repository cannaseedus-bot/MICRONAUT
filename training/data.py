"""
Curriculum data loader for S7-LLM-MOE-300M.

Two loading modes:

  1. Mixed batches (Phase 1 / Phase 2 default):
     Each batch is sampled from the domain mix in config.domain_ratios.
     The model sees all domains interleaved → router learns from mixture.

  2. Domain-curriculum batches (Phase 2 specialist pressure):
     Batches of a single domain are fed to push routing toward domain experts.
     Controlled by curriculum_phase=True in TrainConfig.

Domain sources (configurable; paths resolved at runtime):
    code      → The Stack v2 / CodeParrot (tokenized)
    math      → MathPile + AMPS + OpenWebMath (tokenized)
    reasoning → OpenHermes / CoT traces (tokenized)
    general   → RedPajama-v2 / Dolma / C4 (tokenized)

All datasets are pre-tokenized .bin files (uint16 token ids, row-major).
One row = one sequence of length seq_len.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional, Tuple


# ── Binary Token Dataset ──────────────────────────────────────────────────────

class TokenBinDataset(Dataset):
    """
    Memory-mapped pre-tokenized binary dataset.
    File format: uint16 tokens, shape [N, seq_len+1] (last token = next-token label).
    """
    def __init__(self, path: str, seq_len: int):
        self.seq_len = seq_len
        data = np.memmap(path, dtype=np.uint16, mode="r")
        stride = seq_len + 1
        self.n_seqs = len(data) // stride
        self.data   = data[:self.n_seqs * stride].reshape(self.n_seqs, stride)

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row    = torch.from_numpy(self.data[idx].astype(np.int64))
        tokens = row[:-1]   # [seq_len]
        labels = row[1:]    # [seq_len]
        return tokens, labels


# ── Curriculum Sampler ────────────────────────────────────────────────────────

class CurriculumLoader:
    """
    Yields batches from a weighted mix of domain datasets.

    In domain-curriculum mode (curriculum_step != None):
      Every K steps, a single domain batch is injected to push routing pressure.
    """
    def __init__(
        self,
        domain_paths: Dict[str, str],     # domain → .bin file path
        domain_ratios: Dict[str, float],  # domain → sampling weight
        seq_len: int,
        batch_size: int,
        curriculum_step: Optional[int] = 100,  # None to disable
        seed: int = 42,
    ):
        self.seq_len          = seq_len
        self.batch_size       = batch_size
        self.curriculum_step  = curriculum_step
        self._step            = 0
        self._rng             = random.Random(seed)

        # Build datasets and weighted sampler.
        self.datasets: Dict[str, TokenBinDataset] = {}
        for domain, path in domain_paths.items():
            if os.path.exists(path):
                self.datasets[domain] = TokenBinDataset(path, seq_len)
            else:
                raise FileNotFoundError(
                    f"Dataset file not found: {path}\n"
                    f"Pre-tokenize with: python data.py --tokenize --domain {domain}"
                )

        # Flat list of (domain, local_index) with weighting.
        self.domain_list = list(domain_ratios.keys())
        weights = [domain_ratios[d] for d in self.domain_list]
        total_w = sum(weights)
        self.probs = [w / total_w for w in weights]

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns (tokens, labels, domain_name).
        tokens/labels: [batch_size, seq_len]
        """
        self._step += 1

        # Curriculum injection: every `curriculum_step` steps,
        # yield a pure domain batch.
        if (self.curriculum_step is not None
                and self._step % self.curriculum_step == 0):
            domain = self._rng.choice(self.domain_list)
        else:
            domain = self._rng.choices(self.domain_list, weights=self.probs)[0]

        ds = self.datasets[domain]
        indices = [self._rng.randint(0, len(ds) - 1)
                   for _ in range(self.batch_size)]

        tokens_list, labels_list = zip(*[ds[i] for i in indices])
        tokens = torch.stack(tokens_list)   # [B, T]
        labels = torch.stack(labels_list)   # [B, T]

        return tokens, labels, domain


# ── Tokenizer helper ─────────────────────────────────────────────────────────

def tokenize_files(
    input_dir: str,
    output_path: str,
    vocab_path: str,
    seq_len: int,
    domain: str,
):
    """
    Tokenize raw text files in input_dir into a uint16 binary dataset.
    Requires HuggingFace tokenizers library.

    Usage:
        python data.py --tokenize \
            --input-dir  data/raw/code/ \
            --output     data/tokenized/code.bin \
            --vocab      model/vocab.json \
            --seq-len    2048 \
            --domain     code
    """
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError:
        raise ImportError("pip install tokenizers")

    tokenizer = HFTokenizer.from_file(vocab_path)
    seqs = []
    n_files = 0

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        ids = tokenizer.encode(text).ids
        # Split into non-overlapping windows of seq_len+1.
        for start in range(0, len(ids) - seq_len, seq_len + 1):
            seqs.append(ids[start : start + seq_len + 1])
        n_files += 1

    arr = np.array(seqs, dtype=np.uint16)
    arr.tofile(output_path)
    print(f"[data] domain={domain} files={n_files} seqs={len(seqs)} → {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--input-dir",  default="data/raw/")
    parser.add_argument("--output",     default="data/tokenized/domain.bin")
    parser.add_argument("--vocab",      default="model/vocab.json")
    parser.add_argument("--seq-len",    type=int, default=2048)
    parser.add_argument("--domain",     default="general")
    args = parser.parse_args()

    if args.tokenize:
        tokenize_files(
            args.input_dir, args.output, args.vocab,
            args.seq_len, args.domain,
        )
