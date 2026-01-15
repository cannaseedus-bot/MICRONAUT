# fel_meta_fold.py
"""
META_FOLD Attestation Executor (FEL v1.1)
- Deterministic, replayable attestation chain
- Canonical JSON hashing rules
- ABI + policy pinning (no drift)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# -----------------------------
# Canonicalization (verifier law)
# -----------------------------

TOP_LEVEL_ORDER = [
    "v",
    "tick",
    "type",
    "fold",
    "id",
    "ref",
    "payload",
    "key",
    "value",
    "patch",
    "lane",
    "root",
    "source",
    "policy_hash",
    "meta_hash",
    "abi_hash",
    "prev",
]

EXCLUDED_FROM_HASH = {"timestamp"}  # observational only, never hashed


def _strip_observational(x: Any) -> Any:
    """Remove observational keys recursively (e.g., timestamp)."""
    if isinstance(x, dict):
        return {k: _strip_observational(v) for k, v in x.items() if k not in EXCLUDED_FROM_HASH}
    if isinstance(x, list):
        return [_strip_observational(v) for v in x]
    return x


def canon_json(obj: Dict[str, Any]) -> bytes:
    """
    Canonical JSON bytes:
    - removes observational keys (timestamp)
    - stable key order (top-level per TOP_LEVEL_ORDER, then remaining sorted)
    - nested dict keys sorted
    - minimal separators
    """
    obj = _strip_observational(obj)

    def order_top(d: Dict[str, Any]) -> Dict[str, Any]:
        ordered: Dict[str, Any] = {}
        # top-level prioritized keys
        for k in TOP_LEVEL_ORDER:
            if k in d:
                ordered[k] = d[k]
        # remaining keys sorted
        for k in sorted(d.keys()):
            if k not in ordered:
                ordered[k] = d[k]
        return ordered

    def normalize(x: Any, top: bool = False) -> Any:
        if isinstance(x, dict):
            if top:
                d = order_top(x)
            else:
                d = {k: x[k] for k in sorted(x.keys())}
            return {k: normalize(v, top=False) for k, v in d.items()}
        if isinstance(x, list):
            return [normalize(v, top=False) for v in x]
        return x

    norm = normalize(obj, top=True)
    return json.dumps(norm, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -----------------------------
# META_FOLD stream + pinning
# -----------------------------


@dataclass(frozen=True)
class AttestationRecord:
    tick: int
    att_hash: str
    prev: Optional[str]
    policy_hash: str
    meta_hash: str
    abi_hash: str
    canon_bytes: bytes


class MetaFoldExecutor:
    """
    A deterministic attestation engine for one stream.
    """

    def __init__(self, stream_id: str, policy_hash: str, abi_hash: str):
        self.stream_id = stream_id
        self._pinned_policy = policy_hash
        self._pinned_abi = abi_hash
        self._head: Optional[str] = None
        self._records: List[AttestationRecord] = []

    @property
    def head(self) -> Optional[str]:
        return self._head

    @property
    def records(self) -> List[AttestationRecord]:
        return list(self._records)

    def _enforce_pinning(self, policy_hash: str, abi_hash: str) -> None:
        if policy_hash != self._pinned_policy:
            raise ValueError("policy_hash drift (forbidden in FEL v1.1 stream)")
        if abi_hash != self._pinned_abi:
            raise ValueError("abi_hash drift (forbidden in FEL v1.1 stream)")

    def attest(
        self,
        tick: int,
        meta_hash: str,
        prev: Optional[str] = None,
        *,
        v: str = "fel.v1.1",
    ) -> AttestationRecord:
        """
        Create and append an attestation line and compute its deterministic hash.
        Chain rule:
          - if head is None: prev must be None or "0"*64 (genesis)
          - else: prev must equal current head
        """
        if not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be int >= 0")

        # Pinning
        self._enforce_pinning(self._pinned_policy, self._pinned_abi)

        # Chain enforcement
        if self._head is None:
            if prev in ("0" * 64, None):
                prev_norm = None
            else:
                raise ValueError("genesis prev must be null or 64x0")
        else:
            if prev != self._head:
                raise ValueError("prev must equal current head")
            prev_norm = prev

        line = {
            "v": v,
            "tick": tick,
            "type": "attest",
            "fold": "⟁META_FOLD⟁",
            "policy_hash": self._pinned_policy,
            "meta_hash": meta_hash,
            "abi_hash": self._pinned_abi,
        }
        if prev_norm is not None:
            line["prev"] = prev_norm

        cb = canon_json(line)
        ah = sha256_hex(cb)

        rec = AttestationRecord(
            tick=tick,
            att_hash=ah,
            prev=prev_norm,
            policy_hash=self._pinned_policy,
            meta_hash=meta_hash,
            abi_hash=self._pinned_abi,
            canon_bytes=cb,
        )

        self._records.append(rec)
        self._head = ah
        return rec

    # -----------------------------
    # Recommended meta-hash builder
    # -----------------------------

    def compute_meta_hash(
        self,
        *,
        state_root_sha256: str,
        lane_roots: Optional[Dict[str, str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Domain-separated meta hash:
          meta_hash = sha256( canon({domain, stream_id, state_root, lane_roots, extra}) )
        """
        payload = {
            "domain": "fel.meta.v1.1",
            "stream_id": self.stream_id,
            "state_root_sha256": state_root_sha256,
            "lane_roots": lane_roots or {},
            "extra": extra or {},
        }
        return sha256_hex(canon_json(payload))
