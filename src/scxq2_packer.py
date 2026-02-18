"""
SCXQ2 Full 5-Lane Packer
=========================
Implements the complete SCXQ2 binary container format defined in
docs/scxq2_binary_packing_example.md with all 5 lanes:

  DICT  (1) — symbol tables: DATA, AUTH, META, PATTERN folds
  FIELD (2) — typed scalars: STORAGE, DB, STATE folds
  LANE  (3) — ordered events: UI, EVENTS, TIME folds
  EDGE  (4) — relations: CODE, NETWORK, SPACE, CONTROL folds
  BATCH (5) — ephemeral compute: COMPUTE fold

This module is ADDITIVE — it does NOT replace verifier.py's pack_scx2(),
which remains the golden-pack conformance path.
This packer is used for fold orchestrator state export.

Usage:
    from scxq2_packer import SCXQ2Packer
    from fold_orchestrator import FoldOrchestrator, FoldType

    fo = FoldOrchestrator()
    packer = SCXQ2Packer()
    binary = packer.pack_from_fold_state(fo.folds)
    hash_ = hashlib.sha256(binary).hexdigest()
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Reference implementation (ported verbatim from spec)
# ---------------------------------------------------------------------------

LANE_ID = {"DICT": 1, "FIELD": 2, "LANE": 3, "EDGE": 4, "BATCH": 5}
LANE_BIT = {"DICT": 1, "FIELD": 2, "LANE": 4, "EDGE": 8, "BATCH": 16}


def _u16(n: int) -> bytes:
    return struct.pack("<H", n)


def _u32(n: int) -> bytes:
    return struct.pack("<I", n)


def pack_dict(items: Dict[str, str]) -> bytes:
    out = bytearray()
    for k, v in items.items():
        kb = k.encode("utf-8")
        vb = v.encode("utf-8")
        out += _u16(len(kb)) + kb + _u16(len(vb)) + vb
    return bytes(out)


def pack_field(records: List[Tuple[int, bytes]]) -> bytes:
    out = bytearray()
    for fid, blob in records:
        out += _u32(fid) + _u32(len(blob)) + blob
    return bytes(out)


def pack_lane(events: List[Tuple[int, int, bytes]]) -> bytes:
    out = bytearray()
    for tick, kind, blob in events:
        out += _u32(tick) + _u16(kind) + _u32(len(blob)) + blob
    return bytes(out)


def pack_edge(edges: List[Tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for src, dst, kind in edges:
        out += _u32(src) + _u32(dst) + _u16(kind)
    return bytes(out)


def pack_batch(jobs: List[Tuple[int, bytes]]) -> bytes:
    out = bytearray()
    for job_id, blob in jobs:
        out += _u32(job_id) + _u32(len(blob)) + blob
    return bytes(out)


def scx2_pack(lanes: Dict[str, bytes], add_crc: bool = True) -> bytes:
    """Pack lanes into the SCX2 binary container format (verbatim from spec)."""
    magic = b"SCX2"
    ver = b"\x01"
    bitmask = 0
    for k in lanes:
        bitmask |= LANE_BIT[k]
    header = bytearray(magic + ver + bytes([bitmask]) + b"\x00\x00")

    body = bytearray()
    for name in ["DICT", "FIELD", "LANE", "EDGE", "BATCH"]:
        if name not in lanes:
            continue
        payload = lanes[name]
        tag = LANE_ID[name]
        flags = 0
        count = 0
        body += struct.pack("<BBII", tag, flags, count, len(payload))
        body += payload

    data = bytes(header + body)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return data + (_u32(crc) if add_crc else _u32(0))


# ---------------------------------------------------------------------------
# Fold→Lane routing table
# ---------------------------------------------------------------------------

# Maps fold type value strings to their SCXQ2 lane name
_FOLD_TO_LANE_NAME: Dict[str, str] = {
    "⟁DATA_FOLD⟁":    "DICT",
    "⟁AUTH_FOLD⟁":    "DICT",
    "⟁META_FOLD⟁":    "DICT",
    "⟁PATTERN_FOLD⟁": "DICT",
    "⟁STORAGE_FOLD⟁": "FIELD",
    "⟁DB_FOLD⟁":      "FIELD",
    "⟁STATE_FOLD⟁":   "FIELD",
    "⟁UI_FOLD⟁":      "LANE",
    "⟁EVENTS_FOLD⟁":  "LANE",
    "⟁TIME_FOLD⟁":    "LANE",
    "⟁CODE_FOLD⟁":    "EDGE",
    "⟁NETWORK_FOLD⟁": "EDGE",
    "⟁SPACE_FOLD⟁":   "EDGE",
    "⟁CONTROL_FOLD⟁": "EDGE",
    "⟁COMPUTE_FOLD⟁": "BATCH",
}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# SCXQ2Packer
# ---------------------------------------------------------------------------

class SCXQ2Packer:
    """Pack the full FoldOrchestrator state into a 5-lane SCX2 binary."""

    def pack_from_fold_state(self, folds: Any, tick: int = 0) -> bytes:
        """Build SCX2 binary from the current state of all folds.

        Args:
            folds: dict of FoldType → FoldInterface instances
                   (the FoldOrchestrator.folds dict).
            tick:  optional tick counter for LANE records.

        Returns:
            Raw bytes of the packed SCX2 container.
        """
        dict_items: Dict[str, str] = {}
        field_records: List[Tuple[int, bytes]] = []
        lane_events: List[Tuple[int, int, bytes]] = []
        edge_edges: List[Tuple[int, int, int]] = []
        batch_jobs: List[Tuple[int, bytes]] = []

        field_id_counter = 0
        batch_job_counter = 0

        for fold_type, fold_obj in folds.items():
            fold_name = fold_type.value  # e.g. "⟁DATA_FOLD⟁"
            lane_name = _FOLD_TO_LANE_NAME.get(fold_name, "LANE")
            state = fold_obj.state
            state_blob = _canonical_json(state).encode("utf-8")
            state_hash = _sha256_hex(state_blob)

            if lane_name == "DICT":
                dict_items[fold_name] = state_hash
                for k, v in list(state.items())[:8]:
                    str_v = v if isinstance(v, str) else _canonical_json(v)
                    dict_items[f"{fold_name}.{k}"] = str_v[:255]

            elif lane_name == "FIELD":
                field_records.append((field_id_counter, state_blob))
                field_id_counter += 1

            elif lane_name == "LANE":
                kind = abs(hash(fold_name)) % 65536
                lane_events.append((tick, kind, state_blob))

            elif lane_name == "EDGE":
                src_id = abs(hash(fold_name)) % (2**32)
                dst_id = abs(hash(state_hash)) % (2**32)
                kind = abs(hash(fold_name + "edge")) % 65536
                edge_edges.append((src_id, dst_id, kind))

            elif lane_name == "BATCH":
                batch_jobs.append((batch_job_counter, state_blob))
                batch_job_counter += 1

        lanes: Dict[str, bytes] = {}
        if dict_items:
            lanes["DICT"] = pack_dict(dict_items)
        if field_records:
            lanes["FIELD"] = pack_field(field_records)
        if lane_events:
            lanes["LANE"] = pack_lane(lane_events)
        if edge_edges:
            lanes["EDGE"] = pack_edge(edge_edges)
        if batch_jobs:
            lanes["BATCH"] = pack_batch(batch_jobs)

        return scx2_pack(lanes, add_crc=True)
