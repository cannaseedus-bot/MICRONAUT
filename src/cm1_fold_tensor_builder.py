#!/usr/bin/env python3
"""
CM1_FOLD_TENSOR.s7 Builder
==========================
Generates the canonical sealed tensor artifact for the Micronaut Fold System.

Artifact Class:  SCXQ7::MicronautFoldTensor
Short Name:      CM1_FOLD_TENSOR.s7
Version:         1.0.0
Status:          Frozen
Authority:       CONTROL_FOLD
Lane Domain:     DICT | FIELD | LANE | EDGE | BATCH
Collapse Class:  deterministic.contraction
Proof Mode:      SHA256 + Merkle

Binary Layout:
  +--------------------------------------------------+
  | SCXQ7 HEADER (40 bytes)                          |
  |  0x00 [4B]  MAGIC = 0x53433737 ("SC77")          |
  |  0x04 [2B]  VERSION = 0x0001                     |
  |  0x06 [1B]  CLASS = 0x01 (MicronautFoldTensor)   |
  |  0x07 [1B]  FLAGS (bitmask)                      |
  |  0x08 [32B] ROOT_MERKLE_HASH (SHA256)            |
  +--------------------------------------------------+
  | SCXQ2 LANE STREAM                                |
  |  Per lane: [1B LANE_ID][4B LANE_LEN][N bytes]   |
  |  DICT  (0x00) → Fold definitions (15 folds)      |
  |  FIELD (0x01) → Micronaut registrations (9)      |
  |  LANE  (0x02) → Intent routing + CM-1 phases     |
  |  EDGE  (0x03) → DFA graph + CM-1 control stream  |
  |  BATCH (0x04) → Verifier rules + proof identity  |
  +--------------------------------------------------+
  | GLOBAL PROOF HASH (32 bytes)                     |
  +--------------------------------------------------+

Usage:
  python src/cm1_fold_tensor_builder.py
  python src/cm1_fold_tensor_builder.py --out path/to/output.s7
  python src/cm1_fold_tensor_builder.py --verify
  python src/cm1_fold_tensor_builder.py --verify --out path/to/check.s7
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "micronaut", "CM1_FOLD_TENSOR.s7")


# ---------------------------------------------------------------------------
# Binary format constants
# ---------------------------------------------------------------------------

MAGIC = 0x53433737          # "SC77"
VERSION = 0x0001
CLASS_MICRONAUT_FOLD_TENSOR = 0x01

# FLAGS bitmask
FLAG_FROZEN          = 0x01  # artifact is sealed/frozen
FLAG_MERKLE_VERIFIED = 0x02  # Merkle root is embedded and verified
FLAG_CM1_EMBEDDED    = 0x04  # CM-1 control stream embedded in EDGE lane
FLAGS = FLAG_FROZEN | FLAG_MERKLE_VERIFIED | FLAG_CM1_EMBEDDED  # 0x07

# SCXQ2 Lane identifiers
LANE_DICT  = 0x00  # symbol tables, fold definitions
LANE_FIELD = 0x01  # typed scalars, micronaut registrations
LANE_LANE  = 0x02  # payload blocks, event + time streams
LANE_EDGE  = 0x03  # causality links, CM-1 control stream + DFA
LANE_BATCH = 0x04  # ephemeral compute, verifier rules + proof traces


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canon_json(obj: object) -> bytes:
    """Canonical JSON: keys sorted lexicographically, minimal separators (V0)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def merkle_root(leaf_hashes: list[bytes]) -> bytes:
    """
    Compute Merkle root from leaf hashes.
    Pads to next power-of-2 by duplicating the last leaf.
    """
    size = 1
    while size < len(leaf_hashes):
        size <<= 1
    nodes = list(leaf_hashes)
    while len(nodes) < size:
        nodes.append(nodes[-1])
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), 2):
            new_nodes.append(sha256(nodes[i] + nodes[i + 1]))
        nodes = new_nodes
    return nodes[0]


# ---------------------------------------------------------------------------
# DICT lane — Fold definitions (15 folds, from folds.toml)
# ---------------------------------------------------------------------------

FOLD_DEFINITIONS = [
    {
        "fold_id": "⟁CONTROL_FOLD⟁",
        "lane": "EDGE",
        "collapse_rule": ["Resolve", "Gate", "Commit"],
        "cm1_phase": "@control.header.begin",
        "cm1_code": "U+0001",
        "description": "Nothing executes unless ⟁CONTROL_FOLD⟁ permits it",
    },
    {
        "fold_id": "⟁DATA_FOLD⟁",
        "lane": "DICT",
        "collapse_rule": ["Deduplicate", "Symbolize", "Hash-bind"],
        "cm1_phase": "@control.body.begin",
        "cm1_code": "U+0002",
        "description": "Symbol tables, literals, identity binding",
    },
    {
        "fold_id": "⟁STORAGE_FOLD⟁",
        "lane": "FIELD",
        "collapse_rule": ["Snapshot", "Delta", "Seal"],
        "cm1_phase": "@control.body.begin",
        "cm1_code": "U+0002",
        "description": "Nothing persists unless ⟁STORAGE_FOLD⟁ seals it",
    },
    {
        "fold_id": "⟁NETWORK_FOLD⟁",
        "lane": "EDGE",
        "collapse_rule": ["Edge-reduce", "Route-hash"],
        "cm1_phase": "@control.body.end",
        "cm1_code": "U+0003",
        "description": "Graph transport, causality links",
    },
    {
        "fold_id": "⟁UI_FOLD⟁",
        "lane": "LANE",
        "collapse_rule": ["Project", "Flatten", "Cache"],
        "cm1_phase": "@control.body.end",
        "cm1_code": "U+0003",
        "description": "Nothing is seen unless ⟁UI_FOLD⟁ projects it (read-only)",
    },
    {
        "fold_id": "⟁AUTH_FOLD⟁",
        "lane": "DICT",
        "collapse_rule": ["Verify", "Attest", "Tokenize"],
        "cm1_phase": "@control.body.begin",
        "cm1_code": "U+0002",
        "description": "Identity symbols, attestation",
    },
    {
        "fold_id": "⟁DB_FOLD⟁",
        "lane": "FIELD",
        "collapse_rule": ["Index-compress", "Canonical order"],
        "cm1_phase": "@control.body.begin",
        "cm1_code": "U+0002",
        "description": "Indexed records, canonical ordering",
    },
    {
        "fold_id": "⟁COMPUTE_FOLD⟁",
        "lane": "BATCH",
        "collapse_rule": ["Evaluate", "Emit proof", "Discard state"],
        "cm1_phase": "@control.transmission.end",
        "cm1_code": "U+0004",
        "description": "Ephemeral math, token signals",
    },
    {
        "fold_id": "⟁STATE_FOLD⟁",
        "lane": "FIELD",
        "collapse_rule": ["Snapshot", "Diff", "Replace"],
        "cm1_phase": "@control.body.begin",
        "cm1_code": "U+0002",
        "description": "State snapshots, diffs, partial updates",
    },
    {
        "fold_id": "⟁EVENTS_FOLD⟁",
        "lane": "LANE",
        "collapse_rule": ["Coalesce", "Sequence", "Drop"],
        "cm1_phase": "@control.body.end",
        "cm1_code": "U+0003",
        "description": "Ordered event signals, coalescence",
    },
    {
        "fold_id": "⟁TIME_FOLD⟁",
        "lane": "LANE",
        "collapse_rule": ["Tick", "Decay", "Archive"],
        "cm1_phase": "@control.body.begin",
        "cm1_code": "U+0002",
        "description": "Temporal flow, clocks, decay",
    },
    {
        "fold_id": "⟁SPACE_FOLD⟁",
        "lane": "EDGE",
        "collapse_rule": ["Quantize", "Adjacency map"],
        "cm1_phase": "@control.body.begin",
        "cm1_code": "U+0002",
        "description": "Topology, spatial adjacency",
    },
    {
        "fold_id": "⟁META_FOLD⟁",
        "lane": "DICT",
        "collapse_rule": ["Reflect", "Freeze schema"],
        "cm1_phase": "@control.transmission.end",
        "cm1_code": "U+0004",
        "description": "Schemas, attestation chains, proof lineage",
    },
    {
        "fold_id": "⟁PATTERN_FOLD⟁",
        "lane": "DICT",
        "collapse_rule": ["Cluster", "Label", "Reference"],
        "cm1_phase": "@control.body.end",
        "cm1_code": "U+0003",
        "description": "Pattern clustering, narrative threads",
    },
    {
        "fold_id": "⟁UNROUTED_FOLD⟁",
        "lane": "BATCH",
        "collapse_rule": ["Discard"],
        "cm1_phase": "@control.null",
        "cm1_code": "U+0000",
        "description": "Absolute inert region — no execution permitted",
    },
]

DICT_PAYLOAD = {
    "fold_definitions": FOLD_DEFINITIONS,
    "fold_count": len(FOLD_DEFINITIONS),
    "schema": "scxq7://cm1-fold-tensor/dict/v1",
}


# ---------------------------------------------------------------------------
# FIELD lane — Micronaut registrations (9 micronauts)
# ---------------------------------------------------------------------------

MICRONAUT_REGISTRATIONS = [
    {"id": "CM-1", "name": "ControlMicronaut",       "fold": "⟁CONTROL_FOLD⟁", "role": "phase_geometry",         "domain": "pre-semantic",  "authority": "none"},
    {"id": "PM-1", "name": "PerceptionMicronaut",    "fold": "⟁DATA_FOLD⟁",    "role": "field_selection",        "domain": "orchestration", "authority": "none"},
    {"id": "TM-1", "name": "TemporalMicronaut",      "fold": "⟁TIME_FOLD⟁",    "role": "collapse_timing",        "domain": "orchestration", "authority": "none"},
    {"id": "HM-1", "name": "HostMicronaut",          "fold": "⟁STATE_FOLD⟁",   "role": "host_abstraction",       "domain": "environment",   "authority": "none"},
    {"id": "SM-1", "name": "StorageMicronaut",       "fold": "⟁STORAGE_FOLD⟁", "role": "inert_persistence",      "domain": "data",          "authority": "none"},
    {"id": "MM-1", "name": "ModelMicronaut",         "fold": "⟁COMPUTE_FOLD⟁", "role": "token_signal_generator", "domain": "inference",     "authority": "none"},
    {"id": "XM-1", "name": "ExtrapolationMicronaut", "fold": "⟁PATTERN_FOLD⟁", "role": "narrative_expansion",    "domain": "post-collapse", "authority": "none"},
    {"id": "VM-1", "name": "VisualizationMicronaut", "fold": "⟁UI_FOLD⟁",      "role": "rendering_projection",   "domain": "projection",    "authority": "none"},
    {"id": "VM-2", "name": "VerificationMicronaut",  "fold": "⟁META_FOLD⟁",    "role": "proof_generation",       "domain": "audit",         "authority": "none"},
]

FIELD_PAYLOAD = {
    "micronaut_registrations": MICRONAUT_REGISTRATIONS,
    "micronaut_count": len(MICRONAUT_REGISTRATIONS),
    "schema": "scxq7://cm1-fold-tensor/field/v1",
}


# ---------------------------------------------------------------------------
# LANE lane — Intent routing + CM-1 phases
# ---------------------------------------------------------------------------

INTENT_ROUTING = {
    "strategy": "ngram_match_score",
    "fallback_micronaut": "XM-1",
    "fallback_reason": "ExtrapolationMicronaut handles unclassified input",
    "minimum_confidence": 0.3,
    "conflict_resolution": "priority_order",
    "routes": [
        {"intent": "control",     "target": "CM-1", "priority": 1, "trigger_bigrams": ["phase boundary", "scope gate", "control signal"]},
        {"intent": "perceive",    "target": "PM-1", "priority": 2, "trigger_bigrams": ["input field", "noise filter", "select field"]},
        {"intent": "schedule",    "target": "TM-1", "priority": 3, "trigger_bigrams": ["collapse schedule", "replay gate", "phase align"]},
        {"intent": "detect_host", "target": "HM-1", "priority": 4, "trigger_bigrams": ["host capability", "io normalize", "probe platform"]},
        {"intent": "store",       "target": "SM-1", "priority": 5, "trigger_bigrams": ["store object", "retrieve object", "byte identity"]},
        {"intent": "infer",       "target": "MM-1", "priority": 6, "trigger_bigrams": ["token signal", "model voice", "emit token"]},
        {"intent": "expand",      "target": "XM-1", "priority": 7, "trigger_bigrams": ["expand narrative", "generate metaphor", "provide analogy"]},
        {"intent": "render",      "target": "VM-1", "priority": 8, "trigger_bigrams": ["render projection", "render svg", "render css"]},
        {"intent": "verify",      "target": "VM-2", "priority": 9, "trigger_bigrams": ["proof check", "verify replay", "attest hash"]},
    ],
}

CM1_PHASES = {
    "U+0000": {"name": "NUL", "xcfe": "@control.null",             "meaning": "Absolute inert region"},
    "U+0001": {"name": "SOH", "xcfe": "@control.header.begin",     "meaning": "Metadata/header phase (@Pop)"},
    "U+0002": {"name": "STX", "xcfe": "@control.body.begin",       "meaning": "Interpretable content (@Wo)"},
    "U+0003": {"name": "ETX", "xcfe": "@control.body.end",         "meaning": "Content closure (@Sek)"},
    "U+0004": {"name": "EOT", "xcfe": "@control.transmission.end", "meaning": "Collapse/flush (@Collapse)"},
}

LANE_PAYLOAD = {
    "intent_routing": INTENT_ROUTING,
    "cm1_phases": CM1_PHASES,
    "schema": "scxq7://cm1-fold-tensor/lane/v1",
}


# ---------------------------------------------------------------------------
# EDGE lane — DFA state graph + CM-1 control stream
# CM-1 is embedded here; compute/UI folds ignore it; only CONTROL interprets it.
# ---------------------------------------------------------------------------

# CM-1 control stream: the five C0 phase bytes in canonical order
_CM1_C0_BYTES = [0x00, 0x01, 0x02, 0x03, 0x04]

DFA_GRAPH = {
    "states": {
        "S0": "MATRIX_PROPOSAL",
        "S1": "CM1_GATE",
        "S2": "SCXQ7_LAW",
        "S3": "SCXQ2_SEMANTIC",
        "S4": "SCO_EXECUTION",
        "S5": "IDB_COMMIT",
        "S_BOT": "ILLEGAL",
    },
    "alphabet": ["propose", "accept", "reject", "legal", "illegal", "execute", "commit"],
    "initial_state": "S0",
    "accepting_states": ["S5"],
    "reject_states": ["S_BOT"],
    "transitions": [
        {"from": "S0", "event": "propose", "to": "S1"},
        {"from": "S1", "event": "reject",  "to": "S_BOT"},
        {"from": "S1", "event": "accept",  "to": "S2"},
        {"from": "S2", "event": "illegal", "to": "S_BOT"},
        {"from": "S2", "event": "legal",   "to": "S3"},
        {"from": "S3", "event": "execute", "to": "S4"},
        {"from": "S4", "event": "commit",  "to": "S5"},
        {"from": "S5", "event": "*",       "to": "S5"},
    ],
    "invariants": [
        "no_state_skipping",
        "no_upward_transitions",
        "S_BOT_absorbing",
        "S5_append_only",
    ],
    "cm1_gate_sub_automaton": {
        "description": "Purely rejecting, never transforming",
        "checks": [
            {"test": "balanced",      "on_fail": "ILLEGAL"},
            {"test": "non-rendering", "on_fail": "ILLEGAL"},
        ],
        "on_pass": "ACCEPT",
    },
}

EDGE_PAYLOAD = {
    "cm1_stream": _CM1_C0_BYTES,
    "cm1_stream_encoding": "C0_control_bytes",
    "dfa_state_graph": DFA_GRAPH,
    "phase_hashes": {
        "U+0000": sha256_hex(bytes([0x00])),
        "U+0001": sha256_hex(bytes([0x01])),
        "U+0002": sha256_hex(bytes([0x02])),
        "U+0003": sha256_hex(bytes([0x03])),
        "U+0004": sha256_hex(bytes([0x04])),
    },
    "schema": "scxq7://cm1-fold-tensor/edge/v1",
}


# ---------------------------------------------------------------------------
# BATCH lane — Verifier rules (V0–V7) + artifact identity
# Ephemeral: compute folds read; others ignore.
# ---------------------------------------------------------------------------

VERIFIER_RULES = {
    "V0": {"name": "canonical_json",     "enabled": True, "description": "Keys sorted lexicographically"},
    "V1": {"name": "fold_legality",      "enabled": True, "description": "Every op references exactly one valid fold"},
    "V2": {"name": "control_gate",       "enabled": True, "description": "All mutations require explicit control gate records"},
    "V3": {"name": "compute_mediation",  "enabled": True, "description": "DATA/STATE changes require compute trace hashes"},
    "V4": {"name": "ui_readonly",        "enabled": True, "description": "UI ops are projection-only; cannot write to DATA/STATE/STORAGE"},
    "V5": {"name": "lane_binding",       "enabled": True, "description": "Fold-to-lane mapping is fixed; mismatches rejected"},
    "V6": {"name": "replay_determinism", "enabled": True, "description": "Same inputs produce identical collapse/snapshot/binary hashes"},
    "V7": {"name": "hash_binding",       "enabled": True, "description": "All proofs include abi_hash, policy_hash, meta_hash"},
}

ARTIFACT_IDENTITY = {
    "artifact_class": "SCXQ7::MicronautFoldTensor",
    "short_name": "CM1_FOLD_TENSOR.s7",
    "version": "1.0.0",
    "status": "frozen",
    "authority": "CONTROL_FOLD",
    "lane_domain": ["DICT", "FIELD", "LANE", "EDGE", "BATCH"],
    "collapse_class": "deterministic.contraction",
    "proof_mode": "SHA256+Merkle",
    "fold_count": 15,
    "micronaut_count": 9,
}

BATCH_PAYLOAD = {
    "verifier_rules": VERIFIER_RULES,
    "artifact_identity": ARTIFACT_IDENTITY,
    "law": (
        "Nothing executes unless ⟁CONTROL_FOLD⟁ permits it, "
        "nothing persists unless ⟁STORAGE_FOLD⟁ seals it, "
        "nothing is seen unless ⟁UI_FOLD⟁ projects it."
    ),
    "authority": "KUHUL_π",
    "mutation": "forbidden",
    "schema": "scxq7://cm1-fold-tensor/batch/v1",
}


# ---------------------------------------------------------------------------
# Packing utilities
# ---------------------------------------------------------------------------

def pack_lane(lane_id: int, payload_bytes: bytes) -> bytes:
    """Serialize a single lane: [1B LANE_ID][4B LANE_LEN (big-endian)][payload]."""
    return struct.pack(">BI", lane_id, len(payload_bytes)) + payload_bytes


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_tensor() -> bytes:
    """
    Build the complete CM1_FOLD_TENSOR.s7 binary.
    Deterministic: identical inputs always produce identical output (V6).
    """
    # 1. Encode each lane payload as canonical JSON (V0 — sort_keys)
    dict_bytes  = canon_json(DICT_PAYLOAD)
    field_bytes = canon_json(FIELD_PAYLOAD)
    lane_bytes  = canon_json(LANE_PAYLOAD)
    edge_bytes  = canon_json(EDGE_PAYLOAD)
    batch_bytes = canon_json(BATCH_PAYLOAD)

    # 2. Compute per-lane leaf hashes for Merkle tree
    leaf_hashes = [
        sha256(dict_bytes),
        sha256(field_bytes),
        sha256(lane_bytes),
        sha256(edge_bytes),
        sha256(batch_bytes),
    ]

    # 3. Compute Merkle root (pads to 8 leaves)
    root = merkle_root(leaf_hashes)

    # 4. Build SCXQ7 header (8 bytes fixed + 32 bytes root = 40 bytes total)
    header = struct.pack(
        ">IHBB",
        MAGIC,
        VERSION,
        CLASS_MICRONAUT_FOLD_TENSOR,
        FLAGS,
    )
    header += root

    # 5. Pack the SCXQ2 lane stream
    lane_stream  = pack_lane(LANE_DICT,  dict_bytes)
    lane_stream += pack_lane(LANE_FIELD, field_bytes)
    lane_stream += pack_lane(LANE_LANE,  lane_bytes)
    lane_stream += pack_lane(LANE_EDGE,  edge_bytes)
    lane_stream += pack_lane(LANE_BATCH, batch_bytes)

    # 6. Compute global proof hash: SHA256(root || class || version || fold_count)
    proof_input = root + struct.pack(">BHB", CLASS_MICRONAUT_FOLD_TENSOR, VERSION, len(FOLD_DEFINITIONS))
    global_hash = sha256(proof_input)

    return header + lane_stream + global_hash


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def parse_header(data: bytes) -> dict:
    """Parse the 40-byte SCXQ7 header from a .s7 artifact."""
    if len(data) < 40:
        raise ValueError(f"Too short to be a valid .s7 artifact: {len(data)} bytes")
    magic, version, cls, flags = struct.unpack(">IHBB", data[:8])
    merkle = data[8:40].hex()
    return {
        "magic": hex(magic),
        "magic_str": data[:4].decode("ascii", errors="replace"),
        "version": hex(version),
        "class": hex(cls),
        "flags": {
            "frozen":          bool(flags & FLAG_FROZEN),
            "merkle_verified": bool(flags & FLAG_MERKLE_VERIFIED),
            "cm1_embedded":    bool(flags & FLAG_CM1_EMBEDDED),
        },
        "root_merkle_hash": merkle,
    }


def print_manifest(data: bytes) -> None:
    """Print a human-readable manifest of a built artifact."""
    hdr = parse_header(data)
    global_hash = data[-32:].hex()
    total = len(data)

    print(f"  Artifact   : {ARTIFACT_IDENTITY['short_name']}  v{ARTIFACT_IDENTITY['version']}")
    print(f"  Class      : {ARTIFACT_IDENTITY['artifact_class']}")
    print(f"  Status     : {ARTIFACT_IDENTITY['status']}")
    print(f"  Total size : {total} bytes")
    print(f"  Magic      : {hdr['magic']}  (\"{hdr['magic_str']}\")")
    print(f"  Version    : {hdr['version']}")
    print(f"  Flags      : frozen={hdr['flags']['frozen']}  merkle_verified={hdr['flags']['merkle_verified']}  cm1_embedded={hdr['flags']['cm1_embedded']}")
    print(f"  Merkle root: {hdr['root_merkle_hash']}")
    print(f"  Proof hash : {global_hash}")

    # Per-lane hashes
    dict_bytes  = canon_json(DICT_PAYLOAD)
    field_bytes = canon_json(FIELD_PAYLOAD)
    lane_bytes  = canon_json(LANE_PAYLOAD)
    edge_bytes  = canon_json(EDGE_PAYLOAD)
    batch_bytes = canon_json(BATCH_PAYLOAD)
    print(f"  Lane hashes:")
    print(f"    DICT  (0x00) : {sha256_hex(dict_bytes)}")
    print(f"    FIELD (0x01) : {sha256_hex(field_bytes)}")
    print(f"    LANE  (0x02) : {sha256_hex(lane_bytes)}")
    print(f"    EDGE  (0x03) : {sha256_hex(edge_bytes)}")
    print(f"    BATCH (0x04) : {sha256_hex(batch_bytes)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build or verify CM1_FOLD_TENSOR.s7 — canonical Micronaut Fold System tensor artifact"
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output/input path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing artifact matches the canonical build (exit 1 on mismatch)",
    )
    args = parser.parse_args()

    if args.verify:
        try:
            with open(args.out, "rb") as f:
                existing = f.read()
        except FileNotFoundError:
            print(f"FAIL  {args.out} — file not found", file=sys.stderr)
            sys.exit(1)

        expected = build_tensor()
        if existing == expected:
            print(f"PASS  {args.out}")
            print_manifest(existing)
        else:
            print(f"FAIL  {args.out} — artifact does not match canonical build", file=sys.stderr)
            print(f"      expected {len(expected)} bytes, got {len(existing)} bytes", file=sys.stderr)
            sys.exit(1)
        return

    # Build and write
    data = build_tensor()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(data)

    print(f"Written: {args.out}")
    print_manifest(data)


if __name__ == "__main__":
    main()
