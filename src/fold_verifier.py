#!/usr/bin/env python3
"""Deterministic verifier for Micronaut Fold Law frozen contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_FOLDS = {
    "⟁DATA_FOLD⟁",
    "⟁CODE_FOLD⟁",
    "⟁STORAGE_FOLD⟁",
    "⟁NETWORK_FOLD⟁",
    "⟁UI_FOLD⟁",
    "⟁AUTH_FOLD⟁",
    "⟁DB_FOLD⟁",
    "⟁COMPUTE_FOLD⟁",
    "⟁STATE_FOLD⟁",
    "⟁EVENTS_FOLD⟁",
    "⟁TIME_FOLD⟁",
    "⟁SPACE_FOLD⟁",
    "⟁META_FOLD⟁",
    "⟁CONTROL_FOLD⟁",
    "⟁PATTERN_FOLD⟁",
}

VALID_LANES = {"DICT", "FIELD", "LANE", "EDGE", "BATCH"}


class VerificationError(ValueError):
    """Raised when fold law verification fails."""


def canonical_hash(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_structure(data: dict[str, Any]) -> None:
    if data["system"]["version"] != "1.0.0":
        raise VerificationError("system.version must be 1.0.0")

    if data["system"]["law"] != "identity-preserving contraction":
        raise VerificationError(
            "system.law must be 'identity-preserving contraction'"
        )

    folds = set(data["folds"])
    if folds != REQUIRED_FOLDS:
        raise VerificationError("fold set does not match required 15 folds")

    lanes = data["lanes"]
    if set(lanes.keys()) != REQUIRED_FOLDS:
        raise VerificationError("lane keys must exactly equal required folds")

    for fold, lane in lanes.items():
        if lane not in VALID_LANES:
            raise VerificationError(f"invalid lane '{lane}' for fold '{fold}'")

    for idx, event in enumerate(data["events"]):
        if event["fold"] != "⟁EVENTS_FOLD⟁":
            raise VerificationError(
                f"event index {idx} must emit from ⟁EVENTS_FOLD⟁"
            )


def verify_idempotent_collapse(data: dict[str, Any]) -> str:
    h1 = canonical_hash(data)
    h2 = canonical_hash(data)
    if h1 != h2:
        raise VerificationError("collapse hash must be idempotent")
    return h1


def verify(data: dict[str, Any]) -> dict[str, str]:
    verify_structure(data)
    proof = verify_idempotent_collapse(data)
    return {"status": "valid", "proof_hash": proof}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Micronaut Fold Law frozen contract input."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="system.json",
        help="Path to input JSON (default: system.json)",
    )
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = verify(data)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
