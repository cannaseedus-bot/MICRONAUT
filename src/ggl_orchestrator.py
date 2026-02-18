"""
GGL Orchestrator
================
Wires the GGL spec code from docs/ggl/ into the fold system.

- Validates GGL programs using ggl_oracle_v1.validate_ggl()
- Routes valid programs through ComputeFold
- Extracts emitted nodes/edges from parsed GGL AST
- Provides SCXQ2 GGL frame packing (scxq2_ggl_packer integration)

Usage:
    from ggl_orchestrator import GGLOrchestrator
    from fold_orchestrator import FoldOrchestrator

    fo = FoldOrchestrator()
    ggl = GGLOrchestrator(fo)
    result = ggl.run_ggl("Make a cube.", "GGL.begin\\nnode Cube { size: 1; }\\nemit Cube;\\nGGL.end\\n")
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Load ggl_oracle_v1 from docs/ggl/ without modifying it
# ---------------------------------------------------------------------------

def _load_ggl_oracle():
    here = os.path.dirname(os.path.abspath(__file__))
    oracle_path = os.path.join(here, "..", "docs", "ggl", "ggl_oracle_v1.py")
    oracle_path = os.path.normpath(oracle_path)
    if not os.path.exists(oracle_path):
        raise ImportError(f"ggl_oracle_v1.py not found at {oracle_path}")
    spec = importlib.util.spec_from_file_location("ggl_oracle_v1", oracle_path)
    module = importlib.util.module_from_spec(spec)
    # Must register in sys.modules before exec_module so @dataclass resolves module namespace
    sys.modules["ggl_oracle_v1"] = module
    spec.loader.exec_module(module)
    return module


try:
    _oracle = _load_ggl_oracle()
    validate_ggl = _oracle.validate_ggl
    _ORACLE_AVAILABLE = True
except Exception as _e:
    _ORACLE_AVAILABLE = False
    validate_ggl = None


# ---------------------------------------------------------------------------
# GGL AST node/edge extractor
# ---------------------------------------------------------------------------

def _extract_emitted_nodes(ggl_src: str) -> List[str]:
    """Extract names of emitted nodes/graphs from GGL source.

    Looks for `emit <Name>;` patterns. Does not execute the GGL program;
    only performs structural extraction.
    """
    import re
    # Match `emit Name;` at statement level
    pattern = re.compile(r'\bemit\s+([A-Za-z_][A-Za-z0-9_]*)\s*;')
    return [m.group(1) for m in pattern.finditer(ggl_src)]


def _extract_node_definitions(ggl_src: str) -> List[Dict[str, Any]]:
    """Extract `node Name { ... }` definitions from GGL source."""
    import re
    pattern = re.compile(r'\bnode\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{([^}]*)\}', re.DOTALL)
    nodes = []
    for m in pattern.finditer(ggl_src):
        name = m.group(1)
        body = m.group(2).strip()
        props: Dict[str, str] = {}
        for line in body.splitlines():
            line = line.strip().rstrip(";")
            if ":" in line:
                k, _, v = line.partition(":")
                props[k.strip()] = v.strip()
        nodes.append({"name": name, "properties": props})
    return nodes


def _extract_edge_definitions(ggl_src: str) -> List[Dict[str, str]]:
    """Extract `edge A -> B { ... }` definitions from GGL source."""
    import re
    pattern = re.compile(
        r'\bedge\s+([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{([^}]*)\}',
        re.DOTALL,
    )
    edges = []
    for m in pattern.finditer(ggl_src):
        src, dst, body = m.group(1), m.group(2), m.group(3).strip()
        props: Dict[str, str] = {}
        for line in body.splitlines():
            line = line.strip().rstrip(";")
            if ":" in line:
                k, _, v = line.partition(":")
                props[k.strip()] = v.strip()
        edges.append({"src": src, "dst": dst, "properties": props})
    return edges


# ---------------------------------------------------------------------------
# Canonical helpers (mirrors fold_orchestrator)
# ---------------------------------------------------------------------------

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


# ---------------------------------------------------------------------------
# GGLOrchestrator
# ---------------------------------------------------------------------------

class GGLOrchestrator:
    """Routes GGL programs through the fold system via ComputeFold."""

    def __init__(self, fold_orchestrator: Any) -> None:
        self._fo = fold_orchestrator

    def run_ggl(self, prompt: str, ggl_code: str) -> Dict[str, Any]:
        """Validate a GGL program and route through ComputeFold.

        Steps:
        1. Validate using ggl_oracle_v1.validate_ggl()
        2. If invalid: return oracle errors, route to XM-1 (expansion) territory
        3. If valid: extract emitted nodes/edges; run through ComputeFold; return proof

        Returns:
            {
                "valid": bool,
                "prompt": str,
                "oracle": dict,           # full oracle result
                "emitted_nodes": list,    # names from `emit X;`
                "node_definitions": list, # {name, properties}
                "edge_definitions": list, # {src, dst, properties}
                "proof": dict | None,     # CollapseProof.to_dict() if valid
                "trace_hash": str | None,
            }
        """
        if not _ORACLE_AVAILABLE or validate_ggl is None:
            raise RuntimeError("GGL oracle not available — check docs/ggl/ggl_oracle_v1.py")

        oracle_result = validate_ggl(ggl_code)
        is_valid = oracle_result.get("valid", False)

        if not is_valid:
            return {
                "valid": False,
                "prompt": prompt,
                "oracle": oracle_result,
                "emitted_nodes": [],
                "node_definitions": [],
                "edge_definitions": [],
                "proof": None,
                "trace_hash": None,
            }

        emitted = _extract_emitted_nodes(ggl_code)
        nodes = _extract_node_definitions(ggl_code)
        edges = _extract_edge_definitions(ggl_code)

        # Route through ComputeFold via FoldOrchestrator
        from fold_orchestrator import FoldEvent, FoldType
        input_hash = _sha256_hex(_canonical_json({
            "prompt": prompt, "ggl_code": ggl_code,
        }).encode("utf-8"))

        payload = {
            "expression": ggl_code,
            "type": "ggl_program",
            "input_hash": input_hash,
            "emitted_nodes": emitted,
        }
        event = FoldEvent(
            event_id=input_hash[:16],
            fold=FoldType.EVENTS,
            data={
                "target_fold": FoldType.COMPUTE.value,
                "payload": payload,
                "expression": ggl_code,
                "action": "evaluate",
            },
            source_agent="GGLOrchestrator",
            requires_proof=True,
        )

        proof = None
        trace_hash = None
        try:
            proof_obj = self._fo.receive_event(event)
            if proof_obj is not None:
                proof = proof_obj.to_dict()
                # trace_hash is embedded in ComputeFold result
                trace_hash = _sha256_hex(_canonical_json({
                    "input_hash": input_hash, "result_hash": proof.get("result_hash"),
                }).encode("utf-8"))
        except Exception as exc:
            proof = {"error": str(exc)}

        return {
            "valid": True,
            "prompt": prompt,
            "oracle": oracle_result,
            "emitted_nodes": emitted,
            "node_definitions": nodes,
            "edge_definitions": edges,
            "proof": proof,
            "trace_hash": trace_hash,
        }

    # ------------------------------------------------------------------
    # Bootstrap vector validation

    def validate_bootstrap_vectors(self, bootstrap_path: str) -> Dict[str, Any]:
        """Run all GGL bootstrap vectors from *bootstrap_path* (JSONL).

        Returns a summary with pass/fail counts and any failures.
        """
        results = []
        with open(bootstrap_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vec = json.loads(line)
                oracle_result = validate_ggl(vec["target"])
                results.append({
                    "id": vec["id"],
                    "prompt": vec["prompt"],
                    "valid": oracle_result["valid"],
                    "errors": oracle_result.get("errors", []),
                    "fingerprint": oracle_result.get("fingerprint", ""),
                })

        passes = sum(1 for r in results if r["valid"])
        failures = [r for r in results if not r["valid"]]
        return {
            "total": len(results),
            "passed": passes,
            "failed": len(failures),
            "failures": failures,
            "all_passed": len(failures) == 0,
        }
