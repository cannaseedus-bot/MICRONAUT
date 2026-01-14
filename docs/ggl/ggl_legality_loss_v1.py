from __future__ import annotations

from typing import Any, Dict


def legality_penalty(oracle_result: Dict[str, Any], lam: float = 0.25) -> float:
    return 0.0 if oracle_result.get("valid") else float(lam)


def oracle_weighted_penalty(oracle_result: Dict[str, Any], lam: float = 0.25) -> float:
    if oracle_result.get("valid"):
        return 0.0
    n = len(oracle_result.get("errors", []))
    return float(lam) * (1.0 + min(4.0, n / 4.0))
