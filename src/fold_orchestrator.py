"""
⟁ FOLD ORCHESTRATOR ⟁
A headless governor that enforces fold collapse laws.
Nothing executes unless ⟁CONTROL_FOLD⟁ permits it,
nothing persists unless ⟁STORAGE_FOLD⟁ seals it,
nothing is seen unless ⟁UI_FOLD⟁ projects it.
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from abc import ABC, abstractmethod

# CM-1 parser is imported lazily to avoid circular deps; used in ControlFold
try:
    from cm1_parser import CM1Parser, project_cm1_to_css
    _CM1_AVAILABLE = True
except ImportError:
    _CM1_AVAILABLE = False


class FoldType(Enum):
    """All 15 fold types."""

    DATA = "⟁DATA_FOLD⟁"
    CODE = "⟁CODE_FOLD⟁"
    STORAGE = "⟁STORAGE_FOLD⟁"
    NETWORK = "⟁NETWORK_FOLD⟁"
    UI = "⟁UI_FOLD⟁"
    AUTH = "⟁AUTH_FOLD⟁"
    DB = "⟁DB_FOLD⟁"
    COMPUTE = "⟁COMPUTE_FOLD⟁"
    STATE = "⟁STATE_FOLD⟁"
    EVENTS = "⟁EVENTS_FOLD⟁"
    TIME = "⟁TIME_FOLD⟁"
    SPACE = "⟁SPACE_FOLD⟁"
    META = "⟁META_FOLD⟁"
    CONTROL = "⟁CONTROL_FOLD⟁"
    PATTERN = "⟁PATTERN_FOLD⟁"


class SCXQ2Lane(Enum):
    DICT = "DICT"
    EDGE = "EDGE"
    FIELD = "FIELD"
    LANE = "LANE"
    BATCH = "BATCH"


COLLAPSE_RULES = {
    FoldType.DATA: ["Deduplicate", "Symbolize", "Hash-bind"],
    FoldType.CODE: ["Inline", "Normalize", "Freeze"],
    FoldType.STORAGE: ["Snapshot", "Delta", "Seal"],
    FoldType.NETWORK: ["Edge-reduce", "Route-hash"],
    FoldType.UI: ["Project", "Flatten", "Cache"],
    FoldType.AUTH: ["Verify", "Attest", "Tokenize"],
    FoldType.DB: ["Index-compress", "Canonical order"],
    FoldType.COMPUTE: ["Evaluate", "Emit proof", "Discard state"],
    FoldType.STATE: ["Snapshot", "Diff", "Replace"],
    FoldType.EVENTS: ["Coalesce", "Sequence", "Drop"],
    FoldType.TIME: ["Tick", "Decay", "Archive"],
    FoldType.SPACE: ["Quantize", "Adjacency map"],
    FoldType.META: ["Reflect", "Freeze schema"],
    FoldType.CONTROL: ["Resolve", "Gate", "Commit"],
    FoldType.PATTERN: ["Cluster", "Label", "Reference"],
}


FOLD_TO_LANE = {
    FoldType.DATA: SCXQ2Lane.DICT,
    FoldType.CODE: SCXQ2Lane.EDGE,
    FoldType.STORAGE: SCXQ2Lane.FIELD,
    FoldType.NETWORK: SCXQ2Lane.EDGE,
    FoldType.UI: SCXQ2Lane.LANE,
    FoldType.AUTH: SCXQ2Lane.DICT,
    FoldType.DB: SCXQ2Lane.FIELD,
    FoldType.COMPUTE: SCXQ2Lane.BATCH,
    FoldType.STATE: SCXQ2Lane.FIELD,
    FoldType.EVENTS: SCXQ2Lane.LANE,
    FoldType.TIME: SCXQ2Lane.LANE,
    FoldType.SPACE: SCXQ2Lane.EDGE,
    FoldType.META: SCXQ2Lane.DICT,
    FoldType.CONTROL: SCXQ2Lane.EDGE,
    FoldType.PATTERN: SCXQ2Lane.DICT,
}


SVG_MAPPINGS = {
    FoldType.DATA: "<defs>",
    FoldType.CODE: "<path>",
    FoldType.STORAGE: "<g data-snapshot>",
    FoldType.NETWORK: "<line>/<edge>",
    FoldType.UI: "<foreignObject>",
    FoldType.AUTH: "<shield>",
    FoldType.DB: "<grid>",
    FoldType.COMPUTE: "<animate>",
    FoldType.STATE: "<use>",
    FoldType.EVENTS: "<marker>",
    FoldType.TIME: "<timeline>",
    FoldType.SPACE: "<viewBox>",
    FoldType.META: "<metadata>",
    FoldType.CONTROL: "<mask>",
    FoldType.PATTERN: "<pattern>",
}


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass
class FoldEvent:
    """Event flowing through folds."""

    event_id: str
    fold: FoldType
    data: Dict[str, Any]
    source_agent: Optional[str] = None
    requires_proof: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "fold": self.fold.value,
            "data": self.data,
            "source_agent": self.source_agent,
            "requires_proof": self.requires_proof,
        }


@dataclass
class CollapseProof:
    """Proof of legal fold collapse."""

    fold: FoldType
    result_hash: str
    lane: SCXQ2Lane
    steps_applied: List[str]
    prev_hash: Optional[str] = None
    svg_projection: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold.value,
            "result_hash": self.result_hash,
            "lane": self.lane.value,
            "steps_applied": self.steps_applied,
            "prev_hash": self.prev_hash,
            "svg_projection": self.svg_projection,
        }

    def proof_hash(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()).encode("utf-8"))


@dataclass
class AgentContract:
    """Fold-scoped agent permissions."""

    agent_id: str
    allowed_folds: Set[FoldType]
    read_only_folds: Set[FoldType]
    emit_folds: Set[FoldType]
    forbidden_folds: Set[FoldType]
    proof_required: bool = True

    def can_access(self, fold: FoldType, mode: str = "read") -> bool:
        if fold in self.forbidden_folds:
            return False
        if fold == FoldType.CONTROL and mode != "read":
            return False
        if mode == "write":
            return fold in self.allowed_folds and fold not in self.read_only_folds
        if mode == "emit":
            return fold in self.emit_folds
        return fold in self.allowed_folds


class FoldInterface(ABC):
    """Abstract base for all folds."""

    def __init__(self, fold_type: FoldType):
        self.fold_type = fold_type
        self.state: Dict[str, Any] = {}
        self.collapse_history: List[CollapseProof] = []

    @abstractmethod
    def process(self, event: FoldEvent) -> Dict[str, Any]:
        """Process an event according to fold rules."""

    @abstractmethod
    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        """Apply fold-specific collapse rules."""

    def project_to_svg(self) -> str:
        svg_base = SVG_MAPPINGS[self.fold_type]
        return f'{svg_base} id="{self.fold_type.value}" data-hash="{hash(canonical_json(self.state))}"'


class NullFold(FoldInterface):
    """Explicitly refuses execution for unsupported folds."""

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        raise RuntimeError(f"{self.fold_type.value} has no executor")

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        raise RuntimeError(f"{self.fold_type.value} cannot collapse")


class EventsFold(FoldInterface):
    """⟁EVENTS_FOLD⟁: causal event timeline."""

    def __init__(self) -> None:
        super().__init__(FoldType.EVENTS)
        self.events: List[FoldEvent] = []

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        self.events.append(event)
        return {"event_id": event.event_id, "fold": event.fold.value}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class ControlFold(FoldInterface):
    """⟁CONTROL_FOLD⟁: Resolve → Gate → Commit (no hashing)."""

    def __init__(self) -> None:
        super().__init__(FoldType.CONTROL)
        self.decisions: List[Dict[str, Any]] = []

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        request = event.data.get("request", {})
        resolved = self._resolve(request)
        gated = self._gate(resolved)
        committed = self._commit(gated)
        return {
            "request_id": event.event_id,
            "resolved": resolved,
            "gate_decision": gated["allowed"],
            "target_fold": gated.get("target"),
            "decision_token": committed["decision_token"],
        }

    def _resolve(self, request: Dict[str, Any]) -> Dict[str, Any]:
        target_fold = request.get("target_fold")
        # CM-1 pre-semantic layer: parse cm1_stream if present
        cm1_result: Optional[Dict[str, Any]] = None
        if _CM1_AVAILABLE and "cm1_stream" in request:
            raw = request["cm1_stream"]
            stream_bytes = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
            parser = CM1Parser()
            cm1_result = parser.parse(stream_bytes)
        resolved: Dict[str, Any] = {}
        if target_fold:
            resolved = {"target": FoldType(target_fold), "action": request.get("action")}
        else:
            resolved = {"action": "deny", "reason": "no_target"}
        if cm1_result is not None:
            resolved["cm1_phase"] = cm1_result["phase"]
            resolved["cm1_events"] = cm1_result["events"]
        return resolved

    def _gate(self, resolved: Dict[str, Any]) -> Dict[str, Any]:
        target = resolved.get("target")
        protected_folds = {FoldType.CONTROL, FoldType.AUTH, FoldType.META}
        if target in protected_folds:
            return {"allowed": False, "reason": "protected_fold"}
        return {
            "allowed": True,
            "target": target,
            "lane": FOLD_TO_LANE[target].value if target else None,
        }

    def _commit(self, gated: Dict[str, Any]) -> Dict[str, Any]:
        decision = {
            "gate_id": sha256_hex(canonical_json(gated).encode("utf-8"))[:16],
            "allowed": gated.get("allowed", False),
        }
        self.decisions.append(decision)
        return {"decision_token": decision["gate_id"]}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class StorageFold(FoldInterface):
    """⟁STORAGE_FOLD⟁: Snapshot → Delta → Seal."""

    def __init__(self) -> None:
        super().__init__(FoldType.STORAGE)
        self.seals: Dict[str, str] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        payload = event.data.get("lane_payload", event.data.get("data", {}))
        lane = event.data.get("lane", FOLD_TO_LANE[self.fold_type].value)
        source = event.data.get("source", {})
        source_tick = source.get("tick", 0)
        canonical_payload = canonical_json(payload).encode("utf-8")
        root = self._seal(lane, source_tick, canonical_payload)
        return {
            "lane": lane,
            "root": root,
            "source": {
                "fold": source.get("fold", FoldType.STATE.value),
                "tick": source_tick,
            },
        }

    def _seal(self, lane: str, source_tick: int, payload: bytes) -> str:
        seal_material = {
            "lane": lane,
            "source_tick": source_tick,
            "payload_sha256": sha256_hex(payload),
        }
        root = sha256_hex(canonical_json(seal_material).encode("utf-8"))
        seal_key = f"{lane}:{source_tick}"
        if seal_key in self.seals and self.seals[seal_key] != root:
            raise ValueError("seal immutability violated")
        self.seals[seal_key] = root
        return root

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class MetaFold(FoldInterface):
    """⟁META_FOLD⟁: attestation chain."""

    def __init__(self) -> None:
        super().__init__(FoldType.META)
        self.last_attestation: Optional[str] = None
        self.policy_hash: Optional[str] = None
        self.abi_hash: Optional[str] = None

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        payload = event.data
        policy_hash = payload.get("policy_hash")
        meta_hash = payload.get("meta_hash")
        abi_hash = payload.get("abi_hash")
        prev = payload.get("prev", self.last_attestation)
        if self.policy_hash is None:
            self.policy_hash = policy_hash
        elif policy_hash != self.policy_hash:
            raise ValueError("policy_hash drift not allowed")
        if self.abi_hash is None:
            self.abi_hash = abi_hash
        elif abi_hash != self.abi_hash:
            raise ValueError("abi_hash drift not allowed")
        attestation = {
            "policy_hash": policy_hash,
            "meta_hash": meta_hash,
            "abi_hash": abi_hash,
            "prev": prev,
        }
        att_hash = sha256_hex(canonical_json(attestation).encode("utf-8"))
        self.last_attestation = att_hash
        return {"attestation": attestation, "attestation_hash": att_hash}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class UIFold(FoldInterface):
    """⟁UI_FOLD⟁: projection only (read-only)."""

    def __init__(self) -> None:
        super().__init__(FoldType.UI)
        self.projections: List[Dict[str, Any]] = []

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        fmt = event.data.get("format", "")
        if fmt in ("css", "dom", "terminal", "svg"):
            return self.render(fmt, event)
        # CM-1 CSS projection
        if event.data.get("type") == "cm1_projection":
            cm1_state = event.data.get("cm1_state", {})
            return {"css": self.project_cm1_to_css(cm1_state), "type": "cm1_projection"}
        payload = {
            "source_fold": event.data.get("source_fold"),
            "state": event.data.get("state", {}),
            "proof": event.data.get("proof"),
        }
        self.projections.append(payload)
        return {"projection": payload, "projection_count": len(self.projections)}

    def render(self, fmt: str, event: FoldEvent) -> Dict[str, Any]:
        """Dispatch to the correct renderer based on *fmt*."""
        fold_state = event.data.get("state", self.state)
        tick = event.data.get("tick", 0)
        state_hash = sha256_hex(canonical_json(fold_state).encode("utf-8"))
        fold_name = event.data.get("source_fold", self.fold_type.value)
        if fmt == "css":
            return {"output": self.project_to_css(fold_state), "format": "css"}
        if fmt == "dom":
            return {"output": self.project_to_dom(fold_state, fold_name, state_hash), "format": "dom"}
        if fmt == "terminal":
            return {"output": self.project_to_terminal(fold_state, fold_name, tick, state_hash), "format": "terminal"}
        # svg: minimal element string (extended in verifier for full replay)
        return {"output": self.project_to_svg(), "format": "svg"}

    def project_to_css(self, fold_state: Dict[str, Any]) -> str:
        """Render fold state as CSS custom properties block."""
        lines = [":root {"]
        for k, v in sorted(fold_state.items()):
            css_val = f'"{v}"' if isinstance(v, str) else str(v)
            css_key = k.replace("_", "-").lower()
            lines.append(f"  --{css_key}: {css_val};")
        lines.append("}")
        return "\n".join(lines)

    def project_to_dom(self, fold_state: Dict[str, Any], fold_name: str, state_hash: str) -> Dict[str, Any]:
        """Render fold state as a read-only virtual DOM tree."""
        children = []
        for k, v in sorted(fold_state.items()):
            children.append({
                "tag": "span",
                "attrs": {"data-key": k},
                "children": [str(v)],
            })
        return {
            "tag": "div",
            "attrs": {
                "data-fold": fold_name,
                "data-hash": state_hash,
            },
            "children": children,
        }

    def project_to_terminal(
        self, fold_state: Dict[str, Any], fold_name: str, tick: int, state_hash: str
    ) -> str:
        """Render fold state as ANSI-safe terminal text."""
        lines = [
            f"\u27c1 {fold_name} \u27c1  tick={tick}",
            f"state_root: {state_hash}",
            "-" * 60,
        ]
        for k, v in sorted(fold_state.items()):
            key_col = k.ljust(24)
            lines.append(f"  {key_col} {v}")
        return "\n".join(lines)

    def project_cm1_to_css(self, cm1_state: Dict[str, Any]) -> str:
        """Generate CSS custom properties for CM-1 phase state (spec §9)."""
        if _CM1_AVAILABLE:
            return project_cm1_to_css(cm1_state)
        phase = cm1_state.get("phase", "init") or "init"
        scope_depth = len(cm1_state.get("scope", []))
        mode = cm1_state.get("mode", "normal")
        literal = "true" if cm1_state.get("literal", False) else "false"
        return (
            f':root {{\n  --cm1-phase: "{phase}";\n  --cm1-scope-depth: {scope_depth};\n'
            f'  --cm1-mode: "{mode}";\n  --cm1-literal: {literal};\n}}'
        )

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class DataFold(FoldInterface):
    """⟁DATA_FOLD⟁: Deduplicate → Symbolize → Hash-bind."""

    def __init__(self) -> None:
        super().__init__(FoldType.DATA)
        self.symbols: Dict[str, str] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        payload = event.data.get("payload", event.data)
        if not isinstance(payload, dict):
            payload = {"value": payload}
        deduped: Dict[str, Any] = {}
        for k, v in payload.items():
            canonical_v = canonical_json(v) if isinstance(v, (dict, list)) else str(v)
            sym_key = sha256_hex(canonical_v.encode("utf-8"))[:12]
            self.symbols[sym_key] = canonical_v
            deduped[k] = sym_key
        binding = sha256_hex(canonical_json(deduped).encode("utf-8"))
        self.state.update({"symbols": dict(self.symbols), "binding": binding})
        return {"symbols": deduped, "dedup_count": len(deduped), "hash_binding": binding}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class AuthFold(FoldInterface):
    """⟁AUTH_FOLD⟁: Verify → Attest → Tokenize."""

    def __init__(self) -> None:
        super().__init__(FoldType.AUTH)
        self.tokens: Dict[str, str] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        agent_id = event.source_agent or event.data.get("agent_id", "unknown")
        fold_str = event.data.get("fold", self.fold_type.value)
        tick = event.data.get("tick", 0)
        attest_input = canonical_json({"agent": agent_id, "fold": fold_str, "tick": tick})
        token = sha256_hex(attest_input.encode("utf-8"))
        self.tokens[agent_id] = token
        self.state[agent_id] = token
        return {"agent_id": agent_id, "attestation_token": token, "fold": fold_str}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class PatternFold(FoldInterface):
    """⟁PATTERN_FOLD⟁: Cluster → Label → Reference."""

    def __init__(self) -> None:
        super().__init__(FoldType.PATTERN)
        self.clusters: Dict[str, List[str]] = {}

    def _extract_bigrams(self, text: str) -> List[str]:
        words = text.lower().split()
        return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        payload = event.data.get("payload", event.data)
        texts: List[str] = []
        if isinstance(payload, str):
            texts = [payload]
        elif isinstance(payload, list):
            texts = [str(t) for t in payload]
        elif isinstance(payload, dict):
            texts = [str(v) for v in payload.values()]

        reference_map: Dict[str, str] = {}
        for text in texts:
            bigrams = self._extract_bigrams(text)
            if bigrams:
                label = bigrams[0]
            else:
                label = sha256_hex(text.encode("utf-8"))[:8]
            self.clusters.setdefault(label, []).append(text)
            ref = sha256_hex(canonical_json({"label": label, "text": text}).encode("utf-8"))[:12]
            reference_map[ref] = label

        self.state["clusters"] = {k: len(v) for k, v in self.clusters.items()}
        return {"clusters": dict(self.clusters), "references": reference_map}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class StateFold(FoldInterface):
    """⟁STATE_FOLD⟁: Snapshot → Diff → Replace."""

    def __init__(self) -> None:
        super().__init__(FoldType.STATE)
        self.snapshot: Dict[str, Any] = {}

    def _apply_set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def _apply_delta(self, patch: Dict[str, Any]) -> None:
        op = patch["op"]
        path = patch["path"]
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("patch.path must start with '/'")
        parts = [p for p in path.split("/") if p]
        if not parts:
            raise ValueError("patch.path must not be empty")
        cur = self.state
        for k in parts[:-1]:
            if not isinstance(cur.get(k), dict):
                cur[k] = {}
            cur = cur[k]
        leaf = parts[-1]
        if op in ("add", "replace"):
            cur[leaf] = patch["value"]
        elif op == "remove":
            cur.pop(leaf, None)
        else:
            raise ValueError(f"Unsupported op: {op}")

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        prev_snapshot = dict(self.snapshot)
        typ = event.data.get("type", event.data.get("action", "set"))
        if typ == "set":
            key = event.data.get("key", "")
            value = event.data.get("value")
            self._apply_set(key, value)
        elif typ == "delta":
            patch = event.data.get("patch", {})
            self._apply_delta(patch)
        self.snapshot = dict(self.state)
        state_root = sha256_hex(canonical_json(self.state).encode("utf-8"))
        diff_keys = [k for k in self.state if self.state.get(k) != prev_snapshot.get(k)]
        self.state["_state_root"] = state_root
        return {"state_root_sha256": state_root, "diff_keys": diff_keys, "snapshot": dict(self.snapshot)}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class TimeFold(FoldInterface):
    """⟁TIME_FOLD⟁: Tick → Decay → Archive."""

    DEFAULT_TTL = 10

    def __init__(self) -> None:
        super().__init__(FoldType.TIME)
        self.clock: int = 0
        self.windows: Dict[str, Any] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        self.clock += 1
        entries = event.data.get("entries", {})
        ttl = event.data.get("ttl", self.DEFAULT_TTL)
        for k, v in entries.items():
            self.windows[k] = {"value": v, "expiry_tick": self.clock + ttl}
        decayed = [k for k, w in list(self.windows.items()) if self.clock > w["expiry_tick"]]
        archived = {}
        for k in decayed:
            archived[k] = self.windows.pop(k)
        self.state = {"clock": self.clock, "windows": dict(self.windows)}
        return {"clock": self.clock, "decayed": decayed, "archived": archived, "active_count": len(self.windows)}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class SpaceFold(FoldInterface):
    """⟁SPACE_FOLD⟁: Quantize → Adjacency map."""

    def __init__(self, resolution: int = 1) -> None:
        super().__init__(FoldType.SPACE)
        self.resolution = resolution
        self.positions: List[tuple] = []

    def _quantize(self, coord: Any) -> int:
        return int(round(float(coord) / self.resolution)) * self.resolution

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        raw_positions = event.data.get("positions", [])
        quantized = []
        for pos in raw_positions:
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                q = tuple(self._quantize(c) for c in pos)
            elif isinstance(pos, dict):
                q = tuple(self._quantize(v) for v in sorted(pos.values()))
            else:
                q = (self._quantize(pos),)
            quantized.append(q)
        self.positions.extend(quantized)
        adjacency: Dict[str, List[str]] = {}
        for i, p in enumerate(quantized):
            neighbors = []
            for j, other in enumerate(quantized):
                if i != j and all(abs(p[d] - other[d]) <= self.resolution for d in range(min(len(p), len(other)))):
                    neighbors.append(str(other))
            adjacency[str(p)] = neighbors
        self.state = {"positions": [str(p) for p in self.positions], "adjacency": adjacency}
        return {"quantized_positions": [str(p) for p in quantized], "adjacency": adjacency}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class NetworkFold(FoldInterface):
    """⟁NETWORK_FOLD⟁: Edge-reduce → Route-hash. No payload inspection."""

    def __init__(self) -> None:
        super().__init__(FoldType.NETWORK)
        self.edges: Dict[str, Dict[str, Any]] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        raw_edges = event.data.get("edges", [])
        for edge in raw_edges:
            src = str(edge.get("src", ""))
            dst = str(edge.get("dst", ""))
            kind = str(edge.get("kind", "default"))
            edge_key = f"{src}->{dst}:{kind}"
            if edge_key not in self.edges:
                self.edges[edge_key] = {"src": src, "dst": dst, "kind": kind}
        reduced = list(self.edges.values())
        route_hash = sha256_hex(canonical_json(sorted(self.edges.keys())).encode("utf-8"))
        self.state = {"edge_count": len(reduced), "route_hash": route_hash}
        return {"reduced_edges": reduced, "route_hash": route_hash, "edge_count": len(reduced)}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class ComputeFold(FoldInterface):
    """⟁COMPUTE_FOLD⟁: Evaluate → Emit proof → Discard state (ephemeral)."""

    def __init__(self) -> None:
        super().__init__(FoldType.COMPUTE)

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        payload = event.data.get("payload", event.data)
        input_hash = sha256_hex(canonical_json(payload).encode("utf-8"))
        expression = event.data.get("expression", str(payload))
        output = {"expression": expression, "evaluated": True}
        output_hash = sha256_hex(canonical_json(output).encode("utf-8"))
        trace_hash = sha256_hex(
            canonical_json({"input": input_hash, "output": output_hash}).encode("utf-8")
        )
        proof_record = {
            "compute.trace_hash": trace_hash,
            "input_hash": input_hash,
            "output_hash": output_hash,
        }
        self.state = {}
        return {"proof": proof_record, "output": output, "trace_hash": trace_hash}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        self.state = {}
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class DBFold(FoldInterface):
    """⟁DB_FOLD⟁: Index-compress → Canonical order."""

    def __init__(self) -> None:
        super().__init__(FoldType.DB)
        self.index: Dict[str, Any] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        records = event.data.get("records", event.data.get("payload", {}))
        if not isinstance(records, dict):
            records = {"value": records}
        value_map: Dict[str, List[str]] = {}
        for k, v in records.items():
            canonical_v = canonical_json(v) if isinstance(v, (dict, list)) else str(v)
            value_map.setdefault(canonical_v, []).append(k)
        compressed: Dict[str, Any] = {}
        for canonical_v, keys in value_map.items():
            representative_key = sorted(keys)[0]
            try:
                compressed[representative_key] = json.loads(canonical_v)
            except (json.JSONDecodeError, ValueError):
                compressed[representative_key] = canonical_v
        self.index = dict(sorted(compressed.items()))
        self.state = {"index_size": len(self.index), "keys": sorted(self.index.keys())}
        return {"index": self.index, "key_count": len(self.index)}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class CodeFold(FoldInterface):
    """⟁CODE_FOLD⟁: Inline → Normalize → Freeze."""

    def __init__(self) -> None:
        super().__init__(FoldType.CODE)
        self.frozen: Dict[str, str] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        fragment = event.data.get("fragment", event.data.get("payload", ""))
        if not isinstance(fragment, str):
            fragment = canonical_json(fragment)
        symbols = event.data.get("symbols", {})
        inlined = fragment
        for sym, replacement in symbols.items():
            inlined = inlined.replace(sym, str(replacement))
        normalized = " ".join(inlined.split()).lower()
        content_hash = sha256_hex(normalized.encode("utf-8"))
        if content_hash in self.frozen and self.frozen[content_hash] != normalized:
            raise ValueError(f"CodeFold: hash collision or mutation attempt on frozen fragment {content_hash}")
        self.frozen[content_hash] = normalized
        self.state["frozen_count"] = len(self.frozen)
        return {"normalized": normalized, "content_hash": content_hash, "frozen": True}

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self.project_to_svg(),
        )


class FoldAgent:
    """Fold-scoped agent with limited permissions."""

    def __init__(self, contract: AgentContract) -> None:
        self.contract = contract
        self.event_history: List[FoldEvent] = []

    def create_event(self, fold: FoldType, data: Dict[str, Any]) -> Optional[FoldEvent]:
        if not self.contract.can_access(fold, mode="emit"):
            return None
        event = FoldEvent(
            event_id=sha256_hex(canonical_json({"agent": self.contract.agent_id, "data": data}).encode("utf-8"))[:12],
            fold=fold,
            data=data,
            source_agent=self.contract.agent_id,
            requires_proof=self.contract.proof_required,
        )
        self.event_history.append(event)
        return event


class FoldOrchestrator:
    """Headless governor enforcing fold laws and event routing."""

    def __init__(self) -> None:
        self.folds = {
            FoldType.DATA: DataFold(),
            FoldType.CODE: CodeFold(),
            FoldType.STORAGE: StorageFold(),
            FoldType.NETWORK: NetworkFold(),
            FoldType.UI: UIFold(),
            FoldType.AUTH: AuthFold(),
            FoldType.DB: DBFold(),
            FoldType.COMPUTE: ComputeFold(),
            FoldType.STATE: StateFold(),
            FoldType.EVENTS: EventsFold(),
            FoldType.TIME: TimeFold(),
            FoldType.SPACE: SpaceFold(),
            FoldType.META: MetaFold(),
            FoldType.CONTROL: ControlFold(),
            FoldType.PATTERN: PatternFold(),
        }
        self.proofs: List[CollapseProof] = []
        self.agents: Dict[str, FoldAgent] = {}

    def register_agent(self, contract: AgentContract) -> str:
        agent = FoldAgent(contract)
        self.agents[contract.agent_id] = agent
        return contract.agent_id

    def receive_event(self, event: FoldEvent) -> Optional[CollapseProof]:
        if event.fold != FoldType.EVENTS:
            raise ValueError("All events must originate from EVENTS_FOLD")
        self.folds[FoldType.EVENTS].process(event)
        decision = self.control_gate(event)
        if not decision["allowed"]:
            return None
        target = decision["target"]
        return self.dispatch(event, target)

    def control_gate(self, event: FoldEvent) -> Dict[str, Any]:
        request = {
            "target_fold": event.data.get("target_fold"),
            "action": event.data.get("action", "process"),
            "source_agent": event.source_agent,
        }
        control_event = FoldEvent(
            event_id=f"ctrl:{event.event_id}",
            fold=FoldType.EVENTS,
            data={"request": request, "target_fold": FoldType.CONTROL.value, "internal": True},
        )
        control_result = self.folds[FoldType.CONTROL].process(control_event)
        allowed = control_result.get("gate_decision", False)
        target = control_result.get("target_fold")
        return {"allowed": allowed, "target": FoldType(target) if target else None}

    def dispatch(self, event: FoldEvent, target_fold: Optional[FoldType]) -> CollapseProof:
        if target_fold is None:
            raise ValueError("No target fold provided")
        result = self.folds[target_fold].process(event)
        proof = self.collapse(target_fold, result)
        self.proofs.append(proof)
        if target_fold != FoldType.UI:
            self._project_to_ui(target_fold, result, proof)
        return proof

    def collapse(self, fold: FoldType, result: Dict[str, Any]) -> CollapseProof:
        fold_processor = self.folds[fold]
        proof = fold_processor.collapse(result)
        expected_steps = COLLAPSE_RULES[fold]
        if proof.steps_applied != expected_steps:
            raise ValueError(f"Illegal collapse steps for {fold.value}")
        expected_lane = FOLD_TO_LANE[fold]
        if proof.lane != expected_lane:
            raise ValueError(f"Invalid lane mapping for {fold.value}")
        proof.prev_hash = self.proofs[-1].proof_hash() if self.proofs else None
        return proof

    def _project_to_ui(self, source_fold: FoldType, result: Dict[str, Any], proof: CollapseProof) -> None:
        ui_event = FoldEvent(
            event_id=f"ui:{proof.result_hash[:12]}",
            fold=FoldType.EVENTS,
            data={
                "target_fold": FoldType.UI.value,
                "source_fold": source_fold.value,
                "state": result,
                "proof": proof.to_dict(),
                "internal": True,
            },
        )
        self.folds[FoldType.EVENTS].process(ui_event)
        self.folds[FoldType.UI].process(ui_event)

    def get_svg_projection(self, fold_type: FoldType) -> str:
        fold = self.folds[fold_type]
        return fold.project_to_svg()

    def get_fold_status(self) -> Dict[str, Any]:
        status = {}
        for fold_type, fold in self.folds.items():
            status[fold_type.value] = {
                "state_hash": sha256_hex(canonical_json(fold.state).encode("utf-8")),
                "collapse_count": len(fold.collapse_history),
                "svg": fold.project_to_svg(),
                "lane": FOLD_TO_LANE[fold_type].value,
            }
        return status
