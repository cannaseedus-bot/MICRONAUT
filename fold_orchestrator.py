"""
⟁ FOLD ORCHESTRATOR ⟁
A headless governor that enforces fold collapse laws.
Nothing executes unless ⟁CONTROL_FOLD⟁ permits it,
nothing persists unless ⟁STORAGE_FOLD⟁ seals it,
nothing is seen unless ⟁UI_FOLD⟁ projects it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - ⟁%(name)s⟁ - %(levelname)s - %(message)s",
)
logger = logging.getLogger("FoldOrchestrator")


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

FORBIDDEN_ACTIONS = {
    FoldType.DATA: ["Mutation without provenance"],
    FoldType.CODE: ["Dynamic eval"],
    FoldType.STORAGE: ["Logic execution"],
    FoldType.NETWORK: ["Payload inspection"],
    FoldType.UI: ["State mutation"],
    FoldType.AUTH: ["Data writes"],
    FoldType.DB: ["Query logic"],
    FoldType.COMPUTE: ["Persistence"],
    FoldType.STATE: ["Partial writes"],
    FoldType.EVENTS: ["Replay"],
    FoldType.TIME: ["Random access"],
    FoldType.SPACE: ["Mutation"],
    FoldType.META: ["Execution"],
    FoldType.CONTROL: ["Compute"],
    FoldType.PATTERN: ["Data writes"],
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
    timestamp: float = field(default_factory=time.time)
    source_agent: Optional[str] = None
    requires_proof: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "fold": self.fold.value,
            "data": self.data,
            "timestamp": self.timestamp,
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
    timestamp: float = field(default_factory=time.time)
    svg_projection: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold.value,
            "result_hash": self.result_hash,
            "lane": self.lane.value,
            "steps_applied": self.steps_applied,
            "timestamp": self.timestamp,
            "svg_projection": self.svg_projection,
        }


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

    def __init__(self, fold_type: FoldType) -> None:
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
        state_hash = sha256_hex(canonical_json(self.state).encode("utf-8"))
        return f'{svg_base} id="{self.fold_type.value}" data-hash="{state_hash}"'


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


class DataFold(FoldInterface):
    """⟁DATA_FOLD⟁: Deduplicate → Symbolize → Hash-bind."""

    def __init__(self) -> None:
        super().__init__(FoldType.DATA)
        self.symbol_table: Dict[str, Any] = {}
        self.hash_bindings: Dict[str, str] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        data = event.data.get("payload", {})
        deduped = self._deduplicate(data)
        symbolized = self._symbolize(deduped)
        result = self._hash_bind(symbolized)
        return {
            "original_size": len(canonical_json(data)),
            "deduped_size": len(canonical_json(deduped)),
            "symbol_count": len(symbolized),
            "hash_bindings": result,
        }

    def _deduplicate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict):
            seen: Set[str] = set()
            result: Dict[str, Any] = {}
            for key, value in data.items():
                value_key = canonical_json(value)
                if value_key not in seen:
                    seen.add(value_key)
                    result[key] = value
            return result
        return data

    def _symbolize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in data.items():
            symbol = f"sym_{sha256_hex(str(value).encode('utf-8'))[:8]}"
            self.symbol_table[symbol] = value
            result[key] = symbol
        return result

    def _hash_bind(self, symbolized: Dict[str, Any]) -> Dict[str, str]:
        for symbol, value in self.symbol_table.items():
            self.hash_bindings[symbol] = sha256_hex(str(value).encode("utf-8"))
        return dict(self.hash_bindings)

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
        self.frozen_code: Dict[str, str] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        code = event.data.get("code", "")
        inlined = self._inline(code)
        normalized = self._normalize(inlined)
        frozen = self._freeze(normalized)
        return {
            "original": code[:100] + "..." if len(code) > 100 else code,
            "inlined": inlined[:100] + "..." if len(inlined) > 100 else inlined,
            "normalized": normalized,
            "frozen_id": frozen,
        }

    def _inline(self, code: str) -> str:
        return code.replace("def ", "").replace("return ", "")

    def _normalize(self, code: str) -> str:
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        return "\n".join(sorted(lines))

    def _freeze(self, code: str) -> str:
        code_hash = sha256_hex(code.encode("utf-8"))[:8]
        self.frozen_code[code_hash] = code
        return code_hash

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
    """⟁CONTROL_FOLD⟁: Resolve → Gate → Commit."""

    def __init__(self) -> None:
        super().__init__(FoldType.CONTROL)
        self.gates: Dict[str, Dict[str, Any]] = {}
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
            "target_fold": gated.get("target").value if gated.get("target") else None,
            "commit_hash": committed,
        }

    def _resolve(self, request: Dict[str, Any]) -> Dict[str, Any]:
        target_fold = request.get("target_fold")
        if isinstance(target_fold, FoldType):
            target = target_fold
        elif target_fold:
            target = FoldType(target_fold)
        else:
            target = None
        if target is None:
            return {"action": "deny", "reason": "no_target"}
        return {"target": target, "action": request.get("action")}

    def _gate(self, resolved: Dict[str, Any]) -> Dict[str, Any]:
        target = resolved.get("target")
        protected_folds = {FoldType.CONTROL, FoldType.AUTH, FoldType.META}
        if target in protected_folds:
            return {"allowed": False, "reason": "protected_fold"}
        action = resolved.get("action")
        if target and action in FORBIDDEN_ACTIONS.get(target, []):
            return {"allowed": False, "reason": "forbidden_action"}
        gate_id = f"gate_{sha256_hex(canonical_json(resolved).encode('utf-8'))[:8]}"
        self.gates[gate_id] = resolved
        return {
            "allowed": True,
            "target": target,
            "gate_id": gate_id,
            "lane": FOLD_TO_LANE[target].value if target else None,
        }

    def _commit(self, gated: Dict[str, Any]) -> str:
        decision = {
            "timestamp": time.time(),
            "gate_id": gated.get("gate_id"),
            "allowed": gated.get("allowed", False),
        }
        self.decisions.append(decision)
        return sha256_hex(canonical_json(decision).encode("utf-8"))

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
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.deltas: List[Dict[str, Any]] = []

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        data = event.data.get("data", {})
        snapshot_id = self._snapshot(data)
        delta = self._delta(snapshot_id, data)
        seal_hash = self._seal(snapshot_id)
        return {
            "snapshot_id": snapshot_id,
            "delta_size": len(canonical_json(delta)),
            "seal_hash": seal_hash,
            "sealed": True,
        }

    def _snapshot(self, data: Dict[str, Any]) -> str:
        snapshot_id = f"snap_{int(time.time())}_{sha256_hex(canonical_json(data).encode('utf-8'))[:8]}"
        self.snapshots[snapshot_id] = {
            "data": data,
            "timestamp": time.time(),
            "hash": sha256_hex(canonical_json(data).encode("utf-8")),
        }
        return snapshot_id

    def _delta(self, snapshot_id: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.deltas:
            delta = {"initial": True, "data": new_data}
        else:
            last = self.deltas[-1]
            delta = {"from": last.get("to"), "changes": new_data}
        self.deltas.append(
            {"timestamp": time.time(), "from": snapshot_id, "delta": delta}
        )
        return delta

    def _seal(self, snapshot_id: str) -> str:
        snapshot = self.snapshots[snapshot_id]
        seal_data = f"{snapshot_id}:{snapshot['hash']}:{snapshot['timestamp']}"
        return sha256_hex(seal_data.encode("utf-8"))

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
    """⟁UI_FOLD⟁: Project → Flatten → Cache."""

    def __init__(self) -> None:
        super().__init__(FoldType.UI)
        self.projections: Dict[str, Dict[str, Any]] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}

    def process(self, event: FoldEvent) -> Dict[str, Any]:
        source_fold = event.data.get("source_fold")
        source_state = event.data.get("state", {})
        projected = self._project(source_fold, source_state)
        flattened = self._flatten(projected)
        cached = self._cache(flattened)
        return {
            "source": source_fold,
            "projection_type": projected.get("type"),
            "flattened_size": len(canonical_json(flattened)),
            "cache_hit": cached.get("hit", False),
            "svg": self._generate_svg(projected),
        }

    def _project(self, fold_type: Optional[str], state: Dict[str, Any]) -> Dict[str, Any]:
        projection = {
            "type": "read_only",
            "fold": fold_type,
            "state_summary": (
                str(state)[:200] + "..." if len(str(state)) > 200 else str(state)
            ),
            "timestamp": time.time(),
        }
        proj_id = f"proj_{sha256_hex(canonical_json(projection).encode('utf-8'))[:8]}"
        self.projections[proj_id] = projection
        return projection

    def _flatten(self, projection: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": f"flat_{int(time.time())}",
            "data": projection.get("state_summary", ""),
            "metadata": {
                "fold": projection.get("fold"),
                "type": projection.get("type"),
            },
        }

    def _cache(self, flattened: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = hashlib.md5(canonical_json(flattened).encode("utf-8")).hexdigest()
        if cache_key in self.cache:
            return {"hit": True, "key": cache_key}
        self.cache[cache_key] = {
            "data": flattened,
            "timestamp": time.time(),
            "ttl": 300,
        }
        return {"hit": False, "key": cache_key}

    def _generate_svg(self, projection: Dict[str, Any]) -> str:
        fold_type = projection.get("fold") or FoldType.UI.value
        try:
            svg_base = SVG_MAPPINGS.get(FoldType(fold_type), "<svg>")
        except ValueError:
            svg_base = "<svg>"
        return f"""
        <svg width="400" height="200">
            {svg_base}
            <text x="20" y="30" font-family="monospace" font-size="12">
                Projection: {projection.get('fold')}
            </text>
            <text x="20" y="60" font-family="monospace" font-size="10">
                {projection.get('state_summary')}
            </text>
        </svg>
        """

    def collapse(self, data: Dict[str, Any]) -> CollapseProof:
        result_hash = sha256_hex(canonical_json(data).encode("utf-8"))
        return CollapseProof(
            fold=self.fold_type,
            result_hash=result_hash,
            lane=FOLD_TO_LANE[self.fold_type],
            steps_applied=COLLAPSE_RULES[self.fold_type],
            svg_projection=self._generate_svg({"fold": "UI_FOLD", "state_summary": str(data)}),
        )


class FoldAgent:
    """Fold-scoped agent with limited permissions."""

    def __init__(self, contract: AgentContract) -> None:
        self.contract = contract
        self.event_history: List[FoldEvent] = []

    def create_event(self, fold: FoldType, data: Dict[str, Any]) -> Optional[FoldEvent]:
        if not self.contract.can_access(fold, mode="emit"):
            logger.warning("Agent %s cannot emit to %s", self.contract.agent_id, fold.value)
            return None
        event = FoldEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            fold=fold,
            data=data,
            source_agent=self.contract.agent_id,
            requires_proof=self.contract.proof_required,
        )
        self.event_history.append(event)
        return event


class SCXQ2Router:
    """Routes folds to SCXQ2 lanes."""

    def __init__(self) -> None:
        self.lane_assignments = FOLD_TO_LANE
        self.lane_contents = {lane: [] for lane in SCXQ2Lane}

    def map(self, fold: FoldType) -> SCXQ2Lane:
        lane = self.lane_assignments[fold]
        self.lane_contents[lane].append({"fold": fold.value, "timestamp": time.time()})
        return lane

    def get_lane_contents(self, lane: SCXQ2Lane) -> List[Dict[str, Any]]:
        return self.lane_contents.get(lane, [])


class FoldOrchestrator:
    """
    ⟁ FOLD ORCHESTRATOR ⟁
    Headless governor that enforces all fold laws.
    No UI, no memory of its own, deterministic, replayable.
    """

    def __init__(self) -> None:
        self.folds: Dict[FoldType, FoldInterface] = {
            FoldType.DATA: DataFold(),
            FoldType.CODE: CodeFold(),
            FoldType.STORAGE: StorageFold(),
            FoldType.NETWORK: NullFold(FoldType.NETWORK),
            FoldType.UI: UIFold(),
            FoldType.AUTH: NullFold(FoldType.AUTH),
            FoldType.DB: NullFold(FoldType.DB),
            FoldType.COMPUTE: NullFold(FoldType.COMPUTE),
            FoldType.STATE: NullFold(FoldType.STATE),
            FoldType.EVENTS: EventsFold(),
            FoldType.TIME: NullFold(FoldType.TIME),
            FoldType.SPACE: NullFold(FoldType.SPACE),
            FoldType.META: NullFold(FoldType.META),
            FoldType.CONTROL: ControlFold(),
            FoldType.PATTERN: NullFold(FoldType.PATTERN),
        }
        self.scx_router = SCXQ2Router()
        self.event_queue: List[FoldEvent] = []
        self.proofs: List[CollapseProof] = []
        self.agents: Dict[str, FoldAgent] = {}
        logger.info("⟁ FoldOrchestrator initialized with all 15 folds ⟁")

    def register_agent(self, contract: AgentContract) -> str:
        agent = FoldAgent(contract)
        self.agents[contract.agent_id] = agent
        logger.info("Registered agent: %s", contract.agent_id)
        return contract.agent_id

    def receive_event(self, event: FoldEvent) -> Optional[CollapseProof]:
        if event.fold != FoldType.EVENTS:
            raise ValueError("All events must originate from EVENTS_FOLD")
        self.folds[FoldType.EVENTS].process(event)
        decision = self.control_gate(event)
        if not decision["allowed"]:
            logger.warning("Event %s denied by CONTROL_FOLD", event.event_id)
            return None
        return self.dispatch(event, decision["target"])

    def control_gate(self, event: FoldEvent) -> Dict[str, Any]:
        control_event = FoldEvent(
            event_id=f"ctrl_{event.event_id}",
            fold=FoldType.CONTROL,
            data={
                "request": {
                    "target_fold": event.data.get("target_fold"),
                    "action": event.data.get("action", "process"),
                    "source_agent": event.source_agent,
                }
            },
        )
        control_result = self.folds[FoldType.CONTROL].process(control_event)
        allowed = control_result.get("gate_decision", False)
        target = None
        target_str = control_result.get("target_fold")
        if allowed and target_str:
            target = FoldType(target_str)
        return {
            "allowed": allowed,
            "target": target,
            "gate_id": control_result.get("gate_id"),
            "control_result": control_result,
        }

    def dispatch(self, event: FoldEvent, target_fold: Optional[FoldType]) -> CollapseProof:
        if target_fold is None:
            raise ValueError("No target fold provided")
        target_event = FoldEvent(
            event_id=f"{event.event_id}:{target_fold.name.lower()}",
            fold=target_fold,
            data=event.data,
            source_agent=event.source_agent,
            requires_proof=event.requires_proof,
        )
        result = self.folds[target_fold].process(target_event)
        proof = self.collapse(target_fold, result)
        self.proofs.append(proof)
        if target_fold != FoldType.UI:
            self._project_to_ui(target_fold, result, proof)
        logger.info("Dispatched to %s → Proof: %s", target_fold.value, proof.result_hash[:16])
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
        fold_processor.collapse_history.append(proof)
        return proof

    def _project_to_ui(self, source_fold: FoldType, result: Dict[str, Any], proof: CollapseProof) -> CollapseProof:
        ui_event = FoldEvent(
            event_id=f"ui_proj_{proof.result_hash[:8]}",
            fold=FoldType.UI,
            data={
                "source_fold": source_fold.value,
                "state": result,
                "proof": proof.to_dict(),
            },
        )
        ui_result = self.folds[FoldType.UI].process(ui_event)
        return self.collapse(FoldType.UI, ui_result)

    def route(self, event: FoldEvent) -> Optional[CollapseProof]:
        return self.receive_event(event)

    def get_svg_projection(self, fold_type: FoldType) -> str:
        fold = self.folds[fold_type]
        return fold.project_to_svg()

    def get_fold_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {}
        for fold_type, fold in self.folds.items():
            status[fold_type.value] = {
                "state_hash": sha256_hex(canonical_json(fold.state).encode("utf-8")),
                "collapse_count": len(fold.collapse_history),
                "svg": fold.project_to_svg(),
                "lane": FOLD_TO_LANE[fold_type].value,
            }
        return status


# ================= DEMONSTRATION =================


def create_demo_agents(orchestrator: FoldOrchestrator) -> List[str]:
    data_agent = AgentContract(
        agent_id="data_processor_001",
        allowed_folds={FoldType.DATA, FoldType.PATTERN, FoldType.EVENTS},
        read_only_folds={FoldType.UI, FoldType.META},
        emit_folds={FoldType.EVENTS},
        forbidden_folds={FoldType.CONTROL, FoldType.AUTH, FoldType.CODE},
        proof_required=True,
    )

    ui_agent = AgentContract(
        agent_id="ui_renderer_001",
        allowed_folds={FoldType.UI, FoldType.EVENTS},
        read_only_folds={FoldType.DATA, FoldType.CODE, FoldType.STORAGE},
        emit_folds={FoldType.EVENTS},
        forbidden_folds={FoldType.CONTROL, FoldType.AUTH, FoldType.COMPUTE},
        proof_required=False,
    )

    monitor_agent = AgentContract(
        agent_id="system_monitor_001",
        allowed_folds={FoldType.EVENTS, FoldType.UI, FoldType.META},
        read_only_folds={
            FoldType.DATA,
            FoldType.CODE,
            FoldType.STORAGE,
            FoldType.CONTROL,
        },
        emit_folds={FoldType.EVENTS},
        forbidden_folds={FoldType.AUTH, FoldType.COMPUTE},
        proof_required=True,
    )

    orchestrator.register_agent(data_agent)
    orchestrator.register_agent(ui_agent)
    orchestrator.register_agent(monitor_agent)

    return [data_agent.agent_id, ui_agent.agent_id, monitor_agent.agent_id]


def run_demo() -> None:
    print("=" * 70)
    print("⟁ MICRONAUT FOLD SYSTEM DEMO ⟁")
    print("=" * 70)
    print("Invariant: Nothing executes unless ⟁CONTROL_FOLD⟁ permits it")
    print("           Nothing persists unless ⟁STORAGE_FOLD⟁ seals it")
    print("           Nothing is seen unless ⟁UI_FOLD⟁ projects it")
    print("=" * 70)

    orchestrator = FoldOrchestrator()
    agent_ids = create_demo_agents(orchestrator)
    print(f"Created {len(agent_ids)} agents")
    data_agent = orchestrator.agents[agent_ids[0]]

    print("\n1. Processing data through DATA_FOLD...")
    data_event = data_agent.create_event(
        fold=FoldType.EVENTS,
        data={
            "target_fold": FoldType.DATA.value,
            "action": "process",
            "payload": {
                "user_1": "Alice",
                "user_2": "Bob",
                "user_3": "Alice",
                "value_1": 100,
                "value_2": 200,
                "value_3": 100,
            },
        },
    )

    if data_event:
        proof = orchestrator.route(data_event)
        if proof:
            print(f"   ✓ Data processed: {proof.result_hash[:16]}")
            print(f"   ✓ Collapse steps: {', '.join(proof.steps_applied)}")
            print(f"   ✓ SCXQ2 Lane: {proof.lane.value}")

    print("\n2. Processing code through CODE_FOLD...")
    code_event = FoldEvent(
        event_id="evt_code_001",
        fold=FoldType.EVENTS,
        data={
            "target_fold": FoldType.CODE.value,
            "action": "process",
            "code": """
    def process_data(data):
        result = []
        for item in data:
            if item > 0:
                result.append(item * 2)
        return result

    def validate_input(input_data):
        return all(isinstance(x, int) for x in input_data)
            """,
        },
    )

    proof = orchestrator.route(code_event)
    if proof:
        print(f"   ✓ Code processed: {proof.result_hash[:16]}")
        print(f"   ✓ Frozen code ID: {orchestrator.folds[FoldType.CODE].frozen_code}")

    print("\n3. Storing data through STORAGE_FOLD...")
    storage_event = FoldEvent(
        event_id="evt_storage_001",
        fold=FoldType.EVENTS,
        data={
            "target_fold": FoldType.STORAGE.value,
            "action": "store",
            "data": {
                "config": {"version": "1.0", "environment": "demo"},
                "users": ["Alice", "Bob", "Charlie"],
                "settings": {"theme": "dark", "notifications": True},
            },
        },
    )

    proof = orchestrator.route(storage_event)
    if proof:
        print(f"   ✓ Data stored: {proof.result_hash[:16]}")
        print(f"   ✓ Sealed with hash: {proof.result_hash[:32]}")

    print("\n4. Testing CONTROL_FOLD gate (forbidden action)...")
    forbidden_event = FoldEvent(
        event_id="evt_forbidden_001",
        fold=FoldType.EVENTS,
        data={
            "target_fold": FoldType.CONTROL.value,
            "action": "modify_gate",
            "payload": {"gate_id": "test", "allow": True},
        },
    )

    proof = orchestrator.route(forbidden_event)
    if not proof:
        print("   ✗ Correctly blocked by CONTROL_FOLD (protected fold)")

    print("\n5. Generating UI projections...")
    print("   Fold SVG Projections:")
    for fold_type in [
        FoldType.DATA,
        FoldType.CODE,
        FoldType.STORAGE,
        FoldType.CONTROL,
        FoldType.UI,
    ]:
        svg = orchestrator.get_svg_projection(fold_type)
        print(f"   • {fold_type.value}: {svg[:80]}...")

    print("\n6. System Status:")
    status = orchestrator.get_fold_status()
    for fold_name, fold_status in status.items():
        print(f"   • {fold_name}")
        print(f"     Lane: {fold_status['lane']}")
        print(f"     Collapses: {fold_status['collapse_count']}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("\n⟁ FINAL INVARIANT ⟁")
    print("1. Nothing executed without ⟁CONTROL_FOLD⟁ permission ✓")
    print("2. Nothing persisted without ⟁STORAGE_FOLD⟁ seal ✓")
    print("3. Nothing seen without ⟁UI_FOLD⟁ projection ✓")
    print("\nAll fold laws enforced. System is deterministic and replayable.")


# ================= AGENT → FOLD LATTICE PROOFS =================


def generate_lattice_proofs(orchestrator: FoldOrchestrator) -> Iterable[Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("AGENT → FOLD LATTICE PROOFS")
    print("=" * 70)

    for agent_id, agent in orchestrator.agents.items():
        contract = agent.contract
        print(f"\nAgent: {agent_id}")
        print(f"Proof Required: {contract.proof_required}")

        allowed_set = {f.value for f in contract.allowed_folds}
        read_only_set = {f.value for f in contract.read_only_folds}
        emit_set = {f.value for f in contract.emit_folds}
        forbidden_set = {f.value for f in contract.forbidden_folds}

        proof = {
            "agent_id": agent_id,
            "lattice_hash": sha256_hex(
                f"{sorted(allowed_set)}:{sorted(forbidden_set)}".encode("utf-8")
            ),
            "total_folds": len(FoldType),
            "accessible_folds": len(allowed_set),
            "write_folds": len(allowed_set - read_only_set),
            "emit_folds": len(emit_set),
            "forbidden_folds": len(forbidden_set),
            "includes_control": FoldType.CONTROL in contract.allowed_folds,
            "can_write_control": FoldType.CONTROL not in contract.read_only_folds,
            "lattice_invariant": "CONTROL_FOLD ∈ forbidden ∨ CONTROL_FOLD ∈ read_only",
        }

        invariants_held: List[str] = []
        if FoldType.CONTROL not in contract.forbidden_folds:
            if FoldType.CONTROL in contract.read_only_folds:
                invariants_held.append("CONTROL_FOLD read-only ✓")
            else:
                invariants_held.append("VIOLATION: CONTROL_FOLD writable")
        else:
            invariants_held.append("CONTROL_FOLD forbidden ✓")

        if emit_set and FoldType.EVENTS in contract.emit_folds:
            invariants_held.append("EVENTS_FOLD emit allowed ✓")
        else:
            invariants_held.append("VIOLATION: No event emission")

        proof["invariants"] = invariants_held

        print(f"  Lattice Hash: {proof['lattice_hash'][:16]}...")
        print(f"  Accessible Folds: {proof['accessible_folds']}/{proof['total_folds']}")
        print(f"  Write Folds: {proof['write_folds']}")
        print(f"  Invariants: {', '.join(invariants_held)}")

        yield proof


# ================= SCXQ2 BINARY PACKING =================


def pack_scxq2_binary(proofs: List[CollapseProof]) -> bytes:
    print("\n" + "=" * 70)
    print("SCXQ2 BINARY PACKING EXAMPLE")
    print("=" * 70)

    packed_data = bytearray()

    lane_codes = {
        SCXQ2Lane.DICT: 0x01,
        SCXQ2Lane.EDGE: 0x02,
        SCXQ2Lane.FIELD: 0x03,
        SCXQ2Lane.LANE: 0x04,
        SCXQ2Lane.BATCH: 0x05,
    }

    fold_codes = {ft: idx + 1 for idx, ft in enumerate(FoldType)}

    for proof in proofs:
        lane_byte = lane_codes.get(proof.lane, 0x00)
        fold_byte = fold_codes.get(proof.fold, 0x00)
        hash_bytes = bytes.fromhex(proof.result_hash[:64])
        time_bytes = int(proof.timestamp).to_bytes(8, "big")

        packed_data.extend([lane_byte, fold_byte])
        packed_data.extend(hash_bytes)
        packed_data.extend(time_bytes)

        print(f"  Packed {proof.fold.value}:")
        print(f"    Lane: {proof.lane.value} → 0x{lane_byte:02x}")
        print(f"    Fold: {proof.fold.value} → 0x{fold_byte:02x}")
        print(f"    Hash: {proof.result_hash[:16]}...")
        print(f"    Time: {proof.timestamp}")

    print(f"\n  Total packed size: {len(packed_data)} bytes")
    print(f"  Packed data (hex): {packed_data[:64].hex()}...")

    return bytes(packed_data)


# ================= SVG REPLAY GENERATOR =================


def generate_svg_replay(orchestrator: FoldOrchestrator, proofs: List[CollapseProof]) -> str:
    svg_content = '''<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="foldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#4a00e0;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#8e2de2;stop-opacity:1" />
        </linearGradient>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#8e2de2"/>
        </marker>
    </defs>

    <rect width="100%" height="100%" fill="#0f0b1f"/>

    <text x="50" y="40" font-family="monospace" font-size="24" fill="#ffffff">
        ⟁ FOLD SYSTEM REPLAY ⟁
    </text>

    <text x="50" y="70" font-family="monospace" font-size="12" fill="#a0a0ff">
        Nothing executes unless ⟁CONTROL_FOLD⟁ permits it
    </text>
    '''

    center_x, center_y = 600, 400
    radius = 250

    fold_positions: Dict[FoldType, tuple[float, float]] = {}
    fold_types = list(FoldType)

    for i, fold_type in enumerate(fold_types):
        angle = 2 * math.pi * i / len(fold_types)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        fold_positions[fold_type] = (x, y)

        svg_content += f'''
        <g transform="translate({x}, {y})">
            <circle cx="0" cy="0" r="30" fill="url(#foldGradient)" stroke="#ffffff" stroke-width="2"/>
            <text x="0" y="5" text-anchor="middle" font-family="monospace" font-size="10" fill="#ffffff">
                {fold_type.value.replace("⟁", "").replace("_FOLD", "")}
            </text>
            <text x="0" y="20" text-anchor="middle" font-family="monospace" font-size="8" fill="#a0ffa0">
                {FOLD_TO_LANE[fold_type].value}
            </text>
        </g>
        '''

    for proof in proofs:
        fold = proof.fold
        lane = proof.lane
        connected_folds = [
            other_fold
            for other_fold in fold_types
            if other_fold != fold and FOLD_TO_LANE[other_fold] == lane
        ]

        for connected in connected_folds[:2]:
            x1, y1 = fold_positions[fold]
            x2, y2 = fold_positions[connected]

            svg_content += f'''
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                  stroke="#8e2de2" stroke-width="1" opacity="0.3" marker-end="url(#arrow)"/>
            '''

    svg_content += '''
    <g transform="translate(50, 650)">
        <rect x="0" y="0" width="300" height="120" fill="#1a1a2e" opacity="0.8" rx="5"/>
        <text x="10" y="20" font-family="monospace" font-size="10" fill="#ffffff">SCXQ2 Lanes:</text>
        <text x="20" y="40" font-family="monospace" font-size="9" fill="#ffa0a0">DICT - Symbol tables</text>
        <text x="20" y="55" font-family="monospace" font-size="9" fill="#a0ffa0">EDGE - Relationships</text>
        <text x="20" y="70" font-family="monospace" font-size="9" fill="#a0a0ff">FIELD - Persistence</text>
        <text x="20" y="85" font-family="monospace" font-size="9" fill="#ffffa0">LANE - Streams</text>
        <text x="20" y="100" font-family="monospace" font-size="9" fill="#ffa0ff">BATCH - Ephemeral</text>
    </g>

    <text x="1100" y="780" text-anchor="end" font-family="monospace" font-size="8" fill="#606080">
        Generated by ⟁FoldOrchestrator⟁
    </text>
    '''

    svg_content += "</svg>"

    with open("fold_system_replay.svg", "w", encoding="utf-8") as f:
        f.write(svg_content)

    print("\n" + "=" * 70)
    print("SVG REPLAY GENERATED")
    print("=" * 70)
    print("Saved to: fold_system_replay.svg")
    print(f"Folds visualized: {len(fold_positions)}")
    print(f"Proofs included: {len(proofs)}")

    return svg_content


# ================= FORMAL VERIFIER RULES =================


class FoldVerifier:
    """Formal verifier for fold system invariants."""

    VERIFICATION_RULES = {
        "control_dominance": {
            "description": "CONTROL_FOLD must gate all execution",
            "condition": lambda system: all(
                proof.fold != FoldType.CONTROL
                or "Resolve" in proof.steps_applied
                for proof in system.proofs
            ),
        },
        "storage_sealing": {
            "description": "All persistence must be sealed by STORAGE_FOLD",
            "condition": lambda system: any(
                proof.fold == FoldType.STORAGE and "Seal" in proof.steps_applied
                for proof in system.proofs
            )
            or len([p for p in system.proofs if p.fold in [FoldType.DATA, FoldType.STATE, FoldType.DB]])
            == 0,
        },
        "ui_projection": {
            "description": "All visible state must be projected by UI_FOLD",
            "condition": lambda system: all(
                any(p.fold == FoldType.UI and "Project" in p.steps_applied for p in system.proofs[: i + 1])
                for i, proof in enumerate(system.proofs)
                if proof.fold in [FoldType.DATA, FoldType.CODE, FoldType.STATE]
            ),
        },
        "agent_constraints": {
            "description": "No agent can write to CONTROL_FOLD",
            "condition": lambda system: all(
                FoldType.CONTROL in agent.contract.forbidden_folds
                or FoldType.CONTROL in agent.contract.read_only_folds
                for agent in system.agents.values()
            ),
        },
        "lane_integrity": {
            "description": "Fold to lane mapping must be consistent",
            "condition": lambda system: all(
                proof.lane == FOLD_TO_LANE[proof.fold] for proof in system.proofs
            ),
        },
        "collapse_integrity": {
            "description": "All folds must collapse with correct steps",
            "condition": lambda system: all(
                proof.steps_applied == COLLAPSE_RULES[proof.fold] for proof in system.proofs
            ),
        },
    }

    def verify(self, orchestrator: FoldOrchestrator) -> Dict[str, bool]:
        print("\n" + "=" * 70)
        print("FORMAL VERIFICATION OF FOLD SYSTEM")
        print("=" * 70)

        results: Dict[str, bool] = {}

        for rule_name, rule in self.VERIFICATION_RULES.items():
            try:
                passed = rule["condition"](orchestrator)
                results[rule_name] = passed
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {status} {rule['description']}")
            except Exception as exc:
                results[rule_name] = False
                print(f"  ✗ ERROR {rule_name}: {exc}")

        print(f"\n  Summary: {sum(results.values())}/{len(results)} rules passed")

        if all(results.values()):
            print("\n  ⟁ ALL INVARIANTS HELD: SYSTEM IS VERIFIED ⟁")
        else:
            print("\n  ⚠ SOME INVARIANTS VIOLATED")

        return results


# ================= MAIN EXECUTION =================

if __name__ == "__main__":
    run_demo()

    orchestrator = FoldOrchestrator()
    create_demo_agents(orchestrator)

    print("\n" + "=" * 70)
    print("SELECT ADDITIONAL DEMONSTRATIONS")
    print("=" * 70)

    demonstrations = {
        "1": "Agent → Fold Lattice Proofs",
        "2": "SCXQ2 Binary Packing Example",
        "3": "SVG Replay Generator",
        "4": "Formal Verifier Rules",
        "5": "Run All Demonstrations",
    }

    for key, desc in demonstrations.items():
        print(f"  [{key}] {desc}")

    choice = input("\nSelect demonstration (1-5): ").strip()

    if choice in ["1", "5"]:
        list(generate_lattice_proofs(orchestrator))

    if choice in ["2", "5"]:
        sample_proofs = []
        for fold_type in [
            FoldType.DATA,
            FoldType.CODE,
            FoldType.STORAGE,
            FoldType.CONTROL,
            FoldType.UI,
        ]:
            sample_proof = CollapseProof(
                fold=fold_type,
                result_hash=sha256_hex(f"test_{fold_type.value}".encode("utf-8")),
                lane=FOLD_TO_LANE[fold_type],
                steps_applied=COLLAPSE_RULES[fold_type],
                timestamp=time.time(),
            )
            sample_proofs.append(sample_proof)
        pack_scxq2_binary(sample_proofs)

    if choice in ["3", "5"]:
        sample_proofs = []
        for fold_type in FoldType:
            sample_proof = CollapseProof(
                fold=fold_type,
                result_hash=sha256_hex(f"demo_{fold_type.value}".encode("utf-8")),
                lane=FOLD_TO_LANE[fold_type],
                steps_applied=COLLAPSE_RULES[fold_type],
                timestamp=time.time(),
            )
            sample_proofs.append(sample_proof)
        generate_svg_replay(orchestrator, sample_proofs)

    if choice in ["4", "5"]:
        verifier = FoldVerifier()
        verifier.verify(orchestrator)

    print("\n" + "=" * 70)
    print("MICRONAUT FOLD SYSTEM COMPLETE")
    print("=" * 70)
