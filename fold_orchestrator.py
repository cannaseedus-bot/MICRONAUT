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
        if target_fold:
            return {"target": FoldType(target_fold), "action": request.get("action")}
        return {"action": "deny", "reason": "no_target"}

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
        payload = {
            "source_fold": event.data.get("source_fold"),
            "state": event.data.get("state", {}),
            "proof": event.data.get("proof"),
        }
        self.projections.append(payload)
        return {"projection": payload, "projection_count": len(self.projections)}

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
            FoldType.DATA: NullFold(FoldType.DATA),
            FoldType.CODE: NullFold(FoldType.CODE),
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
            FoldType.META: MetaFold(),
            FoldType.CONTROL: ControlFold(),
            FoldType.PATTERN: NullFold(FoldType.PATTERN),
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
