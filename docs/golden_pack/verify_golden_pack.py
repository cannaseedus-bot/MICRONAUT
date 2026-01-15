import hashlib
import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# -----------------------------
# Golden expected hashes (SHA-256)
# -----------------------------
EXPECTED_REPLAY_0001_SVG_SHA256 = "905fa675040e66d9f9695b71da669c961e13822e476328fb20b0b6961fb3aeba"
EXPECTED_SCX2_BIN_SHA256 = "544e2899b93c9e0e26f3092d94aafc5ee194a47a066b8db91172326c8846613d"

# -----------------------------
# Canonical fold map (law)
# -----------------------------
COMPLETE_FOLD_MAP = {
    "⟁DATA_FOLD⟁": "All data structures and values",
    "⟁CODE_FOLD⟁": "All execution logic and functions",
    "⟁STORAGE_FOLD⟁": "All persistence mechanisms",
    "⟁NETWORK_FOLD⟁": "All communication protocols",
    "⟁UI_FOLD⟁": "All user interface elements",
    "⟁AUTH_FOLD⟁": "All security and identity",
    "⟁DB_FOLD⟁": "All data organization",
    "⟁COMPUTE_FOLD⟁": "All processing operations",
    "⟁STATE_FOLD⟁": "All application state",
    "⟁EVENTS_FOLD⟁": "All event handling",
    "⟁TIME_FOLD⟁": "All temporal operations",
    "⟁SPACE_FOLD⟁": "All spatial relationships",
    "⟁META_FOLD⟁": "Fold operations themselves",
    "⟁CONTROL_FOLD⟁": "Execution flow control",
    "⟁PATTERN_FOLD⟁": "Pattern recognition operations",
}

# -----------------------------
# Fixed fold → SCXQ2 lane mapping (law)
# -----------------------------
FOLD_TO_LANE = {
    "⟁DATA_FOLD⟁": "DICT",
    "⟁CODE_FOLD⟁": "EDGE",
    "⟁STORAGE_FOLD⟁": "FIELD",
    "⟁NETWORK_FOLD⟁": "EDGE",
    "⟁UI_FOLD⟁": "LANE",
    "⟁AUTH_FOLD⟁": "DICT",
    "⟁DB_FOLD⟁": "FIELD",
    "⟁COMPUTE_FOLD⟁": "BATCH",
    "⟁STATE_FOLD⟁": "FIELD",
    "⟁EVENTS_FOLD⟁": "LANE",
    "⟁TIME_FOLD⟁": "LANE",
    "⟁SPACE_FOLD⟁": "EDGE",
    "⟁META_FOLD⟁": "DICT",
    "⟁CONTROL_FOLD⟁": "EDGE",
    "⟁PATTERN_FOLD⟁": "DICT",
}

# -----------------------------
# Helpers
# -----------------------------

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def die(msg: str) -> None:
    raise SystemExit(f"❌ VERIFY FAIL: {msg}")


# -----------------------------
# Read events.jsonl
# -----------------------------

def read_events_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# -----------------------------
# Invariant checks (one pass)
# -----------------------------

def assert_invariants(events: List[Dict[str, Any]]) -> None:
    # V0: canonical types
    for i, e in enumerate(events):
        if not isinstance(e, dict):
            die(f"event[{i}] is not an object")
        if "tick" not in e or not isinstance(e["tick"], int):
            die(f"event[{i}] missing tick:int")
        if "type" not in e or not isinstance(e["type"], str):
            die(f"event[{i}] missing type:str")
        if "fold" not in e or not isinstance(e["fold"], str):
            die(f"event[{i}] missing fold:str")

        fold = e["fold"]
        if fold not in COMPLETE_FOLD_MAP:
            die(f"event[{i}] fold not in COMPLETE_FOLD_MAP: {fold}")

        # V1/V2: no CONTROL fold in events stream (agents never target it)
        if fold == "⟁CONTROL_FOLD⟁":
            die(f"event[{i}] illegally targets ⟁CONTROL_FOLD⟁")

        # V4: UI read-only: no writes to UI fold
        # For this golden pack, any event with fold == UI is illegal.
        if fold == "⟁UI_FOLD⟁":
            die(f"event[{i}] illegally targets ⟁UI_FOLD⟁ (projection must be derived, not written)")

        # “set” is only legal for STATE in this minimal reducer
        if e["type"] == "set" and fold != "⟁STATE_FOLD⟁":
            die(f"event[{i}] type=set must target ⟁STATE_FOLD⟁, got {fold}")

    # V6: replay determinism requires ticks non-decreasing (input order allowed but must be stable)
    ticks = [e["tick"] for e in events]
    if ticks != sorted(ticks):
        die("ticks must be non-decreasing for this golden replay pack")

    # V5: fold→lane map must be total for all folds referenced
    for e in events:
        if e["fold"] not in FOLD_TO_LANE:
            die(f"missing lane mapping for fold {e['fold']}")


# -----------------------------
# Snapshot + SVG replay (deterministic)
# -----------------------------

@dataclass
class Snapshot:
    tick: int
    state: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


def build_snapshots(events: List[Dict[str, Any]]) -> List[Snapshot]:
    ticks = sorted({e["tick"] for e in events})
    tick_to_events: Dict[int, List[Dict[str, Any]]] = {t: [] for t in ticks}
    for e in events:
        tick_to_events[e["tick"]].append(e)

    state: Dict[str, Any] = {}
    snaps: List[Snapshot] = []
    for t in ticks:
        evs = tick_to_events[t]
        # deterministic reducer: apply set events in input order
        for e in evs:
            if e["type"] == "set":
                state[e["key"]] = e["value"]
        snaps.append(Snapshot(tick=t, state=dict(state), events=evs))
    return snaps


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def text_block(x, y, lines, line_h=18, color="#e8f5ff", size=14):
    t = [
        f'<text x="{x}" y="{y}" fill="{color}" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
        f'font-size="{size}">' 
    ]
    yy = y
    for line in lines:
        t.append(f'<tspan x="{x}" y="{yy}">{esc(line)}</tspan>')
        yy += line_h
    t.append("</text>")
    return "\n".join(t)


def svg_frame(snapshot: Snapshot, width=1200, height=675) -> str:
    bg = "#060b14"
    fg = "#e8f5ff"
    accent = "#16f2aa"

    # canonical compact json for display lines (no spaces)
    state_lines = []
    for k in sorted(snapshot.state.keys()):
        v = snapshot.state[k]
        state_lines.append(f"{k}: {json.dumps(v, ensure_ascii=False, separators=(',',':'))}")

    event_lines = []
    for e in snapshot.events[:40]:
        fold = e.get("fold", "")
        typ = e.get("type", "")
        payload = e.get("payload", {})
        event_lines.append(
            f"[{typ}] {fold} {json.dumps(payload, ensure_ascii=False, separators=(',',':'))}"
        )

    meta = [
        f"tick: {snapshot.tick}",
        "replay: svg.replay.v1",
        "projection: fold→svg (read-only)",
    ]

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>
  <!-- Header -->
  <rect x="24" y="18" width="{width-48}" height="60" rx="14" fill="rgba(22,242,170,0.08)" stroke="rgba(22,242,170,0.25)"/>
  {text_block(44, 44, meta, line_h=18, color=fg, size=14)}

  <!-- Panels -->
  <rect x="24" y="96" width="{int(width*0.46)}" height="{height-120}" rx="16" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.10)"/>
  <rect x="{int(width*0.50)}" y="96" width="{width-int(width*0.50)-24}" height="{height-120}" rx="16" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.10)"/>

  <!-- Titles -->
  <text x="44" y="128" fill="{accent}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="16">STATE (⟁STATE_FOLD⟁)</text>
  <text x="{int(width*0.50)+20}" y="128" fill="{accent}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="16">EVENTS (⟁EVENTS_FOLD⟁)</text>

  <!-- Content -->
  {text_block(44, 156, state_lines[:42], line_h=18, color=fg, size=13)}
  {text_block(int(width*0.50)+20, 156, event_lines[:42], line_h=18, color=fg, size=13)}

  <!-- Provenance -->
  <metadata>
    {esc(json.dumps({"tick": snapshot.tick, "state_keys": sorted(snapshot.state.keys())}, ensure_ascii=False, separators=(',',':')))}
  </metadata>
</svg>'''


# -----------------------------
# SCX2 binary pack (deterministic example)
# -----------------------------

LANE_ID = {"DICT": 1, "FIELD": 2, "LANE": 3, "EDGE": 4, "BATCH": 5}
LANE_BIT = {"DICT": 1, "FIELD": 2, "LANE": 4, "EDGE": 8, "BATCH": 16}


def _u16(n):
    return struct.pack("<H", n)


def _u32(n):
    return struct.pack("<I", n)


def pack_dict(items: Dict[str, str]) -> bytes:
    out = bytearray()
    # canonical order
    for k in sorted(items.keys()):
        v = items[k]
        kb = k.encode("utf-8")
        vb = v.encode("utf-8")
        out += _u16(len(kb)) + kb + _u16(len(vb)) + vb
    return bytes(out)


def pack_lane(events_list: List[Tuple[int, int, bytes]]) -> bytes:
    out = bytearray()
    for tick, kind, blob in events_list:
        out += _u32(tick) + _u16(kind) + _u32(len(blob)) + blob
    return bytes(out)


def scx2_pack(lanes: Dict[str, Tuple[int, bytes]], add_crc: bool = True) -> bytes:
    bitmask = 0
    for k in lanes:
        bitmask |= LANE_BIT[k]

    header = bytearray(b"SCX2" + b"\x01" + bytes([bitmask]) + b"\x00\x00")

    body = bytearray()
    for name in ["DICT", "FIELD", "LANE", "EDGE", "BATCH"]:
        if name not in lanes:
            continue
        count, payload = lanes[name]
        body += struct.pack("<BBII", LANE_ID[name], 0, count, len(payload))
        body += payload

    data = bytes(header + body)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return data + (_u32(crc) if add_crc else _u32(0))


def build_scx2(events: List[Dict[str, Any]]) -> bytes:
    # DICT lane contains fold→lane binding (law)
    dict_payload = pack_dict(FOLD_TO_LANE)

    # LANE lane contains only EVENTS fold events
    kind_map = {"emit": 100, "set": 101}
    lane_events: List[Tuple[int, int, bytes]] = []
    for e in events:
        if e["fold"] == "⟁EVENTS_FOLD⟁":
            blob = json.dumps(e, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            lane_events.append((e["tick"], kind_map.get(e["type"], 999), blob))

    lane_payload = pack_lane(lane_events)

    return scx2_pack(
        {
            "DICT": (len(FOLD_TO_LANE), dict_payload),
            "LANE": (len(lane_events), lane_payload),
        }
    )


# -----------------------------
# Main verify (one pass)
# -----------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("events_jsonl", help="Path to events.jsonl")
    ap.add_argument("--write", action="store_true", help="Write replay_0001.svg and scx2.bin outputs")
    args = ap.parse_args()

    events = read_events_jsonl(args.events_jsonl)

    # 1) invariants
    assert_invariants(events)

    # 2) replay_0001.svg
    snaps = build_snapshots(events)
    if not snaps:
        die("no snapshots produced")
    svg1 = svg_frame(snaps[0]).encode("utf-8")
    svg1_hash = sha256_hex(svg1)
    if svg1_hash != EXPECTED_REPLAY_0001_SVG_SHA256:
        die(
            "replay_0001.svg sha256 mismatch\n"
            f"  got: {svg1_hash}\n"
            f"  exp: {EXPECTED_REPLAY_0001_SVG_SHA256}"
        )

    # 3) scx2.bin
    scx_bin = build_scx2(events)
    scx_hash = sha256_hex(scx_bin)
    if scx_hash != EXPECTED_SCX2_BIN_SHA256:
        die(
            "scx2.bin sha256 mismatch\n"
            f"  got: {scx_hash}\n"
            f"  exp: {EXPECTED_SCX2_BIN_SHA256}"
        )

    # Optional write outputs for inspection
    if args.write:
        with open("replay_0001.svg", "wb") as f:
            f.write(svg1)
        with open("scx2.bin", "wb") as f:
            f.write(scx_bin)

    print("✅ VERIFY PASS")
    print("replay_0001.svg sha256:", svg1_hash)
    print("scx2.bin        sha256:", scx_hash)


if __name__ == "__main__":
    main()
