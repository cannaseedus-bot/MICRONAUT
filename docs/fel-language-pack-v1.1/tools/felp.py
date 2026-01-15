import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Snapshot:
    tick: int
    state: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


def read_events_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def apply_delta(state: Dict[str, Any], patch: Dict[str, Any]) -> None:
    op = patch.get("op")
    path = patch.get("path")
    value = patch.get("value")

    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("patch.path must start with '/'")

    parts = [segment for segment in path.split("/")[1:] if segment]
    if not parts:
        raise ValueError("patch.path must not be empty")

    cursor: Any = state
    for key in parts[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError("patch path traversed non-object")
        cursor = cursor.setdefault(key, {})

    last = parts[-1]
    if not isinstance(cursor, dict):
        raise ValueError("patch path traversed non-object")

    if op in {"add", "replace"}:
        cursor[last] = value
    elif op == "remove":
        cursor.pop(last, None)
    else:
        raise ValueError("patch.op must be add|replace|remove")


def build_snapshots(events: List[Dict[str, Any]]) -> List[Snapshot]:
    ticks = sorted({int(e.get("tick", 0)) for e in events})
    tick_to_events = {t: [] for t in ticks}
    for event in events:
        tick_to_events[int(event.get("tick", 0))].append(event)

    state: Dict[str, Any] = {}
    snaps: List[Snapshot] = []
    for tick in ticks:
        evs = tick_to_events[tick]
        for event in evs:
            if event.get("type") == "set":
                state[event["key"]] = event.get("value")
            elif event.get("type") == "delta":
                apply_delta(state, event.get("patch", {}))
        snaps.append(Snapshot(tick=tick, state=dict(state), events=evs))
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

    state_lines = []
    for key in sorted(snapshot.state.keys()):
        value = snapshot.state[key]
        state_lines.append(f"{key}: {json.dumps(value, ensure_ascii=False, separators=(',',':'))}")

    event_lines = []
    for event in snapshot.events[:40]:
        fold = event.get("fold", "")
        event_type = event.get("type", "")
        payload = event.get("payload", {})
        event_lines.append(
            f"[{event_type}] {fold} {json.dumps(payload, ensure_ascii=False, separators=(',',':'))}"
        )

    meta = [
        f"tick: {snapshot.tick}",
        "replay: svg.replay.v1",
        "projection: fold→svg (read-only)",
    ]

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>
  <rect x="24" y="18" width="{width-48}" height="60" rx="14" fill="rgba(22,242,170,0.08)" stroke="rgba(22,242,170,0.25)"/>
  {text_block(44, 44, meta, line_h=18, color=fg, size=14)}

  <rect x="24" y="96" width="{int(width*0.46)}" height="{height-120}" rx="16" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.10)"/>
  <rect x="{int(width*0.50)}" y="96" width="{width-int(width*0.50)-24}" height="{height-120}" rx="16" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.10)"/>

  <text x="44" y="128" fill="{accent}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="16">STATE (⟁STATE_FOLD⟁)</text>
  <text x="{int(width*0.50)+20}" y="128" fill="{accent}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="16">EVENTS (⟁EVENTS_FOLD⟁)</text>

  {text_block(44, 156, state_lines[:42], line_h=18, color=fg, size=13)}
  {text_block(int(width*0.50)+20, 156, event_lines[:42], line_h=18, color=fg, size=13)}

  <metadata>
    {esc(json.dumps({"tick": snapshot.tick, "state_keys": sorted(snapshot.state.keys())}, ensure_ascii=False, separators=(',',':')))}
  </metadata>
</svg>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events_jsonl")
    ap.add_argument("--out", default="replay_out")
    args = ap.parse_args()

    events = read_events_jsonl(args.events_jsonl)
    snaps = build_snapshots(events)

    os.makedirs(args.out, exist_ok=True)
    for i, snap in enumerate(snaps, 1):
        path = os.path.join(args.out, f"replay_{i:04d}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_frame(snap))

    print(f"Wrote {len(snaps)} SVG frames → {args.out}/")


if __name__ == "__main__":
    main()
