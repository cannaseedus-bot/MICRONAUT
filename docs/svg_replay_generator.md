# SVG replay generator (events.jsonl → replay_0001.svg…replay_N.svg)

This is a single-file Python tool.
Assumes each JSONL line looks like:

```json
{"tick": 1, "type":"emit", "fold":"⟁EVENTS_FOLD⟁", "payload":{"msg":"boot"}}
```

## `svg_replay.py`

```python
import json, os
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

def build_snapshots(events: List[Dict[str, Any]]) -> List[Snapshot]:
    # Minimal deterministic reducer:
    # - group by tick
    # - apply "set" events to state (if present)
    # - collect events list
    ticks = sorted({int(e.get("tick", 0)) for e in events})
    tick_to_events: Dict[int, List[Dict[str, Any]]] = {t: [] for t in ticks}
    for e in events:
        t = int(e.get("tick", 0))
        tick_to_events.setdefault(t, []).append(e)

    state: Dict[str, Any] = {}
    snaps: List[Snapshot] = []

    for t in ticks:
        evs = tick_to_events.get(t, [])
        # Apply state updates deterministically in input order
        for e in evs:
            if e.get("type") == "set":
                key = e.get("key")
                state[key] = e.get("value")
        snaps.append(Snapshot(tick=t, state=dict(state), events=evs))
    return snaps

def esc(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
             .replace('"',"&quot;").replace("'","&apos;"))

def svg_frame(snapshot: Snapshot, width=1200, height=675) -> str:
    # A clean replay layout:
    # - left: fold/state panel
    # - right: event log panel
    # - top: provenance metadata
    bg = "#060b14"
    fg = "#e8f5ff"
    accent = "#16f2aa"

    # Render state lines
    state_lines = []
    for k in sorted(snapshot.state.keys()):
        v = snapshot.state[k]
        state_lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")

    # Render event lines
    event_lines = []
    for e in snapshot.events[:40]:
        fold = e.get("fold","")
        typ = e.get("type","")
        payload = e.get("payload", {})
        event_lines.append(f"[{typ}] {fold} {json.dumps(payload, ensure_ascii=False)}")

    # Simple text layout
    def text_block(x, y, lines, line_h=18, color=fg, size=14):
        t = [f'<text x="{x}" y="{y}" fill="{color}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="{size}">']
        yy = y
        for line in lines:
            t.append(f'<tspan x="{x}" y="{yy}">{esc(line)}</tspan>')
            yy += line_h
        t.append("</text>")
        return "\n".join(t)

    meta = [
        f"tick: {snapshot.tick}",
        "replay: svg.replay.v1",
        "projection: fold→svg (read-only)",
    ]

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
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
    {esc(json.dumps({"tick": snapshot.tick, "state_keys": sorted(snapshot.state.keys())}, ensure_ascii=False))}
  </metadata>
</svg>'''
    return svg

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("events_jsonl", help="Path to events.jsonl")
    ap.add_argument("--out", default="replay_out", help="Output directory")
    args = ap.parse_args()

    events = read_events_jsonl(args.events_jsonl)
    snaps = build_snapshots(events)

    os.makedirs(args.out, exist_ok=True)
    for i, snap in enumerate(snaps, 1):
        path = os.path.join(args.out, f"replay_{i:04d}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_frame(snap))
    print(f"Wrote {len(snaps)} SVG frames to {args.out}/")

if __name__ == "__main__":
    main()
```

Run:

```bash
python svg_replay.py events.jsonl --out replay
```
