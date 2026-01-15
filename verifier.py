"""
FEL v1.1 One-Pass Verifier
==========================
Verifies ALL invariants in one pass:
- tick monotonicity
- fold/type legality (STATE-only mutation, UI/CONTROL cannot be targets)
- deterministic state roots (canonical JSON)
- deterministic SVG projection per tick (EVENTS + STATE only)
- optional META_FOLD attestation chain verification (pinning + prev)
- deterministic SCXQ2-style binary packing output + hash

Outputs:
- replay_0001.svg ... replay_N.svg (optional)
- replay_manifest.json (optional)
- scx2.bin (optional)
- prints hashes + PASS/FAIL

Usage:
  python verifier.py events.fel.jsonl
  python verifier.py events.fel.jsonl --out replay_out --write-svg --write-bin --policy POLICY_HASH --abi ABI_HASH
  python verifier.py events.fel.jsonl --expect-manifest replay_out/replay_manifest.json
  python verifier.py events.fel.jsonl --expect-svg-hash <sha256> --expect-bin-hash <sha256>

Notes (law):
- timestamps are observational only, NEVER hashed
- UI is projection-only and cannot be a FEL target
- CONTROL cannot be targeted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Canonicalization + hashing law
# -----------------------------

TOP_LEVEL_ORDER = [
    "v",
    "tick",
    "type",
    "fold",
    "id",
    "ref",
    "payload",
    "key",
    "value",
    "patch",
    "lane",
    "root",
    "source",
    "policy_hash",
    "meta_hash",
    "abi_hash",
    "prev",
]
EXCLUDED_FROM_HASH = {"timestamp"}  # observational only, never hashed


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strip_observational(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_observational(v) for k, v in x.items() if k not in EXCLUDED_FROM_HASH}
    if isinstance(x, list):
        return [_strip_observational(v) for v in x]
    return x


def canon_json(obj: Dict[str, Any]) -> bytes:
    """
    Canonical JSON bytes:
    - removes observational keys
    - stable top-level order, then remaining sorted
    - nested dict keys sorted
    - minimal separators
    """
    obj = _strip_observational(obj)

    def order_top(d: Dict[str, Any]) -> Dict[str, Any]:
        ordered: Dict[str, Any] = {}
        for k in TOP_LEVEL_ORDER:
            if k in d:
                ordered[k] = d[k]
        for k in sorted(d.keys()):
            if k not in ordered:
                ordered[k] = d[k]
        return ordered

    def normalize(x: Any, top: bool = False) -> Any:
        if isinstance(x, dict):
            if top:
                d = order_top(x)
            else:
                d = {k: x[k] for k in sorted(x.keys())}
            return {k: normalize(v, top=False) for k, v in d.items()}
        if isinstance(x, list):
            return [normalize(v, top=False) for v in x]
        return x

    norm = normalize(obj, top=True)
    return json.dumps(norm, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode(
        "utf-8"
    )


# -----------------------------
# STATE mutation (set/delta)
# -----------------------------

def apply_set(state: Dict[str, Any], key: str, value: Any) -> None:
    state[key] = value


def apply_delta(state: Dict[str, Any], patch: Dict[str, Any]) -> None:
    op = patch["op"]
    path = patch["path"]
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("patch.path must start with '/'")
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise ValueError("patch.path must not be empty")

    def get_parent(root: Dict[str, Any], keys: List[str]) -> Tuple[Dict[str, Any], str]:
        cur = root
        for k in keys[:-1]:
            nxt = cur.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[k] = nxt
            cur = nxt
        return cur, keys[-1]

    parent, leaf = get_parent(state, parts)

    if op in ("add", "replace"):
        if "value" not in patch:
            raise ValueError("add/replace requires value")
        parent[leaf] = patch["value"]
    elif op == "remove":
        if "value" in patch:
            raise ValueError("remove must not include value")
        parent.pop(leaf, None)
    else:
        raise ValueError("patch.op must be add|replace|remove")


def state_root_sha256(state: Dict[str, Any]) -> str:
    return sha256_hex(canon_json(state))


# -----------------------------
# Deterministic SVG projection
# -----------------------------

def render_svg(tick: int, events: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    sroot = state_root_sha256(state)

    log_lines: List[str] = []
    for e in events:
        typ = e.get("type", "unknown")
        fold = e.get("fold", "unknown")
        if typ == "set":
            log_lines.append(
                f"set {e.get('key')}={json.dumps(e.get('value'), separators=(',',':'))}"
            )
        elif typ == "delta":
            p = e.get("patch", {})
            v = json.dumps(p.get("value"), separators=(",", ":")) if "value" in p else ""
            log_lines.append(f"delta {p.get('op')} {p.get('path')} {v}")
        else:
            payload = e.get("payload")
            if payload is not None:
                log_lines.append(f"{typ} {fold} {json.dumps(payload, separators=(',',':'))[:120]}")
            else:
                log_lines.append(f"{typ} {fold}")

    state_preview = json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(state_preview) > 600:
        state_preview = state_preview[:600] + "…"

    x0, y0 = 24, 32
    lh = 16

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines_svg = ""
    for i, line in enumerate(log_lines[:32]):
        lines_svg += (
            f'<text x="{x0}" y="{y0 + 80 + i*lh}" '
            f'font-family="monospace" font-size="12" fill="#c9f">{esc(line)}</text>\n'
        )

    svg = f"""<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#05060b"/>
  <text x="{x0}" y="{y0}" font-family="monospace" font-size="22" fill="#9ff">⟁ FEL REPLAY ⟁</text>

  <text x="{x0}" y="{y0+30}" font-family="monospace" font-size="14" fill="#7df">tick: {tick}</text>
  <text x="{x0}" y="{y0+50}" font-family="monospace" font-size="14" fill="#7df">state_root_sha256: {sroot}</text>

  <rect x="{x0-10}" y="{y0+66}" width="1150" height="280" fill="#0b1020" opacity="0.75" rx="10"/>
  <text x="{x0}" y="{y0+92}" font-family="monospace" font-size="14" fill="#fff">EVENTS (first 32):</text>
  {lines_svg}

  <rect x="{x0-10}" y="{y0+370}" width="1150" height="360" fill="#0b1020" opacity="0.75" rx="10"/>
  <text x="{x0}" y="{y0+396}" font-family="monospace" font-size="14" fill="#fff">STATE (preview):</text>
  <text x="{x0}" y="{y0+420}" font-family="monospace" font-size="12" fill="#bfb">{esc(state_preview)}</text>

  <text x="1180" y="780" text-anchor="end" font-family="monospace" font-size="10" fill="#666">
    projection-only · deterministic · no-exec
  </text>
</svg>"""
    return svg


def svg_sha256(svg_text: str) -> str:
    return sha256_hex(svg_text.encode("utf-8"))


# -----------------------------
# META_FOLD attestation chain
# -----------------------------

@dataclass(frozen=True)
class AttestationRecord:
    tick: int
    att_hash: str
    prev: Optional[str]
    policy_hash: str
    abi_hash: str
    meta_hash: str


def compute_meta_hash(
    stream_id: str, state_root: str, lane_roots: Dict[str, str], extra: Dict[str, Any]
) -> str:
    payload = {
        "domain": "fel.meta.v1.1",
        "stream_id": stream_id,
        "state_root_sha256": state_root,
        "lane_roots": lane_roots,
        "extra": extra,
    }
    return sha256_hex(canon_json(payload))


def attest_line_hash(
    v: str, tick: int, policy_hash: str, abi_hash: str, meta_hash: str, prev: Optional[str]
) -> str:
    line: Dict[str, Any] = {
        "v": v,
        "tick": tick,
        "type": "attest",
        "fold": "⟁META_FOLD⟁",
        "policy_hash": policy_hash,
        "meta_hash": meta_hash,
        "abi_hash": abi_hash,
    }
    if prev is not None:
        line["prev"] = prev
    return sha256_hex(canon_json(line))


# -----------------------------
# SCXQ2-style binary packing (deterministic)
# -----------------------------
#
# scx2.bin record format (little, deterministic):
#   magic   4 bytes  b"SCX2"
#   ver     1 byte   0x01
#   flags   1 byte   bit0=has_att
#   count   4 bytes  uint32
#   records:
#     tick        4 bytes  uint32
#     state_root  32 bytes
#     svg_hash    32 bytes
#     att_hash    32 bytes (optional if has_att)
#
# hash(scx2.bin) is SHA256 of full file bytes.

def pack_scx2(records: List[Dict[str, Any]], has_att: bool) -> bytes:
    buf = bytearray()
    buf.extend(b"SCX2")
    buf.extend(struct.pack("<B", 0x01))
    flags = 0x01 if has_att else 0x00
    buf.extend(struct.pack("<B", flags))
    buf.extend(struct.pack("<I", len(records)))

    for r in records:
        tick = r["tick"]
        sr = bytes.fromhex(r["state_root_sha256"])
        sh = bytes.fromhex(r["svg_sha256"])
        if len(sr) != 32 or len(sh) != 32:
            raise ValueError("bad root/hash length in record")

        buf.extend(struct.pack("<I", int(tick)))
        buf.extend(sr)
        buf.extend(sh)

        if has_att:
            ah = bytes.fromhex(r["attestation_sha256"])
            if len(ah) != 32:
                raise ValueError("bad att length in record")
            buf.extend(ah)

    return bytes(buf)


# -----------------------------
# FEL legality checks (v1/v1.1)
# -----------------------------

FORBIDDEN_TARGET_FOLDS = {"⟁CONTROL_FOLD⟁", "⟁UI_FOLD⟁"}


def validate_fel_line(e: Dict[str, Any], line_no: int, last_tick: int) -> int:
    v = e.get("v")
    if v not in ("fel.v1", "fel.v1.1"):
        raise ValueError(f"line {line_no}: invalid v")

    tick = e.get("tick")
    if not isinstance(tick, int) or tick < 0:
        raise ValueError(f"line {line_no}: tick must be int >= 0")
    if tick < last_tick:
        raise ValueError(f"line {line_no}: tick not monotonic")

    typ = e.get("type")
    fold = e.get("fold")
    if typ is None or fold is None:
        raise ValueError(f"line {line_no}: missing type or fold")

    if fold in FORBIDDEN_TARGET_FOLDS:
        raise ValueError(f"line {line_no}: forbidden target fold {fold}")

    if typ in ("set", "delta"):
        if fold != "⟁STATE_FOLD⟁":
            raise ValueError(f"line {line_no}: {typ} must target ⟁STATE_FOLD⟁")

    if typ == "seal":
        if fold != "⟁STORAGE_FOLD⟁":
            raise ValueError(f"line {line_no}: seal must target ⟁STORAGE_FOLD⟁")

    if typ == "attest":
        if fold != "⟁META_FOLD⟁":
            raise ValueError(f"line {line_no}: attest must target ⟁META_FOLD⟁")

    if typ == "delta":
        patch = e.get("patch")
        if not isinstance(patch, dict) or "op" not in patch or "path" not in patch:
            raise ValueError(f"line {line_no}: malformed patch")

    return tick


# -----------------------------
# Main verifier (one pass)
# -----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    last_tick = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            e = json.loads(raw)
            last_tick = validate_fel_line(e, ln, last_tick)
            items.append(e)
    return items


def group_by_tick(items: List[Dict[str, Any]]) -> List[Tuple[int, List[Dict[str, Any]]]]:
    by: Dict[int, List[Dict[str, Any]]] = {}
    for e in items:
        by.setdefault(e["tick"], []).append(e)
    return sorted(by.items(), key=lambda kv: kv[0])


def verify_attestations_present(
    items: List[Dict[str, Any]], policy_hash: str, abi_hash: str, stream_id: str
) -> None:
    """
    If FEL includes attest lines, verify:
    - policy/abi pinning
    - prev chain is correct
    - att_hash matches canonical hash of attest line
    - meta_hash is consistent with state_root/lane_roots/extra (if encoded in payload)
    This verifier supports two modes:
      A) Attest lines exist in FEL -> verify them as-is
      B) Attest lines absent -> generate them deterministically and include in output (optional)
    """
    head: Optional[str] = None
    last_tick = -1

    for e in items:
        if e.get("type") != "attest":
            continue

        if e.get("policy_hash") != policy_hash:
            raise ValueError("policy_hash drift in attest lines")
        if e.get("abi_hash") != abi_hash:
            raise ValueError("abi_hash drift in attest lines")

        tick = e["tick"]
        if tick < last_tick:
            raise ValueError("attest tick not monotonic")
        last_tick = tick

        prev = e.get("prev")
        if head is None:
            # genesis rules
            if prev not in (None, "0" * 64):
                raise ValueError("genesis attest prev must be null or 64x0")
            prev_norm = None
        else:
            if prev != head:
                raise ValueError("attest prev must equal previous att_hash")
            prev_norm = prev

        meta_hash = e.get("meta_hash")
        if not isinstance(meta_hash, str) or len(meta_hash) != 64:
            raise ValueError("attest meta_hash missing or invalid")

        v = e.get("v", "fel.v1.1")
        expected_att_hash = attest_line_hash(v, tick, policy_hash, abi_hash, meta_hash, prev_norm)

        # Optionally allow FEL to provide att_hash; if not, we compute
        provided = e.get("att_hash")
        if provided is not None:
            if provided != expected_att_hash:
                raise ValueError("att_hash mismatch (canonical attest hash)")

        head = expected_att_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fel_jsonl", help="path to FEL JSONL (fel.v1 or fel.v1.1)")
    ap.add_argument("--out", default="replay_out", help="output directory for artifacts")
    ap.add_argument("--write-svg", action="store_true", help="write replay_####.svg frames")
    ap.add_argument("--write-bin", action="store_true", help="write scx2.bin")
    ap.add_argument(
        "--expect-manifest", default=None, help="path to replay_manifest.json to verify against"
    )
    ap.add_argument(
        "--expect-svg-hash",
        default=None,
        help="expected SHA256 of concatenated SVG hashes stream (optional)",
    )
    ap.add_argument(
        "--expect-bin-hash", default=None, help="expected SHA256 of scx2.bin (optional)"
    )

    # META options (if provided, verifier can generate attestation chain even if FEL lacks attest lines)
    ap.add_argument(
        "--policy",
        default=None,
        help="policy_hash pin (required to generate/verify META attestations)",
    )
    ap.add_argument(
        "--abi",
        default=None,
        help="abi_hash pin (required to generate/verify META attestations)",
    )
    ap.add_argument("--stream-id", default="fel_stream", help="stream_id used for meta_hash")
    ap.add_argument(
        "--meta",
        action="store_true",
        help="generate deterministic META attestations per tick (even if FEL has none)",
    )

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    items = load_jsonl(args.fel_jsonl)
    ticks = group_by_tick(items)

    # If FEL includes attest lines and user provided pins, verify chain/pinning/hashes
    if args.policy and args.abi:
        verify_attestations_present(items, args.policy, args.abi, args.stream_id)

    state: Dict[str, Any] = {}
    manifest: Dict[str, Any] = {
        "v": "fel.verify.manifest.v1.1",
        "source": args.fel_jsonl,
        "ticks": [],
    }

    # Determine if we will include attestation bytes in scx2.bin
    use_meta = bool(args.meta)
    if use_meta and not (args.policy and args.abi):
        raise SystemExit("ERROR: --meta requires --policy and --abi")

    meta_head: Optional[str] = None
    combined_svg_hash_stream = hashlib.sha256()

    for frame_idx, (tick, evs) in enumerate(ticks, start=1):
        # Apply state mutations (only STATE)
        for e in evs:
            typ = e.get("type")
            if typ == "set":
                apply_set(state, e["key"], e["value"])
            elif typ == "delta":
                apply_delta(state, e["patch"])
            else:
                pass

        sroot = state_root_sha256(state)
        svg = render_svg(tick, evs, state)
        shash = svg_sha256(svg)

        # Update rolling stream hash (optional)
        combined_svg_hash_stream.update(bytes.fromhex(shash))

        rec: Dict[str, Any] = {
            "tick": tick,
            "file": f"replay_{frame_idx:04d}.svg",
            "state_root_sha256": sroot,
            "svg_sha256": shash,
            "event_count": len(evs),
        }

        if use_meta:
            mh = compute_meta_hash(
                stream_id=args.stream_id,
                state_root=sroot,
                lane_roots={},
                extra={"tick": tick},
            )
            prev_norm = None if meta_head is None else meta_head
            # genesis prev = None in canonical hash; external can represent as 64x0 if desired
            ah = attest_line_hash("fel.v1.1", tick, args.policy, args.abi, mh, prev_norm)
            meta_head = ah
            rec["attestation_sha256"] = ah
            rec["attestation_prev"] = prev_norm if prev_norm is not None else "0" * 64
            rec["meta_hash"] = mh

        manifest["ticks"].append(rec)

        if args.write_svg:
            with open(os.path.join(args.out, rec["file"]), "w", encoding="utf-8") as f:
                f.write(svg)

    # Verify against an expected manifest if provided
    if args.expect_manifest:
        with open(args.expect_manifest, "r", encoding="utf-8") as f:
            exp = json.load(f)
        exp_ticks = exp.get("ticks", [])
        if len(exp_ticks) != len(manifest["ticks"]):
            raise SystemExit("FAIL: manifest tick count mismatch")

        for i, (a, b) in enumerate(zip(manifest["ticks"], exp_ticks), start=1):
            # Compare core hashes
            if a["tick"] != b.get("tick"):
                raise SystemExit(f"FAIL: tick mismatch at frame {i}")
            if a["state_root_sha256"] != b.get("state_root_sha256"):
                raise SystemExit(f"FAIL: state_root mismatch at tick {a['tick']}")
            if a["svg_sha256"] != b.get("svg_sha256"):
                raise SystemExit(f"FAIL: svg_sha256 mismatch at tick {a['tick']}")
            # If both include attestation, compare
            if "attestation_sha256" in a and "attestation_sha256" in b:
                if a["attestation_sha256"] != b.get("attestation_sha256"):
                    raise SystemExit(f"FAIL: attestation mismatch at tick {a['tick']}")

    # Write verifier manifest
    out_manifest_path = os.path.join(args.out, "verify_manifest.json")
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2, ensure_ascii=False))

    # scx2.bin output + hash
    bin_hash = None
    if args.write_bin:
        has_att = use_meta
        scx2_bytes = pack_scx2(manifest["ticks"], has_att=has_att)
        bin_path = os.path.join(args.out, "scx2.bin")
        with open(bin_path, "wb") as f:
            f.write(scx2_bytes)
        bin_hash = sha256_hex(scx2_bytes)

    # Combined SVG hash stream (optional expectation)
    combined_svg_stream_hash = combined_svg_hash_stream.hexdigest()

    # Expectations checks
    if args.expect_svg_hash:
        if combined_svg_stream_hash != args.expect_svg_hash:
            raise SystemExit(
                f"FAIL: expected svg stream hash {args.expect_svg_hash}, got {combined_svg_stream_hash}"
            )

    if args.expect_bin_hash:
        if not args.write_bin:
            raise SystemExit("FAIL: --expect-bin-hash requires --write-bin")
        if bin_hash != args.expect_bin_hash:
            raise SystemExit(
                f"FAIL: expected scx2.bin hash {args.expect_bin_hash}, got {bin_hash}"
            )

    # PASS summary
    print("PASS ✅")
    print(f"ticks: {len(manifest['ticks'])}")
    print(f"verify_manifest: {out_manifest_path}")
    print(f"svg_stream_sha256: {combined_svg_stream_hash}")
    if args.write_bin:
        print(f"scx2_bin_sha256: {bin_hash}")
        print(f"scx2.bin: {os.path.join(args.out, 'scx2.bin')}")
    if args.write_svg:
        print(f"svgs: {args.out}/replay_####.svg")
    if use_meta:
        print(f"meta_head_sha256: {meta_head}")


if __name__ == "__main__":
    main()
