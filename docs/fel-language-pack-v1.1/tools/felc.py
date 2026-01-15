import argparse
import json
import struct
import zlib
from collections import OrderedDict

# --- Fold→Lane law (frozen) ---
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

CANON_KEY_ORDER = [
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


def sort_nested(obj):
    if isinstance(obj, dict):
        return {k: sort_nested(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [sort_nested(x) for x in obj]
    return obj


def canon_fel_line(event):
    ordered = OrderedDict()
    for key in CANON_KEY_ORDER:
        if key in event:
            ordered[key] = sort_nested(event[key])
    extra = sorted([key for key in event.keys() if key not in ordered])
    for key in extra:
        ordered[key] = sort_nested(event[key])
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def read_events_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


LANE_ID = {"DICT": 1, "FIELD": 2, "LANE": 3, "EDGE": 4, "BATCH": 5}
LANE_BIT = {"DICT": 1, "FIELD": 2, "LANE": 4, "EDGE": 8, "BATCH": 16}


def _u16(n):
    return struct.pack("<H", n)


def _u32(n):
    return struct.pack("<I", n)


def pack_dict(items):
    out = bytearray()
    for key in sorted(items.keys()):
        value = items[key]
        kb = key.encode("utf-8")
        vb = value.encode("utf-8")
        out += _u16(len(kb)) + kb + _u16(len(vb)) + vb
    return bytes(out)


def pack_lane(events_list):
    out = bytearray()
    for tick, kind, blob in events_list:
        out += _u32(tick) + _u16(kind) + _u32(len(blob)) + blob
    return bytes(out)


def scx2_pack(lanes, add_crc=True):
    bitmask = 0
    for name in lanes:
        bitmask |= LANE_BIT[name]

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


def compile_fel_to_scx2(events):
    dict_payload = pack_dict(FOLD_TO_LANE)

    kind_map = {"emit": 100, "set": 101, "delta": 102, "seal": 200, "attest": 250}
    lane_events = []
    for event in events:
        if event.get("fold") == "⟁EVENTS_FOLD⟁":
            blob = canon_fel_line(event).encode("utf-8")
            lane_events.append((int(event["tick"]), kind_map.get(event.get("type", ""), 999), blob))

    lane_payload = pack_lane(lane_events)

    return scx2_pack(
        {
            "DICT": (len(FOLD_TO_LANE), dict_payload),
            "LANE": (len(lane_events), lane_payload),
        },
        add_crc=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events_jsonl")
    ap.add_argument("--out", default="scx2.bin")
    args = ap.parse_args()

    events = read_events_jsonl(args.events_jsonl)
    blob = compile_fel_to_scx2(events)

    with open(args.out, "wb") as f:
        f.write(blob)

    print(f"Wrote {len(blob)} bytes → {args.out}")


if __name__ == "__main__":
    main()
