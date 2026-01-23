# SCXQ2 binary packing example (DICT/FIELD/LANE/EDGE/BATCH)

This is a **minimal, deterministic binary container** you can implement anywhere.
It is not the full SCXQ2 spec—this is the example pack that proves the lane model.

## Container layout (little-endian)

```
MAGIC(4)  = "SCX2"
VER(1)    = 0x01
LANES(1)  = bitmask (DICT=1, FIELD=2, LANE=4, EDGE=8, BATCH=16)
RESV(2)

For each lane present:
  TAG(1)        lane id (1..5)
  FLAGS(1)      0x00 (reserved)
  COUNT(u32)    number of records
  BYTES(u32)    payload bytes
  PAYLOAD       lane payload
CRC32(u32) optional (can be 0 in this example)
```

## Lane payload formats (example)

### DICT payload: key/value symbols

```
repeat COUNT:
  key_len(u16) key(bytes utf-8)
  val_len(u16) val(bytes utf-8)
```

### FIELD payload: records (key → blob)

```
repeat COUNT:
  field_id(u32)
  blob_len(u32) blob(bytes)
```

### LANE payload: ordered events (tick, kind, blob)

```
repeat COUNT:
  tick(u32)
  kind(u16)
  blob_len(u32) blob(bytes)
```

### EDGE payload: relations (src, dst, kind)

```
repeat COUNT:
  src(u32) dst(u32) kind(u16)
```

### BATCH payload: ephemeral compute chunks

```
repeat COUNT:
  job_id(u32)
  blob_len(u32) blob(bytes)
```

## Reference implementation (Python)

```python
import struct, zlib
from typing import Dict, List, Tuple

LANE_ID = {"DICT": 1, "FIELD": 2, "LANE": 3, "EDGE": 4, "BATCH": 5}
LANE_BIT = {"DICT": 1, "FIELD": 2, "LANE": 4, "EDGE": 8, "BATCH": 16}

def _u16(n): return struct.pack("<H", n)
def _u32(n): return struct.pack("<I", n)

def pack_dict(items: Dict[str, str]) -> bytes:
    out = bytearray()
    for k, v in items.items():
        kb = k.encode("utf-8"); vb = v.encode("utf-8")
        out += _u16(len(kb)) + kb + _u16(len(vb)) + vb
    return bytes(out)

def pack_field(records: List[Tuple[int, bytes]]) -> bytes:
    out = bytearray()
    for fid, blob in records:
        out += _u32(fid) + _u32(len(blob)) + blob
    return bytes(out)

def pack_lane(events: List[Tuple[int, int, bytes]]) -> bytes:
    out = bytearray()
    for tick, kind, blob in events:
        out += _u32(tick) + _u16(kind) + _u32(len(blob)) + blob
    return bytes(out)

def pack_edge(edges: List[Tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for src, dst, kind in edges:
        out += _u32(src) + _u32(dst) + _u16(kind)
    return bytes(out)

def pack_batch(jobs: List[Tuple[int, bytes]]) -> bytes:
    out = bytearray()
    for job_id, blob in jobs:
        out += _u32(job_id) + _u32(len(blob)) + blob
    return bytes(out)

def scx2_pack(lanes: Dict[str, bytes], add_crc: bool = True) -> bytes:
    # lanes keys: DICT/FIELD/LANE/EDGE/BATCH
    magic = b"SCX2"
    ver = b"\x01"
    bitmask = 0
    for k in lanes:
        bitmask |= LANE_BIT[k]
    header = bytearray(magic + ver + bytes([bitmask]) + b"\x00\x00")

    body = bytearray()
    for name in ["DICT", "FIELD", "LANE", "EDGE", "BATCH"]:
        if name not in lanes:
            continue
        payload = lanes[name]
        tag = LANE_ID[name]
        flags = 0
        # "COUNT" is not directly inferable from bytes here, so store 0 if unknown in this demo.
        # In production, compute COUNT at lane-build time and pass it separately.
        count = 0
        body += struct.pack("<BBII", tag, flags, count, len(payload))
        body += payload

    data = bytes(header + body)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return data + (_u32(crc) if add_crc else _u32(0))

if __name__ == "__main__":
    lanes = {
        "DICT": pack_dict({"⟁DATA_FOLD⟁":"DICT", "⟁EVENTS_FOLD⟁":"LANE"}),
        "LANE": pack_lane([(1, 100, b'{"type":"boot"}'), (2, 101, b'{"type":"tick"}')]),
        "EDGE": pack_edge([(1, 2, 7)]),
    }
    blob = scx2_pack(lanes)
    print("bytes:", len(blob), "crc32:", blob[-4:].hex())
```
