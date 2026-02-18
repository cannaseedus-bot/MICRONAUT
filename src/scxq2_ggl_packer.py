"""
SCXQ2 GGL Frame Packer
======================
Python implementation of the GGL SCXQ2 binary frame layout defined in
docs/ggl/scxq2_ggl_frames_layout_v1.js.

Frame types:
  HDR    (lane 1) — stream header: spec_v, policy_hash, dict
  PACK   (lane 2) — model + tensor binding
  SEED   (lane 3) — optional seed context
  START  (lane 4) — infer.start
  END    (lane 5) — infer.end with proof
  ERR    (lane 6) — infer.error
  ENDSTREAM (lane 7) — stream terminator

All multi-byte integers are little-endian. Magic is b'GGL1'.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any, Dict, List, Optional


MAGIC = b"GGL1"
VERSION = 0x01

# Lane IDs
LANE_HDR = 1
LANE_PACK = 2
LANE_SEED = 3
LANE_START = 4
LANE_END = 5
LANE_ERR = 6
LANE_ENDSTREAM = 7


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _encode_str(s: str) -> bytes:
    """Length-prefixed UTF-8 string (u32 LE + bytes)."""
    encoded = s.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _encode_bytes_blob(data: bytes) -> bytes:
    """Length-prefixed bytes blob (u32 LE + bytes)."""
    return struct.pack("<I", len(data)) + data


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------

def build_hdr_frame(
    spec_v: str = "ggl.frames.v1",
    policy_hash: str = "0" * 64,
    dict_entries: Optional[List[str]] = None,
    t_ms: int = 0,
) -> bytes:
    """Build a HDR (lane 1) frame."""
    dict_payload = b""
    if dict_entries:
        dict_entries_encoded = [_encode_str(e) for e in dict_entries]
        dict_payload = b"".join(dict_entries_encoded)

    buf = bytearray()
    buf.extend(MAGIC)
    buf.extend(struct.pack("<B", VERSION))
    buf.extend(struct.pack("<B", LANE_HDR))
    buf.extend(struct.pack("<I", t_ms))                    # t_ms
    buf.extend(_encode_str(spec_v))                        # spec_v
    buf.extend(_encode_str(policy_hash))                   # policy_hash
    buf.extend(struct.pack("<I", len(dict_entries or []))) # dict_count
    buf.extend(_encode_bytes_blob(dict_payload))           # dict_bytes
    return bytes(buf)


def build_pack_frame(
    pack_hash: str,
    model_id: str,
    tensor_hash: str,
    glyph_hash: str,
    abi_hash: str,
    t_ms: int = 0,
) -> bytes:
    """Build a PACK (lane 2) frame — binds model + tensor + glyph + abi."""
    buf = bytearray()
    buf.extend(MAGIC)
    buf.extend(struct.pack("<B", VERSION))
    buf.extend(struct.pack("<B", LANE_PACK))
    buf.extend(struct.pack("<I", t_ms))
    buf.extend(_encode_str(pack_hash))
    buf.extend(_encode_str(model_id))
    buf.extend(_encode_str(tensor_hash))
    buf.extend(_encode_str(glyph_hash))
    buf.extend(_encode_str(abi_hash))
    return bytes(buf)


def build_start_frame(
    call_id: str,
    prompt_hash: str,
    t_ms: int = 0,
) -> bytes:
    """Build a START (lane 4) frame — infer.start."""
    buf = bytearray()
    buf.extend(MAGIC)
    buf.extend(struct.pack("<B", VERSION))
    buf.extend(struct.pack("<B", LANE_START))
    buf.extend(struct.pack("<I", t_ms))
    buf.extend(_encode_str(call_id))
    buf.extend(_encode_str(prompt_hash))
    return bytes(buf)


def build_end_frame(
    call_id: str,
    output_hash: str,
    proof_hash: str,
    token_count: int = 0,
    t_ms: int = 0,
) -> bytes:
    """Build an END (lane 5) frame — infer.end with proof."""
    buf = bytearray()
    buf.extend(MAGIC)
    buf.extend(struct.pack("<B", VERSION))
    buf.extend(struct.pack("<B", LANE_END))
    buf.extend(struct.pack("<I", t_ms))
    buf.extend(_encode_str(call_id))
    buf.extend(_encode_str(output_hash))
    buf.extend(_encode_str(proof_hash))
    buf.extend(struct.pack("<I", token_count))
    return bytes(buf)


def build_err_frame(
    call_id: str,
    error_msg: str,
    t_ms: int = 0,
) -> bytes:
    """Build an ERR (lane 6) frame — infer.error."""
    buf = bytearray()
    buf.extend(MAGIC)
    buf.extend(struct.pack("<B", VERSION))
    buf.extend(struct.pack("<B", LANE_ERR))
    buf.extend(struct.pack("<I", t_ms))
    buf.extend(_encode_str(call_id))
    buf.extend(_encode_str(error_msg))
    return bytes(buf)


def build_endstream_frame(
    stream_hash: str,
    frame_count: int,
    t_ms: int = 0,
) -> bytes:
    """Build an ENDSTREAM (lane 7) frame — stream terminator."""
    buf = bytearray()
    buf.extend(MAGIC)
    buf.extend(struct.pack("<B", VERSION))
    buf.extend(struct.pack("<B", LANE_ENDSTREAM))
    buf.extend(struct.pack("<I", t_ms))
    buf.extend(_encode_str(stream_hash))
    buf.extend(struct.pack("<I", frame_count))
    return bytes(buf)


# ---------------------------------------------------------------------------
# High-level packer
# ---------------------------------------------------------------------------

class SCXQ2GGLPacker:
    """Pack a GGL inference session into a binary SCXQ2-GGL frame stream."""

    def pack_ggl_session(
        self,
        prompt: str,
        ggl_code: str,
        oracle_result: Dict[str, Any],
        emitted_nodes: List[str],
        proof_hash: str,
        model_id: str = "ggl.oracle.v1",
        policy_hash: str = "0" * 64,
        abi_hash: str = "0" * 64,
    ) -> bytes:
        """Pack a complete GGL inference session into binary frames.

        Returns the raw bytes of the frame stream, which can be written to
        a .bin file or appended to a SCXQ2 BATCH lane.
        """
        prompt_hash = _sha256_hex(prompt.encode("utf-8"))
        ggl_hash = _sha256_hex(ggl_code.encode("utf-8"))
        output_obj = {"emitted_nodes": emitted_nodes, "oracle": oracle_result}
        output_hash = _sha256_hex(_canonical_json(output_obj).encode("utf-8"))
        call_id = _sha256_hex(_canonical_json({"prompt_hash": prompt_hash, "ggl_hash": ggl_hash}).encode("utf-8"))[:16]
        tensor_hash = ggl_hash
        glyph_hash = _sha256_hex(_canonical_json(emitted_nodes).encode("utf-8"))

        frames: List[bytes] = []

        # HDR frame
        dict_entries = [model_id, "ggl.frames.v1", prompt[:64]]
        frames.append(build_hdr_frame(
            spec_v="ggl.frames.v1",
            policy_hash=policy_hash,
            dict_entries=dict_entries,
        ))

        # PACK frame
        frames.append(build_pack_frame(
            pack_hash=_sha256_hex(_canonical_json({
                "model": model_id, "tensor": tensor_hash,
                "glyph": glyph_hash, "abi": abi_hash,
            }).encode("utf-8")),
            model_id=model_id,
            tensor_hash=tensor_hash,
            glyph_hash=glyph_hash,
            abi_hash=abi_hash,
        ))

        # START frame
        frames.append(build_start_frame(call_id=call_id, prompt_hash=prompt_hash))

        # END frame (or ERR if oracle invalid)
        if oracle_result.get("valid", False):
            frames.append(build_end_frame(
                call_id=call_id,
                output_hash=output_hash,
                proof_hash=proof_hash,
                token_count=len(emitted_nodes),
            ))
        else:
            err_msg = "; ".join(oracle_result.get("errors", ["oracle rejected"]))
            frames.append(build_err_frame(call_id=call_id, error_msg=err_msg))

        # ENDSTREAM frame
        stream_content = b"".join(frames)
        stream_hash = _sha256_hex(stream_content)
        frames.append(build_endstream_frame(stream_hash=stream_hash, frame_count=len(frames)))

        return b"".join(frames)
