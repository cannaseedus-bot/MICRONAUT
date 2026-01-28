"""
WebSocket research bot that consumes research intents and emits verified outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from typing import Any, Dict, List

import websockets

from policy_research import validate_research_result
from verifier import (
    apply_delta,
    render_svg,
    state_root_sha256,
    load_registry,
    index_registry,
    assert_registry_pins,
    check_event_against_registry,
)

STATE: Dict[str, Any] = {}
LAST_TICK = -1
SVG_STREAM = hashlib.sha256()
AGENTS: Dict[str, Any] = {}


def build_research_result(request: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "request": request,
        "status": "complete",
        "citations": request.get("citations", []),
        "claims": request.get("claims", []),
        "summary": request.get("summary", ""),
    }


async def handle_intent(msg: Dict[str, Any]) -> Dict[str, Any]:
    global LAST_TICK

    fel = msg.get("fel")
    if not fel:
        raise ValueError("missing fel")

    tick = int(fel.get("tick", -1))
    if tick < LAST_TICK:
        raise ValueError("tick must be monotonic")
    LAST_TICK = tick

    check_event_against_registry(fel, AGENTS)

    if fel.get("fold") != "⟁STATE_FOLD⟁":
        raise ValueError("only STATE_FOLD intents allowed")

    if fel.get("type") != "delta":
        raise ValueError("only delta intents supported")

    patch = fel.get("patch")
    if not isinstance(patch, dict):
        raise ValueError("patch required")

    if patch.get("path") == "/research/queue/-" and patch.get("op") == "add":
        request = patch.get("value", {})
        result = build_research_result(request)
        validate_research_result(result)
        apply_delta(STATE, patch)
        apply_delta(
            STATE,
            {
                "op": "replace",
                "path": f"/research/results/{request.get('id', 'unknown')}",
                "value": result,
            },
        )
    else:
        apply_delta(STATE, patch)

    svg = render_svg(tick=tick, events=[fel], state=STATE)
    svg_sha = hashlib.sha256(svg.encode("utf-8")).hexdigest()
    state_sha = state_root_sha256(STATE)

    SVG_STREAM.update(bytes.fromhex(svg_sha))
    svg_stream_sha = SVG_STREAM.hexdigest()

    return {
        "v": "micronaut.ws.v1",
        "type": "verified",
        "session": msg.get("session"),
        "tick": tick,
        "state": STATE,
        "proof": {
            "state_root_sha256": state_sha,
            "svg_sha256": svg_sha,
            "svg_stream_sha256": svg_stream_sha,
            "attestation_sha256": None,
        },
        "svg": svg,
    }


async def ws_handler(websocket: websockets.WebSocketServerProtocol) -> None:
    async for raw in websocket:
        try:
            msg = json.loads(raw)
            if msg.get("v") != "micronaut.ws.v1":
                raise ValueError("bad version")
            if msg.get("type") != "intent":
                raise ValueError("only intent accepted")

            reply = await handle_intent(msg)
            await websocket.send(json.dumps(reply, separators=(",", ":"), ensure_ascii=False))
        except Exception as exc:
            err = {
                "v": "micronaut.ws.v1",
                "type": "error",
                "session": msg.get("session") if isinstance(msg, dict) else None,
                "error": str(exc),
            }
            await websocket.send(json.dumps(err, separators=(",", ":"), ensure_ascii=False))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="docs/micronaut.registry.json")
    parser.add_argument("--policy-pin", default=None)
    parser.add_argument("--abi-pin", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    assert_registry_pins(registry, args.policy_pin, args.abi_pin)

    global AGENTS
    AGENTS = index_registry(registry)

    print(f"WS research bot on ws://{args.host}:{args.port}")
    async with websockets.serve(ws_handler, args.host, args.port, max_size=2**20):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
