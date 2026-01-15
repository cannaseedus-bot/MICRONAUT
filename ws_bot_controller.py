"""
WebSocket bot controller for CSS Micronaut intents.

Receives FEL intents, validates them against the registry, applies lawful
STATE updates, and emits verified state + SVG projections.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from typing import Any, Dict

import websockets

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

    typ = fel.get("type")
    if typ == "set":
        key = fel["key"]
        STATE[key] = fel["value"]
    elif typ == "delta":
        apply_delta(STATE, fel["patch"])
    else:
        raise ValueError("unsupported intent type")

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
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    policy_pin = args.policy_pin
    abi_pin = args.abi_pin
    assert_registry_pins(registry, policy_pin, abi_pin)

    global AGENTS
    AGENTS = index_registry(registry)

    print(f"WS bot controller on ws://{args.host}:{args.port}")
    async with websockets.serve(ws_handler, args.host, args.port, max_size=2**20):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
