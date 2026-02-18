"""
CM-1 Parser — Control Micronaut 1
===================================
Implements the reference state machine from docs/control_micronaut_1.md §11.1.

Enforces invariants CM1-S1 through CM1-S5 and the projection invariant.

Usage:
    from cm1_parser import CM1Parser, strip_cm1, verify_projection_invariant

    parser = CM1Parser()
    result = parser.parse(b"\\x01\\x02hello\\x03\\x04")
    stripped = strip_cm1(b"\\x01\\x02hello\\x03\\x04")
    verify_projection_invariant(original_stream, stripped_stream)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Control character constants
# ---------------------------------------------------------------------------

NUL = 0x00  # Non-observable
SOH = 0x01  # @control.header.begin
STX = 0x02  # @control.body.begin
ETX = 0x03  # @control.body.end
EOT = 0x04  # @control.transmission.end
ENQ = 0x05  # @control.query
ACK = 0x06  # @control.ack
BEL = 0x07  # @control.attention
SO  = 0x0E  # @control.scope.push
SI  = 0x0F  # @control.scope.pop
DLE = 0x10  # @control.literal.escape
NAK = 0x15  # @control.nak
ESC = 0x1B  # @control.mode.switch
FS  = 0x1C  # @control.file.sep
GS  = 0x1D  # @control.group.sep
RS  = 0x1E  # @control.record.sep
US  = 0x1F  # @control.unit.sep

# Forbidden bytes (hard ban per §4.3)
FORBIDDEN = {0x08, 0x0B, 0x0C, 0x18, 0x1A}

# All CM-1 control bytes (non-content)
CM1_BYTES = {NUL, SOH, STX, ETX, EOT, ENQ, ACK, BEL, SO, SI, DLE, NAK, ESC, FS, GS, RS, US}

# Separator set
SEPARATORS = {FS, GS, RS, US}

# XCFE canonical mapping
XCFE_MAP: Dict[int, str] = {
    NUL: "@control.null",
    SOH: "@control.header.begin",
    STX: "@control.body.begin",
    ETX: "@control.body.end",
    EOT: "@control.transmission.end",
    ENQ: "@control.query",
    ACK: "@control.ack",
    BEL: "@control.attention",
    SO:  "@control.scope.push",
    SI:  "@control.scope.pop",
    DLE: "@control.literal.escape",
    NAK: "@control.nak",
    ESC: "@control.mode.switch",
    FS:  "@control.file.sep",
    GS:  "@control.group.sep",
    RS:  "@control.record.sep",
    US:  "@control.unit.sep",
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class CM1Parser:
    """State machine parser for CM-1 control streams.

    Accepts bytes (or bytearray). Returns a result dict describing the parsed
    stream, the phase state, and any emitted events.

    Raises ValueError on invariant violations.
    """

    def parse(self, stream: bytes) -> Dict[str, Any]:
        """Parse *stream* and enforce all CM1-S invariants.

        Returns:
            {
                "phase": str | None,
                "scope": list,
                "mode": str,
                "literal": bool,
                "events": list[dict],
                "passthrough": bytes,
            }
        """
        state: Dict[str, Any] = {
            "phase": None,
            "scope": [],
            "mode": "normal",
            "literal": False,
        }
        events: List[Dict[str, Any]] = []
        passthrough = bytearray()

        stx_count = 0     # CM1-S1: count unmatched STX
        sep_context: Optional[int] = None  # CM1-S3: current separator level

        for i, byte in enumerate(stream):
            # Forbidden byte check (always, regardless of literal mode)
            if byte in FORBIDDEN:
                raise ValueError(
                    f"CM1 invariant violated: forbidden byte 0x{byte:02X} (BS/VT/FF/CAN/SUB) at position {i}"
                )

            # DLE literal-escape pass-through (CM1-S4: ESC cannot appear inside DLE region)
            if state["literal"] and byte != DLE:
                if byte == ESC:
                    raise ValueError(
                        f"CM1-S4 violated: ESC (0x1B) inside DLE-escaped region at position {i}"
                    )
                passthrough.extend(bytes([byte]))
                continue

            # Dispatch on control byte
            if byte == NUL:
                # CM1-S5: non-observable; do not emit
                continue

            elif byte == SOH:
                state["phase"] = "header"
                events.append({"byte": byte, "xcfe_mapping": XCFE_MAP[byte], "phase": "header"})

            elif byte == STX:
                if state["phase"] != "header":
                    raise ValueError(
                        f"CM1-S1 violated: STX at position {i} requires prior SOH (phase='{state['phase']}')"
                    )
                state["phase"] = "body"
                stx_count += 1
                events.append({"byte": byte, "xcfe_mapping": XCFE_MAP[byte], "phase": "body"})

            elif byte == ETX:
                if state["phase"] != "body":
                    raise ValueError(
                        f"CM1-S1 violated: ETX at position {i} without matching STX (phase='{state['phase']}')"
                    )
                state["phase"] = "closing"
                stx_count -= 1
                events.append({"byte": byte, "xcfe_mapping": XCFE_MAP[byte], "phase": "closing"})

            elif byte == EOT:
                state["phase"] = "collapsed"
                events.append({"byte": byte, "xcfe_mapping": XCFE_MAP[byte], "phase": "collapsed"})

            elif byte == SO:
                state["scope"].append({"mode": state["mode"], "phase": state["phase"]})
                events.append({
                    "byte": byte, "xcfe_mapping": XCFE_MAP[byte],
                    "scope_depth": len(state["scope"]),
                })

            elif byte == SI:
                # CM1-S2: SO/SI must be balanced
                if not state["scope"]:
                    raise ValueError(
                        f"CM1-S2 violated: SI (scope pop) at position {i} with empty scope stack"
                    )
                state["scope"].pop()
                events.append({
                    "byte": byte, "xcfe_mapping": XCFE_MAP[byte],
                    "scope_depth": len(state["scope"]),
                })

            elif byte == ESC:
                # CM1-S4: cannot appear inside DLE region (handled above for literal=True)
                modes = ["normal", "grammar", "parser", "restricted"]
                idx = modes.index(state["mode"]) if state["mode"] in modes else 0
                state["mode"] = modes[(idx + 1) % len(modes)]
                events.append({"byte": byte, "xcfe_mapping": XCFE_MAP[byte], "mode": state["mode"]})

            elif byte == DLE:
                state["literal"] = not state["literal"]
                events.append({"byte": byte, "xcfe_mapping": XCFE_MAP[byte], "literal": state["literal"]})

            elif byte in SEPARATORS:
                # CM1-S3: FS resets separator context; others sequence within it
                if byte == FS:
                    sep_context = FS
                else:
                    if sep_context is not None and byte < sep_context:
                        raise ValueError(
                            f"CM1-S3 violated: illegal separator nesting 0x{byte:02X} inside "
                            f"0x{sep_context:02X} context at position {i}"
                        )
                    sep_context = byte
                events.append({"byte": byte, "xcfe_mapping": XCFE_MAP[byte]})

            elif byte in XCFE_MAP:
                # Other mapped control bytes (ENQ, ACK, BEL, NAK) — emit event
                events.append({"byte": byte, "xcfe_mapping": XCFE_MAP[byte]})

            else:
                # Non-control passthrough
                passthrough.extend(bytes([byte]))

        # Final invariant checks
        if stx_count != 0:
            raise ValueError(
                f"CM1-S1 violated: {stx_count} unmatched STX/ETX pair(s) at end of stream"
            )
        if state["scope"]:
            raise ValueError(
                f"CM1-S2 violated: scope stack not empty at end of stream "
                f"(depth={len(state['scope'])})"
            )

        return {
            "phase": state["phase"],
            "scope": list(state["scope"]),
            "mode": state["mode"],
            "literal": state["literal"],
            "events": events,
            "passthrough": bytes(passthrough),
        }


# ---------------------------------------------------------------------------
# Projection invariant helpers
# ---------------------------------------------------------------------------

def strip_cm1(stream: bytes) -> bytes:
    """Remove all CM-1 control characters from *stream*, returning passthrough bytes only."""
    return bytes(b for b in stream if b not in CM1_BYTES and b not in FORBIDDEN)


def verify_projection_invariant(original: bytes, stripped: bytes) -> None:
    """Assert that *stripped* equals strip_cm1(*original*).

    The projection invariant: removing CM-1 bytes must not change the
    non-control content of the stream.

    Raises ValueError if the invariant is violated.
    """
    expected = strip_cm1(original)
    if stripped != expected:
        raise ValueError(
            "CM-1 projection invariant violated: strip_cm1(original) != stripped. "
            f"Expected {expected!r}, got {stripped!r}"
        )


# ---------------------------------------------------------------------------
# CSS projection helper (for UIFold §9 binding)
# ---------------------------------------------------------------------------

def project_cm1_to_css(phase_state: Dict[str, Any]) -> str:
    """Generate CSS custom properties for a CM-1 phase state dict.

    Per spec §9, binds CM-1 phase geometry to CSS custom properties and
    data-attribute selectors.
    """
    phase = phase_state.get("phase", "init") or "init"
    scope_depth = len(phase_state.get("scope", []))
    mode = phase_state.get("mode", "normal")
    literal = "true" if phase_state.get("literal", False) else "false"

    css_lines = [
        ":root {",
        f'  --cm1-phase: "{phase}";',
        f"  --cm1-scope-depth: {scope_depth};",
        f'  --cm1-mode: "{mode}";',
        f"  --cm1-literal: {literal};",
        "}",
        "",
        '[data-cm1-phase="header"]::before {',
        "  content: \"\";",
        "  /* SOH received - header phase active */",
        "}",
        "",
        '[data-cm1-phase="body"]::before {',
        "  content: \"\";",
        "  /* STX received - body phase active */",
        "}",
        "",
        '[data-cm1-phase="closing"]::before {',
        "  content: \"\";",
        "  /* ETX received - closing phase active */",
        "}",
        "",
        '[data-cm1-phase="collapsed"]::before {',
        "  content: \"\";",
        "  /* EOT received - collapsed (flushed) */",
        "}",
    ]
    return "\n".join(css_lines)
