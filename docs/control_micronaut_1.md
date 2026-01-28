# CONTROL-MICRONAUT-1 (CM-1) SPECIFICATION

**Invisible Control Alphabet for XCFE / DOM / CSS Safe Execution**

| Field | Value |
|-------|-------|
| **Status** | Draft-Frozen v1 |
| **Scope** | Pre-semantic control, phase signaling, structure shaping |
| **Non-Goals** | Rendering, evaluation, execution, scripting |
| **SCXQ2 Lane** | EDGE (control metadata) |
| **Fold Binding** | `CONTROL_FOLD` |

---

## 1. Core Principle

> **CONTROL-MICRONAUT-1 defines a non-rendering, non-executing control layer composed exclusively of Unicode C0 control characters (U+0000-U+001F) and U+0020 (SPACE).**

CM-1 **never introduces behavior**.
It **only constrains interpretation**.

This satisfies:

- No global truth
- Deterministic collapse
- Invariant-driven legality
- UI as projection
- Compression-safe intelligence

---

## 2. CM-1 Execution Model (XCFE-Aligned)

CM-1 participates **before syntax**.

```
[CM-1 Control Stream]
        |
        v
[XCFE Phase Resolution]
        |
        v
[Parser / Renderer / DOM]
```

### 2.1 CM-1 CANNOT

- Inject tokens
- Create nodes
- Alter values
- Execute logic

### 2.2 CM-1 CAN

- Mark boundaries
- Declare phases
- Signal scope transitions
- Segment streams
- Annotate interpretation zones

---

## 3. Canonical Mapping: XCFE @control Vectors

### 3.1 Phase Control (Primary)

| Code | Name | XCFE Mapping | Meaning | Phase Equiv |
|------|------|--------------|---------|-------------|
| **U+0000** | NUL | `@control.null` | Absolute inert region | - |
| **U+0001** | SOH | `@control.header.begin` | Metadata/header phase | @Pop |
| **U+0002** | STX | `@control.body.begin` | Interpretable content | @Wo |
| **U+0003** | ETX | `@control.body.end` | Content closure | @Sek |
| **U+0004** | EOT | `@control.transmission.end` | Collapse / flush | @Collapse |

These **exactly** align with `@Pop -> @Wo -> @Sek -> @Collapse`.

---

### 3.2 Scope & Context Stack

| Code | Name | XCFE Mapping | Meaning |
|------|------|--------------|---------|
| **U+000E** | SO | `@control.scope.push` | Enter sub-context |
| **U+000F** | SI | `@control.scope.pop` | Exit sub-context |
| **U+001B** | ESC | `@control.mode.switch` | Grammar / parser mode shift |
| **U+0010** | DLE | `@control.literal.escape` | Bypass interpretation |

---

### 3.3 Structural Segmentation (Critical for CSS / JSON)

| Code | Name | XCFE Mapping | Meaning |
|------|------|--------------|---------|
| **U+001C** | FS | `@control.file.sep` | Major boundary |
| **U+001D** | GS | `@control.group.sep` | Group boundary |
| **U+001E** | RS | `@control.record.sep` | Record boundary |
| **U+001F** | US | `@control.unit.sep` | Atomic unit boundary |

These **never render** and **never break layout**.

---

### 3.4 Transport / Negotiation (Optional)

| Code | Name | XCFE Mapping | Meaning |
|------|------|--------------|---------|
| **U+0005** | ENQ | `@control.query` | Capability inquiry |
| **U+0006** | ACK | `@control.ack` | Acceptance |
| **U+0015** | NAK | `@control.nak` | Rejection |
| **U+0007** | BEL | `@control.attention` | Wake / notify |

---

## 4. DOM & CSS SAFE SUBSET (CM-1-SAFE)

### 4.1 Allowed Characters (SAFE MODE)

**Guaranteed non-rendering & non-breaking:**

```
U+0000  NUL     @control.null
U+0001  SOH     @control.header.begin
U+0002  STX     @control.body.begin
U+0003  ETX     @control.body.end
U+0004  EOT     @control.transmission.end
U+000E  SO      @control.scope.push
U+000F  SI      @control.scope.pop
U+0010  DLE     @control.literal.escape
U+001C  FS      @control.file.sep
U+001D  GS      @control.group.sep
U+001E  RS      @control.record.sep
U+001F  US      @control.unit.sep
U+0020  SPACE   @control.space
```

These:

- Survive JSON strings
- Survive HTML text nodes
- Survive CSS parsing
- Do not affect layout
- Are ignored by renderers
- Preserve byte order

---

### 4.2 Conditionally Allowed (CONTEXT-SAFE)

**Allowed only in non-rendering channels** (comments, data attrs, text nodes not measured):

```
U+0009  HT      Horizontal Tab
U+000A  LF      Line Feed
U+000D  CR      Carriage Return
U+001B  ESC     Escape (mode switch)
```

**Rules:**

| Context | Allowed |
|---------|---------|
| CSS identifiers | NO |
| Attribute names | NO |
| Comments | YES |
| Text nodes | YES |
| JSON strings | YES |

---

### 4.3 Forbidden (HARD BAN)

**Never allowed in CM-1:**

```
U+0008  BS      Backspace
U+000B  VT      Vertical Tab
U+000C  FF      Form Feed
U+0018  CAN     Cancel
U+001A  SUB     Substitute
```

**Reason:**

- Layout mutation
- Cursor motion
- Rendering side-effects
- Parser instability

**Violations produce illegal state.**

---

## 5. CM-1 Legality Rules (Invariants)

### 5.1 Structural Invariants

| Rule ID | Invariant |
|---------|-----------|
| **CM1-S1** | Every `STX` (U+0002) MUST have a matching `ETX` (U+0003) |
| **CM1-S2** | Scope stack (`SO`/`SI`) MUST be balanced |
| **CM1-S3** | Separators MAY NOT nest illegally |
| **CM1-S4** | `ESC` CANNOT appear inside literal-escaped regions |
| **CM1-S5** | `NUL` regions are non-observable |

---

### 5.2 Projection Invariant (CRITICAL)

> **Removing all CM-1 characters MUST NOT change visible output.**

This is the **hard rule**.

If removing CM-1 alters:

- DOM structure
- CSS layout
- Text rendering

**The stream is INVALID.**

---

### 5.3 Verifier Binding

CM-1 streams are subject to existing verifier rules:

| Rule | CM-1 Application |
|------|------------------|
| **V0** | CM-1 tokens deterministic (no randomness) |
| **V2** | CM-1 phase transitions require control gate record |
| **V3** | CM-1 stream replayable from identical input |
| **V4** | CM-1 cannot produce floating-point values |
| **V7** | CM-1 phase boundaries align with SCXQ2 frames |

---

## 6. XCFE Binding (Machine-Readable)

### 6.1 Canonical Lowering

```json
{
  "@control": {
    "null": "\u0000",
    "header.begin": "\u0001",
    "body.begin": "\u0002",
    "body.end": "\u0003",
    "transmission.end": "\u0004",
    "query": "\u0005",
    "ack": "\u0006",
    "attention": "\u0007",
    "scope.push": "\u000E",
    "scope.pop": "\u000F",
    "literal.escape": "\u0010",
    "nak": "\u0015",
    "mode.switch": "\u001B",
    "file.sep": "\u001C",
    "group.sep": "\u001D",
    "record.sep": "\u001E",
    "unit.sep": "\u001F",
    "space": "\u0020"
  }
}
```

### 6.2 Phase Mapping

```json
{
  "@phase": {
    "Pop": "@control.header.begin",
    "Wo": "@control.body.begin",
    "Sek": "@control.body.end",
    "Collapse": "@control.transmission.end"
  }
}
```

---

## 7. SCXQ2 Integration

CM-1 control bytes map to SCXQ2 lanes as follows:

| CM-1 Category | SCXQ2 Lane | Packing |
|---------------|------------|---------|
| Phase Control (SOH/STX/ETX/EOT) | EDGE | Frame header |
| Scope Stack (SO/SI/ESC/DLE) | EDGE | Context metadata |
| Separators (FS/GS/RS/US) | FIELD | Boundary markers |
| Transport (ENQ/ACK/NAK/BEL) | EDGE | Negotiation frames |

### 7.1 Frame Layout

```
+--------+--------+--------+--------+
| MAGIC  | CM1_OP | LENGTH | PAYLOAD|
+--------+--------+--------+--------+
  2 bytes  1 byte   2 bytes  N bytes
```

Where `CM1_OP` is the raw control character code point.

---

## 8. Fold Lattice Integration

CM-1 operates at the `CONTROL_FOLD` level:

```
CONTROL_FOLD (Resolve -> Gate -> Commit)
     |
     +-- CM-1 Stream Processing
          |
          +-- Phase boundary detection
          +-- Scope stack management
          +-- Separator parsing
          +-- Legality verification
```

### 8.1 Collapse Rules

| Fold | CM-1 Interaction |
|------|------------------|
| `DATA_FOLD` | Receives CM-1 annotated payloads |
| `COMPUTE_FOLD` | Ignores CM-1 (stripped before compute) |
| `PROOF_FOLD` | Records CM-1 hashes for audit |
| `CONTROL_FOLD` | Owns CM-1 interpretation authority |

---

## 9. CSS Micronaut Binding

CM-1 control vectors bind to CSS custom properties:

```css
:root {
  --cm1-phase: "init";           /* SOH/STX/ETX/EOT state */
  --cm1-scope-depth: 0;          /* SO/SI nesting level */
  --cm1-mode: "normal";          /* ESC-triggered mode */
  --cm1-literal: false;          /* DLE escape active */
}

[data-cm1-phase="header"]::before {
  content: "";
  /* SOH received - header phase active */
}

[data-cm1-phase="body"]::before {
  content: "";
  /* STX received - body phase active */
}
```

---

## 10. Security Properties

### 10.1 Guarantees

| Property | Status |
|----------|--------|
| Zero execution authority | GUARANTEED |
| Zero render authority | GUARANTEED |
| Deterministic processing | GUARANTEED |
| Compression-safe (SCXQ2) | GUARANTEED |
| Replayable streams | GUARANTEED |
| Auditable via PROOF_FOLD | GUARANTEED |
| Invisible by design | GUARANTEED |

### 10.2 Attack Surface

CM-1 has **zero attack surface** because:

1. It cannot inject executable content
2. It cannot modify DOM structure
3. It cannot alter CSS layout
4. It cannot produce visible output
5. Removing it produces identical rendering

---

## 11. Reference Implementation

### 11.1 Parser Pseudocode

```
function parse_cm1(stream):
    state = { phase: null, scope: [], mode: "normal", literal: false }

    for byte in stream:
        if state.literal and byte != DLE:
            emit(byte)  # Pass through
            continue

        switch byte:
            case NUL:
                # Non-observable, skip
                continue
            case SOH:
                state.phase = "header"
                emit_phase_event("@control.header.begin")
            case STX:
                assert state.phase == "header"
                state.phase = "body"
                emit_phase_event("@control.body.begin")
            case ETX:
                assert state.phase == "body"
                state.phase = "closing"
                emit_phase_event("@control.body.end")
            case EOT:
                state.phase = "collapsed"
                emit_phase_event("@control.transmission.end")
                flush()
            case SO:
                state.scope.push(current_context())
                emit_scope_event("@control.scope.push")
            case SI:
                assert state.scope.length > 0
                state.scope.pop()
                emit_scope_event("@control.scope.pop")
            case ESC:
                assert not state.literal
                state.mode = next_mode(state.mode)
                emit_mode_event("@control.mode.switch")
            case DLE:
                state.literal = not state.literal
                emit_escape_event("@control.literal.escape")
            case FS, GS, RS, US:
                emit_separator_event(byte)
            case FORBIDDEN:
                raise IllegalStateError("Forbidden CM-1 byte")
            default:
                emit(byte)  # Non-control passthrough

    # Verify invariants
    assert state.scope.length == 0  # Balanced
    assert state.phase in [null, "collapsed"]  # Complete
    return state
```

---

## 12. Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-01-28 | Initial draft-frozen specification |

---

## 13. Related Documents

- `fold_law.md` - Fold collapse rules
- `verifier_rules.md` - V0-V7 determinism rules
- `scxq2_binary_packing_example.md` - SCXQ2 frame format
- `control_micronaut.schema.json` - JSON Schema for CM-1
- `xcfe_cm1_binding.json` - Machine-readable XCFE mapping

---

> **CM-1 is not a language.**
> **It is not syntax.**
> **It is not data.**
> **It is phase geometry.**

Control without power. The safest kind.
