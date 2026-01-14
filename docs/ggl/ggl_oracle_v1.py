from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Tuple

TOKEN_RE = re.compile(
    r"""
    (?P<NL>\n)|
    (?P<WS>[ \t\r]+)|
    (?P<LINECOMMENT>//[^\n]*)|
    (?P<BLOCKCOMMENT>/\*[\s\S]*?\*/)|
    (?P<ARROW>->)|
    (?P<OP>==|!=|<=|>=|[=+\-*/%<>])|
    (?P<PUNC>[{}\(\)\[\];:,\.\"])
    |(?P<STRING>\"([^\"\\\\]|\\\\.)*\"|'([^'\\\\]|\\\\.)*')
    |(?P<NUMBER>-?(0|[1-9][0-9]*)(\.[0-9]+)?)
    |(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

KEYWORDS = {
    "GGL.begin",
    "GGL.end",
    "node",
    "graph",
    "edge",
    "emit",
    "let",
    "set",
    "if",
    "else",
    "for",
    "return",
    "true",
    "false",
    "null",
}

TOPLEVEL = {"node", "graph", "edge", "emit", "let", "if", "for", "return"}


@dataclass(frozen=True)
class Tok:
    t: str
    v: str
    i: int


def lex(src: str) -> List[Tok]:
    out: List[Tok] = []
    i = 0
    n = len(src)
    while i < n:
        match = TOKEN_RE.match(src, i)
        if not match:
            out.append(Tok("BAD", src[i], i))
            i += 1
            continue
        kind = match.lastgroup or "BAD"
        text = match.group(0)
        if kind in ("WS", "LINECOMMENT", "BLOCKCOMMENT"):
            i = match.end()
            continue
        if kind == "IDENT" and text in KEYWORDS:
            out.append(Tok("KW", text, i))
        else:
            out.append(Tok(kind, text, i))
        i = match.end()
    return out


def _balanced_check(tokens: List[Tok]) -> List[str]:
    stack: List[Tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    opens = set(pairs.values())
    closes = set(pairs.keys())
    errs: List[str] = []
    for tok in tokens:
        if tok.v in opens:
            stack.append((tok.v, tok.i))
        elif tok.v in closes:
            if not stack:
                errs.append(f"Unmatched closing {tok.v} at {tok.i}")
            else:
                top, pos = stack.pop()
                if pairs[tok.v] != top:
                    errs.append(
                        f"Mismatched closing {tok.v} at {tok.i} (opened {top} at {pos})"
                    )
    for top, pos in reversed(stack):
        errs.append(f"Unclosed {top} opened at {pos}")
    return errs


def _must_wrap(src: str) -> List[str]:
    s = src.strip()
    errs: List[str] = []
    if not s.startswith("GGL.begin"):
        errs.append("Missing leading GGL.begin")
    if not s.endswith("GGL.end"):
        errs.append("Missing trailing GGL.end")
    return errs


def _toplevel_form(tokens: List[Tok]) -> List[str]:
    errs: List[str] = []
    kws = [t for t in tokens if t.t == "KW"]
    if not any(t.v == "GGL.begin" for t in kws) or not any(t.v == "GGL.end" for t in kws):
        return errs

    seen_begin = False
    statement_start = True
    block_depth = 0
    for tok in tokens:
        if tok.t == "KW" and tok.v == "GGL.begin":
            seen_begin = True
            statement_start = True
            continue
        if tok.t == "KW" and tok.v == "GGL.end":
            break
        if not seen_begin:
            continue
        if tok.v == "{":
            block_depth += 1
        if tok.v == "}":
            block_depth = max(0, block_depth - 1)
            statement_start = True
        if tok.t == "NL":
            statement_start = True
            continue
        if statement_start and block_depth == 0:
            if tok.t == "KW" and tok.v in TOPLEVEL:
                statement_start = False
                continue
            if tok.t not in ("NL",):
                errs.append(f"Top-level statement must start with keyword at {tok.i}")
                statement_start = False
    return errs


def _semicolon_sanity(src: str) -> List[str]:
    errs: List[str] = []
    lines = src.splitlines()
    in_wrap = False
    for ln_no, line in enumerate(lines, start=1):
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("/*"):
            continue
        if s == "GGL.begin":
            in_wrap = True
            continue
        if s == "GGL.end":
            in_wrap = False
            continue
        if not in_wrap:
            continue
        if (
            s.endswith("{")
            or s.endswith("}")
            or s.startswith("if ")
            or s.startswith("for ")
            or s.startswith("graph ")
            or s.startswith("node ")
        ):
            continue
        if not s.endswith(";"):
            errs.append(f"Line {ln_no}: missing ';' (got: {s[:80]})")
    return errs


def validate_ggl(src: str) -> Dict[str, Any]:
    tokens = lex(src)
    errs: List[str] = []
    errs += _must_wrap(src)
    errs += _balanced_check(tokens)
    errs += _semicolon_sanity(src)
    errs += _toplevel_form(tokens)
    fingerprint = (
        f"tok={len(tokens)};bad={sum(1 for t in tokens if t.t == 'BAD')};"
        f"errs={len(errs)}"
    )
    return {
        "@type": "ggl.oracle.result.v1",
        "valid": len(errs) == 0,
        "errors": errs[:32],
        "fingerprint": fingerprint,
    }
