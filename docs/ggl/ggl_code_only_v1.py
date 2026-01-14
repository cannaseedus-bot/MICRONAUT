from __future__ import annotations

from typing import Any, Dict, Tuple


def extract_ggl_block(text: str) -> Tuple[bool, str]:
    start = text.find("GGL.begin")
    end = text.rfind("GGL.end")
    if start == -1 or end == -1 or end < start:
        return False, ""
    end_inclusive = end + len("GGL.end")
    return True, text[start:end_inclusive].strip() + "\n"


def code_only_response(text: str) -> Dict[str, Any]:
    ok, block = extract_ggl_block(text)
    if not ok:
        return {
            "@type": "ggl.output.v1",
            "ok": False,
            "error": "NO_GGL_BLOCK",
            "code": "GGL.begin\n// ERROR: model did not emit GGL block\nGGL.end\n",
        }
    return {
        "@type": "ggl.output.v1",
        "ok": True,
        "code": block,
    }
