"""
Research policy checks for verified research output.

These checks enforce:
- source allowlist per domain
- credibility/recency thresholds
- minimum source counts
- citation coverage constraints
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Tuple

ALLOWED_SOURCES = {
    "technology": {
        "arxiv.org",
        "github.com",
        "news.ycombinator.com",
        "stackoverflow.com",
    },
    "science": {
        "pubmed.ncbi.nlm.nih.gov",
        "nature.com",
        "science.org",
        "plos.org",
    },
    "finance": {
        "sec.gov",
        "reuters.com",
        "bloomberg.com",
        "finance.yahoo.com",
    },
    "general_knowledge": {
        "wikipedia.org",
        "britannica.com",
        "plato.stanford.edu",
        "archive.org",
    },
}


def extract_host(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    return host[4:] if host.startswith("www.") else host


def check_source_allowlist(domain: str, citations: Iterable[Dict[str, Any]]) -> None:
    allowed = ALLOWED_SOURCES.get(domain, set())
    for citation in citations:
        url = citation.get("url")
        if not url:
            raise ValueError("citation missing url")
        host = extract_host(url)
        if host not in allowed:
            raise ValueError(f"source host {host} not allowed for domain {domain}")


def check_thresholds(
    citations: Iterable[Dict[str, Any]],
    min_sources: int,
    credibility_threshold: float,
    recency_days_limit: int,
) -> None:
    citation_list = list(citations)
    if len(citation_list) < min_sources:
        raise ValueError("insufficient sources used")
    for citation in citation_list:
        credibility = float(citation.get("credibility", 0))
        recency_days = int(citation.get("recency_days", 10**9))
        if credibility < credibility_threshold:
            raise ValueError("citation below credibility threshold")
        if recency_days > recency_days_limit:
            raise ValueError("citation beyond recency limit")


def check_claim_coverage(claims: Iterable[Dict[str, Any]]) -> None:
    for claim in claims:
        claim_type = claim.get("type")
        if claim_type != "fact":
            continue
        support = claim.get("supporting_sources", [])
        primary = claim.get("primary_source")
        if len(support) < 2 and not primary:
            raise ValueError("fact claim lacks sufficient source coverage")


def validate_research_result(result: Dict[str, Any]) -> None:
    request = result.get("request", {})
    citations = result.get("citations", [])
    claims = result.get("claims", [])

    domain = request.get("domain")
    min_sources = int(request.get("min_sources", 1))
    credibility_threshold = float(request.get("credibility_threshold", 0.0))
    recency_days = int(request.get("recency_days", 10**9))

    if not domain:
        raise ValueError("research result missing domain")

    check_source_allowlist(domain, citations)
    check_thresholds(citations, min_sources, credibility_threshold, recency_days)
    check_claim_coverage(claims)


def validate_research_batch(results: Iterable[Dict[str, Any]]) -> None:
    for result in results:
        validate_research_result(result)
