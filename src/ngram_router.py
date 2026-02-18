"""
Ngram Intent Router
===================
Routes input text to the correct micronaut by scoring bigrams (weight 1.0) and
trigrams (weight 1.5) against trigger lists in meta-intent-map.json.

The sealed brain file is read-only; this module never writes to it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class NgramRouter:
    """Load the intent map and score text against ngram triggers."""

    BIGRAM_WEIGHT = 1.0
    TRIGRAM_WEIGHT = 1.5
    FALLBACK = "XM-1"

    def __init__(self, intent_map_path: Optional[str] = None) -> None:
        if intent_map_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            intent_map_path = os.path.join(here, "..", "micronaut", "brains", "meta-intent-map.json")
        with open(intent_map_path, "r", encoding="utf-8") as f:
            self._map: Dict[str, Any] = json.load(f)
        self._intents: Dict[str, Any] = self._map.get("intents", {})
        routing = self._map.get("routing", {})
        self._min_confidence: float = float(routing.get("minimum_confidence", 0.3))
        self._fallback: str = routing.get("fallback", self.FALLBACK)

    # ------------------------------------------------------------------
    # Ngram extraction

    @staticmethod
    def _extract_bigrams(text: str) -> List[str]:
        words = text.lower().split()
        return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    @staticmethod
    def _extract_trigrams(text: str) -> List[str]:
        words = text.lower().split()
        return [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]

    # ------------------------------------------------------------------
    # Scoring

    def _score_intent(self, text: str, intent: Dict[str, Any]) -> float:
        bigrams = self._extract_bigrams(text)
        trigrams = self._extract_trigrams(text)
        score = 0.0
        for bg in intent.get("trigger_bigrams", []):
            if bg in bigrams:
                score += self.BIGRAM_WEIGHT
        for tg in intent.get("trigger_trigrams", []):
            if tg in trigrams:
                score += self.TRIGRAM_WEIGHT
        return score

    # ------------------------------------------------------------------
    # Public API

    def route(self, text: str) -> Dict[str, Any]:
        """Return routing decision for *text*.

        Returns:
            {
                "target": micronaut_id,
                "intent": intent_name,
                "target_fold": fold_string,
                "tool": first_tool_in_intent,
                "confidence": score,
            }
        Falls back to XM-1 when confidence < minimum_confidence.
        """
        scores: List[Tuple[float, str, Dict[str, Any]]] = []
        for intent_name, intent in self._intents.items():
            score = self._score_intent(text, intent)
            if score > 0:
                scores.append((score, intent_name, intent))

        scores.sort(key=lambda x: (-x[0], x[2].get("priority", 99)))

        if scores and scores[0][0] >= self._min_confidence:
            best_score, best_name, best_intent = scores[0]
            tools = best_intent.get("tools", [])
            return {
                "target": best_intent.get("target", self._fallback),
                "intent": best_name,
                "target_fold": best_intent.get("fold", ""),
                "tool": tools[0] if tools else None,
                "confidence": best_score,
            }

        # Fallback — find fallback micronaut entry
        fallback_intent = next(
            (v for v in self._intents.values() if v.get("target") == self._fallback),
            {},
        )
        fallback_tools = fallback_intent.get("tools", [])
        return {
            "target": self._fallback,
            "intent": "expand",
            "target_fold": fallback_intent.get("fold", "PATTERN_FOLD"),
            "tool": fallback_tools[0] if fallback_tools else None,
            "confidence": 0.0,
        }

    def all_scores(self, text: str) -> Dict[str, float]:
        """Return a dict of intent_name → score for all intents."""
        return {name: self._score_intent(text, intent) for name, intent in self._intents.items()}
