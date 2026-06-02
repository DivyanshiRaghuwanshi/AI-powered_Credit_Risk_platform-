from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MatchDecision:
    confidence: float
    status: str  # auto_accepted | manual_review | rejected
    reasons: List[str]


class EntityResolver:
    """Simple confidence-based resolver for Phase 1 baseline."""

    def __init__(self, auto_accept_threshold: float = 0.98, manual_review_threshold: float = 0.90) -> None:
        if auto_accept_threshold <= manual_review_threshold:
            raise ValueError("auto_accept_threshold must be greater than manual_review_threshold")
        self.auto_accept_threshold = auto_accept_threshold
        self.manual_review_threshold = manual_review_threshold

    def score_pair(self, a: Dict[str, object], b: Dict[str, object]) -> MatchDecision:
        score = 0.0
        reasons: List[str] = []

        if _same(a, b, "gov_id"):
            score += 0.70
            reasons.append("gov_id_exact")
        if _same(a, b, "phone"):
            score += 0.12
            reasons.append("phone_exact")
        if _same(a, b, "dob"):
            score += 0.08
            reasons.append("dob_exact")
        if _same(a, b, "email"):
            score += 0.06
            reasons.append("email_exact")
        if _same(a, b, "address_hash"):
            score += 0.04
            reasons.append("address_hash_exact")

        # penalty for strong contradiction
        if _conflict(a, b, "gov_id"):
            score -= 0.80
            reasons.append("gov_id_conflict")
        if _conflict(a, b, "dob"):
            score -= 0.20
            reasons.append("dob_conflict")

        confidence = round(max(0.0, min(1.0, score)), 6)
        if confidence >= self.auto_accept_threshold:
            status = "auto_accepted"
        elif confidence >= self.manual_review_threshold:
            status = "manual_review"
        else:
            status = "rejected"

        return MatchDecision(confidence=confidence, status=status, reasons=reasons)

    def resolve_cluster(self, records: Iterable[Dict[str, object]]) -> MatchDecision:
        rows = list(records)
        if len(rows) < 2:
            return MatchDecision(confidence=1.0, status="auto_accepted", reasons=["single_record"])

        pair_scores: List[MatchDecision] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                pair_scores.append(self.score_pair(rows[i], rows[j]))

        min_conf = round(min(d.confidence for d in pair_scores), 6)
        combined_reasons: List[str] = []
        for d in pair_scores:
            combined_reasons.extend(d.reasons)

        # conservative cluster decision by weakest link
        if min_conf >= self.auto_accept_threshold:
            status = "auto_accepted"
        elif min_conf >= self.manual_review_threshold:
            status = "manual_review"
        else:
            status = "rejected"

        return MatchDecision(confidence=min_conf, status=status, reasons=sorted(set(combined_reasons)))


def _same(a: Dict[str, object], b: Dict[str, object], key: str) -> bool:
    av = a.get(key)
    bv = b.get(key)
    return av is not None and bv is not None and str(av).strip() != "" and av == bv


def _conflict(a: Dict[str, object], b: Dict[str, object], key: str) -> bool:
    av = a.get(key)
    bv = b.get(key)
    return av is not None and bv is not None and str(av).strip() != "" and str(bv).strip() != "" and av != bv
