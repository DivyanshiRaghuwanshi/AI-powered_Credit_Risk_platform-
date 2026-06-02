from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterable, List, Mapping, MutableMapping


MASKED = "***MASKED***"


@dataclass(frozen=True)
class AccessPolicy:
    role: str
    can_unmask_pii: bool
    can_approve_rules: bool


ROLE_POLICIES = {
    "analyst": AccessPolicy(role="analyst", can_unmask_pii=False, can_approve_rules=False),
    "senior_analyst": AccessPolicy(role="senior_analyst", can_unmask_pii=True, can_approve_rules=False),
    "risk_manager": AccessPolicy(role="risk_manager", can_unmask_pii=True, can_approve_rules=True),
    "auditor": AccessPolicy(role="auditor", can_unmask_pii=False, can_approve_rules=False),
}

PII_FIELDS = {"name", "phone", "email", "national_id", "customer_id", "address"}


def mask_pii(record: Mapping[str, object], role: str) -> Dict[str, object]:
    policy = ROLE_POLICIES.get(role)
    if policy is None:
        raise ValueError(f"Unknown role: {role}")

    out: Dict[str, object] = dict(record)
    if policy.can_unmask_pii:
        return out

    for k in list(out.keys()):
        if k in PII_FIELDS:
            out[k] = MASKED
    return out


def create_audit_event(user_id: str, role: str, request_id: str, tool_calls: Iterable[str], evidence_ids: Iterable[str], response_text: str) -> Dict[str, object]:
    payload = {
        "user_id": user_id,
        "role": role,
        "request_id": request_id,
        "tool_calls": list(tool_calls),
        "evidence_ids": list(evidence_ids),
    }
    payload["response_hash"] = sha256(response_text.encode("utf-8")).hexdigest()
    return payload


def validate_claim_evidence(claims: Iterable[str], evidence_map: Mapping[str, List[str]]) -> List[str]:
    """Return claims that have no evidence ids attached."""
    unsupported: List[str] = []
    for claim in claims:
        ids = evidence_map.get(claim, [])
        if not ids:
            unsupported.append(claim)
    return unsupported


def enforce_no_unsupported_claims(claims: Iterable[str], evidence_map: Mapping[str, List[str]]) -> None:
    missing = validate_claim_evidence(claims, evidence_map)
    if missing:
        raise ValueError(f"Unsupported claims without evidence: {missing}")
