from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from hybrid_risk.phase1.governance import ROLE_POLICIES, create_audit_event, validate_claim_evidence
from hybrid_risk.phase4.service import Phase4InvestigationService
from hybrid_risk.phase4.types import Phase4InvestigationRequest

from .types import Phase5CopilotRequest


POLICY_PROFILES: Dict[str, Dict[str, Any]] = {
    "balanced": {"include_sql": True, "include_graph": True, "include_vector": False, "include_model": True, "top_k": 5},
    "fraud_deep_dive": {"include_sql": True, "include_graph": True, "include_vector": True, "include_model": True, "top_k": 10},
    "credit_fast": {"include_sql": True, "include_graph": False, "include_vector": False, "include_model": True, "top_k": 5},
    "semantic_review": {"include_sql": False, "include_graph": False, "include_vector": True, "include_model": True, "top_k": 8},
}

INTENT_TO_PROFILE: Dict[str, str] = {
    "fraud_investigation": "fraud_deep_dive",
    "customer_risk_reasoning": "balanced",
    "exact_customer_profile": "credit_fast",
    "semantic_notes": "semantic_review",
    "full_investigation": "balanced",
}


def _resolve_profile(request: Phase5CopilotRequest) -> str:
    if request.policy_profile and request.policy_profile in POLICY_PROFILES:
        return request.policy_profile
    return INTENT_TO_PROFILE.get(request.intent, "balanced")


def _apply_profile(request: Phase5CopilotRequest, profile: str) -> Dict[str, Any]:
    p = POLICY_PROFILES[profile]
    if request.strict_route_policy:
        return {
            "include_sql": bool(p["include_sql"]),
            "include_graph": bool(p["include_graph"]),
            "include_vector": bool(p["include_vector"]),
            "include_model": bool(p["include_model"]),
            "top_k": int(p["top_k"]),
        }
    return {
        "include_sql": bool(request.include_sql),
        "include_graph": bool(request.include_graph),
        "include_vector": bool(request.include_vector),
        "include_model": bool(request.include_model),
        "top_k": int(request.top_k),
    }


def _mask_record_for_role(record: Dict[str, Any], role: str) -> Dict[str, Any]:
    policy = ROLE_POLICIES.get(role)
    if policy is None:
        raise ValueError(f"Unknown role: {role}")
    if policy.can_unmask_pii:
        return dict(record)

    out = dict(record)
    for k in list(out.keys()):
        lk = k.lower()
        if lk in {"customer_gid", "customer_id", "sk_id_curr", "sk_id_prev", "sk_id_bureau"}:
            out[k] = "***MASKED***"
            continue
        if any(tok in lk for tok in ("name", "email", "phone", "address", "national_id")):
            out[k] = "***MASKED***"
    return out


def _mask_responses_for_role(responses: List[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    masked: List[Dict[str, Any]] = []
    for item in responses:
        recs = [_mask_record_for_role(r, role) for r in item.get("records", [])]
        clone = dict(item)
        clone["records"] = recs
        masked.append(clone)
    return masked


class Phase5CopilotService:
    """Policy-driven orchestration + governance trace over Phase 4 investigation."""

    def __init__(self, phase4_service: Phase4InvestigationService) -> None:
        self.phase4_service = phase4_service

    def health(self) -> Dict[str, Any]:
        inner = self.phase4_service.health()
        return {
            "ok": bool(inner.get("ok")),
            "checks": list(inner.get("checks", []))
            + [{"name": "phase5_policy_engine", "ok": True, "message": "loaded"}],
        }

    async def investigate(self, request: Phase5CopilotRequest) -> Dict[str, Any]:
        profile = _resolve_profile(request)
        policy_flags = _apply_profile(request, profile)

        p4_req = Phase4InvestigationRequest(
            intent=request.intent,
            customer_gid=request.customer_gid,
            customer_id=request.customer_id,
            model_id=request.model_id,
            filters=request.filters or {},
            query_text=request.query_text,
            query_embedding=request.query_embedding,
            top_k=policy_flags["top_k"],
            include_sql=policy_flags["include_sql"],
            include_graph=policy_flags["include_graph"],
            include_vector=policy_flags["include_vector"],
            include_model=policy_flags["include_model"],
            include_explanation=request.include_explanation,
            explanation_top_k=request.explanation_top_k,
            apply_calibration=request.apply_calibration,
        )
        p4 = await self.phase4_service.investigate(p4_req)

        responses = _mask_responses_for_role(list(p4.get("responses", [])), request.role)
        routes = list(p4.get("routes", []))

        evidence_ids: List[str] = []
        claims: List[str] = []
        evidence_map: Dict[str, List[str]] = {}
        for response in responses:
            src = str(response.get("source"))
            recs = list(response.get("records", []))
            ids = [f"ev_{src}_{i+1}" for i in range(len(recs))]
            evidence_ids.extend(ids)
            if src == "model" and recs:
                rec = recs[0]
                claim = (
                    f"Model predicts risk_band={rec.get('risk_band')} "
                    f"with probability={rec.get('predicted_probability')}"
                )
                claims.append(claim)
                evidence_map[claim] = ids[:1] if ids else []
            elif src == "graph":
                claim = f"Graph traversal returned {int(response.get('count', 0))} linked paths."
                claims.append(claim)
                evidence_map[claim] = ids[:1] if ids else []
            elif src == "sql":
                claim = f"SQL retrieval returned {int(response.get('count', 0))} candidate records."
                claims.append(claim)
                evidence_map[claim] = ids[:1] if ids else []
            elif src == "vector":
                claim = f"Vector retrieval returned {int(response.get('count', 0))} similar documents."
                claims.append(claim)
                evidence_map[claim] = ids[:1] if ids else []

        unsupported_claims = validate_claim_evidence(claims, evidence_map)
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        trace = {
            "trace_id": trace_id,
            "profile_selected": profile,
            "strict_route_policy": request.strict_route_policy,
            "policy_flags": policy_flags,
            "tool_routes": routes,
            "unsupported_claims": unsupported_claims,
        }

        response_text = json.dumps({"summary": p4.get("summary", {}), "claims": claims}, ensure_ascii=True)
        audit = create_audit_event(
            user_id=request.user_id,
            role=request.role,
            request_id=request.request_id,
            tool_calls=routes,
            evidence_ids=evidence_ids,
            response_text=response_text,
        )

        return {
            "routes": routes,
            "responses": responses,
            "summary": p4.get("summary", {}),
            "claims": claims,
            "evidence_map": evidence_map,
            "trace": trace,
            "audit_event": audit,
            "total_latency_ms": float(p4.get("total_latency_ms", 0.0)),
        }
