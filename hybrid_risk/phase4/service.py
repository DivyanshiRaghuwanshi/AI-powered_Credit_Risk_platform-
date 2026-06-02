from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from hybrid_risk.phase3.service import Phase3HybridService
from hybrid_risk.phase3.types import Phase3RetrievalRequest

from .adapters import ModelAdapter
from .types import Phase4InvestigationRequest


class Phase4InvestigationService:
    """Combines Phase 3 evidence retrieval with model scoring + explanation."""

    def __init__(self, phase3_service: Phase3HybridService, model_adapter: ModelAdapter) -> None:
        self.phase3_service = phase3_service
        self.model_adapter = model_adapter

    def health(self) -> Dict[str, Any]:
        phase3_health = self.phase3_service.health()
        model_health = self.model_adapter.health()
        checks = list(phase3_health.get("checks", [])) + [
            {"name": model_health.name, "ok": model_health.ok, "message": model_health.message}
        ]
        return {"ok": bool(phase3_health.get("ok")) and model_health.ok, "checks": checks}

    async def investigate(self, request: Phase4InvestigationRequest) -> Dict[str, Any]:
        t0 = time.perf_counter()
        p3_req = Phase3RetrievalRequest(
            intent=request.intent,
            customer_gid=request.customer_gid,
            filters=request.filters or {},
            query_text=request.query_text,
            query_embedding=request.query_embedding,
            top_k=request.top_k,
            include_sql=request.include_sql,
            include_graph=request.include_graph,
            include_vector=request.include_vector,
        )
        retrieval_out = await self.phase3_service.retrieve(p3_req)

        responses = list(retrieval_out.get("responses", []))
        routes = list(retrieval_out.get("routes", []))

        if request.include_model:
            routes.append("model")
            try:
                model_res = await asyncio.to_thread(self.model_adapter.score, request, retrieval_out)
                responses.append(
                    {
                        "source": model_res.source,
                        "count": 1 if model_res.record else 0,
                        "latency_ms": model_res.latency_ms,
                        "note": model_res.note,
                        "records": [model_res.record] if model_res.record else [],
                        "ok": True,
                    }
                )
            except Exception as exc:
                responses.append(
                    {
                        "source": "model",
                        "count": 0,
                        "latency_ms": 0.0,
                        "note": f"error: {exc}",
                        "records": [],
                        "ok": False,
                    }
                )

        model_record = {}
        for r in responses:
            if r.get("source") == "model" and r.get("records"):
                model_record = r["records"][0]
                break
        summary = {
            "risk_band": model_record.get("risk_band", "unknown"),
            "predicted_probability": model_record.get("predicted_probability"),
            "evidence_sources": len(routes),
        }

        return {
            "routes": routes,
            "responses": responses,
            "summary": summary,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 3),
        }

