from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from .graph_retriever import GraphRetriever
from .sql_retriever import SqlRetriever
from .types import RetrievalRequest, RetrievalResponse
from .vector_retriever import VectorRetriever


class HybridRetrievalOrchestrator:
    """Deterministic Phase 2 tool orchestrator for SQL/Graph/Vector retrieval."""

    def __init__(self) -> None:
        self.sql = SqlRetriever()
        self.graph = GraphRetriever()
        self.vector = VectorRetriever()

    def route(self, request: RetrievalRequest) -> List[str]:
        intent = (request.intent or "").strip().lower()
        if intent in {"exact_customer_profile", "portfolio_aggregate"}:
            return ["sql"]
        if intent in {"fraud_investigation", "multi_hop_linkage"}:
            return ["graph", "sql"]
        if intent in {"semantic_notes", "complaint_context"}:
            return ["vector"]
        if intent in {"full_investigation", "customer_risk_reasoning"}:
            return ["sql", "graph", "vector"]
        return ["sql"]

    def retrieve(
        self,
        request: RetrievalRequest,
        sql_rows: Iterable[Mapping[str, Any]],
        graph_edges: Iterable[Mapping[str, Any]],
        vector_docs: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        routes = self.route(request)
        responses: List[RetrievalResponse] = []

        if "sql" in routes:
            responses.append(self.sql.retrieve(sql_rows, filters=request.filters or {}, limit=request.top_k * 20))

        if "graph" in routes:
            if not request.customer_gid:
                responses.append(RetrievalResponse(source="graph", records=[], latency_ms=0.0, notes="customer_gid_missing"))
            else:
                responses.append(
                    self.graph.traverse(
                        graph_edges,
                        start_node=request.customer_gid,
                        max_hops=2,
                        allowed_rels=None,
                    )
                )

        if "vector" in routes:
            q = request.query_embedding or []
            if not q:
                responses.append(RetrievalResponse(source="vector", records=[], latency_ms=0.0, notes="query_embedding_missing"))
            else:
                responses.append(
                    self.vector.search(
                        docs=vector_docs,
                        query_embedding=q,
                        metadata_filter=request.filters or {},
                        top_k=request.top_k,
                    )
                )

        return {
            "routes": routes,
            "responses": [
                {
                    "source": r.source,
                    "latency_ms": r.latency_ms,
                    "count": len(r.records),
                    "notes": r.notes,
                    "records": r.records,
                }
                for r in responses
            ],
        }
