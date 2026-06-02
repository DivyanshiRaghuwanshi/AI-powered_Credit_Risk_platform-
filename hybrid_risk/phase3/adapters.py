from __future__ import annotations

from typing import Any, Dict, List, Protocol

from .types import AdapterHealth, AdapterResponse, Phase3RetrievalRequest


class SqlAdapter(Protocol):
    def health(self) -> AdapterHealth: ...

    def retrieve(self, request: Phase3RetrievalRequest) -> AdapterResponse: ...


class GraphAdapter(Protocol):
    def health(self) -> AdapterHealth: ...

    def retrieve(self, request: Phase3RetrievalRequest) -> AdapterResponse: ...


class VectorAdapter(Protocol):
    def health(self) -> AdapterHealth: ...

    def retrieve(self, request: Phase3RetrievalRequest) -> AdapterResponse: ...


class InMemorySqlAdapter:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def health(self) -> AdapterHealth:
        return AdapterHealth(name="sql_in_memory", ok=True, message="in-memory sql adapter")

    def retrieve(self, request: Phase3RetrievalRequest) -> AdapterResponse:
        import time

        t0 = time.perf_counter()
        records = list(self.rows)
        filters = request.filters or {}
        for k, v in filters.items():
            records = [r for r in records if r.get(k) == v]
        if request.customer_gid:
            records = [r for r in records if str(r.get("customer_gid")) == str(request.customer_gid)]
        records = records[: max(0, request.top_k * 20)]
        return AdapterResponse(
            source="sql",
            records=records,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            note=f"rows={len(records)}",
        )


class InMemoryGraphAdapter:
    def __init__(self, edges: List[Dict[str, Any]]) -> None:
        self.edges = edges

    def health(self) -> AdapterHealth:
        return AdapterHealth(name="graph_in_memory", ok=True, message="in-memory graph adapter")

    def retrieve(self, request: Phase3RetrievalRequest) -> AdapterResponse:
        import time

        t0 = time.perf_counter()
        if not request.customer_gid:
            return AdapterResponse(source="graph", records=[], latency_ms=0.0, note="customer_gid_missing")
        cid = str(request.customer_gid)
        out = [e for e in self.edges if str(e.get("source")) == cid or str(e.get("target")) == cid]
        return AdapterResponse(
            source="graph",
            records=out[: max(0, request.top_k * 10)],
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            note=f"edges={len(out)}",
        )


class InMemoryVectorAdapter:
    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs

    def health(self) -> AdapterHealth:
        return AdapterHealth(name="vector_in_memory", ok=True, message="in-memory vector adapter")

    def retrieve(self, request: Phase3RetrievalRequest) -> AdapterResponse:
        import time

        t0 = time.perf_counter()
        filters = request.filters or {}
        out = []
        for d in self.docs:
            meta = d.get("metadata", {})
            if all(meta.get(k) == v for k, v in filters.items()):
                out.append(d)
        return AdapterResponse(
            source="vector",
            records=out[: max(0, request.top_k)],
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            note=f"hits={len(out)}",
        )
