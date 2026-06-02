from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from .adapters import GraphAdapter, SqlAdapter, VectorAdapter
from .types import AdapterHealth, Phase3RetrievalRequest


class Phase3HybridService:
    """Async fan-out retrieval over sql/graph/vector adapters."""

    def __init__(self, sql_adapter: SqlAdapter, graph_adapter: GraphAdapter, vector_adapter: VectorAdapter) -> None:
        self.sql_adapter = sql_adapter
        self.graph_adapter = graph_adapter
        self.vector_adapter = vector_adapter

    def health(self) -> Dict[str, Any]:
        checks = [self.sql_adapter.health(), self.graph_adapter.health(), self.vector_adapter.health()]
        overall = all(c.ok for c in checks)
        return {
            "ok": overall,
            "checks": [{"name": c.name, "ok": c.ok, "message": c.message} for c in checks],
        }

    async def retrieve(self, request: Phase3RetrievalRequest) -> Dict[str, Any]:
        t0 = time.perf_counter()
        tasks = []
        routes = []

        if request.include_sql:
            routes.append("sql")
            tasks.append(asyncio.to_thread(self.sql_adapter.retrieve, request))
        if request.include_graph:
            routes.append("graph")
            tasks.append(asyncio.to_thread(self.graph_adapter.retrieve, request))
        if request.include_vector:
            routes.append("vector")
            tasks.append(asyncio.to_thread(self.vector_adapter.retrieve, request))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for src, res in zip(routes, results):
            if isinstance(res, Exception):
                responses.append(
                    {
                        "source": src,
                        "count": 0,
                        "latency_ms": 0.0,
                        "note": f"error: {res}",
                        "records": [],
                        "ok": False,
                    }
                )
            else:
                responses.append(
                    {
                        "source": res.source,
                        "count": len(res.records),
                        "latency_ms": res.latency_ms,
                        "note": res.note,
                        "records": res.records,
                        "ok": True,
                    }
                )

        return {
            "routes": routes,
            "responses": responses,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 3),
        }
