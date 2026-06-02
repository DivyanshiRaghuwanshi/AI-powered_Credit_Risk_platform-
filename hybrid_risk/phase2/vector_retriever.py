from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from .types import RetrievalResponse


class VectorRetriever:
    """Phase 2 baseline vector retriever with metadata filtering."""

    def search(
        self,
        docs: Iterable[Mapping[str, Any]],
        query_embedding: List[float],
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        t0 = time.perf_counter()
        q = np.asarray(query_embedding, dtype=float)
        if q.ndim != 1 or q.size == 0:
            raise ValueError("query_embedding must be a non-empty 1D vector")

        metadata_filter = metadata_filter or {}
        scored: List[Dict[str, Any]] = []
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            raise ValueError("query_embedding norm is zero")

        for d in docs:
            if not _metadata_match(d.get("metadata", {}), metadata_filter):
                continue

            vec = np.asarray(d.get("embedding", []), dtype=float)
            if vec.shape != q.shape:
                continue
            v_norm = float(np.linalg.norm(vec))
            if v_norm == 0:
                continue
            sim = float(np.dot(q, vec) / (q_norm * v_norm))
            scored.append(
                {
                    "doc_id": d.get("doc_id"),
                    "text": d.get("text"),
                    "metadata": dict(d.get("metadata", {})),
                    "similarity": sim,
                }
            )

        scored = sorted(scored, key=lambda x: x["similarity"], reverse=True)[: max(0, int(top_k))]
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return RetrievalResponse(
            source="vector",
            records=scored,
            latency_ms=latency_ms,
            notes=f"hits={len(scored)}",
        )


def _metadata_match(meta: Mapping[str, Any], flt: Mapping[str, Any]) -> bool:
    for k, v in flt.items():
        if k not in meta:
            return False
        if meta[k] != v:
            return False
    return True
