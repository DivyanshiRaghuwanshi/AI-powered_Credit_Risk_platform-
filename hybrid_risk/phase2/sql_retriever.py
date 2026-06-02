from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from .types import RetrievalResponse


class SqlRetriever:
    """Phase 2 baseline SQL-style retriever over tabular rows (DataFrame simulation)."""

    def retrieve(
        self,
        rows: Sequence[Mapping[str, Any]],
        filters: Dict[str, Any] | None = None,
        limit: int = 100,
    ) -> RetrievalResponse:
        t0 = time.perf_counter()
        df = pd.DataFrame(rows)
        filters = filters or {}

        for key, value in filters.items():
            if key not in df.columns:
                continue
            if isinstance(value, dict):
                lo = value.get("min")
                hi = value.get("max")
                col = pd.to_numeric(df[key], errors="coerce")
                mask = col.notna()
                if lo is not None:
                    mask = mask & (col >= float(lo))
                if hi is not None:
                    mask = mask & (col <= float(hi))
                df = df[mask]
            else:
                df = df[df[key] == value]

        if limit > 0:
            df = df.head(int(limit))

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return RetrievalResponse(
            source="sql",
            records=df.to_dict(orient="records"),
            latency_ms=latency_ms,
            notes=f"rows={len(df)}",
        )
