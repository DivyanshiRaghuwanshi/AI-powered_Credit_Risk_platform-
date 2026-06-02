from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievalRequest:
    intent: str
    customer_gid: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = 5


@dataclass(frozen=True)
class RetrievalResponse:
    source: str
    records: List[Dict[str, Any]]
    latency_ms: float
    notes: str = ""
