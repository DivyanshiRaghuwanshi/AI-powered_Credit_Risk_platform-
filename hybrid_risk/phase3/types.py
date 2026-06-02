from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Phase3RetrievalRequest:
    intent: str
    customer_gid: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = 5
    include_sql: bool = True
    include_graph: bool = True
    include_vector: bool = True


@dataclass(frozen=True)
class AdapterHealth:
    name: str
    ok: bool
    message: str


@dataclass(frozen=True)
class AdapterResponse:
    source: str
    records: List[Dict[str, Any]]
    latency_ms: float
    note: str = ""
