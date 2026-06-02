from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Phase4InvestigationRequest:
    intent: str
    customer_gid: Optional[str] = None
    customer_id: Optional[int] = None
    model_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = 5
    include_sql: bool = True
    include_graph: bool = True
    include_vector: bool = True
    include_model: bool = True
    include_explanation: bool = True
    explanation_top_k: int = 8
    apply_calibration: bool = True


@dataclass(frozen=True)
class Phase4AdapterHealth:
    name: str
    ok: bool
    message: str


@dataclass(frozen=True)
class Phase4ModelResponse:
    source: str
    record: Dict[str, Any]
    latency_ms: float
    note: str = ""

