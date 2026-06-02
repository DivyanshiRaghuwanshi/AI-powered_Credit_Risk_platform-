from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Phase5CopilotRequest:
    intent: str
    customer_gid: Optional[str] = None
    customer_id: Optional[int] = None
    model_id: Optional[str] = None
    role: str = "analyst"
    user_id: str = "demo_user"
    request_id: str = "demo_request"
    policy_profile: Optional[str] = None
    strict_route_policy: bool = True
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
class Phase5CopilotHealth:
    name: str
    ok: bool
    message: str

