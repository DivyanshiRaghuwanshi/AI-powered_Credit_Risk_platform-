from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .types import RetrievalResponse


class GraphRetriever:
    """Phase 2 baseline graph traversal retriever."""

    def traverse(
        self,
        edges: Iterable[Mapping[str, Any]],
        start_node: str,
        max_hops: int = 2,
        allowed_rels: Optional[List[str]] = None,
    ) -> RetrievalResponse:
        t0 = time.perf_counter()
        graph: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for e in edges:
            rel = str(e.get("rel", ""))
            if allowed_rels and rel not in allowed_rels:
                continue
            src = str(e.get("source"))
            dst = str(e.get("target"))
            payload = {"source": src, "target": dst, "rel": rel, "props": dict(e.get("props", {}))}
            graph[src].append(payload)

        visited = {start_node}
        q = deque([(start_node, 0)])
        out: List[Dict[str, Any]] = []

        while q:
            node, depth = q.popleft()
            if depth >= max_hops:
                continue
            for edge in graph.get(node, []):
                out.append(edge)
                nxt = edge["target"]
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, depth + 1))

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return RetrievalResponse(
            source="graph",
            records=out,
            latency_ms=latency_ms,
            notes=f"visited_nodes={len(visited)} edges={len(out)}",
        )
