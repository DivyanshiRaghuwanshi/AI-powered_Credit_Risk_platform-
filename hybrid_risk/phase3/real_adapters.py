from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from .types import AdapterHealth, AdapterResponse, Phase3RetrievalRequest


class PostgresSqlAdapter:
    """Real SQL adapter backed by SQLAlchemy engine."""

    def __init__(self, engine: Any, table_name: str = "master_ews_fibo") -> None:
        self.engine = engine
        self.table_name = table_name

    def health(self) -> AdapterHealth:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return AdapterHealth(name="postgres_sql", ok=True, message="ok")
        except Exception as exc:
            return AdapterHealth(name="postgres_sql", ok=False, message=str(exc))

    def retrieve(self, request: Phase3RetrievalRequest) -> AdapterResponse:
        t0 = time.perf_counter()
        filters = request.filters or {}
        where = []
        params: Dict[str, Any] = {"limit_n": int(max(1, request.top_k * 20))}

        if request.customer_gid:
            where.append('"SK_ID_CURR"::text = :customer_gid')
            params["customer_gid"] = str(request.customer_gid)

        for i, (k, v) in enumerate(filters.items()):
            p = f"p{i}"
            where.append(f'"{k}"::text = :{p}')
            params[p] = str(v)

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = text(f'SELECT * FROM "{self.table_name}"{where_sql} LIMIT :limit_n')

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params)

        return AdapterResponse(
            source="sql",
            records=df.where(pd.notnull(df), None).to_dict(orient="records"),
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            note=f"rows={len(df)}",
        )


class Neo4jGraphAdapter:
    """Real Neo4j adapter (optional dependency)."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database

    def _driver(self):
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as exc:
            raise RuntimeError("neo4j driver missing. Install with: pip install neo4j") from exc
        return GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def health(self) -> AdapterHealth:
        try:
            drv = self._driver()
            with drv.session(database=self.database) as s:
                s.run("RETURN 1 as ok").single()
            drv.close()
            return AdapterHealth(name="neo4j_graph", ok=True, message="ok")
        except Exception as exc:
            return AdapterHealth(name="neo4j_graph", ok=False, message=str(exc))

    def retrieve(self, request: Phase3RetrievalRequest) -> AdapterResponse:
        t0 = time.perf_counter()
        if not request.customer_gid:
            return AdapterResponse(source="graph", records=[], latency_ms=0.0, note="customer_gid_missing")

        q = """
        MATCH (c:Customer {customer_gid: $customer_gid})
        CALL (c) {
          MATCH (c)-[rel]-(n)
          RETURN c AS s, n AS t, type(rel) AS rel_type, 1 AS hop

          UNION

          MATCH (c)-[r1]-(mid)-[r2]-(n2)
          WHERE NOT n2:Customer
          RETURN mid AS s, n2 AS t, type(r2) AS rel_type, 2 AS hop
        }
        WITH DISTINCT s, t, rel_type, hop
        WITH
          s,
          t,
          rel_type,
          hop,
          CASE rel_type
            WHEN 'DEFAULTED_ON' THEN 1
            WHEN 'HAS_BUREAU_RECORD' THEN 2
            WHEN 'APPLIED_FOR' THEN 3
            WHEN 'WORKS_AT' THEN 4
            WHEN 'LINKED_TO_DEVICE' THEN 5
            WHEN 'CONNECTED_TO' THEN 6
            WHEN 'LIVES_IN' THEN 9
            ELSE 7
          END AS rel_rank
        RETURN
          coalesce(
            s.customer_gid,
            s.application_gid,
            s.bureau_record_gid,
            s.name,
            elementId(s)
          ) AS source,
          coalesce(
            t.customer_gid,
            t.application_gid,
            t.bureau_record_gid,
            t.name,
            elementId(t)
          ) AS target,
          rel_type AS rel
        ORDER BY rel_rank ASC, hop ASC
        LIMIT $limit_n
        """
        drv = self._driver()
        rows: List[Dict[str, Any]] = []
        with drv.session(database=self.database) as s:
            rs = s.run(q, customer_gid=str(request.customer_gid), limit_n=max(1, int(request.top_k * 10)))
            for rec in rs:
                rows.append(
                    {
                        "source": str(rec.get("source") or ""),
                        "target": str(rec.get("target") or ""),
                        "rel": str(rec.get("rel") or "UNKNOWN"),
                    }
                )
        drv.close()

        return AdapterResponse(
            source="graph",
            records=rows,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            note=f"paths={len(rows)}",
        )


class PgVectorAdapter:
    """Real pgvector adapter using a metadata+embedding table."""

    def __init__(self, engine: Any, table_name: str = "risk_documents", embedding_col: str = "embedding") -> None:
        self.engine = engine
        self.table_name = table_name
        self.embedding_col = embedding_col

    def health(self) -> AdapterHealth:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return AdapterHealth(name="pgvector", ok=True, message="ok")
        except Exception as exc:
            return AdapterHealth(name="pgvector", ok=False, message=str(exc))

    def retrieve(self, request: Phase3RetrievalRequest) -> AdapterResponse:
        t0 = time.perf_counter()
        emb = request.query_embedding
        if not emb:
            return AdapterResponse(source="vector", records=[], latency_ms=0.0, note="query_embedding_missing")

        # pgvector cosine distance operator: <=>
        emb_literal = "[" + ",".join(str(float(x)) for x in emb) + "]"

        filters = request.filters or {}
        where = []
        params: Dict[str, Any] = {"emb": emb_literal, "limit_n": int(max(1, request.top_k))}
        for i, (k, v) in enumerate(filters.items()):
            p = f"f{i}"
            where.append(f'"{k}"::text = :{p}')
            params[p] = str(v)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        sql = text(
            f'''
            SELECT *, 1 - ("{self.embedding_col}" <=> CAST(:emb AS vector)) AS similarity
            FROM "{self.table_name}"
            {where_sql}
            ORDER BY "{self.embedding_col}" <=> CAST(:emb AS vector)
            LIMIT :limit_n
            '''
        )

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params)

        return AdapterResponse(
            source="vector",
            records=df.where(pd.notnull(df), None).to_dict(orient="records"),
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            note=f"hits={len(df)}",
        )
