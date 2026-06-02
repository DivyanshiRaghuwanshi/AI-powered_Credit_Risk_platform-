import argparse
import json
import logging
import math
import os
import pickle
import re
import time
import uuid
from collections import Counter, defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from hybrid_risk.phase0.bootstrap import build_phase0_manifest, resolve_data_dir, validate_phase0_workspace
from hybrid_risk.phase1.data_contracts import ColumnContract, DataContract, validate_dataframe_contract
from hybrid_risk.phase1.entity_resolution import EntityResolver
from hybrid_risk.phase1.governance import create_audit_event, mask_pii, validate_claim_evidence
from hybrid_risk.phase2.orchestrator import HybridRetrievalOrchestrator
from hybrid_risk.phase2.types import RetrievalRequest
from hybrid_risk.phase3.adapters import InMemoryGraphAdapter, InMemorySqlAdapter, InMemoryVectorAdapter
from hybrid_risk.phase3.real_adapters import Neo4jGraphAdapter, PgVectorAdapter, PostgresSqlAdapter
from hybrid_risk.phase3.service import Phase3HybridService
from hybrid_risk.phase3.types import Phase3RetrievalRequest as Phase3ServiceRequest
from hybrid_risk.phase4.adapters import InMemoryModelAdapter, ShapModelArtifactAdapter
from hybrid_risk.phase4.service import Phase4InvestigationService
from hybrid_risk.phase4.types import Phase4InvestigationRequest as Phase4ServiceRequest
from hybrid_risk.phase5.service import Phase5CopilotService
from hybrid_risk.phase5.types import Phase5CopilotRequest as Phase5ServiceRequest
from hybrid_risk.llm import invoke_llm_json, invoke_llm_text

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
except Exception:
    RandomForestClassifier = None
    roc_auc_score = None
    train_test_split = None
    DecisionTreeClassifier = None

try:
    import shap
except Exception:
    shap = None


RELATIONSHIPS = [
    "application_{train|test}.SK_ID_CURR = bureau.SK_ID_CURR",
    "bureau.SK_ID_BUREAU = bureau_balance.SK_ID_BUREAU",
    "application_{train|test}.SK_ID_CURR = previous_application.SK_ID_CURR",
    "previous_application.SK_ID_PREV = pos_cash_balance.SK_ID_PREV",
    "previous_application.SK_ID_PREV = installments_payments.SK_ID_PREV",
    "previous_application.SK_ID_PREV = credit_card_balance.SK_ID_PREV",
]
METADATA_TABLE = "metadata_column_descriptions"

# In-memory follow-up context (same conversation_id across requests).
_CONVERSATIONS: Dict[str, List[Dict[str, str]]] = defaultdict(list)
_CONV_LOCK = Lock()
_LIME_MODELS: Dict[str, Dict[str, Any]] = {}
_LIME_LOCK = Lock()
_LATEST_LIME_MODEL_ID: Optional[str] = None
_RULE_REGISTRY_LOCK = Lock()
_ENTITY_RESOLVER = EntityResolver()
_PHASE2_ORCH = HybridRetrievalOrchestrator()
_PHASE3_SERVICE = Phase3HybridService(
    sql_adapter=InMemorySqlAdapter(rows=[]),
    graph_adapter=InMemoryGraphAdapter(edges=[]),
    vector_adapter=InMemoryVectorAdapter(docs=[]),
)
_PHASE4_SERVICE = Phase4InvestigationService(
    phase3_service=_PHASE3_SERVICE,
    model_adapter=InMemoryModelAdapter(),
)
_PHASE5_SERVICE = Phase5CopilotService(phase4_service=_PHASE4_SERVICE)


def _build_phase3_real_service() -> Phase3HybridService:
    engine = get_engine()
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
    pgvector_table = os.getenv("PGVECTOR_TABLE", "risk_documents")
    pgvector_col = os.getenv("PGVECTOR_EMBEDDING_COL", "embedding")
    sql_table = os.getenv("PHASE3_SQL_TABLE", "master_ews_fibo")

    return Phase3HybridService(
        sql_adapter=PostgresSqlAdapter(engine=engine, table_name=sql_table),
        graph_adapter=Neo4jGraphAdapter(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_db,
        ),
        vector_adapter=PgVectorAdapter(
            engine=engine,
            table_name=pgvector_table,
            embedding_col=pgvector_col,
        ),
    )

load_dotenv()
_MAX_TURNS_PER_CONVERSATION = int(os.getenv("CONVERSATION_MAX_TURNS", "12"))
app = FastAPI(title="Home Credit NL SQL API", version="1.0.0")


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("nl_sql_qa")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        try:
            log_path = Path(os.getenv("APP_LOG_FILE", str(Path(__file__).resolve().parent / "logs" / "app.log")))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        except Exception as e:
            # Fallback to console logging if file logging fails
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            print(f"Warning: Could not create file logger, using console: {e}")
    return logger


LOGGER = _build_logger()


def get_model_dir() -> Path:
    env_val = os.getenv("LIME_MODEL_DIR", "").strip()
    # Use env var if it's absolute, otherwise use default
    if env_val and env_val != "." and not env_val.startswith("."):
        p = Path(env_val).resolve()
    else:
        p = Path(__file__).resolve().parent / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_phase4_real_service() -> Phase4InvestigationService:
    model_dir = os.getenv("LIME_MODEL_DIR", "").strip() or str(get_model_dir())
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_latest = os.getenv("PHASE4_LATEST_MODEL_FILE", "latest_model_id.txt")
    model_adapter = ShapModelArtifactAdapter(
        engine=get_engine(),
        model_dir=model_dir,
        latest_model_file=model_latest,
    )
    return Phase4InvestigationService(
        phase3_service=_build_phase3_real_service(),
        model_adapter=model_adapter,
    )


def _build_phase5_real_service() -> Phase5CopilotService:
    return Phase5CopilotService(phase4_service=_build_phase4_real_service())


def _model_artifact_path(model_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\\-]", "_", model_id)
    return get_model_dir() / f"{safe}.pkl"


def _persist_lime_model(
    payload: Dict[str, Any],
    update_latest: bool = True,
    latest_file_name: str = "latest_model_id.txt",
) -> Path:
    path = _model_artifact_path(payload["model_id"])
    with path.open("wb") as f:
        pickle.dump(payload, f)
    if update_latest:
        latest_path = get_model_dir() / latest_file_name
        latest_path.write_text(str(payload["model_id"]), encoding="utf-8")
    return path


def _load_latest_model_id_from_disk(latest_file_name: str = "latest_model_id.txt") -> Optional[str]:
    latest_path = get_model_dir() / latest_file_name
    if not latest_path.exists():
        return None
    value = latest_path.read_text(encoding="utf-8").strip()
    return value or None


def _load_lime_model_from_disk(model_id: str) -> Optional[Dict[str, Any]]:
    path = _model_artifact_path(model_id)
    if not path.exists():
        return None
    with path.open("rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, dict):
        return loaded
    return None


def get_rule_registry_path() -> Path:
    env_val = os.getenv("RULE_REGISTRY_PATH", "").strip()
    # Use env var if it's absolute, otherwise use default
    if env_val and env_val != "." and not env_val.startswith("."):
        p = Path(env_val).resolve()
    else:
        p = Path(__file__).resolve().parent / "rules" / "approved_rules.json"
    return p


def _load_rule_registry() -> List[Dict[str, Any]]:
    path = get_rule_registry_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return payload
    return []


def _save_rule_registry(rules: List[Dict[str, Any]]) -> None:
    try:
        path = get_rule_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rules, ensure_ascii=True, indent=2), encoding="utf-8")
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write rules to {path}. Check directory permissions. "
            f"Set RULE_REGISTRY_PATH to a writable location: {e}"
        ) from e
    except Exception as e:
        raise IOError(f"Error saving rule registry to {path}: {e}") from e


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question about loaded tables.")
    limit_rows: int = Field(default=200, ge=1, le=5000)
    conversation_id: Optional[str] = Field(
        default=None,
        description="Reuse the same id across turns so follow-up questions inherit prior Q/SQL/context.",
    )
    profile: bool = Field(
        default=False,
        description="If true, response includes timings_ms with per-step wall times in milliseconds.",
    )


class AskResponse(BaseModel):
    conversation_id: str
    question: str
    sql: str
    answer: str
    row_count: int
    preview: List[dict]
    timings_ms: Optional[Dict[str, float]] = None


class RuleDraftRequest(BaseModel):
    rule_intent: str = Field(..., min_length=4, description="Natural-language rule intent.")
    table_name: str = Field(default="master_ews_fibo")
    target_column: str = Field(default="EWS_LABEL")
    conversation_id: Optional[str] = None
    profile: bool = Field(default=False)


class RuleDraftResponse(BaseModel):
    conversation_id: str
    rule_name: str
    where_clause: str
    rationale: str
    example_sql: str
    timings_ms: Optional[Dict[str, float]] = None


class RuleEvaluateRequest(BaseModel):
    table_name: str = Field(default="master_ews_fibo")
    target_column: str = Field(default="EWS_LABEL")
    positive_value: str = Field(default="1")
    where_clause: str = Field(..., min_length=3)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    alternative: str = Field(default="greater", description="greater | less | two-sided")


class RuleEvaluateResponse(BaseModel):
    table_name: str
    target_column: str
    where_clause: str
    total_n: int
    total_positive_n: int
    rule_n: int
    rule_positive_n: int
    base_rate: float
    rule_rate: float
    lift: float
    z_score: float
    p_value: float
    reject_null: bool
    alternative: str
    summary: str


class RuleRegisterRequest(BaseModel):
    rule_name: str = Field(..., min_length=3)
    where_clause: str = Field(..., min_length=3)
    table_name: str = Field(default="master_ews_fibo")
    target_column: str = Field(default="EWS_LABEL")
    positive_value: str = Field(default="1")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    alternative: str = Field(default="greater", description="greater | less | two-sided")
    min_lift: float = Field(default=1.05, ge=0.0)
    min_rule_n: int = Field(default=200, ge=1)
    require_reject_null: bool = Field(default=True)
    notes: str = Field(default="")
    owner: str = Field(default="risk_analytics")


class RuleRegisterResponse(BaseModel):
    rule_id: str
    status: str
    rule_name: str
    registered_at: str
    where_clause: str
    evaluation: RuleEvaluateResponse
    summary: str


class RuleListResponse(BaseModel):
    count: int
    rules: List[dict]


class RuleApplyCustomerRequest(BaseModel):
    customer_id: int
    table_name: str = Field(default="master_ews_fibo")
    id_column: str = Field(default="SK_ID_CURR")
    include_all_rules: bool = Field(default=False)


class RuleApplyCustomerResponse(BaseModel):
    customer_id: int
    table_name: str
    matched_count: int
    matched_rules: List[dict]
    summary: str


class HypothesisTestRequest(BaseModel):
    hypothesis: str = Field(..., min_length=6)
    null_hypothesis: str = Field(
        default="Customers matching this rule have the same EWS positive rate as others."
    )
    table_name: str = Field(default="master_ews_fibo")
    target_column: str = Field(default="EWS_LABEL")
    positive_value: str = Field(default="1")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    alternative: str = Field(default="greater", description="greater | less | two-sided")
    conversation_id: Optional[str] = None
    profile: bool = Field(default=False)


class HypothesisTestResponse(BaseModel):
    conversation_id: str
    hypothesis: str
    null_hypothesis: str
    drafted_rule_name: str
    where_clause: str
    evaluation: RuleEvaluateResponse
    follow_up_questions: List[str]
    timings_ms: Optional[Dict[str, float]] = None


class LimeTrainRequest(BaseModel):
    table_name: str = Field(default="master_ews_fibo")
    id_column: str = Field(default="SK_ID_CURR")
    target_column: str = Field(default="EWS_LABEL")
    positive_value: str = Field(default="1")
    sample_rows: int = Field(default=120000, ge=2000, le=1000000)
    test_size: float = Field(default=0.20, gt=0.05, lt=0.50)
    random_state: int = Field(default=42)
    exclude_columns: List[str] = Field(default_factory=list)


class LimeTrainResponse(BaseModel):
    model_id: str
    model_path: str
    table_name: str
    id_column: str
    target_column: str
    positive_value: str
    train_rows: int
    feature_count: int
    positive_rate: float
    validation_auc: Optional[float] = None
    status: str


class LimeExplainRequest(BaseModel):
    model_id: Optional[str] = None
    customer_id: int
    top_k: int = Field(default=8, ge=3, le=25)
    conversation_id: Optional[str] = None
    profile: bool = Field(default=False)


class LimeExplainResponse(BaseModel):
    conversation_id: str
    model_id: str
    customer_id: int
    predicted_probability: float
    predicted_label: int
    feature_contributions: List[dict]
    explanation: str
    timings_ms: Optional[Dict[str, float]] = None


class RuleModelTrainRequest(BaseModel):
    table_name: str = Field(default="master_ews_fibo")
    id_column: str = Field(default="SK_ID_CURR")
    target_column: str = Field(default="EWS_LABEL")
    positive_value: str = Field(default="1")
    sample_rows: int = Field(default=120000, ge=2000, le=1000000)
    test_size: float = Field(default=0.20, gt=0.05, lt=0.50)
    random_state: int = Field(default=42)
    max_depth: int = Field(default=4, ge=2, le=12)
    min_samples_leaf: int = Field(default=200, ge=20, le=100000)
    top_k_features: int = Field(default=15, ge=5, le=50)
    max_rules: int = Field(default=12, ge=3, le=50)
    exclude_columns: List[str] = Field(default_factory=list)
    auto_register_extracted_rules: bool = True
    auto_register_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    auto_register_alternative: str = Field(default="greater", description="greater | less | two-sided")
    auto_register_min_lift: float = Field(default=1.05, ge=0.0)
    auto_register_min_rule_n: int = Field(default=200, ge=1)
    auto_register_require_reject_null: bool = True
    auto_register_owner: str = "risk_analytics"
    auto_register_notes_prefix: str = "Auto-registered from decision-tree extracted rule"


class RuleModelTrainResponse(BaseModel):
    model_id: str
    model_path: str
    table_name: str
    id_column: str
    target_column: str
    positive_value: str
    train_rows: int
    feature_count: int
    positive_rate: float
    validation_auc: Optional[float] = None
    top_features: List[dict]
    extracted_rules: List[dict]
    auto_registered_count: int = 0
    auto_registered_rule_ids: List[str] = Field(default_factory=list)
    auto_register_failures: List[dict] = Field(default_factory=list)
    status: str


class Phase0ValidateRequest(BaseModel):
    project_root: str = Field(default=str(Path(__file__).resolve().parent))
    env_raw_data_dir: Optional[str] = None


class Phase0ValidateResponse(BaseModel):
    project_root: str
    is_valid: bool
    missing_dirs: List[str]
    missing_files: List[str]
    resolved_data_dir: str
    manifest: Dict[str, Any]


class Phase1ResolvePairRequest(BaseModel):
    record_a: Dict[str, Any]
    record_b: Dict[str, Any]


class Phase1ResolveClusterRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(default_factory=list)


class Phase1ResolveResponse(BaseModel):
    status: str
    confidence: float
    reasons: List[str]


class ContractColumnRequest(BaseModel):
    name: str
    dtype: str = "float"
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unique: bool = False
    critical: bool = False
    max_null_ratio: Optional[float] = None


class Phase1ContractValidateRequest(BaseModel):
    source_name: str = "home_credit"
    table_name: str = "adhoc_table"
    columns: List[ContractColumnRequest]
    rows: List[Dict[str, Any]]


class Phase1ContractValidateResponse(BaseModel):
    is_valid: bool
    issue_count: int
    issues: List[Dict[str, str]]


class Phase1GovernanceCheckRequest(BaseModel):
    role: str = Field(default="analyst")
    record: Dict[str, Any]
    claims: List[str] = Field(default_factory=list)
    evidence_map: Dict[str, List[str]] = Field(default_factory=dict)
    user_id: str = "demo_user"
    request_id: str = "demo_request"
    tool_calls: List[str] = Field(default_factory=list)
    response_text: str = ""


class Phase1GovernanceCheckResponse(BaseModel):
    masked_record: Dict[str, Any]
    unsupported_claims: List[str]
    audit_event: Dict[str, Any]


class Phase2RetrieveRequest(BaseModel):
    intent: str
    customer_gid: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_embedding: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1, le=200)
    sql_rows: List[Dict[str, Any]] = Field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = Field(default_factory=list)
    vector_docs: List[Dict[str, Any]] = Field(default_factory=list)


class Phase2RetrieveResponse(BaseModel):
    routes: List[str]
    responses: List[Dict[str, Any]]


class Phase3RetrieveRequest(BaseModel):
    intent: str
    customer_gid: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1, le=500)
    include_sql: bool = True
    include_graph: bool = True
    include_vector: bool = True
    sql_rows: List[Dict[str, Any]] = Field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = Field(default_factory=list)
    vector_docs: List[Dict[str, Any]] = Field(default_factory=list)


class Phase3RetrieveResponse(BaseModel):
    routes: List[str]
    responses: List[Dict[str, Any]]
    total_latency_ms: float


class Phase4InvestigateRequest(BaseModel):
    intent: str
    customer_gid: Optional[str] = None
    customer_id: Optional[int] = None
    model_id: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1, le=500)
    include_sql: bool = True
    include_graph: bool = True
    include_vector: bool = True
    include_model: bool = True
    include_explanation: bool = True
    explanation_top_k: int = Field(default=8, ge=1, le=25)
    apply_calibration: bool = True
    sql_rows: List[Dict[str, Any]] = Field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = Field(default_factory=list)
    vector_docs: List[Dict[str, Any]] = Field(default_factory=list)


class Phase4InvestigateResponse(BaseModel):
    routes: List[str]
    responses: List[Dict[str, Any]]
    summary: Dict[str, Any]
    total_latency_ms: float


class Phase5CopilotRequest(BaseModel):
    intent: str
    customer_gid: Optional[str] = None
    customer_id: Optional[int] = None
    model_id: Optional[str] = None
    role: str = Field(default="analyst")
    user_id: str = Field(default="demo_user")
    request_id: str = Field(default="demo_request")
    policy_profile: Optional[str] = None
    strict_route_policy: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1, le=500)
    include_sql: bool = True
    include_graph: bool = True
    include_vector: bool = True
    include_model: bool = True
    include_explanation: bool = True
    explanation_top_k: int = Field(default=8, ge=1, le=25)
    apply_calibration: bool = True
    sql_rows: List[Dict[str, Any]] = Field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = Field(default_factory=list)
    vector_docs: List[Dict[str, Any]] = Field(default_factory=list)


class Phase5CopilotResponse(BaseModel):
    routes: List[str]
    responses: List[Dict[str, Any]]
    summary: Dict[str, Any]
    claims: List[str]
    evidence_map: Dict[str, List[str]]
    trace: Dict[str, Any]
    audit_event: Dict[str, Any]
    total_latency_ms: float


class Phase5CustomerInsightRequest(BaseModel):
    customer_id: Optional[int] = None
    customer_gid: Optional[str] = None
    table_name: str = Field(default="home_credit_future_dpd_labels")
    id_column: str = Field(default="SK_ID_CURR")
    include_all_rules: bool = False
    intent: str = "customer_risk_reasoning"
    model_id: Optional[str] = None
    role: str = Field(default="analyst")
    user_id: str = Field(default="demo_user")
    request_id: str = Field(default="customer_insight")
    policy_profile: Optional[str] = None
    strict_route_policy: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1, le=500)
    include_sql: bool = True
    include_graph: bool = True
    include_vector: bool = False
    include_model: bool = True
    include_explanation: bool = True
    explanation_top_k: int = Field(default=8, ge=1, le=25)
    apply_calibration: bool = True
    sql_rows: List[Dict[str, Any]] = Field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = Field(default_factory=list)
    vector_docs: List[Dict[str, Any]] = Field(default_factory=list)


class Phase5CustomerInsightResponse(BaseModel):
    customer_id: int
    customer_gid: str
    overall_risk_level: str
    overall_risk_score: float
    summary: str
    recommended_actions: List[str]
    next_questions: List[str]
    model_summary: Dict[str, Any]
    rules_summary: Dict[str, Any]
    graph_summary: Dict[str, Any]
    hybrid_detail: Dict[str, Any]


def normalize_table_name(file_name: str) -> str:
    base = Path(file_name).stem.lower()
    return re.sub(r"[^a-z0-9_]+", "_", base)


def build_pg_url() -> str:
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def get_engine() -> Engine:
    return create_engine(build_pg_url())


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _extract_usage(response: object) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _accumulate_usage(
    usage_tracker: Optional[Dict[str, Dict[str, int]]],
    step: str,
    usage: Dict[str, int],
) -> None:
    if usage_tracker is None:
        return
    usage_tracker[step] = usage
    agg = usage_tracker.setdefault("aggregate", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    agg["prompt_tokens"] += usage.get("prompt_tokens", 0)
    agg["completion_tokens"] += usage.get("completion_tokens", 0)
    agg["total_tokens"] += usage.get("total_tokens", 0)


def default_data_dir() -> Path:
    env_dir = os.getenv("RAW_DATA_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if p.exists():
            return p

    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "data" / "raw",
        project_root / "raw_kaggle_data",
        project_root.parent / "MAIN_RULE_EXTRACTION" / "raw_kaggle_data",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def list_data_files(data_dir: Path) -> List[Path]:
    files = [f for f in data_dir.glob("*.csv") if f.name != "HomeCredit_columns_description.csv"]
    if not files:
        raise FileNotFoundError(f"No data CSV files found in: {data_dir}")
    return sorted(files)


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(path, **kwargs)


def load_csvs_to_postgres(data_dir: Path, schema: str = "public", sample_rows: int | None = None) -> None:
    engine = get_engine()
    files = list_data_files(data_dir)
    for csv_path in files:
        table_name = normalize_table_name(csv_path.name)
        print(f"Loading {csv_path.name} -> {schema}.{table_name}")
        if sample_rows:
            df = read_csv_with_fallback(csv_path, nrows=sample_rows)
            df.to_sql(table_name, engine, schema=schema, if_exists="replace", index=False, chunksize=10_000)
        else:
            for i, chunk in enumerate(
                read_csv_with_fallback(csv_path, chunksize=50_000, low_memory=False)
            ):
                chunk.to_sql(
                    table_name,
                    engine,
                    schema=schema,
                    if_exists="replace" if i == 0 else "append",
                    index=False,
                    chunksize=10_000,
                )
    print("Data loading completed.")


def load_column_descriptions(description_csv: Path) -> pd.DataFrame:
    df = read_csv_with_fallback(description_csv)
    raw_table = df["Table"].fillna("").astype(str).str.replace(".csv", "", regex=False)
    table_train = raw_table.str.replace("{train|test}", "train", regex=False).apply(normalize_table_name)
    table_test = raw_table.str.replace("{train|test}", "test", regex=False).apply(normalize_table_name)
    df["table_name"] = table_train
    df["column_name"] = df["Row"].fillna("").astype(str)
    df["description"] = df["Description"].fillna("").astype(str)
    expanded = df[["table_name", "column_name", "description"]].copy()
    # Duplicate application_{train|test} metadata for application_test to keep prompt context consistent.
    train_rows = expanded[expanded["table_name"] == "application_train"].copy()
    if not train_rows.empty:
        train_rows["table_name"] = table_test.loc[train_rows.index]
        expanded = pd.concat([expanded, train_rows], ignore_index=True)
    return expanded


def load_metadata_table(data_dir: Path, schema: str = "public") -> None:
    desc_csv = data_dir / "HomeCredit_columns_description.csv"
    if not desc_csv.exists():
        raise FileNotFoundError(f"Missing metadata file: {desc_csv}")
    desc_df = load_column_descriptions(desc_csv)
    desc_df = desc_df[(desc_df["table_name"] != "") & (desc_df["column_name"] != "")]
    engine = get_engine()
    desc_df.to_sql(METADATA_TABLE, engine, schema=schema, if_exists="replace", index=False, chunksize=10_000)
    print(f"Metadata loaded into {schema}.{METADATA_TABLE}.")


def get_column_descriptions_from_db(schema: str = "public") -> pd.DataFrame:
    sql = text(
        f"""
        SELECT table_name, column_name, description
        FROM {schema}.{METADATA_TABLE}
        """
    )
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def get_db_schema(engine: Engine, schema: str = "public") -> Dict[str, List[str]]:
    sql = text(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name <> :metadata_table
        ORDER BY table_name, ordinal_position
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema_name": schema, "metadata_table": METADATA_TABLE}).fetchall()
    schema_map: Dict[str, List[str]] = {}
    for table_name, column_name in rows:
        schema_map.setdefault(table_name, []).append(column_name)
    return schema_map


def build_semantic_context(schema_map: Dict[str, List[str]], desc_df: pd.DataFrame) -> str:
    available_tables = set(schema_map.keys())
    lines = []
    lines.append(
        'Identifier rule: table and column names are case-sensitive in this database. '
        'Always use double quotes around identifiers exactly as listed below, '
        'e.g. "application_train"."AMT_CREDIT".'
    )
    lines.append("")
    for table_name, columns in schema_map.items():
        lines.append(f"Table: {table_name}")
        lines.append(f'Quoted table: "{table_name}"')
        quoted_columns = ", ".join([f'"{col}"' for col in columns])
        lines.append(f"Columns: {quoted_columns}")
        table_desc = desc_df[desc_df["table_name"] == table_name]
        if not table_desc.empty:
            lines.append("Column descriptions:")
            for _, row in table_desc.iterrows():
                col = row["column_name"]
                desc = row["description"]
                if col and desc:
                    lines.append(f"- {col}: {desc}")
        lines.append("")
    lines.append("Known relationships (from Home Credit ontology, filtered to available tables):")
    for rel in RELATIONSHIPS:
        rel_lower = rel.lower()
        needs_prev = "previous_application" in rel_lower and "previous_application" not in available_tables
        needs_pos = "pos_cash_balance" in rel_lower and "pos_cash_balance" not in available_tables
        needs_installments = "installments_payments" in rel_lower and "installments_payments" not in available_tables
        needs_cc = "credit_card_balance" in rel_lower and "credit_card_balance" not in available_tables
        needs_bureau = "bureau" in rel_lower and "bureau" not in available_tables
        needs_bureau_balance = "bureau_balance" in rel_lower and "bureau_balance" not in available_tables
        needs_application = (
            "application_{train|test}" in rel_lower
            and ("application_train" not in available_tables and "application_test" not in available_tables)
        )
        if any(
            [
                needs_prev,
                needs_pos,
                needs_installments,
                needs_cc,
                needs_bureau,
                needs_bureau_balance,
                needs_application,
            ]
        ):
            continue
        lines.append(f"- {rel}")
    return "\n".join(lines).strip()


def _conversation_subtext(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    lines = [
        "Prior turns in this conversation (use when the current question is a follow-up, "
        "e.g. 'which quarter had the highest', 'same as before', 'drill into that trend'):"
    ]
    for i, turn in enumerate(history[-8:], 1):
        lines.append(f"Turn {i} — User: {turn['user']}")
        lines.append(f"  SQL executed:\n{turn['sql'][:2000]}")
        lines.append(f"  Result preview (markdown, truncated):\n{turn['preview_md'][:3000]}")
        lines.append(f"  Answer given: {turn['answer'][:1200]}")
        lines.append("")
    return "\n".join(lines).strip()


def _record_turn(conversation_id: str, user: str, sql: str, answer: str, result_df: pd.DataFrame) -> None:
    preview_md = dataframe_preview(result_df, max_rows=15) if not result_df.empty else "No rows."
    turn = {
        "user": user,
        "sql": sql,
        "answer": answer[:4000],
        "preview_md": preview_md[:6000],
    }
    with _CONV_LOCK:
        _CONVERSATIONS[conversation_id].append(turn)
        while len(_CONVERSATIONS[conversation_id]) > _MAX_TURNS_PER_CONVERSATION:
            _CONVERSATIONS[conversation_id].pop(0)


def generate_sql_query(
    user_question: str,
    semantic_context: str,
    conversation_context: str = "",
    usage_tracker: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    system_prompt = (
        "You are a PostgreSQL SQL expert. Produce a single SELECT query only. "
        "Do not generate INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE. "
        "Use only tables and columns in provided context. Prefer explicit JOINs. "
        "Use double quotes for all table/column identifiers exactly as provided in the context. "
        "If Prior turns are present, treat follow-up questions as continuing that analysis "
        "(same metrics, time grain, or filters unless the user changes them)."
    )
    prior_block = conversation_context if conversation_context.strip() else "(No prior conversation.)"
    user_prompt = f"""
Database context:
{semantic_context}

{prior_block}

Current user question:
{user_question}

Return JSON exactly in this format:
{{"sql": "<query>"}}
"""
    parsed = invoke_llm_json(system_prompt, user_prompt, temperature=0.0)
    sql = str(parsed.get("sql", "")).strip()
    if not sql:
        raise ValueError("LLM did not return SQL.")
    return sql


def repair_sql_query(
    user_question: str,
    semantic_context: str,
    failed_sql: str,
    db_error: str,
    conversation_context: str = "",
    usage_tracker: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    system_prompt = (
        "You are a PostgreSQL SQL expert. The previous query failed. "
        "Return a corrected single SELECT query only. "
        "Do not generate INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE. "
        "Use double quotes for all identifiers exactly as provided in context. "
        "Respect follow-up intent from Prior turns if provided."
    )
    prior_block = conversation_context if conversation_context.strip() else "(No prior conversation.)"
    user_prompt = f"""
Database context:
{semantic_context}

{prior_block}

Current user question:
{user_question}

Failed SQL:
{failed_sql}

Database error:
{db_error}

Fix the SQL so it runs in PostgreSQL and answers the same question.
Return JSON exactly in this format:
{{"sql": "<query>"}}
"""
    parsed = invoke_llm_json(system_prompt, user_prompt, temperature=0.0)
    sql = str(parsed.get("sql", "")).strip()
    if not sql:
        raise ValueError("LLM did not return repaired SQL.")
    return sql


def generate_rule_where_clause(
    rule_intent: str,
    table_name: str,
    target_column: str,
    semantic_context: str,
    conversation_context: str = "",
    usage_tracker: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, str]:
    system_prompt = (
        "You are a risk analytics SQL assistant. Convert banker intent into a PostgreSQL boolean WHERE clause only. "
        "Do not use SELECT/CTE/JOIN/subqueries. Use only columns from the provided table and context. "
        "The clause must be directly embeddable in CASE WHEN (<where_clause>) THEN 1 ELSE 0 END. "
        "Use double quotes around identifiers."
    )
    prior_block = conversation_context if conversation_context.strip() else "(No prior conversation.)"
    user_prompt = f"""
Database context:
{semantic_context}

{prior_block}

Target table for rule: "{table_name}"
Target label column: "{target_column}"
Rule intent from banker:
{rule_intent}

Return JSON exactly:
{{
  "rule_name": "<short_rule_name>",
  "where_clause": "<boolean_expression_only>",
  "rationale": "<1-2 sentence business rationale>"
}}
"""
    payload = invoke_llm_json(system_prompt, user_prompt, temperature=0.1)
    return {
        "rule_name": str(payload.get("rule_name", "nl_rule")).strip() or "nl_rule",
        "where_clause": str(payload.get("where_clause", "")).strip(),
        "rationale": str(payload.get("rationale", "")).strip(),
    }


def ensure_safe_where_clause(where_clause: str) -> None:
    if not where_clause or not where_clause.strip():
        raise ValueError("where_clause is empty.")
    lowered = where_clause.lower()
    blocked = [
        ";",
        "--",
        "/*",
        "*/",
        " select ",
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " truncate ",
        " create ",
        " union ",
        " with ",
    ]
    padded = f" {lowered} "
    if any(token in padded for token in blocked):
        raise ValueError("Unsafe where_clause detected.")


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _p_value_from_z(z: float, alternative: str) -> float:
    alt = alternative.strip().lower()
    if alt not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be one of: greater, less, two-sided")
    if alt == "greater":
        return max(0.0, min(1.0, 1.0 - _norm_cdf(z)))
    if alt == "less":
        return max(0.0, min(1.0, _norm_cdf(z)))
    return max(0.0, min(1.0, 2.0 * (1.0 - _norm_cdf(abs(z)))))


def evaluate_rule(
    table_name: str,
    target_column: str,
    positive_value: str,
    where_clause: str,
    alpha: float = 0.05,
    alternative: str = "greater",
) -> Dict[str, Any]:
    ensure_safe_where_clause(where_clause)

    engine = get_engine()
    schema_map = get_db_schema(engine)
    if table_name not in schema_map:
        raise ValueError(f'Table "{table_name}" not found in schema.')
    if target_column not in schema_map[table_name]:
        raise ValueError(f'Column "{target_column}" not found in table "{table_name}".')

    sql = text(
        f"""
        SELECT
            COUNT(*)::bigint AS total_n,
            SUM(CASE WHEN "{target_column}"::text = :positive_value THEN 1 ELSE 0 END)::bigint AS total_pos_n,
            SUM(CASE WHEN ({where_clause}) THEN 1 ELSE 0 END)::bigint AS rule_n,
            SUM(CASE WHEN ({where_clause}) AND "{target_column}"::text = :positive_value THEN 1 ELSE 0 END)::bigint AS rule_pos_n
        FROM "{table_name}"
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"positive_value": str(positive_value)}).mappings().first()
    if row is None:
        raise ValueError("Could not evaluate rule (empty SQL result).")

    total_n = int(row["total_n"] or 0)
    total_pos_n = int(row["total_pos_n"] or 0)
    rule_n = int(row["rule_n"] or 0)
    rule_pos_n = int(row["rule_pos_n"] or 0)
    non_rule_n = max(total_n - rule_n, 0)
    non_rule_pos_n = max(total_pos_n - rule_pos_n, 0)

    if total_n <= 0:
        raise ValueError("Table has no rows.")
    if rule_n <= 0:
        raise ValueError("Rule matched 0 rows; broaden the rule.")
    if non_rule_n <= 0:
        raise ValueError("Rule matched all rows; cannot compare against complement.")

    base_rate = total_pos_n / total_n
    rule_rate = rule_pos_n / rule_n
    non_rule_rate = non_rule_pos_n / non_rule_n
    lift = rule_rate / base_rate if base_rate > 0 else 0.0

    pooled = total_pos_n / total_n
    se = math.sqrt(max(pooled * (1.0 - pooled) * ((1.0 / rule_n) + (1.0 / non_rule_n)), 1e-12))
    z_score = (rule_rate - non_rule_rate) / se
    p_value = _p_value_from_z(z_score, alternative=alternative)
    reject_null = p_value < alpha

    direction = "higher" if rule_rate > non_rule_rate else "lower"
    summary = (
        f'Rule segment has {rule_n:,}/{total_n:,} rows ({(rule_n/total_n):.2%}). '
        f'Positive-rate in segment is {rule_rate:.2%} vs {non_rule_rate:.2%} outside '
        f'(lift={lift:.2f}x, z={z_score:.3f}, p={p_value:.4g}). '
        f'At alpha={alpha:.3f}, null is {"rejected" if reject_null else "not rejected"}; '
        f'segment appears {direction} risk than complement.'
    )

    return {
        "table_name": table_name,
        "target_column": target_column,
        "where_clause": where_clause,
        "total_n": total_n,
        "total_positive_n": total_pos_n,
        "rule_n": rule_n,
        "rule_positive_n": rule_pos_n,
        "base_rate": round(base_rate, 6),
        "rule_rate": round(rule_rate, 6),
        "lift": round(lift, 6),
        "z_score": round(z_score, 6),
        "p_value": round(p_value, 8),
        "reject_null": bool(reject_null),
        "alternative": alternative,
        "summary": summary,
    }


def register_rule(
    rule_name: str,
    where_clause: str,
    table_name: str,
    target_column: str,
    positive_value: str,
    alpha: float,
    alternative: str,
    min_lift: float,
    min_rule_n: int,
    require_reject_null: bool,
    owner: str,
    notes: str,
) -> Dict[str, Any]:
    evaluation = evaluate_rule(
        table_name=table_name,
        target_column=target_column,
        positive_value=positive_value,
        where_clause=where_clause,
        alpha=alpha,
        alternative=alternative,
    )

    if evaluation["rule_n"] < int(min_rule_n):
        raise ValueError(
            f'Rule support too low: rule_n={evaluation["rule_n"]} < min_rule_n={int(min_rule_n)}'
        )
    if evaluation["lift"] < float(min_lift):
        raise ValueError(
            f'Rule lift too low: lift={evaluation["lift"]:.4f} < min_lift={float(min_lift):.4f}'
        )
    if require_reject_null and not evaluation["reject_null"]:
        raise ValueError("Null not rejected at provided alpha; rule is not statistically strong enough.")

    rule_id = f"rule_{uuid.uuid4().hex[:12]}"
    entry = {
        "rule_id": rule_id,
        "status": "approved",
        "rule_name": rule_name.strip(),
        "where_clause": where_clause.strip(),
        "table_name": table_name,
        "target_column": target_column,
        "positive_value": str(positive_value),
        "alpha": float(alpha),
        "alternative": alternative,
        "min_lift": float(min_lift),
        "min_rule_n": int(min_rule_n),
        "require_reject_null": bool(require_reject_null),
        "owner": owner.strip(),
        "notes": notes.strip(),
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation": evaluation,
    }
    with _RULE_REGISTRY_LOCK:
        rules = _load_rule_registry()
        rules.append(entry)
        _save_rule_registry(rules)
    return entry


def list_rules(status: Optional[str] = None) -> List[Dict[str, Any]]:
    with _RULE_REGISTRY_LOCK:
        rules = _load_rule_registry()
    if status and status.strip():
        wanted = status.strip().lower()
        rules = [r for r in rules if str(r.get("status", "")).lower() == wanted]
    return rules


def apply_rules_to_customer(
    customer_id: int,
    table_name: str,
    id_column: str,
    include_all_rules: bool,
) -> Dict[str, Any]:
    engine = get_engine()
    schema_map = get_db_schema(engine)
    if table_name not in schema_map:
        raise ValueError(f'Table "{table_name}" not found in schema.')
    if id_column not in schema_map[table_name]:
        raise ValueError(f'Column "{id_column}" not found in table "{table_name}".')

    rules = list_rules(status=None if include_all_rules else "approved")
    candidate_rules = [r for r in rules if r.get("table_name") == table_name]
    if not candidate_rules:
        return {
            "customer_id": int(customer_id),
            "table_name": table_name,
            "matched_count": 0,
            "matched_rules": [],
            "summary": "No registered rules available for this table.",
        }

    matched_rules: List[Dict[str, Any]] = []
    with engine.connect() as conn:
        for r in candidate_rules:
            wc = str(r.get("where_clause", "")).strip()
            if not wc:
                continue
            ensure_safe_where_clause(wc)
            sql = text(
                f"""
                SELECT CASE WHEN ({wc}) THEN 1 ELSE 0 END AS matched
                FROM "{table_name}"
                WHERE "{id_column}" = :cid
                LIMIT 1
                """
            )
            row = conn.execute(sql, {"cid": int(customer_id)}).fetchone()
            if row is None:
                raise ValueError(
                    f'Customer id {customer_id} not found in table "{table_name}" using id column "{id_column}".'
                )
            if int(row[0] or 0) == 1:
                matched_rules.append(
                    {
                        "rule_id": r.get("rule_id"),
                        "rule_name": r.get("rule_name"),
                        "status": r.get("status"),
                        "where_clause": wc,
                        "lift": (r.get("evaluation") or {}).get("lift"),
                        "rule_rate": (r.get("evaluation") or {}).get("rule_rate"),
                        "base_rate": (r.get("evaluation") or {}).get("base_rate"),
                        "summary": (r.get("evaluation") or {}).get("summary", ""),
                    }
                )

    matched_rules = sorted(
        matched_rules,
        key=lambda x: float(x.get("lift") or 0.0),
        reverse=True,
    )
    if matched_rules:
        summary = (
            f"Customer {customer_id} matched {len(matched_rules)} rule(s). "
            f"Top rule: {matched_rules[0]['rule_name']} (lift={matched_rules[0].get('lift', 0):.2f}x)."
        )
    else:
        summary = f"Customer {customer_id} did not match any registered rule."

    return {
        "customer_id": int(customer_id),
        "table_name": table_name,
        "matched_count": int(len(matched_rules)),
        "matched_rules": matched_rules,
        "summary": summary,
    }


def _require_lime_dependencies() -> None:
    if shap is None or RandomForestClassifier is None or train_test_split is None:
        raise ValueError(
            "SHAP dependency missing. Install with: pip install shap scikit-learn"
        )


def _load_table_sample(table_name: str, sample_rows: int) -> pd.DataFrame:
    engine = get_engine()
    schema_map = get_db_schema(engine)
    if table_name not in schema_map:
        raise ValueError(f'Table "{table_name}" not found in schema.')
    sql = text(f'SELECT * FROM "{table_name}" LIMIT :n')
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"n": int(sample_rows)})


def _prepare_numeric_training_matrix(
    df: pd.DataFrame,
    id_column: str,
    target_column: str,
    exclude_columns: Optional[List[str]] = None,
    min_non_null_ratio: float = 0.70,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    drop_cols = {
        id_column,
        target_column,
        "TARGET",
        "is_train",
        "EWS_SCORE",
        "ews_bureau_overdue",
        "ews_bureau_balance_severe",
        "ews_pos_dpd",
        "ews_installment_stress",
        "ews_cc_stress",
    }
    if exclude_columns:
        drop_cols.update([str(c) for c in exclude_columns if str(c).strip()])
    candidate_cols = [c for c in df.columns if c not in drop_cols]
    numeric_cols: Dict[str, pd.Series] = {}
    for c in candidate_cols:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() >= float(min_non_null_ratio):
            numeric_cols[c] = num

    X = pd.DataFrame(numeric_cols, index=df.index)

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    return X, medians, pd.Series(candidate_cols)


def train_lime_model(
    table_name: str,
    id_column: str,
    target_column: str,
    positive_value: str,
    sample_rows: int,
    test_size: float,
    random_state: int,
    exclude_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    _require_lime_dependencies()
    # df = _load_table_sample(table_name=table_name, sample_rows=sample_rows)
    df = _load_table_sample(table_name=table_name, sample_rows=sample_rows)
    if df.empty:
        raise ValueError(f'Table "{table_name}" returned no rows.')
    if id_column not in df.columns:
        raise ValueError(f'Column "{id_column}" not found in table "{table_name}".')
    if target_column not in df.columns:
        raise ValueError(f'Column "{target_column}" not found in table "{table_name}".')

    work = df.copy()
    work = work[work[target_column].notna()].copy()
    if work.empty:
        raise ValueError(f'Table "{table_name}" has no non-null target rows for "{target_column}".')

    y = (work[target_column].astype(str) == str(positive_value)).astype(int)
    pos_rate = float(y.mean()) if len(y) else 0.0
    if y.nunique() < 2:
        raise ValueError("Target column has only one class in sampled rows; increase sample or review label.")

    drop_cols = {
        id_column,
        target_column,
        "TARGET",
        "is_train",
        "EWS_SCORE",
        "ews_bureau_overdue",
        "ews_bureau_balance_severe",
        "ews_pos_dpd",
        "ews_installment_stress",
        "ews_cc_stress",
    }
    if exclude_columns:
        drop_cols.update([str(c) for c in exclude_columns if str(c).strip()])
    candidate_cols = [c for c in work.columns if c not in drop_cols]

    X = pd.DataFrame(index=work.index)
    for c in candidate_cols:
        num = pd.to_numeric(work[c], errors="coerce")
        # keep columns with enough numeric signal
        if num.notna().mean() >= 0.70:
            X[c] = num
    if X.shape[1] < 5:
        raise ValueError("Too few numeric features available for SHAP model training.")

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=350,
        max_depth=16,
        min_samples_leaf=30,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    valid_prob = rf.predict_proba(X_valid)[:, 1]
    auc = None
    if roc_auc_score is not None:
        try:
            auc = float(roc_auc_score(y_valid, valid_prob))
        except Exception:
            auc = None

    explainer = shap.TreeExplainer(rf)

    model_id = f"shap_{table_name}_{uuid.uuid4().hex[:10]}"
    payload = {
        "model_id": model_id,
        "table_name": table_name,
        "id_column": id_column,
        "target_column": target_column,
        "positive_value": str(positive_value),
        "feature_cols": X_train.columns.tolist(),
        "medians": medians.to_dict(),
        "model": rf,
        "explainer": explainer,
        "validation_auc": auc,
        "train_rows": int(len(X)),
        "positive_rate": pos_rate,
        "random_state": random_state,
    }
    artifact_path = _persist_lime_model(payload)
    payload["model_path"] = str(artifact_path)
    global _LATEST_LIME_MODEL_ID
    with _LIME_LOCK:
        _LIME_MODELS[model_id] = payload
        _LATEST_LIME_MODEL_ID = model_id
    return payload


def _require_rule_model_dependencies() -> None:
    if DecisionTreeClassifier is None or train_test_split is None:
        raise ValueError(
            "Decision tree dependency missing. Install with: pip install scikit-learn"
        )


def _format_sql_threshold(v: float) -> str:
    return f"{float(v):.10g}"


def _extract_tree_rules(
    model: Any,
    feature_cols: List[str],
    base_rate: float,
    max_rules: int,
) -> List[Dict[str, Any]]:
    tree = model.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    feature_idx = tree.feature
    threshold = tree.threshold
    values = tree.value
    samples = tree.n_node_samples
    rows: List[Dict[str, Any]] = []

    def walk(node_id: int, conditions: List[str]) -> None:
        is_leaf = children_left[node_id] == children_right[node_id]
        if is_leaf:
            counts = values[node_id][0]
            neg_w = float(counts[0]) if len(counts) >= 1 else 0.0
            pos_w = float(counts[1]) if len(counts) >= 2 else 0.0
            leaf_n = int(samples[node_id])
            if leaf_n <= 0:
                return
            total_w = float(neg_w + pos_w)
            leaf_rate = float(pos_w / total_w) if total_w > 0 else 0.0
            pos_n = int(round(leaf_rate * leaf_n))
            pred_label = int(leaf_rate >= 0.5)
            if pred_label != 1:
                return
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            lift = float(leaf_rate / base_rate) if base_rate > 0 else 0.0
            rows.append(
                {
                    "rule_name": f"Tree high-risk leaf {len(rows) + 1}",
                    "where_clause": where_clause,
                    "leaf_n": int(leaf_n),
                    "positive_n": int(pos_n),
                    "positive_rate": round(leaf_rate, 6),
                    "lift": round(lift, 6),
                    "support_rate": round(float(samples[node_id] / max(samples[0], 1)), 6),
                    "predicted_label": 1,
                }
            )
            return

        feat_name = feature_cols[int(feature_idx[node_id])]
        thr = _format_sql_threshold(float(threshold[node_id]))
        left_cond = conditions + [f'"{feat_name}" <= {thr}']
        right_cond = conditions + [f'"{feat_name}" > {thr}']
        walk(int(children_left[node_id]), left_cond)
        walk(int(children_right[node_id]), right_cond)

    walk(0, [])
    rows = sorted(
        rows,
        key=lambda r: (float(r["positive_rate"]), float(r["lift"]), int(r["leaf_n"])),
        reverse=True,
    )
    trimmed = rows[: int(max_rules)]
    for idx, row in enumerate(trimmed, start=1):
        row["rule_name"] = f"Tree high-risk rule {idx}"
    return trimmed


def train_rule_model(
    table_name: str,
    id_column: str,
    target_column: str,
    positive_value: str,
    sample_rows: int,
    test_size: float,
    random_state: int,
    max_depth: int,
    min_samples_leaf: int,
    top_k_features: int,
    max_rules: int,
    exclude_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    _require_rule_model_dependencies()
    df = _load_table_sample(table_name=table_name, sample_rows=sample_rows)
    if df.empty:
        raise ValueError(f'Table "{table_name}" returned no rows.')
    if id_column not in df.columns:
        raise ValueError(f'Column "{id_column}" not found in table "{table_name}".')
    if target_column not in df.columns:
        raise ValueError(f'Column "{target_column}" not found in table "{table_name}".')

    work = df[df[target_column].notna()].copy()
    if work.empty:
        raise ValueError(f'Table "{table_name}" has no non-null target rows for "{target_column}".')

    y = (work[target_column].astype(str) == str(positive_value)).astype(int)
    pos_rate = float(y.mean()) if len(y) else 0.0
    if y.nunique() < 2:
        raise ValueError("Target column has only one class in sampled rows; increase sample or review label.")

    X, medians, _ = _prepare_numeric_training_matrix(
        df=work,
        id_column=id_column,
        target_column=target_column,
        exclude_columns=exclude_columns,
    )
    if X.shape[1] < 5:
        raise ValueError("Too few numeric features available for rule model training.")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )
    dt.fit(X_train, y_train)

    valid_prob = dt.predict_proba(X_valid)[:, 1]
    auc = None
    if roc_auc_score is not None:
        try:
            auc = float(roc_auc_score(y_valid, valid_prob))
        except Exception:
            auc = None

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns.tolist(),
            "importance": dt.feature_importances_.tolist(),
        }
    ).sort_values("importance", ascending=False)
    importance_df = importance_df[importance_df["importance"] > 0].head(int(top_k_features))
    top_features = [
        {
            "rank": int(i + 1),
            "feature": str(r["feature"]),
            "importance": round(float(r["importance"]), 6),
        }
        for i, r in importance_df.reset_index(drop=True).iterrows()
    ]

    extracted_rules = _extract_tree_rules(
        model=dt,
        feature_cols=X_train.columns.tolist(),
        base_rate=pos_rate,
        max_rules=max_rules,
    )

    model_id = f"rule_tree_{table_name}_{uuid.uuid4().hex[:10]}"
    payload = {
        "model_id": model_id,
        "model_type": "rule_tree",
        "table_name": table_name,
        "id_column": id_column,
        "target_column": target_column,
        "positive_value": str(positive_value),
        "feature_cols": X_train.columns.tolist(),
        "medians": medians.to_dict(),
        "model": dt,
        "validation_auc": auc,
        "train_rows": int(len(X)),
        "positive_rate": pos_rate,
        "random_state": random_state,
        "top_features": top_features,
        "extracted_rules": extracted_rules,
    }
    artifact_path = _persist_lime_model(payload, update_latest=False)
    latest_rule_path = get_model_dir() / "latest_rule_model_id.txt"
    latest_rule_path.write_text(str(model_id), encoding="utf-8")
    payload["model_path"] = str(artifact_path)
    return payload


def _get_lime_model(model_id: Optional[str]) -> Dict[str, Any]:
    global _LATEST_LIME_MODEL_ID
    with _LIME_LOCK:
        mid = (model_id or "").strip() or (_LATEST_LIME_MODEL_ID or "")
        if not mid:
            mid = _load_latest_model_id_from_disk() or ""
            if mid:
                _LATEST_LIME_MODEL_ID = mid
        if not mid:
            raise ValueError("No SHAP model available. Train one via /ews/shap/train first.")
        if mid in _LIME_MODELS:
            return _LIME_MODELS[mid]
        loaded = _load_lime_model_from_disk(mid)
        if loaded is None:
            raise ValueError(f'Model id "{mid}" not found.')
        _LIME_MODELS[mid] = loaded
        _LATEST_LIME_MODEL_ID = mid
        return loaded


def _get_customer_feature_row(model_payload: Dict[str, Any], customer_id: int) -> pd.DataFrame:
    table_name = model_payload["table_name"]
    id_column = model_payload["id_column"]
    feature_cols = model_payload["feature_cols"]
    medians = model_payload["medians"]

    engine = get_engine()
    wanted_cols = [id_column] + feature_cols
    quoted = ", ".join([f'"{c}"' for c in wanted_cols])
    sql = text(
        f'SELECT {quoted} FROM "{table_name}" WHERE "{id_column}" = :cid LIMIT 1'
    )
    with engine.connect() as conn:
        row_df = pd.read_sql(sql, conn, params={"cid": int(customer_id)})
    if row_df.empty:
        raise ValueError(f"Customer id {customer_id} not found in table \"{table_name}\".")

    X = pd.DataFrame(index=row_df.index)
    for c in feature_cols:
        num = pd.to_numeric(row_df[c], errors="coerce")
        X[c] = num.fillna(float(medians.get(c, 0.0)))
    return X


def generate_shap_natural_explanation(
    customer_id: int,
    predicted_probability: float,
    predicted_label: int,
    contributions: List[dict],
) -> str:
    try:
        prompt = f"""
You are a credit risk analyst.
Explain why this single customer was selected by an EWS model, in plain banker language.

Customer ID: {customer_id}
Predicted positive probability: {predicted_probability:.6f}
Predicted label: {predicted_label}
Top SHAP local contributions (positive pushes to high-risk, negative pushes down):
{json.dumps(contributions[:10], ensure_ascii=True)}

Return:
1) One short summary sentence
2) Top 3 risk drivers (bullets)
3) One suggested action for a relationship manager
"""
        text_out = invoke_llm_text(
            system_prompt="Be concise, factual, and business-friendly.",
            user_prompt=prompt,
            temperature=0.2,
        )
        if text_out:
            return text_out
    except Exception:
        pass

    bullets = "\n".join(
        [f"- {c['rule']}: contribution={c['weight']:.4f}" for c in contributions[:6]]
    )
    return (
        f"Customer {customer_id} predicted risk={predicted_probability:.2%} "
        f"(label={predicted_label}). Main local drivers:\n{bullets}"
    )


def explain_customer_with_shap(
    customer_id: int,
    model_id: Optional[str] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    payload = _get_lime_model(model_id)
    model = payload["model"]
    explainer = payload["explainer"]
    feature_cols = payload["feature_cols"]
    X_row = _get_customer_feature_row(payload, customer_id=int(customer_id))

    pred_prob = float(model.predict_proba(X_row.values)[0, 1])
    pred_label = int(pred_prob >= 0.5)

    shap_values_raw = explainer.shap_values(X_row)
    shap_arr: np.ndarray
    if isinstance(shap_values_raw, list):
        # common format: list[class] -> (n_samples, n_features)
        idx = 1 if len(shap_values_raw) > 1 else 0
        shap_arr = np.asarray(shap_values_raw[idx])
    else:
        shap_arr = np.asarray(shap_values_raw)

    if shap_arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        cls_idx = 1 if shap_arr.shape[2] > 1 else 0
        shap_row = shap_arr[0, :, cls_idx]
    elif shap_arr.ndim == 2:
        shap_row = shap_arr[0]
    elif shap_arr.ndim == 1:
        shap_row = shap_arr
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_arr.shape}")

    values = X_row.iloc[0]
    rows = []
    for feat, sv in zip(feature_cols, shap_row):
        rows.append(
            {
                "feature": feat,
                "feature_value": float(values[feat]),
                "weight": float(sv),
                "abs_weight": float(abs(sv)),
                "direction": "increases_risk" if float(sv) >= 0 else "decreases_risk",
                "rule": f'{feat}={float(values[feat]):.5g}',
            }
        )
    rows = sorted(rows, key=lambda x: x["abs_weight"], reverse=True)[: int(top_k)]
    contributions = [{k: v for k, v in r.items() if k != "abs_weight"} for r in rows]

    explanation = generate_shap_natural_explanation(
        customer_id=int(customer_id),
        predicted_probability=pred_prob,
        predicted_label=pred_label,
        contributions=contributions,
    )
    return {
        "model_id": payload["model_id"],
        "customer_id": int(customer_id),
        "predicted_probability": round(pred_prob, 6),
        "predicted_label": int(pred_label),
        "feature_contributions": contributions,
        "explanation": explanation,
    }


def ensure_safe_select(sql_query: str) -> None:
    if not sql_query:
        raise ValueError("Generated SQL is empty.")
    lowered = sql_query.lower().strip()
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    blocked = ["insert ", "update ", "delete ", "drop ", "alter ", "truncate ", "create "]
    if any(token in lowered for token in blocked):
        raise ValueError("Unsafe SQL detected.")


def run_query(sql_query: str, limit_rows: int = 200) -> pd.DataFrame:
    engine = get_engine()
    ensure_safe_select(sql_query)
    safe_query = f"SELECT * FROM ({sql_query.rstrip(';')}) q LIMIT {limit_rows}"
    with engine.connect() as conn:
        return pd.read_sql(text(safe_query), conn)


def dataframe_preview(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "No rows returned."
    return df.head(max_rows).to_markdown(index=False)


def generate_natural_answer(
    user_question: str,
    sql_query: str,
    query_df: pd.DataFrame,
    conversation_context: str = "",
    usage_tracker: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    preview = dataframe_preview(query_df)
    prior_block = (
        f"\nPrior conversation (for follow-up context):\n{conversation_context}\n"
        if conversation_context.strip()
        else ""
    )
    prompt = f"""
You are a data analyst.
{prior_block}
Current question: {user_question}
SQL used:
{sql_query}

SQL output sample:
{preview}

Write a concise natural-language answer. If output is empty, say data is unavailable.
For follow-ups, relate explicitly to the prior question when helpful.
"""
    answer = invoke_llm_text(
        system_prompt="Answer clearly with business-friendly wording.",
        user_prompt=prompt,
        temperature=0.2,
    )
    return answer or "Data was returned, but the model did not produce a response."


def answer_question(
    user_question: str,
    limit_rows: int = 200,
    conversation_id: Optional[str] = None,
    profile: bool = False,
) -> Dict[str, object]:
    request_id = str(uuid.uuid4())
    t_total = time.perf_counter()
    timings: Dict[str, float] = {}
    usage_tracker: Dict[str, Dict[str, int]] = {}

    cid = (conversation_id or "").strip() or str(uuid.uuid4())
    t0 = time.perf_counter()
    with _CONV_LOCK:
        prior_snapshot = [dict(t) for t in _CONVERSATIONS.get(cid, [])]
    conversation_context = _conversation_subtext(prior_snapshot)
    if profile:
        timings["conversation_snapshot_ms"] = _elapsed_ms(t0)

    engine = get_engine()
    t0 = time.perf_counter()
    schema_map = get_db_schema(engine)
    if profile:
        timings["load_schema_ms"] = _elapsed_ms(t0)
    if not schema_map:
        raise ValueError("No tables found in PostgreSQL public schema. Run ingestion first.")

    t0 = time.perf_counter()
    desc_df = get_column_descriptions_from_db()
    if profile:
        timings["load_metadata_ms"] = _elapsed_ms(t0)
    if desc_df.empty:
        raise ValueError(
            f"No metadata found in {METADATA_TABLE}. Load it once using --load-metadata."
        )

    t0 = time.perf_counter()
    context = build_semantic_context(schema_map, desc_df)
    if profile:
        timings["build_semantic_context_ms"] = _elapsed_ms(t0)

    t0 = time.perf_counter()
    sql_query = generate_sql_query(
        user_question,
        context,
        conversation_context=conversation_context,
        usage_tracker=usage_tracker,
    )
    if profile:
        timings["llm_generate_sql_ms"] = _elapsed_ms(t0)

    execute_ms = 0.0
    repair_ms = 0.0
    try:
        t0 = time.perf_counter()
        results_df = run_query(sql_query, limit_rows=limit_rows)
        execute_ms += _elapsed_ms(t0)
    except Exception as first_error:
        execute_ms += _elapsed_ms(t0)
        t0 = time.perf_counter()
        repaired_sql = repair_sql_query(
            user_question=user_question,
            semantic_context=context,
            failed_sql=sql_query,
            db_error=str(first_error),
            conversation_context=conversation_context,
            usage_tracker=usage_tracker,
        )
        repair_ms += _elapsed_ms(t0)
        t0 = time.perf_counter()
        results_df = run_query(repaired_sql, limit_rows=limit_rows)
        execute_ms += _elapsed_ms(t0)
        sql_query = repaired_sql
    if profile:
        timings["execute_sql_ms"] = round(execute_ms, 2)
        timings["llm_repair_sql_ms"] = round(repair_ms, 2)

    t0 = time.perf_counter()
    answer = generate_natural_answer(
        user_question,
        sql_query,
        results_df,
        conversation_context=conversation_context,
        usage_tracker=usage_tracker,
    )
    if profile:
        timings["llm_natural_answer_ms"] = _elapsed_ms(t0)

    preview_df = results_df.head(50).where(pd.notnull(results_df.head(50)), None)

    t0 = time.perf_counter()
    _record_turn(cid, user_question, sql_query, answer, results_df)
    if profile:
        timings["record_turn_ms"] = _elapsed_ms(t0)
        timings["total_ms"] = round(_elapsed_ms(t_total), 2)

    result: Dict[str, object] = {
        "conversation_id": cid,
        "question": user_question,
        "sql": sql_query,
        "answer": answer,
        "row_count": int(len(results_df)),
        "preview": preview_df.to_dict(orient="records"),
    }
    if profile:
        result["timings_ms"] = timings
    LOGGER.info(
        json.dumps(
            {
                "event": "ask_success",
                "request_id": request_id,
                "conversation_id": cid,
                "question": user_question[:1000],
                "sql": sql_query[:4000],
                "row_count": int(len(results_df)),
                "timings_ms": timings,
                "token_usage": usage_tracker,
            },
            ensure_ascii=True,
        )
    )
    return result


def _get_conversation_snapshot(conversation_id: Optional[str]) -> tuple[str, str]:
    cid = (conversation_id or "").strip() or str(uuid.uuid4())
    with _CONV_LOCK:
        prior_snapshot = [dict(t) for t in _CONVERSATIONS.get(cid, [])]
    return cid, _conversation_subtext(prior_snapshot)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/phase0/validate-workspace", response_model=Phase0ValidateResponse)
def phase0_validate_workspace(payload: Phase0ValidateRequest) -> Phase0ValidateResponse:
    try:
        project_root = Path(payload.project_root).expanduser().resolve()
        result = validate_phase0_workspace(project_root)
        resolved_data_dir = resolve_data_dir(project_root, payload.env_raw_data_dir)
        manifest = build_phase0_manifest(project_root)
        return Phase0ValidateResponse(
            project_root=str(project_root),
            is_valid=result.is_valid,
            missing_dirs=result.missing_dirs,
            missing_files=result.missing_files,
            resolved_data_dir=str(resolved_data_dir),
            manifest=manifest,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase1/entity/resolve-pair", response_model=Phase1ResolveResponse)
def phase1_resolve_pair(payload: Phase1ResolvePairRequest) -> Phase1ResolveResponse:
    try:
        decision = _ENTITY_RESOLVER.score_pair(payload.record_a, payload.record_b)
        return Phase1ResolveResponse(
            status=decision.status,
            confidence=round(float(decision.confidence), 6),
            reasons=decision.reasons,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase1/entity/resolve-cluster", response_model=Phase1ResolveResponse)
def phase1_resolve_cluster(payload: Phase1ResolveClusterRequest) -> Phase1ResolveResponse:
    try:
        decision = _ENTITY_RESOLVER.resolve_cluster(payload.records)
        return Phase1ResolveResponse(
            status=decision.status,
            confidence=round(float(decision.confidence), 6),
            reasons=decision.reasons,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase1/contracts/validate", response_model=Phase1ContractValidateResponse)
def phase1_validate_contract(payload: Phase1ContractValidateRequest) -> Phase1ContractValidateResponse:
    try:
        contract = DataContract(
            source_name=payload.source_name,
            table_name=payload.table_name,
            columns=[
                ColumnContract(
                    name=c.name,
                    dtype=c.dtype,
                    nullable=c.nullable,
                    min_value=c.min_value,
                    max_value=c.max_value,
                    unique=c.unique,
                    critical=c.critical,
                    max_null_ratio=c.max_null_ratio,
                )
                for c in payload.columns
            ],
        )
        df = pd.DataFrame(payload.rows)
        is_valid, issues = validate_dataframe_contract(df, contract)
        issue_payload = [{"level": i.level, "code": i.code, "message": i.message} for i in issues]
        return Phase1ContractValidateResponse(
            is_valid=is_valid,
            issue_count=len(issue_payload),
            issues=issue_payload,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase1/governance/check", response_model=Phase1GovernanceCheckResponse)
def phase1_governance_check(payload: Phase1GovernanceCheckRequest) -> Phase1GovernanceCheckResponse:
    try:
        masked_record = mask_pii(payload.record, payload.role)
        unsupported_claims = validate_claim_evidence(payload.claims, payload.evidence_map)
        audit_event = create_audit_event(
            user_id=payload.user_id,
            role=payload.role,
            request_id=payload.request_id,
            tool_calls=payload.tool_calls,
            evidence_ids=[eid for ids in payload.evidence_map.values() for eid in ids],
            response_text=payload.response_text,
        )
        return Phase1GovernanceCheckResponse(
            masked_record=masked_record,
            unsupported_claims=unsupported_claims,
            audit_event=audit_event,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase2/retrieve", response_model=Phase2RetrieveResponse)
def phase2_retrieve(payload: Phase2RetrieveRequest) -> Phase2RetrieveResponse:
    try:
        req = RetrievalRequest(
            intent=payload.intent,
            customer_gid=payload.customer_gid,
            filters=payload.filters,
            query_embedding=payload.query_embedding,
            top_k=payload.top_k,
        )
        result = _PHASE2_ORCH.retrieve(
            request=req,
            sql_rows=payload.sql_rows,
            graph_edges=payload.graph_edges,
            vector_docs=payload.vector_docs,
        )
        return Phase2RetrieveResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/phase3/health")
def phase3_health() -> Dict[str, Any]:
    return _PHASE3_SERVICE.health()


@app.get("/phase3/health-real")
def phase3_health_real() -> Dict[str, Any]:
    svc = _build_phase3_real_service()
    return svc.health()


@app.post("/phase3/retrieve-live", response_model=Phase3RetrieveResponse)
async def phase3_retrieve_live(payload: Phase3RetrieveRequest) -> Phase3RetrieveResponse:
    try:
        # For local/dev we bind request-provided data to in-memory adapters.
        # Later phases can switch this wiring to real Postgres/Neo4j/pgvector adapters.
        temp_service = Phase3HybridService(
            sql_adapter=InMemorySqlAdapter(rows=payload.sql_rows),
            graph_adapter=InMemoryGraphAdapter(edges=payload.graph_edges),
            vector_adapter=InMemoryVectorAdapter(docs=payload.vector_docs),
        )
        req = Phase3ServiceRequest(
            intent=payload.intent,
            customer_gid=payload.customer_gid,
            filters=payload.filters,
            query_text=payload.query_text,
            query_embedding=payload.query_embedding,
            top_k=payload.top_k,
            include_sql=payload.include_sql,
            include_graph=payload.include_graph,
            include_vector=payload.include_vector,
        )
        out = await temp_service.retrieve(req)
        return Phase3RetrieveResponse(**out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase3/retrieve-real", response_model=Phase3RetrieveResponse)
async def phase3_retrieve_real(payload: Phase3RetrieveRequest) -> Phase3RetrieveResponse:
    try:
        svc = _build_phase3_real_service()
        req = Phase3ServiceRequest(
            intent=payload.intent,
            customer_gid=payload.customer_gid,
            filters=payload.filters,
            query_text=payload.query_text,
            query_embedding=payload.query_embedding,
            top_k=payload.top_k,
            include_sql=payload.include_sql,
            include_graph=payload.include_graph,
            include_vector=payload.include_vector,
        )
        out = await svc.retrieve(req)
        return Phase3RetrieveResponse(**out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/phase4/health")
def phase4_health() -> Dict[str, Any]:
    return _PHASE4_SERVICE.health()


@app.get("/phase4/health-real")
def phase4_health_real() -> Dict[str, Any]:
    svc = _build_phase4_real_service()
    return svc.health()


@app.post("/phase4/investigate-live", response_model=Phase4InvestigateResponse)
async def phase4_investigate_live(payload: Phase4InvestigateRequest) -> Phase4InvestigateResponse:
    try:
        phase3_service = Phase3HybridService(
            sql_adapter=InMemorySqlAdapter(rows=payload.sql_rows),
            graph_adapter=InMemoryGraphAdapter(edges=payload.graph_edges),
            vector_adapter=InMemoryVectorAdapter(docs=payload.vector_docs),
        )
        phase4_service = Phase4InvestigationService(
            phase3_service=phase3_service,
            model_adapter=InMemoryModelAdapter(),
        )
        req = Phase4ServiceRequest(
            intent=payload.intent,
            customer_gid=payload.customer_gid,
            customer_id=payload.customer_id,
            model_id=payload.model_id,
            filters=payload.filters,
            query_text=payload.query_text,
            query_embedding=payload.query_embedding,
            top_k=payload.top_k,
            include_sql=payload.include_sql,
            include_graph=payload.include_graph,
            include_vector=payload.include_vector,
            include_model=payload.include_model,
            include_explanation=payload.include_explanation,
            explanation_top_k=payload.explanation_top_k,
            apply_calibration=payload.apply_calibration,
        )
        out = await phase4_service.investigate(req)
        return Phase4InvestigateResponse(**out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase4/investigate-real", response_model=Phase4InvestigateResponse)
async def phase4_investigate_real(payload: Phase4InvestigateRequest) -> Phase4InvestigateResponse:
    try:
        svc = _build_phase4_real_service()
        req = Phase4ServiceRequest(
            intent=payload.intent,
            customer_gid=payload.customer_gid,
            customer_id=payload.customer_id,
            model_id=payload.model_id,
            filters=payload.filters,
            query_text=payload.query_text,
            query_embedding=payload.query_embedding,
            top_k=payload.top_k,
            include_sql=payload.include_sql,
            include_graph=payload.include_graph,
            include_vector=payload.include_vector,
            include_model=payload.include_model,
            include_explanation=payload.include_explanation,
            explanation_top_k=payload.explanation_top_k,
            apply_calibration=payload.apply_calibration,
        )
        out = await svc.investigate(req)
        return Phase4InvestigateResponse(**out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/phase5/health")
def phase5_health() -> Dict[str, Any]:
    return _PHASE5_SERVICE.health()


@app.get("/phase5/health-real")
def phase5_health_real() -> Dict[str, Any]:
    svc = _build_phase5_real_service()
    return svc.health()


def _resolve_customer_identity(customer_id: Optional[int], customer_gid: Optional[str]) -> Tuple[int, str]:
    gid = (customer_gid or "").strip()
    cid = customer_id
    if cid is None and gid:
        try:
            cid = int(gid)
        except Exception as exc:
            raise ValueError("customer_id is required when customer_gid is non-numeric.") from exc
    if cid is None:
        raise ValueError("Provide customer_id or customer_gid.")
    if not gid:
        gid = str(int(cid))
    return int(cid), gid


def _extract_model_record(phase5_payload: Dict[str, Any]) -> Dict[str, Any]:
    for r in phase5_payload.get("responses", []):
        if r.get("source") == "model" and r.get("records"):
            return dict(r["records"][0])
    return {}


def _extract_graph_count(phase5_payload: Dict[str, Any]) -> int:
    for r in phase5_payload.get("responses", []):
        if r.get("source") == "graph":
            try:
                return int(r.get("count", 0) or 0)
            except Exception:
                return 0
    return 0


def _extract_graph_records(phase5_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for r in phase5_payload.get("responses", []):
        if r.get("source") == "graph":
            recs = r.get("records", [])
            if isinstance(recs, list):
                return [dict(x) for x in recs if isinstance(x, dict)]
            return []
    return []


def _summarize_graph_records(graph_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not graph_records:
        return {
            "top_relationship_types": [],
            "sample_paths": [],
            "unique_counterparties": 0,
        }

    rel_counter: Counter[str] = Counter()
    counterparties: set[str] = set()
    sample_paths: List[Dict[str, Any]] = []
    for rec in graph_records:
        rel = str(rec.get("rel", "UNKNOWN"))
        src = str(rec.get("source", ""))
        tgt = str(rec.get("target", ""))
        rel_counter[rel] += 1
        if src:
            counterparties.add(src)
        if tgt:
            counterparties.add(tgt)
        if len(sample_paths) < 8:
            sample_paths.append({"source": src, "target": tgt, "rel": rel})

    return {
        "top_relationship_types": [
            {"relationship": rel, "count": int(cnt)}
            for rel, cnt in rel_counter.most_common(5)
        ],
        "sample_paths": sample_paths,
        "unique_counterparties": int(len(counterparties)),
    }


def _generate_customer_insight_llm(
    customer_id: int,
    customer_gid: str,
    risk_level: str,
    score: float,
    model: Dict[str, Any],
    matched_rules: List[Dict[str, Any]],
    graph_count: int,
    graph_summary: Dict[str, Any],
    fallback_summary: str,
    fallback_actions: List[str],
    fallback_questions: List[str],
) -> Dict[str, Any]:
    prompt_payload = {
        "customer_id": customer_id,
        "customer_gid": customer_gid,
        "computed_risk_level": risk_level,
        "computed_risk_score": round(float(score), 3),
        "model_prediction": {
            "probability": model.get("predicted_probability"),
            "risk_band": model.get("risk_band"),
            "predicted_label": model.get("predicted_label"),
            "top_factors": model.get("feature_contributions", [])[:8],
        },
        "matched_rules": matched_rules[:8],
        "graph_evidence": {
            "linked_paths": graph_count,
            "top_relationship_types": graph_summary.get("top_relationship_types", []),
            "unique_counterparties": graph_summary.get("unique_counterparties", 0),
            "sample_paths": graph_summary.get("sample_paths", []),
        },
    }
    prompt = f"""
You are a senior banking early-warning analyst.
Use the provided hybrid evidence to explain customer risk.
IMPORTANT: you must explicitly use graph_evidence (relationship structure) in the reasoning.

Evidence payload:
{json.dumps(prompt_payload, ensure_ascii=True)}

Return STRICT JSON with keys:
- summary: string (2-3 sentences, business language)
- recommended_actions: array of exactly 3 short actions
- next_questions: array of exactly 3 follow-up investigation questions
Do not include markdown.
"""
    try:
        parsed = invoke_llm_json(
            system_prompt="You are concise, evidence-grounded, and avoid speculation.",
            user_prompt=prompt,
            temperature=0.1,
        )
        summary = str(parsed.get("summary", "")).strip()
        actions = parsed.get("recommended_actions", [])
        questions = parsed.get("next_questions", [])
        if not summary:
            raise ValueError("Empty summary from LLM.")
        if not isinstance(actions, list) or not actions:
            actions = fallback_actions
        if not isinstance(questions, list) or not questions:
            questions = fallback_questions
        return {
            "summary": summary,
            "recommended_actions": [str(x) for x in actions][:3],
            "next_questions": [str(x) for x in questions][:3],
            "llm_used": True,
            "llm_reason": "",
        }
    except Exception as exc:
        return {
            "summary": fallback_summary,
            "recommended_actions": fallback_actions,
            "next_questions": fallback_questions,
            "llm_used": False,
            "llm_reason": f"fallback_due_to_error: {exc}",
        }


def _build_customer_insight(
    customer_id: int,
    customer_gid: str,
    phase5_result: Dict[str, Any],
    rules_result: Dict[str, Any],
) -> Dict[str, Any]:
    model = _extract_model_record(phase5_result)
    graph_count = _extract_graph_count(phase5_result)
    graph_records = _extract_graph_records(phase5_result)
    graph_evidence_summary = _summarize_graph_records(graph_records)
    matched_rules = list(rules_result.get("matched_rules", []))
    matched_count = int(rules_result.get("matched_count", 0) or 0)
    prob = float(model.get("predicted_probability", 0.0) or 0.0)

    score = 0.0
    if prob >= 0.70:
        score += 2.0
    elif prob >= 0.40:
        score += 1.0
    if matched_count >= 3:
        score += 2.0
    elif matched_count >= 1:
        score += 1.0
    if graph_count >= 50:
        score += 1.0
    if any(float((r.get("lift") or 0.0)) >= 2.0 for r in matched_rules):
        score += 1.0

    if score >= 4.0:
        risk_level = "high"
    elif score >= 2.0:
        risk_level = "medium"
    else:
        risk_level = "low"

    top_rule = matched_rules[0] if matched_rules else {}
    top_rule_name = str(top_rule.get("rule_name", "none"))
    top_rule_lift = float(top_rule.get("lift", 0.0) or 0.0)
    summary = (
        f"Customer {customer_id} is assessed as {risk_level.upper()} risk "
        f"(score={score:.1f}, model_probability={prob:.2%}, matched_rules={matched_count}, graph_links={graph_count}). "
        f"Top matched rule: {top_rule_name} (lift={top_rule_lift:.2f}x)."
    )

    if risk_level == "high":
        actions = [
            "Immediate RM outreach within 24 hours and payment capacity check.",
            "Place account on enhanced monitoring and tighten credit line changes.",
            "Offer structured repayment plan and collect early warning documentation.",
        ]
    elif risk_level == "medium":
        actions = [
            "Schedule RM follow-up within 7 days with focused repayment reminders.",
            "Monitor delinquency and utilization weekly for trend break detection.",
            "Pre-approve contingency restructuring options if stress indicators rise.",
        ]
    else:
        actions = [
            "Continue monthly monitoring and keep in early-warning watchlist.",
            "Nudge customer with preventive payment reminders before due dates.",
            "Re-check graph/rule triggers on next cycle for any adverse changes.",
        ]

    next_questions = [
        "Which specific facilities are driving this risk and what is the near-term DPD path?",
        "Are there linked customers/devices with recent deterioration that raise contagion risk?",
        "Which intervention is most likely to reduce the 90+ DPD transition risk in 30 days?",
    ]
    llm_block = _generate_customer_insight_llm(
        customer_id=customer_id,
        customer_gid=customer_gid,
        risk_level=risk_level,
        score=score,
        model=model,
        matched_rules=matched_rules,
        graph_count=graph_count,
        graph_summary=graph_evidence_summary,
        fallback_summary=summary,
        fallback_actions=actions,
        fallback_questions=next_questions,
    )

    return {
        "customer_id": int(customer_id),
        "customer_gid": customer_gid,
        "overall_risk_level": risk_level,
        "overall_risk_score": round(float(score), 3),
        "summary": llm_block["summary"],
        "recommended_actions": llm_block["recommended_actions"],
        "next_questions": llm_block["next_questions"],
        "model_summary": {
            "model_id": model.get("model_id"),
            "predicted_probability": model.get("predicted_probability"),
            "predicted_label": model.get("predicted_label"),
            "risk_band": model.get("risk_band"),
            "top_factors": model.get("feature_contributions", []),
        },
        "rules_summary": {
            "matched_count": matched_count,
            "matched_rules": matched_rules,
            "rules_message": rules_result.get("summary", ""),
        },
        "graph_summary": {
            "linked_paths": graph_count,
            "top_relationship_types": graph_evidence_summary.get("top_relationship_types", []),
            "sample_paths": graph_evidence_summary.get("sample_paths", []),
            "unique_counterparties": graph_evidence_summary.get("unique_counterparties", 0),
            "graph_note": "Higher linked-path counts may indicate relationship/contagion risk.",
        },
        "hybrid_detail": {
            "phase5": phase5_result,
            "claims": phase5_result.get("claims", []),
            "trace": phase5_result.get("trace", {}),
            "audit_event": phase5_result.get("audit_event", {}),
            "llm_customer_insight": {
                "used": bool(llm_block.get("llm_used", False)),
                "reason": str(llm_block.get("llm_reason", "")),
            },
        },
    }


@app.post("/phase5/investigate-live", response_model=Phase5CopilotResponse)
async def phase5_investigate_live(payload: Phase5CopilotRequest) -> Phase5CopilotResponse:
    try:
        phase3_service = Phase3HybridService(
            sql_adapter=InMemorySqlAdapter(rows=payload.sql_rows),
            graph_adapter=InMemoryGraphAdapter(edges=payload.graph_edges),
            vector_adapter=InMemoryVectorAdapter(docs=payload.vector_docs),
        )
        phase4_service = Phase4InvestigationService(
            phase3_service=phase3_service,
            model_adapter=InMemoryModelAdapter(),
        )
        phase5_service = Phase5CopilotService(phase4_service=phase4_service)
        req = Phase5ServiceRequest(
            intent=payload.intent,
            customer_gid=payload.customer_gid,
            customer_id=payload.customer_id,
            model_id=payload.model_id,
            role=payload.role,
            user_id=payload.user_id,
            request_id=payload.request_id,
            policy_profile=payload.policy_profile,
            strict_route_policy=payload.strict_route_policy,
            filters=payload.filters,
            query_text=payload.query_text,
            query_embedding=payload.query_embedding,
            top_k=payload.top_k,
            include_sql=payload.include_sql,
            include_graph=payload.include_graph,
            include_vector=payload.include_vector,
            include_model=payload.include_model,
            include_explanation=payload.include_explanation,
            explanation_top_k=payload.explanation_top_k,
            apply_calibration=payload.apply_calibration,
        )
        out = await phase5_service.investigate(req)
        return Phase5CopilotResponse(**out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase5/investigate-real", response_model=Phase5CopilotResponse)
async def phase5_investigate_real(payload: Phase5CopilotRequest) -> Phase5CopilotResponse:
    try:
        svc = _build_phase5_real_service()
        req = Phase5ServiceRequest(
            intent=payload.intent,
            customer_gid=payload.customer_gid,
            customer_id=payload.customer_id,
            model_id=payload.model_id,
            role=payload.role,
            user_id=payload.user_id,
            request_id=payload.request_id,
            policy_profile=payload.policy_profile,
            strict_route_policy=payload.strict_route_policy,
            filters=payload.filters,
            query_text=payload.query_text,
            query_embedding=payload.query_embedding,
            top_k=payload.top_k,
            include_sql=payload.include_sql,
            include_graph=payload.include_graph,
            include_vector=payload.include_vector,
            include_model=payload.include_model,
            include_explanation=payload.include_explanation,
            explanation_top_k=payload.explanation_top_k,
            apply_calibration=payload.apply_calibration,
        )
        out = await svc.investigate(req)
        return Phase5CopilotResponse(**out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase5/customer-insight-live", response_model=Phase5CustomerInsightResponse)
async def phase5_customer_insight_live(payload: Phase5CustomerInsightRequest) -> Phase5CustomerInsightResponse:
    try:
        cid, gid = _resolve_customer_identity(payload.customer_id, payload.customer_gid)
        phase3_service = Phase3HybridService(
            sql_adapter=InMemorySqlAdapter(rows=payload.sql_rows),
            graph_adapter=InMemoryGraphAdapter(edges=payload.graph_edges),
            vector_adapter=InMemoryVectorAdapter(docs=payload.vector_docs),
        )
        phase4_service = Phase4InvestigationService(
            phase3_service=phase3_service,
            model_adapter=InMemoryModelAdapter(),
        )
        phase5_service = Phase5CopilotService(phase4_service=phase4_service)
        req = Phase5ServiceRequest(
            intent=payload.intent,
            customer_gid=gid,
            customer_id=cid,
            model_id=payload.model_id,
            role=payload.role,
            user_id=payload.user_id,
            request_id=payload.request_id,
            policy_profile=payload.policy_profile,
            strict_route_policy=payload.strict_route_policy,
            filters=payload.filters,
            query_text=payload.query_text,
            query_embedding=payload.query_embedding,
            top_k=payload.top_k,
            include_sql=payload.include_sql,
            include_graph=payload.include_graph,
            include_vector=payload.include_vector,
            include_model=payload.include_model,
            include_explanation=payload.include_explanation,
            explanation_top_k=payload.explanation_top_k,
            apply_calibration=payload.apply_calibration,
        )
        phase5_result = await phase5_service.investigate(req)

        try:
            rules_result = apply_rules_to_customer(
                customer_id=cid,
                table_name=payload.table_name,
                id_column=payload.id_column,
                include_all_rules=payload.include_all_rules,
            )
        except Exception as rules_exc:
            rules_result = {
                "customer_id": cid,
                "table_name": payload.table_name,
                "matched_count": 0,
                "matched_rules": [],
                "summary": f"Rules unavailable: {rules_exc}",
            }

        out = _build_customer_insight(
            customer_id=cid,
            customer_gid=gid,
            phase5_result=phase5_result,
            rules_result=rules_result,
        )
        return Phase5CustomerInsightResponse(**out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/phase5/customer-insight-real", response_model=Phase5CustomerInsightResponse)
async def phase5_customer_insight_real(payload: Phase5CustomerInsightRequest) -> Phase5CustomerInsightResponse:
    try:
        cid, gid = _resolve_customer_identity(payload.customer_id, payload.customer_gid)
        svc = _build_phase5_real_service()
        req = Phase5ServiceRequest(
            intent=payload.intent,
            customer_gid=gid,
            customer_id=cid,
            model_id=payload.model_id,
            role=payload.role,
            user_id=payload.user_id,
            request_id=payload.request_id,
            policy_profile=payload.policy_profile,
            strict_route_policy=payload.strict_route_policy,
            filters=payload.filters,
            query_text=payload.query_text,
            query_embedding=payload.query_embedding,
            top_k=payload.top_k,
            include_sql=payload.include_sql,
            include_graph=payload.include_graph,
            include_vector=payload.include_vector,
            include_model=payload.include_model,
            include_explanation=payload.include_explanation,
            explanation_top_k=payload.explanation_top_k,
            apply_calibration=payload.apply_calibration,
        )
        phase5_result = await svc.investigate(req)

        try:
            rules_result = apply_rules_to_customer(
                customer_id=cid,
                table_name=payload.table_name,
                id_column=payload.id_column,
                include_all_rules=payload.include_all_rules,
            )
        except Exception as rules_exc:
            rules_result = {
                "customer_id": cid,
                "table_name": payload.table_name,
                "matched_count": 0,
                "matched_rules": [],
                "summary": f"Rules unavailable: {rules_exc}",
            }

        out = _build_customer_insight(
            customer_id=cid,
            customer_gid=gid,
            phase5_result=phase5_result,
            rules_result=rules_result,
        )
        return Phase5CustomerInsightResponse(**out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/conversation/{conversation_id}")
def clear_conversation(conversation_id: str) -> Dict[str, bool]:
    with _CONV_LOCK:
        _CONVERSATIONS.pop(conversation_id.strip(), None)
    return {"ok": True}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        result = answer_question(
            payload.question,
            limit_rows=payload.limit_rows,
            conversation_id=payload.conversation_id,
            profile=payload.profile,
        )
        return AskResponse(**result)
    except Exception as exc:
        LOGGER.exception(
            json.dumps(
                {
                    "event": "ask_error",
                    "question": payload.question[:1000],
                    "conversation_id": payload.conversation_id,
                    "profile": payload.profile,
                    "detail": str(exc)[:4000],
                },
                ensure_ascii=True,
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rule/draft", response_model=RuleDraftResponse)
def draft_rule(payload: RuleDraftRequest) -> RuleDraftResponse:
    timings: Dict[str, float] = {}
    usage_tracker: Dict[str, Dict[str, int]] = {}
    t_total = time.perf_counter()
    try:
        t0 = time.perf_counter()
        cid, conversation_context = _get_conversation_snapshot(payload.conversation_id)
        if payload.profile:
            timings["conversation_snapshot_ms"] = _elapsed_ms(t0)

        engine = get_engine()
        t0 = time.perf_counter()
        schema_map = get_db_schema(engine)
        if payload.profile:
            timings["load_schema_ms"] = _elapsed_ms(t0)
        if not schema_map:
            raise ValueError("No tables found in PostgreSQL public schema. Run ingestion first.")

        t0 = time.perf_counter()
        desc_df = get_column_descriptions_from_db()
        if payload.profile:
            timings["load_metadata_ms"] = _elapsed_ms(t0)
        if desc_df.empty:
            raise ValueError(f"No metadata found in {METADATA_TABLE}. Run --load-metadata first.")

        t0 = time.perf_counter()
        context = build_semantic_context(schema_map, desc_df)
        if payload.profile:
            timings["build_semantic_context_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        drafted = generate_rule_where_clause(
            rule_intent=payload.rule_intent,
            table_name=payload.table_name,
            target_column=payload.target_column,
            semantic_context=context,
            conversation_context=conversation_context,
            usage_tracker=usage_tracker,
        )
        if payload.profile:
            timings["llm_generate_rule_ms"] = _elapsed_ms(t0)

        ensure_safe_where_clause(drafted["where_clause"])
        example_sql = (
            f'SELECT COUNT(*) AS rule_rows FROM "{payload.table_name}" '
            f'WHERE {drafted["where_clause"]};'
        )

        _record_turn(
            cid,
            user=f"Draft rule: {payload.rule_intent}",
            sql=example_sql,
            answer=f'Rule drafted: {drafted["rule_name"]}. {drafted["rationale"]}',
            result_df=pd.DataFrame([{"where_clause": drafted["where_clause"]}]),
        )
        if payload.profile:
            timings["total_ms"] = _elapsed_ms(t_total)

        resp = {
            "conversation_id": cid,
            "rule_name": drafted["rule_name"],
            "where_clause": drafted["where_clause"],
            "rationale": drafted["rationale"],
            "example_sql": example_sql,
        }
        if payload.profile:
            # recompute safely at end
            pass
        if payload.profile:
            resp["timings_ms"] = timings
        LOGGER.info(
            json.dumps(
                {
                    "event": "rule_draft_success",
                    "conversation_id": cid,
                    "rule_intent": payload.rule_intent[:1000],
                    "table_name": payload.table_name,
                    "target_column": payload.target_column,
                    "where_clause": drafted["where_clause"][:2000],
                    "timings_ms": timings,
                    "token_usage": usage_tracker,
                },
                ensure_ascii=True,
            )
        )
        return RuleDraftResponse(**resp)
    except Exception as exc:
        LOGGER.exception(
            json.dumps(
                {
                    "event": "rule_draft_error",
                    "rule_intent": payload.rule_intent[:1000],
                    "table_name": payload.table_name,
                    "target_column": payload.target_column,
                    "detail": str(exc)[:4000],
                },
                ensure_ascii=True,
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rule/evaluate", response_model=RuleEvaluateResponse)
def rule_evaluate(payload: RuleEvaluateRequest) -> RuleEvaluateResponse:
    try:
        result = evaluate_rule(
            table_name=payload.table_name,
            target_column=payload.target_column,
            positive_value=payload.positive_value,
            where_clause=payload.where_clause,
            alpha=payload.alpha,
            alternative=payload.alternative,
        )
        LOGGER.info(
            json.dumps(
                {
                    "event": "rule_evaluate_success",
                    "table_name": payload.table_name,
                    "target_column": payload.target_column,
                    "where_clause": payload.where_clause[:2000],
                    "rule_n": result.get("rule_n", 0),
                    "total_n": result.get("total_n", 0),
                    "lift": result.get("lift", 0.0),
                    "p_value": result.get("p_value", 1.0),
                    "reject_null": result.get("reject_null", False),
                },
                ensure_ascii=True,
            )
        )
        return RuleEvaluateResponse(**result)
    except Exception as exc:
        LOGGER.exception(
            json.dumps(
                {
                    "event": "rule_evaluate_error",
                    "table_name": payload.table_name,
                    "target_column": payload.target_column,
                    "where_clause": payload.where_clause[:2000],
                    "detail": str(exc)[:4000],
                },
                ensure_ascii=True,
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rule/register", response_model=RuleRegisterResponse)
def rule_register(payload: RuleRegisterRequest) -> RuleRegisterResponse:
    try:
        entry = register_rule(
            rule_name=payload.rule_name,
            where_clause=payload.where_clause,
            table_name=payload.table_name,
            target_column=payload.target_column,
            positive_value=payload.positive_value,
            alpha=payload.alpha,
            alternative=payload.alternative,
            min_lift=payload.min_lift,
            min_rule_n=payload.min_rule_n,
            require_reject_null=payload.require_reject_null,
            owner=payload.owner,
            notes=payload.notes,
        )
        LOGGER.info(
            json.dumps(
                {
                    "event": "rule_register_success",
                    "rule_id": entry["rule_id"],
                    "rule_name": entry["rule_name"],
                    "table_name": entry["table_name"],
                    "target_column": entry["target_column"],
                    "owner": entry["owner"],
                },
                ensure_ascii=True,
            )
        )
        return RuleRegisterResponse(
            rule_id=entry["rule_id"],
            status=entry["status"],
            rule_name=entry["rule_name"],
            registered_at=entry["registered_at"],
            where_clause=entry["where_clause"],
            evaluation=RuleEvaluateResponse(**entry["evaluation"]),
            summary=entry["evaluation"]["summary"],
        )
    except Exception as exc:
        LOGGER.exception(
            json.dumps(
                {
                    "event": "rule_register_error",
                    "rule_name": payload.rule_name[:200],
                    "table_name": payload.table_name,
                    "target_column": payload.target_column,
                    "detail": str(exc)[:4000],
                },
                ensure_ascii=True,
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/rule/list", response_model=RuleListResponse)
def rule_list(status: Optional[str] = None) -> RuleListResponse:
    try:
        rules = list_rules(status=status)
        return RuleListResponse(count=len(rules), rules=rules)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rule/apply-customer", response_model=RuleApplyCustomerResponse)
def rule_apply_customer(payload: RuleApplyCustomerRequest) -> RuleApplyCustomerResponse:
    try:
        result = apply_rules_to_customer(
            customer_id=payload.customer_id,
            table_name=payload.table_name,
            id_column=payload.id_column,
            include_all_rules=payload.include_all_rules,
        )
        return RuleApplyCustomerResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/hypothesis/test", response_model=HypothesisTestResponse)
def hypothesis_test(payload: HypothesisTestRequest) -> HypothesisTestResponse:
    timings: Dict[str, float] = {}
    usage_tracker: Dict[str, Dict[str, int]] = {}
    t_total = time.perf_counter()
    try:
        t0 = time.perf_counter()
        cid, conversation_context = _get_conversation_snapshot(payload.conversation_id)
        if payload.profile:
            timings["conversation_snapshot_ms"] = _elapsed_ms(t0)

        engine = get_engine()
        t0 = time.perf_counter()
        schema_map = get_db_schema(engine)
        if payload.profile:
            timings["load_schema_ms"] = _elapsed_ms(t0)
        if not schema_map:
            raise ValueError("No tables found in PostgreSQL public schema. Run ingestion first.")

        t0 = time.perf_counter()
        desc_df = get_column_descriptions_from_db()
        if payload.profile:
            timings["load_metadata_ms"] = _elapsed_ms(t0)
        if desc_df.empty:
            raise ValueError(f"No metadata found in {METADATA_TABLE}. Run --load-metadata first.")

        t0 = time.perf_counter()
        context = build_semantic_context(schema_map, desc_df)
        if payload.profile:
            timings["build_semantic_context_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        drafted = generate_rule_where_clause(
            rule_intent=payload.hypothesis,
            table_name=payload.table_name,
            target_column=payload.target_column,
            semantic_context=context,
            conversation_context=conversation_context,
            usage_tracker=usage_tracker,
        )
        if payload.profile:
            timings["llm_generate_rule_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        evaluation = evaluate_rule(
            table_name=payload.table_name,
            target_column=payload.target_column,
            positive_value=payload.positive_value,
            where_clause=drafted["where_clause"],
            alpha=payload.alpha,
            alternative=payload.alternative,
        )
        if payload.profile:
            timings["evaluate_rule_ms"] = _elapsed_ms(t0)

        follow_ups = [
            "Can we tighten this rule to improve lift while keeping support above 5%?",
            "How stable is this rule month-over-month or quarter-over-quarter?",
            "Which sub-segments violate this rule (possible exceptions)?",
        ]

        _record_turn(
            cid,
            user=f"Hypothesis: {payload.hypothesis}",
            sql=f'RULE WHERE {drafted["where_clause"]}',
            answer=evaluation["summary"],
            result_df=pd.DataFrame([evaluation]),
        )
        if payload.profile:
            timings["total_ms"] = _elapsed_ms(t_total)

        LOGGER.info(
            json.dumps(
                {
                    "event": "hypothesis_test_success",
                    "conversation_id": cid,
                    "hypothesis": payload.hypothesis[:1000],
                    "null_hypothesis": payload.null_hypothesis[:1000],
                    "table_name": payload.table_name,
                    "target_column": payload.target_column,
                    "where_clause": drafted["where_clause"][:2000],
                    "p_value": evaluation["p_value"],
                    "reject_null": evaluation["reject_null"],
                    "timings_ms": timings,
                    "token_usage": usage_tracker,
                },
                ensure_ascii=True,
            )
        )
        return HypothesisTestResponse(
            conversation_id=cid,
            hypothesis=payload.hypothesis,
            null_hypothesis=payload.null_hypothesis,
            drafted_rule_name=drafted["rule_name"],
            where_clause=drafted["where_clause"],
            evaluation=RuleEvaluateResponse(**evaluation),
            follow_up_questions=follow_ups,
            timings_ms=timings if payload.profile else None,
        )
    except Exception as exc:
        LOGGER.exception(
            json.dumps(
                {
                    "event": "hypothesis_test_error",
                    "hypothesis": payload.hypothesis[:1000],
                    "table_name": payload.table_name,
                    "target_column": payload.target_column,
                    "detail": str(exc)[:4000],
                },
                ensure_ascii=True,
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ews/rule-model/train", response_model=RuleModelTrainResponse)
def rule_model_train(payload: RuleModelTrainRequest) -> RuleModelTrainResponse:
    try:
        model_payload = train_rule_model(
            table_name=payload.table_name,
            id_column=payload.id_column,
            target_column=payload.target_column,
            positive_value=payload.positive_value,
            sample_rows=payload.sample_rows,
            test_size=payload.test_size,
            random_state=payload.random_state,
            max_depth=payload.max_depth,
            min_samples_leaf=payload.min_samples_leaf,
            top_k_features=payload.top_k_features,
            max_rules=payload.max_rules,
            exclude_columns=payload.exclude_columns,
        )
        auto_registered_rule_ids: List[str] = []
        auto_register_failures: List[Dict[str, Any]] = []
        if payload.auto_register_extracted_rules:
            for idx, r in enumerate(model_payload.get("extracted_rules", []), start=1):
                where_clause = str(r.get("where_clause", "")).strip()
                if not where_clause:
                    auto_register_failures.append(
                        {
                            "index": idx,
                            "rule_name": r.get("rule_name"),
                            "reason": "empty_where_clause",
                        }
                    )
                    continue
                base_name = str(r.get("rule_name", f"Tree high-risk rule {idx}")).strip() or f"Tree high-risk rule {idx}"
                auto_rule_name = f"{base_name} [{model_payload['model_id']}]"
                try:
                    entry = register_rule(
                        rule_name=auto_rule_name,
                        where_clause=where_clause,
                        table_name=payload.table_name,
                        target_column=payload.target_column,
                        positive_value=payload.positive_value,
                        alpha=payload.auto_register_alpha,
                        alternative=payload.auto_register_alternative,
                        min_lift=payload.auto_register_min_lift,
                        min_rule_n=payload.auto_register_min_rule_n,
                        require_reject_null=payload.auto_register_require_reject_null,
                        owner=payload.auto_register_owner,
                        notes=f"{payload.auto_register_notes_prefix}; model_id={model_payload['model_id']}; idx={idx}",
                    )
                    auto_registered_rule_ids.append(str(entry.get("rule_id")))
                except Exception as reg_exc:
                    auto_register_failures.append(
                        {
                            "index": idx,
                            "rule_name": auto_rule_name,
                            "where_clause": where_clause,
                            "reason": str(reg_exc),
                        }
                    )
        response = {
            "model_id": model_payload["model_id"],
            "model_path": model_payload["model_path"],
            "table_name": model_payload["table_name"],
            "id_column": model_payload["id_column"],
            "target_column": model_payload["target_column"],
            "positive_value": model_payload["positive_value"],
            "train_rows": model_payload["train_rows"],
            "feature_count": len(model_payload["feature_cols"]),
            "positive_rate": round(float(model_payload["positive_rate"]), 6),
            "validation_auc": (
                round(float(model_payload["validation_auc"]), 6)
                if model_payload["validation_auc"] is not None
                else None
            ),
            "top_features": model_payload["top_features"],
            "extracted_rules": model_payload["extracted_rules"],
            "auto_registered_count": len(auto_registered_rule_ids),
            "auto_registered_rule_ids": auto_registered_rule_ids,
            "auto_register_failures": auto_register_failures,
            "status": "trained",
        }
        LOGGER.info(
            json.dumps(
                {
                    "event": "rule_model_train_success",
                    "model_id": response["model_id"],
                    "model_path": response["model_path"],
                    "table_name": response["table_name"],
                    "target_column": response["target_column"],
                    "train_rows": response["train_rows"],
                    "feature_count": response["feature_count"],
                    "validation_auc": response["validation_auc"],
                    "rule_count": len(response["extracted_rules"]),
                    "auto_registered_count": response["auto_registered_count"],
                    "auto_register_failures_count": len(response["auto_register_failures"]),
                },
                ensure_ascii=True,
            )
        )
        return RuleModelTrainResponse(**response)
    except Exception as exc:
        LOGGER.exception(
            json.dumps(
                {
                    "event": "rule_model_train_error",
                    "table_name": payload.table_name,
                    "target_column": payload.target_column,
                    "detail": str(exc)[:4000],
                },
                ensure_ascii=True,
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ews/shap/train", response_model=LimeTrainResponse)
@app.post("/ews/lime/train", response_model=LimeTrainResponse)  # backward-compatible alias
def shap_train(payload: LimeTrainRequest) -> LimeTrainResponse:
    try:
        model_payload = train_lime_model(
            table_name=payload.table_name,
            id_column=payload.id_column,
            target_column=payload.target_column,
            positive_value=payload.positive_value,
            sample_rows=payload.sample_rows,
            test_size=payload.test_size,
            random_state=payload.random_state,
            exclude_columns=payload.exclude_columns,
        )
        response = {
            "model_id": model_payload["model_id"],
            "model_path": model_payload["model_path"],
            "table_name": model_payload["table_name"],
            "id_column": model_payload["id_column"],
            "target_column": model_payload["target_column"],
            "positive_value": model_payload["positive_value"],
            "train_rows": model_payload["train_rows"],
            "feature_count": len(model_payload["feature_cols"]),
            "positive_rate": round(float(model_payload["positive_rate"]), 6),
            "validation_auc": (
                round(float(model_payload["validation_auc"]), 6)
                if model_payload["validation_auc"] is not None
                else None
            ),
            "status": "trained",
        }
        LOGGER.info(
            json.dumps(
                {
                    "event": "shap_train_success",
                    "model_id": response["model_id"],
                    "model_path": response["model_path"],
                    "table_name": response["table_name"],
                    "target_column": response["target_column"],
                    "train_rows": response["train_rows"],
                    "feature_count": response["feature_count"],
                    "validation_auc": response["validation_auc"],
                },
                ensure_ascii=True,
            )
        )
        return LimeTrainResponse(**response)
    except Exception as exc:
        LOGGER.exception(
            json.dumps(
                {
                    "event": "shap_train_error",
                    "table_name": payload.table_name,
                    "target_column": payload.target_column,
                    "detail": str(exc)[:4000],
                },
                ensure_ascii=True,
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ews/shap/explain", response_model=LimeExplainResponse)
@app.post("/ews/lime/explain", response_model=LimeExplainResponse)  # backward-compatible alias
def shap_explain(payload: LimeExplainRequest) -> LimeExplainResponse:
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()
    try:
        t0 = time.perf_counter()
        cid, _ = _get_conversation_snapshot(payload.conversation_id)
        if payload.profile:
            timings["conversation_snapshot_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        result = explain_customer_with_shap(
            customer_id=payload.customer_id,
            model_id=payload.model_id,
            top_k=payload.top_k,
        )
        if payload.profile:
            timings["shap_explain_ms"] = _elapsed_ms(t0)
            timings["total_ms"] = _elapsed_ms(t_total)

        _record_turn(
            cid,
            user=f"Explain customer {payload.customer_id} via SHAP",
            sql=f'MODEL {result["model_id"]} -> CUSTOMER {payload.customer_id}',
            answer=result["explanation"],
            result_df=pd.DataFrame(result["feature_contributions"]),
        )
        LOGGER.info(
            json.dumps(
                {
                    "event": "shap_explain_success",
                    "conversation_id": cid,
                    "model_id": result["model_id"],
                    "customer_id": payload.customer_id,
                    "predicted_probability": result["predicted_probability"],
                    "predicted_label": result["predicted_label"],
                    "top_k": payload.top_k,
                    "timings_ms": timings,
                },
                ensure_ascii=True,
            )
        )
        return LimeExplainResponse(
            conversation_id=cid,
            model_id=result["model_id"],
            customer_id=result["customer_id"],
            predicted_probability=result["predicted_probability"],
            predicted_label=result["predicted_label"],
            feature_contributions=result["feature_contributions"],
            explanation=result["explanation"],
            timings_ms=timings if payload.profile else None,
        )
    except Exception as exc:
        LOGGER.exception(
            json.dumps(
                {
                    "event": "shap_explain_error",
                    "customer_id": payload.customer_id,
                    "model_id": payload.model_id,
                    "detail": str(exc)[:4000],
                },
                ensure_ascii=True,
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Home Credit NL -> SQL -> NL pipeline")
    parser.add_argument("--data-dir", default=str(default_data_dir()), help="Path to raw_kaggle_data directory")
    parser.add_argument("--ingest", action="store_true", help="Ingest CSV files into Postgres")
    parser.add_argument("--load-metadata", action="store_true", help="Load column description metadata table")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional sampled rows per table")
    parser.add_argument("--question", type=str, default="", help="User question to answer")
    parser.add_argument(
        "--conversation-id",
        type=str,
        default="",
        help="Optional id to chain follow-up questions (same as API conversation_id).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-step timings (same as API profile=true).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if args.ingest:
        load_csvs_to_postgres(data_dir=data_dir, sample_rows=args.sample_rows)
    if args.load_metadata:
        load_metadata_table(data_dir=data_dir)

    if args.question:
        result = answer_question(
            args.question,
            conversation_id=args.conversation_id or None,
            profile=args.profile,
        )

        print("\nConversation id (reuse for follow-ups):\n")
        print(result["conversation_id"])
        print("\nGenerated SQL:\n")
        print(result["sql"])
        print("\nQuery output preview:\n")
        preview_df = pd.DataFrame(result["preview"])
        print(dataframe_preview(preview_df, max_rows=20))
        print("\nNatural language answer:\n")
        print(result["answer"])
        if args.profile and result.get("timings_ms"):
            print("\nTimings (ms):\n")
            print(json.dumps(result["timings_ms"], indent=2))


if __name__ == "__main__":
    main()
