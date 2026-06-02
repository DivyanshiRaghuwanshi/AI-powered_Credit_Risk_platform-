from __future__ import annotations

import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
from sqlalchemy import text

from .types import Phase4AdapterHealth, Phase4InvestigationRequest, Phase4ModelResponse


class ModelAdapter(Protocol):
    def health(self) -> Phase4AdapterHealth: ...

    def score(self, request: Phase4InvestigationRequest, retrieval_output: Dict[str, Any]) -> Phase4ModelResponse: ...


def _calibrate_probability(prob: float, enabled: bool, calibrator: Optional[Dict[str, Any]]) -> tuple[float, str]:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    if not enabled:
        return p, "calibration_disabled"
    if not calibrator:
        return p, "identity_no_calibrator"

    kind = str(calibrator.get("type", "platt")).lower()
    if kind != "platt":
        return p, f"identity_unsupported_{kind}"

    a = float(calibrator.get("a", 1.0))
    b = float(calibrator.get("b", 0.0))
    logit = math.log(p / (1.0 - p))
    calibrated = 1.0 / (1.0 + math.exp(-(a * logit + b)))
    return float(calibrated), "platt"


def _prob_to_band(prob: float) -> str:
    if prob >= 0.70:
        return "high"
    if prob >= 0.40:
        return "medium"
    return "low"


class InMemoryModelAdapter:
    """Simple deterministic model adapter for tests and local demos."""

    def health(self) -> Phase4AdapterHealth:
        return Phase4AdapterHealth(name="model_in_memory", ok=True, message="in-memory model adapter")

    def score(self, request: Phase4InvestigationRequest, retrieval_output: Dict[str, Any]) -> Phase4ModelResponse:
        t0 = time.perf_counter()
        if not request.customer_gid and request.customer_id is None:
            return Phase4ModelResponse(
                source="model",
                record={},
                latency_ms=0.0,
                note="customer_missing",
            )

        sql_rows: List[Dict[str, Any]] = []
        for r in retrieval_output.get("responses", []):
            if r.get("source") == "sql":
                sql_rows = list(r.get("records", []))
                break

        anchor = sql_rows[0] if sql_rows else {}
        dpd = float(anchor.get("dpd", anchor.get("past_dpd_max", 0.0)) or 0.0)
        overdues = float(anchor.get("CREDIT_DAY_OVERDUE", anchor.get("bureau_day_overdue_max", 0.0)) or 0.0)
        utilization = float(anchor.get("cc_utilization_max", 0.0) or 0.0)

        base = 0.08
        contribs = []
        if dpd > 0:
            w = min(0.60, dpd / 120.0)
            base += w
            contribs.append({"feature": "dpd", "feature_value": dpd, "weight": round(w, 6), "direction": "increases_risk"})
        if overdues > 0:
            w = min(0.25, overdues / 365.0)
            base += w
            contribs.append(
                {
                    "feature": "CREDIT_DAY_OVERDUE",
                    "feature_value": overdues,
                    "weight": round(w, 6),
                    "direction": "increases_risk",
                }
            )
        if utilization > 0:
            w = min(0.20, utilization / 2.0)
            base += w
            contribs.append(
                {
                    "feature": "cc_utilization_max",
                    "feature_value": utilization,
                    "weight": round(w, 6),
                    "direction": "increases_risk",
                }
            )
        if not contribs:
            contribs.append({"feature": "history_signal", "feature_value": 0.0, "weight": -0.02, "direction": "decreases_risk"})

        raw_prob = float(min(max(base, 0.01), 0.99))
        calibrated_prob, calib_note = _calibrate_probability(raw_prob, request.apply_calibration, calibrator=None)
        pred_label = int(calibrated_prob >= 0.50)
        band = _prob_to_band(calibrated_prob)
        top_k = max(1, int(request.explanation_top_k))
        top_contribs = sorted(contribs, key=lambda x: abs(float(x["weight"])), reverse=True)[:top_k]

        explanation = (
            f"Customer risk band is {band} with calibrated probability {calibrated_prob:.2%}. "
            f"Primary drivers: {', '.join([c['feature'] for c in top_contribs[:3]])}."
        )

        record = {
            "model_id": request.model_id or "in_memory_heuristic_v1",
            "predicted_probability_raw": round(raw_prob, 6),
            "predicted_probability": round(calibrated_prob, 6),
            "predicted_label": pred_label,
            "risk_band": band,
            "calibration": calib_note,
            "feature_contributions": top_contribs if request.include_explanation else [],
            "explanation": explanation if request.include_explanation else "",
        }
        return Phase4ModelResponse(
            source="model",
            record=record,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            note=f"contributors={len(top_contribs)}",
        )


class ShapModelArtifactAdapter:
    """Loads persisted SHAP model artifacts and explains one customer."""

    def __init__(
        self,
        engine: Any,
        model_dir: str,
        latest_model_file: str = "latest_model_id.txt",
    ) -> None:
        self.engine = engine
        self.model_dir = Path(model_dir)
        self.latest_model_file = latest_model_file

    def health(self) -> Phase4AdapterHealth:
        try:
            if not self.model_dir.exists():
                return Phase4AdapterHealth(name="model_shap_artifact", ok=False, message=f"model_dir_missing:{self.model_dir}")
            latest = self._latest_model_id()
            if not latest:
                return Phase4AdapterHealth(name="model_shap_artifact", ok=False, message="latest_model_id_missing")
            return Phase4AdapterHealth(name="model_shap_artifact", ok=True, message=f"ready:{latest}")
        except Exception as exc:
            return Phase4AdapterHealth(name="model_shap_artifact", ok=False, message=str(exc))

    def _latest_model_id(self) -> Optional[str]:
        p = self.model_dir / self.latest_model_file
        if not p.exists():
            return None
        value = p.read_text(encoding="utf-8").strip()
        return value or None

    def _artifact_path(self, model_id: str) -> Path:
        safe = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in model_id)
        return self.model_dir / f"{safe}.pkl"

    def _load_payload(self, model_id: Optional[str]) -> Dict[str, Any]:
        mid = (model_id or "").strip() or (self._latest_model_id() or "")
        if not mid:
            raise ValueError("No SHAP model available. Train model first via /ews/shap/train.")
        p = self._artifact_path(mid)
        if not p.exists():
            raise ValueError(f'SHAP model artifact not found for "{mid}".')
        with p.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f'Invalid model artifact payload for "{mid}".')
        return payload

    def _resolve_customer_id(self, request: Phase4InvestigationRequest) -> int:
        if request.customer_id is not None:
            return int(request.customer_id)
        if request.customer_gid is not None:
            return int(str(request.customer_gid))
        raise ValueError("customer_id/customer_gid required for model scoring.")

    def _load_customer_row(self, model_payload: Dict[str, Any], customer_id: int) -> pd.DataFrame:
        table_name = model_payload["table_name"]
        id_column = model_payload["id_column"]
        feature_cols = list(model_payload["feature_cols"])
        wanted_cols = [id_column] + feature_cols
        quoted = ", ".join([f'"{c}"' for c in wanted_cols])
        sql = text(f'SELECT {quoted} FROM "{table_name}" WHERE "{id_column}" = :cid LIMIT 1')
        with self.engine.connect() as conn:
            row_df = pd.read_sql(sql, conn, params={"cid": int(customer_id)})
        if row_df.empty:
            raise ValueError(f'Customer id {customer_id} not found in table "{table_name}".')

        medians = model_payload.get("medians", {}) or {}
        X = pd.DataFrame(index=row_df.index)
        for c in feature_cols:
            num = pd.to_numeric(row_df[c], errors="coerce")
            X[c] = num.fillna(float(medians.get(c, 0.0)))
        return X

    @staticmethod
    def _extract_shap_row(shap_values_raw: Any) -> np.ndarray:
        shap_arr: np.ndarray
        if isinstance(shap_values_raw, list):
            idx = 1 if len(shap_values_raw) > 1 else 0
            shap_arr = np.asarray(shap_values_raw[idx])
        else:
            shap_arr = np.asarray(shap_values_raw)

        if shap_arr.ndim == 3:
            cls_idx = 1 if shap_arr.shape[2] > 1 else 0
            return np.asarray(shap_arr[0, :, cls_idx])
        if shap_arr.ndim == 2:
            return np.asarray(shap_arr[0])
        if shap_arr.ndim == 1:
            return np.asarray(shap_arr)
        raise ValueError(f"Unexpected SHAP values shape: {shap_arr.shape}")

    def score(self, request: Phase4InvestigationRequest, retrieval_output: Dict[str, Any]) -> Phase4ModelResponse:
        _ = retrieval_output
        t0 = time.perf_counter()
        payload = self._load_payload(request.model_id)
        model = payload["model"]
        explainer = payload["explainer"]
        feature_cols = list(payload["feature_cols"])
        customer_id = self._resolve_customer_id(request)
        X_row = self._load_customer_row(payload, customer_id)

        raw_prob = float(model.predict_proba(X_row.values)[0, 1])
        calibrator = payload.get("calibration")
        calibrated_prob, calib_note = _calibrate_probability(raw_prob, request.apply_calibration, calibrator)
        pred_label = int(calibrated_prob >= 0.50)
        band = _prob_to_band(calibrated_prob)

        shap_values_raw = explainer.shap_values(X_row)
        shap_row = self._extract_shap_row(shap_values_raw)
        values = X_row.iloc[0]
        contributions: List[Dict[str, Any]] = []
        for feat, sv in zip(feature_cols, shap_row):
            contributions.append(
                {
                    "feature": str(feat),
                    "feature_value": float(values[feat]),
                    "weight": float(sv),
                    "direction": "increases_risk" if float(sv) >= 0 else "decreases_risk",
                }
            )
        top_k = max(1, int(request.explanation_top_k))
        top_contribs = sorted(contributions, key=lambda x: abs(float(x["weight"])), reverse=True)[:top_k]

        explanation = (
            f"Customer {customer_id} is classified as {band} risk "
            f"with calibrated probability {calibrated_prob:.2%}. "
            f"Top drivers: {', '.join([c['feature'] for c in top_contribs[:3]])}."
        )

        record = {
            "model_id": payload.get("model_id", request.model_id),
            "predicted_probability_raw": round(raw_prob, 6),
            "predicted_probability": round(calibrated_prob, 6),
            "predicted_label": pred_label,
            "risk_band": band,
            "calibration": calib_note,
            "feature_contributions": top_contribs if request.include_explanation else [],
            "explanation": explanation if request.include_explanation else "",
        }

        return Phase4ModelResponse(
            source="model",
            record=record,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            note=f"contributors={len(top_contribs)}",
        )

