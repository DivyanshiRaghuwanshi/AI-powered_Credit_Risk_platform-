from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import streamlit as st


st.set_page_config(
    page_title="Risk Intelligence Studio",
    page_icon="R",
    layout="wide",
)


def _apply_branding() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(76, 110, 245, 0.18), transparent 28%),
                linear-gradient(180deg, #101827 0%, #0b1220 100%);
            color: #e5e7eb;
        }
        .stApp header {
            background: rgba(11, 18, 32, 0.85);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }
        .stApp label,
        .stApp p,
        .stApp span,
        .stApp div,
        .stApp .stMarkdown,
        .stApp .stCaption,
        .stApp .stHelp,
        .stApp .st-bq,
        .stApp .st-13f0b0x,
        .stApp [data-testid="stMarkdownContainer"] {
            color: #f8fafc !important;
        }
        .stApp input,
        .stApp textarea,
        .stApp .stSelectbox,
        .stApp .stNumberInput,
        .stApp .stTextInput {
            color: #f8fafc !important;
        }
        .stApp input::placeholder,
        .stApp textarea::placeholder {
            color: #cbd5e1 !important;
            opacity: 1;
        }
        .block-container {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _api_base() -> str:
    return st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000").rstrip("/")


def _post(base: str, path: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    url = f"{base}{path}"
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code} {url}\n{detail}")
    return r.json()


def _get(base: str, path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> Dict[str, Any]:
    url = f"{base}{path}"
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url, params=params or {})
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code} {url}\n{detail}")
    return r.json()


def _health(base: str) -> tuple[bool, str]:
    try:
        data = _get(base, "/health")
        return True, str(data)
    except Exception as exc:
        return False, str(exc)


def section_ask(base: str) -> None:
    st.subheader("Ask Data (Question -> Query -> Insight)")
    c1, c2, c3 = st.columns([6, 2, 2])
    question = c1.text_input("Question", value="Which customers have the highest bureau overdue amount?")
    limit_rows = c2.number_input("Limit Rows", min_value=1, max_value=5000, value=200, step=50)
    profile = c3.checkbox("Profile Timings", value=True)

    conv_id = st.text_input("Conversation ID (optional for follow-ups)", value=st.session_state.get("conversation_id", ""))

    if st.button("Ask", type="primary"):
        try:
            payload = {
                "question": question,
                "limit_rows": int(limit_rows),
                "profile": bool(profile),
            }
            if conv_id.strip():
                payload["conversation_id"] = conv_id.strip()
            data = _post(base, "/ask", payload, timeout=180)
            st.session_state["conversation_id"] = data.get("conversation_id", "")
            st.success(f'Conversation ID: {data.get("conversation_id", "")}')
            st.code(data.get("sql", ""), language="sql")
            st.write(data.get("answer", ""))
            preview = data.get("preview", [])
            if preview:
                st.dataframe(preview, use_container_width=True)
            if data.get("timings_ms"):
                st.json(data["timings_ms"])
        except Exception as exc:
            st.error(str(exc))


def section_eda(base: str) -> None:
    """Display precomputed EDA markdown and images from docs/results."""
    st.subheader("Exploratory Data Analysis (EDA)")
    # Try several ancestor folders to handle nested repo layouts
    md_file = None
    resolved = Path(__file__).resolve()
    repo_root = None
    for i in range(1, 6):
        candidate_root = resolved.parents[i] if i < len(resolved.parents) else None
        if not candidate_root:
            continue
        candidate = candidate_root / "docs" / "results" / "home_credit_eda.md"
        if candidate.exists():
            md_file = candidate
            results_dir = candidate_root / "docs" / "results"
            repo_root = candidate_root
            break
    # fallback: check workspace-relative path (outer repo)
    if md_file is None:
        candidate = Path("docs") / "results" / "home_credit_eda.md"
        if candidate.exists():
            md_file = candidate
            results_dir = Path("docs") / "results"
            repo_root = Path.cwd()

    if md_file and md_file.exists():
        # Show the markdown summary first
        try:
            md_text = md_file.read_text(encoding="utf8")
            st.markdown(md_text)
        except Exception:
            with open(md_file, "r", encoding="utf8") as f:
                st.markdown(f.read())

        # Display images if present
        for img_name in ["target_distribution.png", "missingness_top.png", "corr_heatmap.png"]:
            img_path = results_dir / img_name
            if img_path.exists():
                st.image(str(img_path), caption=img_name, use_column_width=True)

        # Provide link / preview for numeric stats CSV
        stats_csv = (repo_root or Path.cwd()) / "data" / "processed" / "home_credit_numeric_stats.csv"
        if stats_csv.exists():
            st.markdown("**Numeric summary CSV**")
            try:
                df = None
                import pandas as _pd

                df = _pd.read_csv(stats_csv)
                st.dataframe(df.head(20), use_container_width=True)
            except Exception:
                st.write(f"CSV available at: {stats_csv}")
    else:
        st.warning(f"EDA summary not found: {md_file}\nRun `python -m scripts.run_eda_home_credit` to generate it.")


def section_rule_lifecycle(base: str) -> None:
    st.subheader("Rule Lab (Draft -> Evaluate -> Register -> Apply)")
    tabs = st.tabs(["Draft", "Evaluate", "Register", "Library", "Apply"])

    with tabs[0]:
        rule_intent = st.text_area(
            "Rule Intent (Natural Language)",
            value="Customers with high POS delinquency and high credit card utilization should be considered high risk.",
            height=100,
        )
        d1, d2, d3 = st.columns(3)
        table_name = d1.text_input("Table", value="master_ews_fibo", key="draft_table")
        target_col = d2.text_input("Target Column", value="EWS_LABEL", key="draft_target")
        conv = d3.text_input("Conversation ID (optional)", value=st.session_state.get("conversation_id", ""), key="draft_conv")
        profile = st.checkbox("Profile Timings", value=True, key="draft_profile")
        if st.button("Draft Rule", key="btn_draft"):
            try:
                payload = {
                    "rule_intent": rule_intent,
                    "table_name": table_name,
                    "target_column": target_col,
                    "profile": profile,
                }
                if conv.strip():
                    payload["conversation_id"] = conv.strip()
                data = _post(base, "/rule/draft", payload, timeout=120)
                st.session_state["conversation_id"] = data.get("conversation_id", st.session_state.get("conversation_id", ""))
                st.session_state["drafted_where_clause"] = data.get("where_clause", "")
                st.session_state["drafted_rule_name"] = data.get("rule_name", "")
                st.success(f'Drafted: {data.get("rule_name","")}')
                st.code(data.get("where_clause", ""), language="sql")
                st.write(data.get("rationale", ""))
                st.caption(data.get("example_sql", ""))
                if data.get("timings_ms"):
                    st.json(data["timings_ms"])
            except Exception as exc:
                st.error(str(exc))

    with tabs[1]:
        where_clause = st.text_area(
            "WHERE clause to evaluate",
            value=st.session_state.get("drafted_where_clause", '"POS_SK_DPD_MAX" >= 30'),
            height=120,
        )
        e1, e2, e3 = st.columns(3)
        table_name = e1.text_input("Table", value="master_ews_fibo", key="eval_table")
        target_col = e2.text_input("Target Column", value="EWS_LABEL", key="eval_target")
        positive_value = e3.text_input("Positive Value", value="1", key="eval_pos")
        e4, e5 = st.columns(2)
        alpha = e4.number_input("Alpha", min_value=0.001, max_value=0.25, value=0.05, step=0.01, format="%.3f")
        alternative = e5.selectbox("Alternative", ["greater", "less", "two-sided"], index=0)
        if st.button("Evaluate Rule", key="btn_eval"):
            try:
                payload = {
                    "table_name": table_name,
                    "target_column": target_col,
                    "positive_value": positive_value,
                    "where_clause": where_clause,
                    "alpha": float(alpha),
                    "alternative": alternative,
                }
                data = _post(base, "/rule/evaluate", payload, timeout=120)
                st.session_state["last_evaluation"] = data
                st.success("Rule evaluated.")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rule N", f'{data.get("rule_n",0):,}')
                m2.metric("Rule Rate", f'{100*float(data.get("rule_rate",0)):.2f}%')
                m3.metric("Lift", f'{float(data.get("lift",0)):.3f}x')
                m4.metric("p-value", f'{float(data.get("p_value",1)):.4g}')
                st.write(data.get("summary", ""))
                st.json(data)
            except Exception as exc:
                st.error(str(exc))

    with tabs[2]:
        default_rule_name = st.session_state.get("drafted_rule_name", "Candidate Rule")
        default_where = st.session_state.get("drafted_where_clause", "")
        r1, r2 = st.columns(2)
        rule_name = r1.text_input("Rule Name", value=default_rule_name)
        owner = r2.text_input("Owner", value="risk_policy")
        where_clause = st.text_area("WHERE clause", value=default_where, height=120, key="reg_where")
        r3, r4, r5 = st.columns(3)
        table_name = r3.text_input("Table", value="master_ews_fibo", key="reg_table")
        target_col = r4.text_input("Target Column", value="EWS_LABEL", key="reg_target")
        positive_value = r5.text_input("Positive Value", value="1", key="reg_pos")
        r6, r7, r8, r9 = st.columns(4)
        alpha = r6.number_input("Alpha", min_value=0.001, max_value=0.25, value=0.05, step=0.01, format="%.3f", key="reg_alpha")
        min_lift = r7.number_input("Min Lift", min_value=0.0, max_value=10.0, value=1.05, step=0.05)
        min_rule_n = r8.number_input("Min Rule N", min_value=1, max_value=1000000, value=200, step=50)
        alternative = r9.selectbox("Alternative", ["greater", "less", "two-sided"], index=0, key="reg_alt")
        require_reject = st.checkbox("Require Reject Null", value=True)
        notes = st.text_area("Notes", value="", height=80)
        if st.button("Register Approved Rule", key="btn_register"):
            try:
                payload = {
                    "rule_name": rule_name,
                    "where_clause": where_clause,
                    "table_name": table_name,
                    "target_column": target_col,
                    "positive_value": positive_value,
                    "alpha": float(alpha),
                    "alternative": alternative,
                    "min_lift": float(min_lift),
                    "min_rule_n": int(min_rule_n),
                    "require_reject_null": bool(require_reject),
                    "notes": notes,
                    "owner": owner,
                }
                data = _post(base, "/rule/register", payload, timeout=120)
                st.success(f'Registered: {data.get("rule_id","")}')
                st.json(data)
            except Exception as exc:
                st.error(str(exc))

    with tabs[3]:
        status = st.selectbox("Status filter", ["", "approved"], index=1)
        if st.button("Refresh Rules", key="btn_list_rules"):
            try:
                params = {"status": status} if status else {}
                data = _get(base, "/rule/list", params=params, timeout=60)
                st.write(f'Count: {data.get("count", 0)}')
                rules = data.get("rules", [])
                if rules:
                    st.dataframe(rules, use_container_width=True)
                else:
                    st.info("No rules found.")
            except Exception as exc:
                st.error(str(exc))

    with tabs[4]:
        a1, a2, a3 = st.columns(3)
        customer_id = a1.number_input("Customer ID", min_value=1, value=100002, step=1)
        table_name = a2.text_input("Table", value="master_ews_fibo", key="apply_table")
        id_col = a3.text_input("ID Column", value="SK_ID_CURR", key="apply_id_col")
        include_all_rules = st.checkbox("Include non-approved rules", value=False)
        if st.button("Apply Rules To Customer", key="btn_apply_customer"):
            try:
                payload = {
                    "customer_id": int(customer_id),
                    "table_name": table_name,
                    "id_column": id_col,
                    "include_all_rules": include_all_rules,
                }
                data = _post(base, "/rule/apply-customer", payload, timeout=120)
                st.success(data.get("summary", "Done"))
                matched = data.get("matched_rules", [])
                if matched:
                    st.dataframe(matched, use_container_width=True)
                st.json(data)
            except Exception as exc:
                st.error(str(exc))


def section_hypothesis(base: str) -> None:
    st.subheader("Hypothesis Lab")
    hypothesis = st.text_area(
        "Hypothesis",
        value="Borrowers with high delinquency and low payment quality have higher positive target probability.",
        height=90,
    )
    null_hyp = st.text_input(
        "Null Hypothesis",
        value="Matched customers have the same positive target rate as others.",
    )
    h1, h2, h3 = st.columns(3)
    table_name = h1.text_input("Table", value="master_ews_fibo", key="hyp_table")
    target_col = h2.text_input("Target Column", value="EWS_LABEL", key="hyp_target")
    positive_value = h3.text_input("Positive Value", value="1", key="hyp_pos")
    h4, h5 = st.columns(2)
    alpha = h4.number_input("Alpha", min_value=0.001, max_value=0.25, value=0.05, step=0.01, format="%.3f", key="hyp_alpha")
    alternative = h5.selectbox("Alternative", ["greater", "less", "two-sided"], index=0, key="hyp_alt")
    profile = st.checkbox("Profile Timings", value=True, key="hyp_profile")
    conv_id = st.text_input("Conversation ID (optional)", value=st.session_state.get("conversation_id", ""), key="hyp_conv")

    if st.button("Run Hypothesis Test", key="btn_hyp_test"):
        try:
            payload = {
                "hypothesis": hypothesis,
                "null_hypothesis": null_hyp,
                "table_name": table_name,
                "target_column": target_col,
                "positive_value": positive_value,
                "alpha": float(alpha),
                "alternative": alternative,
                "profile": profile,
            }
            if conv_id.strip():
                payload["conversation_id"] = conv_id.strip()
            data = _post(base, "/hypothesis/test", payload, timeout=180)
            st.session_state["conversation_id"] = data.get("conversation_id", st.session_state.get("conversation_id", ""))
            st.success("Hypothesis tested.")
            ev = data.get("evaluation", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Lift", f'{float(ev.get("lift",0)):.3f}x')
            m2.metric("p-value", f'{float(ev.get("p_value",1)):.4g}')
            m3.metric("Reject Null", "Yes" if ev.get("reject_null", False) else "No")
            st.code(data.get("where_clause", ""), language="sql")
            st.write(ev.get("summary", ""))
            if data.get("follow_up_questions"):
                st.write("Suggested follow-up questions:")
                for q in data["follow_up_questions"]:
                    st.markdown(f"- {q}")
            if data.get("timings_ms"):
                st.json(data["timings_ms"])
        except Exception as exc:
            st.error(str(exc))


def section_shap(base: str) -> None:
    st.subheader("Customer Explainability (SHAP)")
    tabs = st.tabs(["Train SHAP Model", "Explain Customer"])

    with tabs[0]:
        t1, t2, t3 = st.columns(3)
        table_name = t1.text_input("Table", value="master_ews_fibo", key="lime_table")
        id_col = t2.text_input("ID Column", value="SK_ID_CURR", key="lime_id_col")
        target_col = t3.text_input("Target Column", value="EWS_LABEL", key="lime_target_col")
        t4, t5, t6 = st.columns(3)
        pos_value = t4.text_input("Positive Value", value="1", key="lime_pos_value")
        sample_rows = t5.number_input("Sample Rows", min_value=2000, max_value=1000000, value=120000, step=10000)
        test_size = t6.number_input("Test Size", min_value=0.05, max_value=0.45, value=0.20, step=0.05, format="%.2f")
        exclude_cols_text = st.text_area(
            "Exclude Columns (comma-separated, for leakage control)",
            value="SK_ID_CURR,TARGET,is_train,future_obs_months,future_dpd_max,FUTURE_DPD30_FLAG,FUTURE_DPD60_FLAG,FUTURE_DEFAULT_90_FLAG,FUTURE_STAGE_MAX,TARGET_TRANSITION_TO_90",
            height=80,
            key="shap_exclude_cols",
        )
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
        if st.button("Train SHAP Model", key="btn_lime_train"):
            try:
                exclude_columns = [c.strip() for c in exclude_cols_text.split(",") if c.strip()]
                payload = {
                    "table_name": table_name,
                    "id_column": id_col,
                    "target_column": target_col,
                    "positive_value": pos_value,
                    "sample_rows": int(sample_rows),
                    "test_size": float(test_size),
                    "random_state": int(random_state),
                    "exclude_columns": exclude_columns,
                }
                data = _post(base, "/ews/shap/train", payload, timeout=300)
                st.session_state["lime_model_id"] = data.get("model_id", "")
                st.success(f'SHAP model trained: {data.get("model_id","")}')
                st.write(f'Model path: `{data.get("model_path","")}`')
                c1, c2, c3 = st.columns(3)
                c1.metric("Train Rows", f'{int(data.get("train_rows",0)):,}')
                c2.metric("Feature Count", f'{int(data.get("feature_count",0)):,}')
                auc = data.get("validation_auc", None)
                c3.metric("Validation AUC", "NA" if auc is None else f"{float(auc):.4f}")
                st.json(data)
            except Exception as exc:
                st.error(str(exc))

    with tabs[1]:
        model_id = st.text_input("Model ID (optional; blank uses latest)", value=st.session_state.get("lime_model_id", ""))
        customer_id = st.number_input("Customer ID", min_value=1, value=100002, step=1, key="lime_customer_id")
        top_k = st.slider("Top K features", min_value=3, max_value=25, value=8)
        conv_id = st.text_input("Conversation ID (optional)", value=st.session_state.get("conversation_id", ""), key="lime_conv")
        profile = st.checkbox("Profile Timings", value=True, key="lime_profile")
        if st.button("Explain Customer", key="btn_lime_explain"):
            try:
                payload = {
                    "customer_id": int(customer_id),
                    "top_k": int(top_k),
                    "profile": bool(profile),
                }
                if model_id.strip():
                    payload["model_id"] = model_id.strip()
                if conv_id.strip():
                    payload["conversation_id"] = conv_id.strip()
                data = _post(base, "/ews/shap/explain", payload, timeout=180)
                st.session_state["conversation_id"] = data.get("conversation_id", st.session_state.get("conversation_id", ""))
                st.success("Customer explanation ready.")
                st.metric("Predicted Probability", f'{100*float(data.get("predicted_probability",0)):.2f}%')
                st.metric("Predicted Label", str(data.get("predicted_label", 0)))
                st.write(data.get("explanation", ""))
                contribs = data.get("feature_contributions", [])
                if contribs:
                    st.dataframe(contribs, use_container_width=True)
                if data.get("timings_ms"):
                    st.json(data["timings_ms"])
            except Exception as exc:
                st.error(str(exc))


def section_signal_model(base: str) -> None:
    st.subheader("Signal Builder (Rules + Key Features)")
    c1, c2, c3 = st.columns(3)
    table_name = c1.text_input("Table", value="master_ews_fibo", key="rule_model_table")
    id_col = c2.text_input("ID Column", value="SK_ID_CURR", key="rule_model_id_col")
    target_col = c3.text_input("Target Column", value="EWS_LABEL", key="rule_model_target_col")

    c4, c5, c6 = st.columns(3)
    pos_value = c4.text_input("Positive Value", value="1", key="rule_model_pos_value")
    sample_rows = c5.number_input("Sample Rows", min_value=2000, max_value=1000000, value=120000, step=10000)
    test_size = c6.number_input("Test Size", min_value=0.05, max_value=0.45, value=0.20, step=0.05, format="%.2f", key="rule_model_test_size")

    c7, c8, c9 = st.columns(3)
    random_state = c7.number_input("Random State", min_value=0, max_value=9999, value=42, step=1, key="rule_model_random_state")
    max_depth = c8.number_input("Tree Max Depth", min_value=2, max_value=12, value=4, step=1)
    min_leaf = c9.number_input("Min Samples Leaf", min_value=20, max_value=100000, value=200, step=20)

    c10, c11 = st.columns(2)
    top_k_features = c10.slider("Top Features", min_value=5, max_value=50, value=15, step=1)
    max_rules = c11.slider("Max Extracted Rules", min_value=3, max_value=50, value=12, step=1)
    a1, a2, a3, a4 = st.columns(4)
    auto_register = a1.checkbox("Auto-register extracted rules", value=True)
    auto_min_lift = a2.number_input("Auto min lift", min_value=0.0, max_value=10.0, value=1.05, step=0.05)
    auto_min_rule_n = a3.number_input("Auto min rule N", min_value=1, max_value=1000000, value=200, step=50)
    auto_require_reject = a4.checkbox("Auto require reject-null", value=True)
    exclude_cols_text = st.text_area(
        "Exclude Columns (comma-separated, for leakage control)",
        value="SK_ID_CURR,TARGET,is_train,future_obs_months,future_dpd_max,FUTURE_DPD30_FLAG,FUTURE_DPD60_FLAG,FUTURE_DEFAULT_90_FLAG,FUTURE_STAGE_MAX,TARGET_TRANSITION_TO_90",
        height=80,
        key="rule_model_exclude_cols",
    )

    if st.button("Train Signal Model", type="primary"):
        try:
            exclude_columns = [c.strip() for c in exclude_cols_text.split(",") if c.strip()]
            payload = {
                "table_name": table_name,
                "id_column": id_col,
                "target_column": target_col,
                "positive_value": pos_value,
                "sample_rows": int(sample_rows),
                "test_size": float(test_size),
                "random_state": int(random_state),
                "max_depth": int(max_depth),
                "min_samples_leaf": int(min_leaf),
                "top_k_features": int(top_k_features),
                "max_rules": int(max_rules),
                "exclude_columns": exclude_columns,
                "auto_register_extracted_rules": bool(auto_register),
                "auto_register_min_lift": float(auto_min_lift),
                "auto_register_min_rule_n": int(auto_min_rule_n),
                "auto_register_require_reject_null": bool(auto_require_reject),
            }
            data = _post(base, "/ews/rule-model/train", payload, timeout=300)
            st.session_state["rule_model_id"] = data.get("model_id", "")
            st.success(f'Rule model trained: {data.get("model_id","")}')
            st.write(f'Model path: `{data.get("model_path","")}`')
            m1, m2, m3 = st.columns(3)
            m1.metric("Train Rows", f'{int(data.get("train_rows",0)):,}')
            m2.metric("Feature Count", f'{int(data.get("feature_count",0)):,}')
            auc = data.get("validation_auc", None)
            m3.metric("Validation AUC", "NA" if auc is None else f"{float(auc):.4f}")

            st.markdown("**Top Important Features**")
            top_features = data.get("top_features", [])
            if top_features:
                st.dataframe(top_features, use_container_width=True)
            else:
                st.info("No feature importance available.")

            st.markdown("**Extracted High-Risk Rules**")
            rules = data.get("extracted_rules", [])
            if rules:
                st.dataframe(rules, use_container_width=True)
            else:
                st.info("No positive high-risk rules were extracted with current tree settings.")

            st.markdown("**Auto-Registration Summary**")
            st.write(f'Auto registered count: {int(data.get("auto_registered_count", 0))}')
            auto_ids = data.get("auto_registered_rule_ids", [])
            if auto_ids:
                st.dataframe([{"rule_id": rid} for rid in auto_ids], use_container_width=True)
            failures = data.get("auto_register_failures", [])
            if failures:
                st.warning("Some extracted rules were not auto-registered.")
                st.dataframe(failures, use_container_width=True)

            st.json(data)
        except Exception as exc:
            st.error(str(exc))


def section_phase3_live(base: str) -> None:
    st.subheader("Live Graph Retrieval")
    st.caption("Run real backend health and retrieval flow using API endpoints /phase3/health-real and /phase3/retrieve-real.")

    c1, c2 = st.columns(2)
    if c1.button("Check Real Backend Health", type="primary"):
        try:
            data = _get(base, "/phase3/health-real", timeout=90)
            st.success("Real backend health check completed.")
            st.json(data)
        except Exception as exc:
            st.error(str(exc))

    if c2.button("Check In-Memory Health"):
        try:
            data = _get(base, "/phase3/health", timeout=60)
            st.success("In-memory backend health check completed.")
            st.json(data)
        except Exception as exc:
            st.error(str(exc))

    st.markdown("---")
    st.markdown("**Live Retrieval Request**")
    i1, i2, i3 = st.columns(3)
    intent = i1.selectbox(
        "Intent",
        ["full_investigation", "exact_customer_profile", "fraud_investigation", "semantic_notes", "customer_risk_reasoning"],
        index=0,
    )
    customer_gid = i2.text_input("Customer GID", value="302629")
    top_k = i3.number_input("Top K", min_value=1, max_value=200, value=5, step=1)

    f1, f2, f3 = st.columns(3)
    include_sql = f1.checkbox("Include SQL", value=True)
    include_graph = f2.checkbox("Include Graph", value=True)
    include_vector = f3.checkbox("Include Vector", value=False)

    filters_text = st.text_area(
        "Filters (JSON object)",
        value='{"segment":"retail","region":"IN"}',
        height=90,
    )
    query_embedding_text = st.text_area(
        "Query Embedding (JSON array, optional)",
        value="[1.0, 0.0, 0.0]",
        height=80,
    )
    sql_rows_text = st.text_area(
        "SQL Rows (JSON array, optional fallback payload)",
        value='[{"customer_gid":"302629","segment":"retail","region":"IN","dpd":40}]',
        height=90,
    )
    graph_edges_text = st.text_area(
        "Graph Edges (JSON array, optional fallback payload)",
        value='[{"source":"302629","target":"dev-302629","rel":"LINKED_TO_DEVICE"}]',
        height=90,
    )
    vector_docs_text = st.text_area(
        "Vector Docs (JSON array, optional fallback payload)",
        value='[{"doc_id":"d1","metadata":{"segment":"retail","region":"IN"},"text":"risk note"}]',
        height=90,
    )

    if st.button("Run Phase 3 Real Retrieval"):
        try:
            filters = json.loads(filters_text.strip()) if filters_text.strip() else {}
            query_embedding = json.loads(query_embedding_text.strip()) if query_embedding_text.strip() else None
            sql_rows = json.loads(sql_rows_text.strip()) if sql_rows_text.strip() else []
            graph_edges = json.loads(graph_edges_text.strip()) if graph_edges_text.strip() else []
            vector_docs = json.loads(vector_docs_text.strip()) if vector_docs_text.strip() else []

            payload = {
                "intent": intent,
                "customer_gid": customer_gid.strip() or None,
                "filters": filters,
                "query_embedding": query_embedding,
                "top_k": int(top_k),
                "include_sql": bool(include_sql),
                "include_graph": bool(include_graph),
                "include_vector": bool(include_vector),
                "sql_rows": sql_rows,
                "graph_edges": graph_edges,
                "vector_docs": vector_docs,
            }
            data = _post(base, "/phase3/retrieve-real", payload, timeout=180)
            st.success("Phase 3 live retrieval completed.")

            routes = data.get("routes", [])
            st.write("Routes:", routes)
            st.metric("Total Latency (ms)", f'{float(data.get("total_latency_ms",0)):.3f}')

            responses = data.get("responses", [])
            if responses:
                st.dataframe(
                    [
                        {
                            "source": r.get("source"),
                            "ok": r.get("ok"),
                            "count": r.get("count"),
                            "latency_ms": r.get("latency_ms"),
                            "note": r.get("note"),
                        }
                        for r in responses
                    ],
                    use_container_width=True,
                )
            st.json(data)
        except Exception as exc:
            st.error(str(exc))


def section_phase4_live(base: str) -> None:
    st.subheader("Investigation (Retrieval + Model Explainability)")
    st.caption("Runs /phase4/investigate-live to combine SQL/graph/vector evidence with model scoring.")

    c1, c2 = st.columns(2)
    if c1.button("Check Phase 4 Health", type="primary"):
        try:
            data = _get(base, "/phase4/health", timeout=60)
            st.success("Phase 4 health check completed.")
            st.json(data)
        except Exception as exc:
            st.error(str(exc))

    if c2.button("Check Phase 4 Real Health"):
        try:
            data = _get(base, "/phase4/health-real", timeout=90)
            st.success("Phase 4 real health check completed.")
            st.json(data)
        except Exception as exc:
            st.error(str(exc))

    st.markdown("---")
    i1, i2, i3 = st.columns(3)
    intent = i1.selectbox(
        "Intent",
        ["full_investigation", "customer_risk_reasoning", "fraud_investigation", "exact_customer_profile"],
        index=0,
        key="p4_intent",
    )
    customer_gid = i2.text_input("Customer GID", value="302629", key="p4_customer_gid")
    customer_id = i3.number_input("Customer ID (optional)", min_value=0, value=302629, step=1)

    f1, f2, f3, f4 = st.columns(4)
    include_sql = f1.checkbox("Include SQL", value=True, key="p4_sql")
    include_graph = f2.checkbox("Include Graph", value=True, key="p4_graph")
    include_vector = f3.checkbox("Include Vector", value=False, key="p4_vector")
    include_model = f4.checkbox("Include Model", value=True, key="p4_model")

    e1, e2, e3 = st.columns(3)
    include_explanation = e1.checkbox("Include Explanation", value=True, key="p4_explain")
    apply_calibration = e2.checkbox("Apply Calibration", value=True, key="p4_calib")
    explanation_top_k = e3.slider("Top K Contributions", min_value=1, max_value=25, value=8, step=1)

    filters_text = st.text_area("Filters (JSON object)", value='{"segment":"retail","region":"IN"}', height=80, key="p4_filters")
    query_embedding_text = st.text_area("Query Embedding (JSON array)", value="[1.0, 0.0, 0.0]", height=80, key="p4_emb")
    model_id = st.text_input("Model ID (optional; real endpoint uses latest if blank)", value=st.session_state.get("lime_model_id", ""))
    sql_rows_text = st.text_area(
        "SQL Rows for in-memory test",
        value='[{"customer_gid":"302629","segment":"retail","region":"IN","dpd":75}]',
        height=90,
        key="p4_sql_rows",
    )
    graph_edges_text = st.text_area(
        "Graph Edges for in-memory test",
        value='[{"source":"302629","target":"dev-302629","rel":"LINKED_TO_DEVICE"}]',
        height=90,
        key="p4_graph_edges",
    )
    vector_docs_text = st.text_area(
        "Vector Docs for in-memory test",
        value='[{"doc_id":"d1","metadata":{"segment":"retail","region":"IN"},"text":"risk note"}]',
        height=90,
        key="p4_vector_docs",
    )

    if st.button("Run Phase 4 Investigation", type="primary"):
        try:
            payload = {
                "intent": intent,
                "customer_gid": customer_gid.strip() or None,
                "customer_id": int(customer_id) if int(customer_id) > 0 else None,
                "model_id": model_id.strip() or None,
                "filters": json.loads(filters_text.strip()) if filters_text.strip() else {},
                "query_embedding": json.loads(query_embedding_text.strip()) if query_embedding_text.strip() else None,
                "top_k": 5,
                "include_sql": bool(include_sql),
                "include_graph": bool(include_graph),
                "include_vector": bool(include_vector),
                "include_model": bool(include_model),
                "include_explanation": bool(include_explanation),
                "apply_calibration": bool(apply_calibration),
                "explanation_top_k": int(explanation_top_k),
                "sql_rows": json.loads(sql_rows_text.strip()) if sql_rows_text.strip() else [],
                "graph_edges": json.loads(graph_edges_text.strip()) if graph_edges_text.strip() else [],
                "vector_docs": json.loads(vector_docs_text.strip()) if vector_docs_text.strip() else [],
            }
            data = _post(base, "/phase4/investigate-live", payload, timeout=180)
            st.success("Phase 4 investigation completed.")
            st.metric("Total Latency (ms)", f'{float(data.get("total_latency_ms",0)):.3f}')
            st.json(data.get("summary", {}))
            st.dataframe(
                [
                    {
                        "source": r.get("source"),
                        "ok": r.get("ok"),
                        "count": r.get("count"),
                        "latency_ms": r.get("latency_ms"),
                        "note": r.get("note"),
                    }
                    for r in data.get("responses", [])
                ],
                use_container_width=True,
            )
            st.json(data)
        except Exception as exc:
            st.error(str(exc))


def section_phase5_copilot(base: str) -> None:
    st.subheader("Copilot (Policy + Trace + Audit)")
    st.caption("Runs /phase5/investigate-live for policy-driven orchestration with evidence and governance trace.")

    c1, c2 = st.columns(2)
    if c1.button("Check Phase 5 Health", type="primary"):
        try:
            data = _get(base, "/phase5/health", timeout=60)
            st.success("Phase 5 health check completed.")
            st.json(data)
        except Exception as exc:
            st.error(str(exc))
    if c2.button("Check Phase 5 Real Health"):
        try:
            data = _get(base, "/phase5/health-real", timeout=90)
            st.success("Phase 5 real health check completed.")
            st.json(data)
        except Exception as exc:
            st.error(str(exc))

    st.markdown("---")
    a1, a2, a3 = st.columns(3)
    intent = a1.selectbox(
        "Intent",
        ["fraud_investigation", "full_investigation", "customer_risk_reasoning", "exact_customer_profile", "semantic_notes"],
        index=0,
        key="p5_intent",
    )
    customer_gid = a2.text_input("Customer GID", value="302629", key="p5_gid")
    customer_id = a3.number_input("Customer ID", min_value=0, value=302629, step=1, key="p5_cid")

    b1, b2, b3, b4 = st.columns(4)
    role = b1.selectbox("Role", ["analyst", "senior_analyst", "risk_manager", "auditor"], index=0)
    user_id = b2.text_input("User ID", value="demo_user")
    request_id = b3.text_input("Request ID", value="demo_request")
    policy_profile = b4.selectbox("Policy Profile", ["", "balanced", "fraud_deep_dive", "credit_fast", "semantic_review"], index=0)

    c3, c4, c5 = st.columns(3)
    strict_policy = c3.checkbox("Strict Route Policy", value=True)
    include_explanation = c4.checkbox("Include Explanation", value=True)
    apply_calibration = c5.checkbox("Apply Calibration", value=True)
    use_real_endpoint = st.checkbox("Use Real Endpoint (/phase5/investigate-real)", value=False)

    filters_text = st.text_area("Filters (JSON object)", value='{"segment":"retail","region":"IN"}', height=80, key="p5_filters")
    query_embedding_text = st.text_area("Query Embedding (JSON array)", value="[1.0,0.0,0.0]", height=80, key="p5_emb")
    model_id = st.text_input("Model ID (optional)", value=st.session_state.get("lime_model_id", ""))
    sql_rows_text = st.text_area(
        "SQL Rows for in-memory run",
        value='[{"customer_gid":"302629","segment":"retail","region":"IN","dpd":72}]',
        height=90,
        key="p5_sql_rows",
    )
    graph_edges_text = st.text_area(
        "Graph Edges for in-memory run",
        value='[{"source":"302629","target":"dev-302629","rel":"LINKED_TO_DEVICE"}]',
        height=90,
        key="p5_graph_edges",
    )
    vector_docs_text = st.text_area(
        "Vector Docs for in-memory run",
        value='[{"doc_id":"d1","metadata":{"segment":"retail","region":"IN"},"text":"linked to previously risky device"}]',
        height=90,
        key="p5_vector_docs",
    )

    if st.button("Run Phase 5 Copilot Investigation", type="primary"):
        try:
            payload = {
                "intent": intent,
                "customer_gid": customer_gid.strip() or None,
                "customer_id": int(customer_id) if int(customer_id) > 0 else None,
                "model_id": model_id.strip() or None,
                "role": role,
                "user_id": user_id,
                "request_id": request_id,
                "policy_profile": policy_profile or None,
                "strict_route_policy": bool(strict_policy),
                "filters": json.loads(filters_text.strip()) if filters_text.strip() else {},
                "query_embedding": json.loads(query_embedding_text.strip()) if query_embedding_text.strip() else None,
                "include_explanation": bool(include_explanation),
                "apply_calibration": bool(apply_calibration),
                "sql_rows": json.loads(sql_rows_text.strip()) if sql_rows_text.strip() else [],
                "graph_edges": json.loads(graph_edges_text.strip()) if graph_edges_text.strip() else [],
                "vector_docs": json.loads(vector_docs_text.strip()) if vector_docs_text.strip() else [],
            }
            target_path = "/phase5/investigate-real" if use_real_endpoint else "/phase5/investigate-live"
            data = _post(base, target_path, payload, timeout=180)
            st.success("Phase 5 investigation completed.")
            st.metric("Total Latency (ms)", f'{float(data.get("total_latency_ms",0)):.3f}')
            st.markdown("**Summary**")
            st.json(data.get("summary", {}))
            st.markdown("**Claims**")
            for claim in data.get("claims", []):
                st.markdown(f"- {claim}")
            st.markdown("**Trace**")
            st.json(data.get("trace", {}))
            st.markdown("**Audit Event**")
            st.json(data.get("audit_event", {}))
            st.markdown("**Evidence Map**")
            st.json(data.get("evidence_map", {}))
            st.markdown("**Source Responses**")
            st.dataframe(
                [
                    {
                        "source": r.get("source"),
                        "ok": r.get("ok"),
                        "count": r.get("count"),
                        "latency_ms": r.get("latency_ms"),
                        "note": r.get("note"),
                    }
                    for r in data.get("responses", [])
                ],
                use_container_width=True,
            )
            st.json(data)
        except Exception as exc:
            st.error(str(exc))



def section_risk_workflow(base: str) -> None:
    st.subheader("Risk Early Warning Workflow")
    st.caption("Identify near-default customers with rule + model workflow, then explain why and what to do next.")

    w1, w2, w3 = st.columns(3)
    table_name = w1.text_input("Working Table", value="home_credit_future_dpd_labels", key="wf_table")
    target_col = w2.text_input("Target Column", value="TARGET_TRANSITION_TO_90", key="wf_target")
    pos_value = w3.text_input("Positive Value", value="1", key="wf_pos")

    t_train, t_eval, t_register, t_apply, t_explain = st.tabs(
        [
            "1) Train + Extract Rules",
            "2) Evaluate Manual Rule",
            "3) Register Manual Rule",
            "4) Apply Rules To Customer",
            "5) Explain Customer (SHAP)",
        ]
    )

    with t_train:
        c1, c2, c3 = st.columns(3)
        sample_rows = c1.number_input("Sample Rows", min_value=2000, max_value=1000000, value=120000, step=10000, key="wf_rows")
        test_size = c2.number_input("Test Size", min_value=0.05, max_value=0.45, value=0.20, step=0.05, format="%.2f", key="wf_test")
        random_state = c3.number_input("Random State", min_value=0, max_value=9999, value=42, step=1, key="wf_seed")
        c4, c5, c6 = st.columns(3)
        max_depth = c4.number_input("Tree Depth", min_value=2, max_value=12, value=4, step=1, key="wf_depth")
        min_leaf = c5.number_input("Min Samples Leaf", min_value=20, max_value=100000, value=200, step=20, key="wf_leaf")
        max_rules = c6.number_input("Max Extracted Rules", min_value=3, max_value=50, value=12, step=1, key="wf_rules")
        leak_cols_text = st.text_area(
            "Leakage columns to exclude",
            value="SK_ID_CURR,TARGET,is_train,future_obs_months,future_dpd_max,FUTURE_DPD30_FLAG,FUTURE_DPD60_FLAG,FUTURE_DEFAULT_90_FLAG,FUTURE_STAGE_MAX,TARGET_TRANSITION_TO_90",
            height=80,
            key="wf_leaks",
        )
        a1, a2, a3 = st.columns(3)
        auto_min_lift = a1.number_input("Auto-register min lift", min_value=0.0, max_value=10.0, value=1.05, step=0.05, key="wf_auto_lift")
        auto_min_n = a2.number_input("Auto-register min rule N", min_value=1, max_value=1000000, value=200, step=50, key="wf_auto_n")
        auto_reject = a3.checkbox("Auto-register require reject-null", value=True, key="wf_auto_reject")

        if st.button("Train Model And Save Extracted Rules", type="primary", key="wf_train_btn"):
            try:
                exclude_cols = [c.strip() for c in leak_cols_text.split(",") if c.strip()]
                payload = {
                    "table_name": table_name,
                    "id_column": "SK_ID_CURR",
                    "target_column": target_col,
                    "positive_value": pos_value,
                    "sample_rows": int(sample_rows),
                    "test_size": float(test_size),
                    "random_state": int(random_state),
                    "max_depth": int(max_depth),
                    "min_samples_leaf": int(min_leaf),
                    "top_k_features": 20,
                    "max_rules": int(max_rules),
                    "exclude_columns": exclude_cols,
                    "auto_register_extracted_rules": True,
                    "auto_register_min_lift": float(auto_min_lift),
                    "auto_register_min_rule_n": int(auto_min_n),
                    "auto_register_require_reject_null": bool(auto_reject),
                }
                data = _post(base, "/ews/rule-model/train", payload, timeout=300)
                st.success(f'Model trained: {data.get("model_id","")}')
                st.session_state["rule_model_id"] = data.get("model_id", "")
                st.metric("Validation AUC", "NA" if data.get("validation_auc") is None else f'{float(data["validation_auc"]):.4f}')
                st.metric("Auto-registered rules", int(data.get("auto_registered_count", 0)))
                if data.get("extracted_rules"):
                    st.dataframe(data["extracted_rules"], use_container_width=True)
                if data.get("auto_register_failures"):
                    st.warning("Some rules could not be auto-registered.")
                    st.dataframe(data["auto_register_failures"], use_container_width=True)
            except Exception as exc:
                st.error(str(exc))


def section_customer_insight(base: str) -> None:
    st.subheader("Customer Insight")
    st.caption("Single-call hybrid insight: SQL + Graph + Rules + SHAP + recommended actions.")

    c1, c2, c3 = st.columns(3)
    customer_id = c1.number_input("Customer ID", min_value=1, value=302629, step=1, key="ci_customer_id")
    table_name = c2.text_input("Rule Table", value="home_credit_future_dpd_labels", key="ci_table")
    id_column = c3.text_input("ID Column", value="SK_ID_CURR", key="ci_id_col")

    c4, c5, c6 = st.columns(3)
    role = c4.selectbox("Role", ["analyst", "senior_analyst", "risk_manager", "auditor"], index=0, key="ci_role")
    policy = c5.selectbox("Policy Profile", ["", "balanced", "fraud_deep_dive", "credit_fast", "semantic_review"], index=0, key="ci_policy")
    use_real = c6.checkbox("Use Real Endpoint", value=True, key="ci_real")
    include_all_rules = st.checkbox("Include non-approved rules", value=False, key="ci_all_rules")

    filters_text = st.text_area("Filters (JSON object)", value="{}", height=70, key="ci_filters")
    query_embedding_text = st.text_area("Query Embedding (JSON array, optional)", value="", height=60, key="ci_emb")

    if st.button("Generate Customer Insight", type="primary", key="ci_run"):
        try:
            payload = {
                "customer_id": int(customer_id),
                "table_name": table_name,
                "id_column": id_column,
                "include_all_rules": bool(include_all_rules),
                "intent": "customer_risk_reasoning",
                "role": role,
                "policy_profile": policy or None,
                "strict_route_policy": True,
                "filters": json.loads(filters_text.strip()) if filters_text.strip() else {},
                "query_embedding": json.loads(query_embedding_text.strip()) if query_embedding_text.strip() else None,
                "include_explanation": True,
            }
            path = "/phase5/customer-insight-real" if use_real else "/phase5/customer-insight-live"
            data = _post(base, path, payload, timeout=180)
            st.success("Customer insight ready.")
            st.metric("Overall Risk", str(data.get("overall_risk_level", "")).upper())
            st.metric("Risk Score", f'{float(data.get("overall_risk_score",0)):.3f}')
            st.write(data.get("summary", ""))

            st.markdown("**Recommended Actions**")
            for action in data.get("recommended_actions", []):
                st.markdown(f"- {action}")

            st.markdown("**Top Model Drivers**")
            model_summary = data.get("model_summary", {})
            if model_summary.get("top_factors"):
                st.dataframe(model_summary["top_factors"], use_container_width=True)

            st.markdown("**Matched Rules**")
            rules_summary = data.get("rules_summary", {})
            st.write(f'Matched: {int(rules_summary.get("matched_count", 0))}')
            if rules_summary.get("matched_rules"):
                st.dataframe(rules_summary["matched_rules"], use_container_width=True)

            st.markdown("**Graph Summary**")
            st.json(data.get("graph_summary", {}))

            with st.expander("Hybrid Detail (Trace/Audit)"):
                st.json(data.get("hybrid_detail", {}))
        except Exception as exc:
            st.error(str(exc))

def main() -> None:
    _apply_branding()
    st.title("Risk Intelligence Studio")
    st.caption("Explore customer risk, test rules, and explain drivers with a cleaner local workflow.")

    base = _api_base()
    ok, detail = _health(base)
    if ok:
        st.sidebar.success("API healthy")
    else:
        st.sidebar.error("API unavailable")
        st.sidebar.code(detail)

    page = st.sidebar.radio(
        "Modules",
        [
            "Overview",
            "Customer Insight",
            "Ask Data",
            "EDA",
            "Signal Builder",
            "Rule Lab",
            "Hypothesis Lab",
            "Explainability",
            "Live Graph",
            "Investigation",
            "Copilot",
        ],
    )

    if page == "Overview":
        section_customer_insight(base)
    elif page == "Customer Insight":
        section_customer_insight(base)
    elif page == "Ask Data":
        section_ask(base)
    elif page == "EDA":
        section_eda(base)
    elif page == "Signal Builder":
        section_signal_model(base)
    elif page == "Rule Lab":
        section_rule_lifecycle(base)
    elif page == "Hypothesis Lab":
        section_hypothesis(base)
    elif page == "Live Graph":
        section_phase3_live(base)
    elif page == "Investigation":
        section_phase4_live(base)
    elif page == "Copilot":
        section_phase5_copilot(base)
    else:
        section_shap(base)

    st.sidebar.markdown("---")
    st.sidebar.write("Conversation ID")
    st.sidebar.code(st.session_state.get("conversation_id", ""))
    st.sidebar.write("Latest SHAP Model ID")
    st.sidebar.code(st.session_state.get("lime_model_id", ""))
    st.sidebar.write("Latest Rule Model ID")
    st.sidebar.code(st.session_state.get("rule_model_id", ""))
    if st.sidebar.button("Clear Local Session State"):
        for k in ["conversation_id", "lime_model_id", "rule_model_id", "drafted_where_clause", "drafted_rule_name"]:
            st.session_state.pop(k, None)
        st.sidebar.success("Cleared")


if __name__ == "__main__":
    main()

