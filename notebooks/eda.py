"""Quick EDA for `home_credit_future_dpd_labels.csv`.

Usage: from repo root run:

    python -m scripts.run_eda_home_credit

Outputs:
- uploads the CSV to Postgres table `home_credit_future_dpd_labels` (if DB available)
- writes summary CSV/JSON under `data/processed/`
- saves plots and a short markdown summary under `docs/results/`

This script is intentionally lightweight so you can run it before submission.
"""
from __future__ import annotations

import os
from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"
DOCS_RES = ROOT / "docs" / "results"
DOCS_RES.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)


def get_csv_path() -> Path:
    # default path
    candidate = DATA_PROC / "home_credit_future_dpd_labels.csv"
    if candidate.exists():
        return candidate
    # fall back to project root raw
    alt = ROOT / "data" / "processed" / "home_credit_future_dpd_labels.csv"
    return alt


def connect_db():
    load_dotenv(ROOT / ".env")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "")
    if not pwd:
        print("POSTGRES_PASSWORD not set in .env — skipping DB upload.")
        return None
    url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    try:
        engine = create_engine(url)
        # quick test
        conn = engine.connect()
        conn.close()
        return engine
    except Exception as exc:
        print("Failed to connect to Postgres:", exc)
        return None


def run_eda(csv_path: Path) -> dict:
    print("Reading:", csv_path)
    df = pd.read_csv(csv_path)
    out = {}
    out["rows"] = int(df.shape[0])
    out["cols"] = int(df.shape[1])
    out["columns"] = list(df.columns)

    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    out["missing_pct"] = missing.to_dict()

    dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
    out["dtypes"] = dtypes

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    out["numeric_count"] = len(numeric_cols)
    out["categorical_count"] = len(cat_cols)

    if numeric_cols:
        stats = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T
        stats_file = DATA_PROC / "home_credit_numeric_stats.csv"
        stats.to_csv(stats_file)
        out["numeric_stats_csv"] = str(stats_file.relative_to(ROOT))

    # detect likely target column
    target_candidates = ["EWS_LABEL", "TARGET", "DPD_LABEL", "future_dpd", "label"]
    target_col = None
    for t in target_candidates:
        if t in df.columns:
            target_col = t
            break
    out["target_col"] = target_col
    if target_col is not None:
        vc = df[target_col].value_counts(dropna=False)
        out["target_counts"] = vc.to_dict()
        # plot target
        plt.figure(figsize=(6,4))
        sns.barplot(x=vc.index.astype(str), y=vc.values)
        plt.title(f"Distribution of {target_col}")
        plt.ylabel("count")
        plt.xticks(rotation=45)
        tgt_plot = DOCS_RES / "target_distribution.png"
        plt.tight_layout()
        plt.savefig(tgt_plot)
        plt.close()
        out["target_plot"] = str(tgt_plot.relative_to(ROOT))

    # missingness plot
    top_missing = missing.head(40)
    plt.figure(figsize=(8, max(4, 0.2 * len(top_missing))))
    sns.barplot(x=top_missing.values, y=top_missing.index)
    plt.xlabel("% missing")
    plt.title("Top missingness")
    miss_plot = DOCS_RES / "missingness_top.png"
    plt.tight_layout()
    plt.savefig(miss_plot)
    plt.close()
    out["missing_plot"] = str(miss_plot.relative_to(ROOT))

    # correlations (if numeric)
    if numeric_cols and target_col and target_col in numeric_cols:
        corr = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        out["top_correlated_with_target"] = corr.head(15).to_dict()
        # heatmap of top correlated
        top_feats = corr.head(12).index.tolist()
        plt.figure(figsize=(8,6))
        sns.heatmap(df[top_feats].corr(), annot=True, fmt=".2f", cmap="vlag")
        corr_plot = DOCS_RES / "corr_heatmap.png"
        plt.tight_layout()
        plt.savefig(corr_plot)
        plt.close()
        out["corr_plot"] = str(corr_plot.relative_to(ROOT)).replace("\\", "/")

    # New EDA 1: Credit amount distribution KDE
    if "AMT_CREDIT" in df.columns and target_col:
        plt.figure(figsize=(7,4))
        sns.kdeplot(data=df, x="AMT_CREDIT", hue=target_col, common_norm=False, fill=True, palette="Set1")
        plt.title("Credit Amount Distribution by Target")
        plt.xlabel("Amount of Credit")
        plt.ylabel("Density")
        cred_plot = DOCS_RES / "credit_distribution.png"
        plt.tight_layout()
        plt.savefig(cred_plot)
        plt.close()
        out["credit_plot"] = str(cred_plot.relative_to(ROOT)).replace("\\", "/")

    # New EDA 2: EXT_SOURCE_2 distribution boxplot
    if "EXT_SOURCE_2" in df.columns and target_col:
        plt.figure(figsize=(7,4))
        sns.boxplot(data=df, x=target_col, y="EXT_SOURCE_2", palette="Set2")
        plt.title("EXT_SOURCE_2 Distribution by Target")
        plt.xlabel("Target (0 = Repaid, 1 = Default)")
        plt.ylabel("EXT_SOURCE_2 Score")
        ext_plot = DOCS_RES / "ext_source_2_distribution.png"
        plt.tight_layout()
        plt.savefig(ext_plot)
        plt.close()
        out["ext_plot"] = str(ext_plot.relative_to(ROOT)).replace("\\", "/")

    # save summary JSON
    summary_file = DATA_PROC / "home_credit_eda_summary.json"
    with open(summary_file, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)

    # write a small markdown summary
    md = DOCS_RES / "home_credit_eda.md"
    with open(md, "w", encoding="utf8") as f:
        f.write("# EDA: home_credit_future_dpd_labels.csv\n\n")
        f.write(f"Rows: {out['rows']}  \\n+Columns: {out['cols']}  \n\n")
        f.write("## Target\n\n")
        if target_col:
            f.write(f"Detected target column: **{target_col}**\n\n")
            f.write(f"![target distribution]({out['target_plot']})\n\n")
        else:
            f.write("No canonical target column detected.\n\n")
        f.write("## Missingness\n\n")
        f.write(f"![missingness]({out['missing_plot']})\n\n")
        if out.get("corr_plot"):
            f.write("## Correlations\n\n")
            f.write(f"![correlations]({out['corr_plot']})\n\n")
        if out.get("credit_plot"):
            f.write("## Credit Amount Distribution\n\n")
            f.write(f"![credit distribution]({out['credit_plot']})\n\n")
        if out.get("ext_plot"):
            f.write("## EXT_SOURCE_2 Distribution\n\n")
            f.write(f"![ext_source_2 distribution]({out['ext_plot']})\n\n")
        f.write("## Notes\n\n- Summary CSV and numeric stats saved under `data/processed/`.\n")

    print("Wrote summary to:", md)
    return out


def main():
    csv_path = get_csv_path()
    if not csv_path.exists():
        print("CSV not found at expected path:", csv_path)
        sys.exit(1)
    engine = connect_db()
    df = pd.read_csv(csv_path)
    if engine is not None:
        try:
            print("Uploading table to Postgres: home_credit_future_dpd_labels")
            df.to_sql("home_credit_future_dpd_labels", engine, if_exists="replace", index=False)
            print("Upload complete")
        except Exception as exc:
            print("Failed to upload to DB:", exc)
    out = run_eda(csv_path)
    print("EDA complete. Summary keys:", list(out.keys()))


if __name__ == "__main__":
    main()
