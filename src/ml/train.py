import pickle
import uuid
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from src.utils.config import MODELS_DIR, get_sqlalchemy_url
from src.data.preprocessor import clean_and_impute
import shap
from sqlalchemy import create_engine, text

def train_shap_model(table_name: str, id_column: str, target_column: str, positive_value: str = "1", sample_rows: int = 100000):
    print(f"Training SHAP model on {table_name}...")
    engine = create_engine(get_sqlalchemy_url())
    with engine.connect() as conn:
        df = pd.read_sql(text(f'SELECT * FROM "{table_name}" LIMIT {sample_rows}'), conn)
        
    if df.empty:
        raise ValueError("Empty table.")
        
    work = df[df[target_column].notna()].copy()
    y = (work[target_column].astype(str) == str(positive_value)).astype(int)
    
    X, medians = clean_and_impute(work, id_column, target_column)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(rf)
    
    model_id = f"shap_{table_name}_{uuid.uuid4().hex[:10]}"
    payload = {
        "model_id": model_id,
        "table_name": table_name,
        "id_column": id_column,
        "target_column": target_column,
        "feature_cols": X_train.columns.tolist(),
        "medians": medians.to_dict(),
        "model": rf,
        "explainer": explainer,
        "train_rows": len(X),
        "positive_rate": float(y.mean())
    }
    
    model_path = Path(MODELS_DIR) / f"{model_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
        
    # Write latest model txt
    (Path(MODELS_DIR) / "latest_model_id.txt").write_text(model_id, encoding="utf-8")
    print(f"Saved SHAP model to {model_path}")
    return model_id

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="master_ews_fibo")
    parser.add_argument("--id", default="SK_ID_CURR")
    parser.add_argument("--target", default="EWS_LABEL")
    args = parser.parse_args()
    
    try:
        train_shap_model(args.table, args.id, args.target)
    except Exception as e:
        print(f"Failed to train model: {e}")
