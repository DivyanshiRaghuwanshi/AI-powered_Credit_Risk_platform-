import pickle
from pathlib import Path
from src.utils.config import MODELS_DIR

def load_latest_model():
    latest_txt = Path(MODELS_DIR) / "latest_model_id.txt"
    if not latest_txt.exists():
        raise FileNotFoundError("No models trained yet.")
    model_id = latest_txt.read_text(encoding="utf-8").strip()
    
    with open(Path(MODELS_DIR) / f"{model_id}.pkl", "rb") as f:
        return pickle.load(f)

def predict_customer_risk(X_row, model_payload) -> tuple[float, int, str]:
    """Given a processed row, score the model and assign risk band."""
    model = model_payload["model"]
    feature_cols = model_payload["feature_cols"]
    
    # Reorder columns to match feature cols
    X_input = X_row[feature_cols]
    prob = float(model.predict_proba(X_input.values)[0, 1])
    label = int(prob >= 0.50)
    
    if prob >= 0.70:
        band = "High"
    elif prob >= 0.40:
        band = "Medium"
    else:
        band = "Low"
        
    return prob, label, band
