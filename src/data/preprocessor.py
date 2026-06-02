import pandas as pd
from typing import Tuple

def clean_and_impute(df: pd.DataFrame, id_col: str, target_col: str, exclude_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Cleans data, drops identifiers, and performs median imputation for numeric columns."""
    work = df.copy()
    
    # Standard columns to ignore during training
    drop_cols = {
        id_col, target_col, "TARGET", "is_train", "EWS_SCORE",
        "ews_bureau_overdue", "ews_bureau_balance_severe",
        "ews_pos_dpd", "ews_installment_stress", "ews_cc_stress"
    }
    if exclude_cols:
        drop_cols.update(exclude_cols)
        
    candidate_cols = [c for c in work.columns if c not in drop_cols]
    numeric_cols = {}
    for c in candidate_cols:
        num = pd.to_numeric(work[c], errors="coerce")
        # Keep features with at least 70% non-null values
        if num.notna().mean() >= 0.70:
            numeric_cols[c] = num
            
    X = pd.DataFrame(numeric_cols, index=work.index)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    
    return X, medians
