import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from src.utils.config import get_sqlalchemy_url
from src.utils.helpers import normalize_table_name

def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(path, **kwargs)

def load_csv_to_postgres(csv_path: Path, sample_rows: int = None):
    engine = create_engine(get_sqlalchemy_url())
    table_name = normalize_table_name(csv_path.name)
    print(f"Loading {csv_path.name} -> {table_name}...")
    
    if sample_rows:
        df = read_csv_with_fallback(csv_path, nrows=sample_rows)
        df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=10000)
    else:
        for i, chunk in enumerate(read_csv_with_fallback(csv_path, chunksize=50000, low_memory=False)):
            chunk.to_sql(
                table_name,
                engine,
                if_exists="replace" if i == 0 else "append",
                index=False,
                chunksize=10000
            )
    print(f"Loaded {table_name}")
