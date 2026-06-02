import pandas as pd
from sqlalchemy import create_engine, text
from src.utils.config import get_sqlalchemy_url

def execute_query(sql_query: str, limit: int = 200) -> pd.DataFrame:
    engine = create_engine(get_sqlalchemy_url())
    lowered = sql_query.strip().lower()
    
    # Basic Safety Check
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
        
    blocked = ["drop", "delete", "insert", "update", "truncate", "alter"]
    if any(b in lowered for b in blocked):
        raise ValueError("Unsafe SQL statement blocked.")
        
    safe_sql = f"SELECT * FROM ({sql_query.rstrip(';')}) q LIMIT {limit}"
    with engine.connect() as conn:
        return pd.read_sql(text(safe_sql), conn)
