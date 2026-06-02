import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Database Config
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")

def get_pg_url() -> str:
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

def get_sqlalchemy_url() -> str:
    return f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# LLM Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Path Settings
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "").strip() or str(PROJECT_ROOT / "data" / "raw")
PROCESSED_DATA_DIR = str(PROJECT_ROOT / "data" / "processed")
MODELS_DIR = os.getenv("LIME_MODEL_DIR", "").strip() or str(PROJECT_ROOT / "models")
RULE_REGISTRY_PATH = os.getenv("RULE_REGISTRY_PATH", "").strip() or str(PROJECT_ROOT / "models" / "approved_rules.json")

Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
