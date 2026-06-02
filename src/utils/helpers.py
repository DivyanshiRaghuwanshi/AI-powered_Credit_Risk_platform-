import re
import time
import math
from pathlib import Path

def normalize_table_name(file_name: str) -> str:
    base = Path(file_name).stem.lower()
    return re.sub(r"[^a-z0-9_]+", "_", base)

def elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 2)

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def p_value_from_z(z: float, alternative: str = "greater") -> float:
    alt = alternative.strip().lower()
    if alt not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be one of: greater, less, two-sided")
    if alt == "greater":
        return max(0.0, min(1.0, 1.0 - norm_cdf(z)))
    if alt == "less":
        return max(0.0, min(1.0, norm_cdf(z)))
    return max(0.0, min(1.0, 2.0 * (1.0 - norm_cdf(abs(z)))))
