import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

def get_logger(name: str = "credit_risk_platform") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        
        # Stream Handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        # File Handler
        try:
            log_dir = Path(__file__).resolve().parents[2] / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "app.log", 
                maxBytes=5_000_000, 
                backupCount=5, 
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file logger: {e}")
            
    return logger
