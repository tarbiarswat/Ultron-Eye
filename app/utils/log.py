import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEF_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

def _resolve_log_dir():
    """
    Try to import LOG_DIR from app.config. If that would cause a circular import
    (e.g., during Streamlit module init), fall back to ../logs relative to repo root.
    """
    try:
        # Lazy import avoids circular imports at module-import time
        from app.config import LOG_DIR  # type: ignore
        return LOG_DIR
    except Exception:
        # Fallback: derive logs/ two levels up from this file
        return Path(__file__).resolve().parents[2] / "logs"

def get_logger(name: str = "ultron_eye"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(_DEF_FMT)

    log_dir = _resolve_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(log_dir / "ultron_eye.log", maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(_DEF_FMT)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
