import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEF_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

def _resolve_log_dir() -> Path:
    try:
        from app.config import LOG_DIR  # lazy import to avoid circulars
        return LOG_DIR
    except Exception:
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