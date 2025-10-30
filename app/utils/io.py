from datetime import datetime
from pathlib import Path
import pandas as pd
import hashlib

def ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def write_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]