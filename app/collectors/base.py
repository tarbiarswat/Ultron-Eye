from abc import ABC, abstractmethod
import pandas as pd

SCHEMA_COLS = ["source", "fetched_at", "term", "url", "title", "text", "author", "region", "category"]

class Collector(ABC):
    name: str = "Base"

    @abstractmethod
    def collect(self, term: str, limit: int = 50) -> pd.DataFrame:
        ...

    def _ensure_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in SCHEMA_COLS:
            if c not in df.columns:
                df[c] = None
        return df[SCHEMA_COLS]