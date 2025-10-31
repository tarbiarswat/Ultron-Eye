from pathlib import Path

APP_NAME = "Ultron Eye"
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"

for p in (RAW_DIR, PROC_DIR, CACHE_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

SOURCES = ["Google RSS", "Yahoo News", "Reddit", "Pinterest"]

CLEANING = {
    "strip_urls": True,
    "strip_emojis": True,
    "strip_punct": True,
    "strip_numbers": False,
    "lowercase": True,
    "remove_stopwords": True,
    "lemmatize": True,
}