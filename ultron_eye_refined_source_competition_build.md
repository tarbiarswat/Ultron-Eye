# Ultron Eye — Refined Source (Competition Build)

> Production‑ready, streamlined version matching your current functionality: two modes (Decision Helper / Go Broad), collectors for RSS/Yahoo/Reddit/**Pinterest (Playwright→Selenium fallback)**, robust cleaning (HTML strip + entity decode + noise blacklist), **progress bar**, **animated word cloud** fix, Windows asyncio policy for Playwright, absolute imports. Drop‑in replacement.

---

## 📁 Project Layout
```
ultron-eye/
  requirements.txt
  run_console.py
  app/
    __init__.py
    ui.py
    config.py
    modes.py
    utils/
      __init__.py
      log.py
      io.py
      text.py
    pipelines/
      __init__.py
      clean.py
      nlp.py
      vis.py
    collectors/
      __init__.py
      base.py
      common.py
      rss.py
      yahoo_news.py
      reddit.py
      pinterest.py
  data/
    raw/
    processed/
    cache/
  logs/
```

---

## 🔧 requirements.txt
```txt
streamlit
pandas
numpy
regex
unidecode
nltk
wordcloud
matplotlib
imageio
feedparser
beautifulsoup4
requests
selenium
webdriver-manager
playwright
```

> After install, run: `python -m playwright install`

---

## ▶️ run_console.py
```python
import pandas as pd
from app.pipelines.clean import clean_frame, token_counts
from app.utils.io import write_df, ts
from app.config import RAW_DIR, PROC_DIR

if __name__ == "__main__":
    demo = pd.DataFrame({"text": [
        "AI camera upgrades dominate the iphone 17 pro chatter!",
        "iphone 17 air praised for portability and battery life.",
        "Users compare iphone 17 pro vs iphone 17 air.",
    ]})
    write_df(demo, RAW_DIR / f"raw_console_{ts()}.csv")
    cleaned = clean_frame(demo, text_col="text")
    write_df(cleaned, PROC_DIR / f"processed_console_{ts()}.csv")
    print(token_counts(cleaned).head(20))
```

---

## 🧩 app/__init__.py
```python
# Makes `app` a package
```

## 🧩 app/utils/__init__.py
```python
# Utils package marker
```

## 🧩 app/pipelines/__init__.py
```python
# Pipelines package marker
```

## 🧩 app/collectors/__init__.py
```python
# Keep passive to avoid premature imports
__all__ = []
```

---

## ⚙️ app/config.py
```python
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
```

---

## 🧭 app/modes.py
```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Mode1Inputs:
    left_term: str
    right_term: str
    sources: List[str]

@dataclass
class Mode2Inputs:
    terms: List[str]
    sources: List[str]

def validate_mode1(left: str, right: str) -> Tuple[bool, str]:
    if not left or not right:
        return False, "Please enter both comparison terms."
    if left.strip().lower() == right.strip().lower():
        return False, "Terms must be different."
    return True, ""

def validate_mode2(terms: List[str]) -> Tuple[bool, str]:
    terms = [t for t in (terms or []) if t.strip()]
    if len(terms) < 1:
        return False, "Enter at least one term."
    return True, ""
```

---

## 🖥️ app/ui.py
```python
# --- Windows asyncio policy fix for Playwright subprocesses ---
import sys
if sys.platform.startswith("win"):
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass
# ----------------------------------------------------------------

# --- bootstrap sys.path so absolute imports like "from app.utils..." work ---
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../ultron-eye
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from pathlib import Path

from app.utils.log import get_logger
from app.utils.io import ts, write_df
from app.pipelines.clean import clean_frame, token_counts
from app.pipelines.vis import wordcloud_from_tokens, animated_wordcloud
from app.modes import validate_mode1, validate_mode2

from app.collectors.rss import RSSCollector
from app.collectors.yahoo_news import YahooNewsCollector
from app.collectors.reddit import RedditCollector
from app.collectors.pinterest import PinterestCollector

from app.config import APP_NAME, SOURCES, RAW_DIR, PROC_DIR

logger = get_logger()

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Real-time data intelligence — Mode 1: Decision Helper | Mode 2: Go Broad")

with st.sidebar:
    mode = st.radio("Choose Mode", ["Decision Helper (compare 2 terms)", "Go Broad (multi-term)"])
    sources = st.multiselect("Select Data Sources", SOURCES, default=["Google RSS", "Yahoo News"])
    limit = st.slider("Items per source per term", 10, 100, 40, 10)
    make_animated = st.checkbox("Generate animated word cloud (GIF)", value=False)
    st.divider()
    st.markdown("**Optional local data (MVP)**")
    raw_text = st.text_area("Paste text (optional)", height=120)
    uploaded = st.file_uploader("Or upload CSV/TXT (optional)", type=["csv", "txt"])
    st.caption("CSV needs a 'text' column; TXT uses lines as records.")
    run = st.button("Run")

if mode.startswith("Decision"):
    c1, c2 = st.columns(2)
    with c1:
        left = st.text_input("Term A", placeholder="iphone 17 pro")
    with c2:
        right = st.text_input("Term B", placeholder="iphone 17 air")
else:
    terms_input = st.text_input("Enter terms (comma-separated)", placeholder="iphone 17, foldable phone, ai camera")

# ---- helpers ----

def advance(progress, step, total, label=""):
    step += 1
    pct = min(int(step / total * 100), 100)
    progress.progress(pct, text=f"{pct}% {label}")
    return step

def build_input_df() -> pd.DataFrame:
    rows = []
    if uploaded is not None:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            if "text" in df.columns:
                rows.extend({"text": t} for t in df["text"].dropna().astype(str).tolist())
            else:
                st.error("CSV must include a 'text' column.")
        else:
            content = uploaded.read().decode("utf-8", errors="ignore")
            for line in content.splitlines():
                if line.strip():
                    rows.append({"text": line.strip()})
    if raw_text.strip():
        for line in raw_text.splitlines():
            if line.strip():
                rows.append({"text": line.strip()})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["text"])

# Collector registry

def make_collectors(selected: list[str]):
    mapping = {
        "Google RSS": RSSCollector(),
        "Yahoo News": YahooNewsCollector(use_selenium=False),
        "Reddit": RedditCollector(),
        "Pinterest": PinterestCollector(scroll_batches=6, headless=True),
    }
    return [mapping[s] for s in selected if s in mapping]

# ---- main run ----
if run:
    if mode.startswith("Decision"):
        ok, msg = validate_mode1(left, right)
        if not ok:
            st.error(msg); st.stop()
        terms = [left, right]
    else:
        terms = [t.strip() for t in (terms_input or "").split(",") if t.strip()]
        ok, msg = validate_mode2(terms)
        if not ok:
            st.error(msg); st.stop()

    # Progress setup
    est_collect = max(1, len(terms) * max(1, len(sources)))
    est_clean   = 1
    est_vis     = 1 + (1 if make_animated else 0)
    TOTAL_STEPS = est_collect + est_clean + est_vis

    progress = st.progress(0, text="Starting…")
    step = 0

    # 1) Collect
    all_rows = []
    collectors = make_collectors(sources)
    for term in terms:
        for col in collectors:
            try:
                df_c = col.collect(term, limit=limit)
                all_rows.append(df_c)
                logger.info(f"Collected {len(df_c)} from {col.name} for '{term}'")
            except Exception as e:
                logger.exception(f"Collector error [{col.name}] term='{term}': {e}")
            step = advance(progress, step, TOTAL_STEPS, label=f"Collected from {col.name} ({term})")

    collected_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["text"])

    # + Optional local data
    local_df = build_input_df()
    if not local_df.empty:
        local_df = local_df.assign(source="Local", fetched_at=pd.Timestamp.utcnow().isoformat(), term=terms[0])
        collected_df = pd.concat([collected_df, local_df], ignore_index=True)

    if collected_df.empty:
        st.warning("No data collected. Try different sources/terms or paste/upload text.")
        st.stop()

    raw_path = RAW_DIR / f"raw_{ts()}.csv"
    write_df(collected_df, raw_path)
    logger.info(f"Saved raw: {raw_path.name} | rows={len(collected_df)}")

    # Ensure term exists for all rows
    if "term" not in collected_df.columns or collected_df["term"].isna().all():
        cyc, k = [], 0
        for _ in range(len(collected_df)):
            cyc.append(terms[k])
            k = (k + 1) % len(terms)
        collected_df["term"] = cyc

    # 2) Clean
    base_text = collected_df.get("text")
    title_fallback = collected_df.get("title")
    cleaned = clean_frame(collected_df.assign(text=base_text.fillna(title_fallback)), text_col="text")
    proc_path = PROC_DIR / f"processed_{ts()}.csv"
    write_df(cleaned, proc_path)
    logger.info(f"Cleaned & saved: {proc_path.name} | rows={len(cleaned)}")
    step = advance(progress, step, TOTAL_STEPS, label="Cleaned data")

    st.success("Processing complete.")

    # ---- UI previews ----
    st.subheader("Raw (sample)")
    st.dataframe(collected_df.head(20), use_container_width=True)

    st.subheader("Cleaned (sample)")
    st.dataframe(cleaned.head(20), use_container_width=True)

    st.subheader("Top Tokens (Overall)")
    overall = token_counts(cleaned)
    st.dataframe(overall.head(30), use_container_width=True)

    st.subheader("Top Tokens by Term")
    by_term = token_counts(cleaned, groupby="term")
    st.dataframe(by_term.groupby("term").head(15), use_container_width=True)

    # Static word cloud
    st.subheader("Word Cloud (Overall)")
    wc_img = wordcloud_from_tokens(cleaned["tokens"].tolist())
    if wc_img is not None:
        st.image(wc_img, use_container_width=True)
        step = advance(progress, step, TOTAL_STEPS, label="Word cloud")
    else:
        st.info("Not enough tokens to create a word cloud.")

    # Animated word cloud (GIF)
    if make_animated:
        st.subheader("Animated Word Cloud (GIF)")
        gif_path = PROC_DIR / f"wordcloud_{ts()}.gif"
        out = animated_wordcloud(cleaned["tokens"].tolist(), gif_path, frames=12, fps=3)
        if out:
            st.image(str(out))
        else:
            st.info("Could not generate animated word cloud.")
        step = advance(progress, step, TOTAL_STEPS, label="Animated word cloud")

    # Console echo
    print(f"[UltronEye] rows={len(cleaned)} | mode={'M1' if mode.startswith('Decision') else 'M2'} | sources={sources}")

    progress.progress(100, text="Done")
```

---

## 🧰 app/utils/log.py
```python
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
```

## 🧰 app/utils/io.py
```python
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
```

## 🧰 app/utils/text.py
```python
import re
import regex as rxx
from unidecode import unidecode

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = rxx.compile(r"\p{Emoji}", flags=rxx.VERSION1)

def normalize(text: str) -> str:
    return unidecode(text or "")

def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)

def strip_emojis(text: str) -> str:
    return EMOJI_RE.sub(" ", text)

def strip_nonletters(text: str, keep_spaces=True) -> str:
    return re.sub(r"[^A-Za-z\s]" if keep_spaces else r"[^A-Za-z]", " ", text)
```

---

## 🧼 app/pipelines/clean.py
```python
from typing import List
import pandas as pd
import re
import nltk

# NLTK setup
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from app.config import CLEANING
from app.utils import text as TX

STOP = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()

def _basic_clean(s: str, cfg=CLEANING) -> str:
    import html
    from bs4 import BeautifulSoup

    # Decode entities and strip HTML
    s = html.unescape(s or "")
    s = BeautifulSoup(s, "html.parser").get_text(" ")

    # Normalize & rules
    s = TX.normalize(s)
    if cfg["strip_urls"]: s = TX.strip_urls(s)
    if cfg["strip_emojis"]: s = TX.strip_emojis(s)
    if cfg["lowercase"]: s = s.lower()
    if cfg["strip_punct"]: s = TX.strip_nonletters(s, keep_spaces=True)

    # Remove residual HTML/formatting noise
    s = re.sub(r"\b(href|target|font|color|nbsp|http|https|style|align|class|id|div|span|border|background)\b", " ", s, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", s).strip()


def _tokenize(s: str) -> List[str]:
    return [t for t in s.split() if len(t) > 2 and t.isalpha()]


def _post_tokens(tokens: List[str], cfg=CLEANING) -> List[str]:
    if cfg["remove_stopwords"]:
        tokens = [t for t in tokens if t not in STOP]
    if cfg["lemmatize"]:
        tokens = [LEMM.lemmatize(t) for t in tokens]
    return tokens


def clean_frame(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    assert text_col in df.columns, f"missing column: {text_col}"
    out = df.copy()
    out["text_clean"] = out[text_col].astype(str).map(_basic_clean)
    out["tokens"] = out["text_clean"].map(_tokenize).map(_post_tokens)
    out = out[out["tokens"].map(len) > 0].reset_index(drop=True)
    return out


def token_counts(df: pd.DataFrame, groupby: str | None = None) -> pd.DataFrame:
    rows = []
    if groupby and groupby in df.columns:
        for key, part in df.groupby(groupby):
            freq = {}
            for toks in part["tokens"]:
                for t in toks:
                    freq[t] = freq.get(t, 0) + 1
            for t, c in sorted(freq.items(), key=lambda x: x[1], reverse=True):
                rows.append({groupby: key, "token": t, "count": c})
    else:
        freq = {}
        for toks in df["tokens"]:
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
        for t, c in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            rows.append({"token": t, "count": c})
    return pd.DataFrame(rows)
```

---

## 🧪 app/pipelines/nlp.py
```python
import pandas as pd
try:
    import nltk
    nltk.data.find('sentiment/vader_lexicon.zip')
except Exception:
    import nltk
    nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

def sentiment_scores(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    out = df.copy()
    out["sent_neg"] = out[text_col].map(lambda s: _sia.polarity_scores(s)["neg"])
    out["sent_neu"] = out[text_col].map(lambda s: _sia.polarity_scores(s)["neu"])
    out["sent_pos"] = out[text_col].map(lambda s: _sia.polarity_scores(s)["pos"])
    out["sent_compound"] = out[text_col].map(lambda s: _sia.polarity_scores(s)["compound"])
    return out
```

---

## 🖼️ app/pipelines/vis.py
```python
from pathlib import Path
from typing import Iterable, List
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def wordcloud_from_tokens(tokens_iter: Iterable[List[str]], width=1000, height=500, bg="white"):
    text_blob = " ".join([" ".join(toks) for toks in tokens_iter])
    if not text_blob.strip():
        return None
    wc = WordCloud(width=width, height=height, background_color=bg).generate(text_blob)
    return wc.to_array()


def animated_wordcloud(tokens_iter: Iterable[List[str]], out_path: Path, frames: int = 12,
                       step: int = 1, width: int = 900, height: int = 400, bg: str = "white", fps: int = 3):
    tokens = list(tokens_iter)
    if not tokens:
        return None

    images = []
    n = len(tokens)

    for i in range(1, frames + 1):
        end = max(1, int(i * n / frames))
        blob = " ".join([" ".join(toks) for toks in tokens[:end:step]])
        if not blob.strip():
            continue

        wc = WordCloud(width=width, height=height, background_color=bg).generate(blob)
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = imageio.imread(buf)
        images.append(img)
        plt.close(fig)

    if not images:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, images, duration=1.0 / fps)
    return out_path
```

---

## 🧱 app/collectors/base.py
```python
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
```

## 🧱 app/collectors/common.py
```python
import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

def now_iso():
    return datetime.utcnow().isoformat()

def soup_from_url(url: str, sleep=(0.5, 1.2)) -> BeautifulSoup:
    time.sleep(random.uniform(*sleep))
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")
```

## 🌐 app/collectors/rss.py
```python
import pandas as pd
import feedparser
from app.collectors.base import Collector
from app.collectors.common import now_iso

class RSSCollector(Collector):
    name = "Google RSS"

    def __init__(self, feed_template: str = "https://news.google.com/rss/search?q={query}"):
        self.feed_template = feed_template

    def collect(self, term: str, limit: int = 50) -> pd.DataFrame:
        url = self.feed_template.format(query=term.replace(" ", "+"))
        feed = feedparser.parse(url)
        rows = []
        for entry in feed.entries[:limit]:
            rows.append({
                "source": self.name,
                "fetched_at": now_iso(),
                "term": term,
                "url": entry.get("link"),
                "title": entry.get("title"),
                "text": entry.get("summary", ""),
                "author": entry.get("author"),
                "region": None,
                "category": None,
            })
        return pd.DataFrame(rows)
```

## 📰 app/collectors/yahoo_news.py
```python
import pandas as pd
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from app.collectors.base import Collector
from app.collectors.common import now_iso, HEADERS

class YahooNewsCollector(Collector):
    name = "Yahoo News"

    def __init__(self, use_selenium: bool = False):
        self.use_selenium = use_selenium

    def _requests_collect(self, term: str, limit: int = 30) -> pd.DataFrame:
        q = term.replace(" ", "+")
        url = f"https://news.search.yahoo.com/search?p={q}"
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        cards = soup.select("div.NewsArticle")
        rows = []
        for c in cards[:limit]:
            a = c.select_one("h4 a")
            title = a.get_text(strip=True) if a else None
            link = a.get("href") if a else None
            snippet = c.select_one("p.s-textLine2").get_text(strip=True) if c.select_one("p.s-textLine2") else None
            rows.append({
                "source": self.name,
                "fetched_at": now_iso(),
                "term": term,
                "url": link,
                "title": title,
                "text": snippet,
                "author": None,
                "region": None,
                "category": None,
            })
        return pd.DataFrame(rows)

    def _selenium_collect(self, term: str, limit: int = 30) -> pd.DataFrame:
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        try:
            q = term.replace(" ", "+")
            driver.get(f"https://news.search.yahoo.com/search?p={q}")
            soup = BeautifulSoup(driver.page_source, "html.parser")
            cards = soup.select("div.NewsArticle")
            rows = []
            for c in cards[:limit]:
                a = c.select_one("h4 a")
                title = a.get_text(strip=True) if a else None
                link = a.get("href") if a else None
                snippet = c.select_one("p.s-textLine2").get_text(strip=True) if c.select_one("p.s-textLine2") else None
                rows.append({
                    "source": self.name,
                    "fetched_at": now_iso(),
                    "term": term,
                    "url": link,
                    "title": title,
                    "text": snippet,
                    "author": None,
                    "region": None,
                    "category": None,
                })
            return pd.DataFrame(rows)
        finally:
            driver.quit()

    def collect(self, term: str, limit: int = 30) -> pd.DataFrame:
        if self.use_selenium:
            return self._selenium_collect(term, limit)
        return self._requests_collect(term, limit)
```

## 🧵 app/collectors/reddit.py
```python
import pandas as pd
import requests
from app.collectors.base import Collector
from app.collectors.common import now_iso, HEADERS

class RedditCollector(Collector):
    name = "Reddit"

    def __init__(self, subreddit: str | None = None):
        self.subreddit = subreddit

    def collect(self, term: str, limit: int = 50) -> pd.DataFrame:
        if self.subreddit:
            url = f"https://www.reddit.com/r/{self.subreddit}/search.json?q={term}&restrict_sr=1&sort=new"
        else:
            url = f"https://www.reddit.com/search.json?q={term}&sort=new"
        r = requests.get(url, headers={**HEADERS, "User-Agent": "Mozilla/5.0 RedditCollector"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        rows = []
        for child in data.get("data", {}).get("children", [])[:limit]:
            d = child.get("data", {})
            rows.append({
                "source": self.name,
                "fetched_at": now_iso(),
                "term": term,
                "url": f"https://www.reddit.com{d.get('permalink')}",
                "title": d.get("title"),
                "text": d.get("selftext") or d.get("title"),
                "author": d.get("author"),
                "region": None,
                "category": d.get("subreddit"),
            })
        return pd.DataFrame(rows)
```

## 📌 app/collectors/pinterest.py
```python
import sys
if sys.platform.startswith("win"):
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from typing import List, Dict
import pandas as pd
from app.collectors.base import Collector
from app.collectors.common import now_iso

# ---------- Playwright path ----------
def _p_collect(term: str, limit: int, headless: bool, scroll_batches: int, locale: str, user_agent: str) -> List[Dict]:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

    PIN_URL = "https://www.pinterest.com/search/pins/?q={query}&rs=typed"

    def _dismiss_overlays(page):
        for selector in [
            "button:has-text('Accept')",
            "button:has-text('Accept all')",
            "button[aria-label='Close']",
            "[data-test-id='fullpage-modal-close']",
            "button[title='Close']",
        ]:
            try:
                page.locator(selector).first.click(timeout=800)
                page.wait_for_timeout(200)
            except Exception:
                pass

    def _extract_from_dom(page, limit: int) -> List[Dict]:
        rows: List[Dict] = []
        pins = page.locator("a[href^='/pin/']")
        seen = set()
        n = pins.count()
        for i in range(min(n, max(limit * 2, 60))):
            try:
                el = pins.nth(i)
                href = el.get_attribute("href") or ""
                if not href or href in seen:
                    continue
                seen.add(href)
                alt = el.get_attribute("aria-label") or (el.inner_text(timeout=800) or "")
                full_url = f"https://www.pinterest.com{href}" if href.startswith("/") else href
                rows.append({
                    "source": "Pinterest",
                    "fetched_at": now_iso(),
                    "term": None,
                    "url": full_url,
                    "title": (alt or "").strip()[:200] or None,
                    "text": (alt or "").strip(),
                    "author": None,
                    "region": None,
                    "category": None,
                })
                if len(rows) >= limit:
                    break
            except Exception:
                continue
        return rows

    rows: List[Dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=user_agent,
            locale=locale,
            viewport={"width": 1366, "height": 900},
            extra_http_headers={"Accept-Language": locale},
        )
        page = context.new_page()
        page.set_default_timeout(30000)
        q = term.replace(" ", "%20")

        try:
            page.goto(PIN_URL.format(query=q), wait_until="networkidle")
        except PWTimeoutError:
            page.goto(PIN_URL.format(query=q))

        _dismiss_overlays(page)
        try:
            page.wait_for_selector("a[href^='/pin/']", timeout=8000)
        except PWTimeoutError:
            pass

        for _ in range(scroll_batches):
            page.mouse.wheel(0, 5000)
            page.wait_for_timeout(1400)

        rows.extend(_extract_from_dom(page, limit))

        context.close()
        browser.close()

    uniq, seen = [], set()
    for r in rows:
        u = r.get("url")
        if u and u not in seen:
            seen.add(u)
            uniq.append(r)
    return uniq[:limit]

# ---------- Selenium fallback ----------
def _s_collect(term: str, limit: int, headless: bool) -> List[Dict]:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup
    import time

    PIN_URL = "https://www.pinterest.com/search/pins/?q={query}&rs=typed"

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1366,900")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    try:
        q = term.replace(" ", "%20")
        driver.get(PIN_URL.format(query=q))
        time.sleep(2.0)

        for css in ["button[aria-label='Close']"]:
            try:
                driver.find_element("css selector", css).click()
                time.sleep(0.2)
            except Exception:
                pass

        for _ in range(5):
            driver.execute_script("window.scrollBy(0, 5000);")
            time.sleep(1.4)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        rows: List[Dict] = []
        seen = set()
        for a in soup.select("a[href^='/pin/']"):
            href = a.get("href")
            if not href or href in seen:
                continue
            seen.add(href)
            full_url = f"https://www.pinterest.com{href}" if href.startswith("/") else href
            alt = a.get("aria-label") or a.get_text(strip=True) or ""
            rows.append({
                "source": "Pinterest",
                "fetched_at": now_iso(),
                "term": None,
                "url": full_url,
                "title": alt[:200] or None,
                "text": alt,
                "author": None,
                "region": None,
                "category": None,
            })
            if len(rows) >= limit:
                break
        return rows
    finally:
        driver.quit()

class PinterestCollector(Collector):
    name = "Pinterest"

    def __init__(self, scroll_batches: int = 5, headless: bool = True, locale: str = "en-US"):
        self.scroll_batches = scroll_batches
        self.headless = headless
        self.locale = locale
        self.ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    def collect(self, term: str, limit: int = 40) -> pd.DataFrame:
        rows: List[Dict] = []
        try:
            rows = _p_collect(term, limit, self.headless, self.scroll_batches, self.locale, self.ua)
        except Exception:
            rows = _s_collect(term, limit, self.headless)

        seen, final = set(), []
        for r in rows:
            r["term"] = term
            u = r.get("url")
            if u and u not in seen:
                seen.add(u)
                final.append(r)
        return pd.DataFrame(final[:limit])
```

