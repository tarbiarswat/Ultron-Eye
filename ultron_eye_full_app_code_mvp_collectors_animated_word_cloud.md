# Ultron Eye — Full App Code (MVP + Collectors + Animated Word Cloud)

> Python + Streamlit app with two modes (Decision Helper / Go Broad), modular collectors (RSS, Yahoo News, Reddit, Pinterest) using **Selenium**, **Playwright**, and **BeautifulSoup**; standardized cleaning pipeline; optional sentiment; static and **animated word cloud**; console + file logs. Designed to extend with NLP, timelines, heatmaps, and AI summaries.

---

## 📁 Project Structure
```
ultron-eye/
  requirements.txt
  run_console.py
  app/
    app.py
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
  logs/
  data/
    raw/
    processed/
    cache/
```

> **Notes**
> - Pinterest/Reddit selectors & anti-bot may change; this code is best-effort and demonstrates patterns. Respect each site's Terms of Service & robots.txt.
> - Playwright requires a one-time: `playwright install`.
> - Selenium requires a driver (e.g., `webdriver-manager`).

---

## 🔧 `requirements.txt`
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

> After installing requirements: run `python -m playwright install` to install browser engines.

---

## ▶️ `run_console.py`
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
    print("Top tokens:")
    print(token_counts(cleaned).head(20))
```

---

## ⚙️ `app/config.py`
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

# Registered sources (checkboxes). Keys must match collectors mapping in app.py
SOURCES = ["Reddit", "Yahoo News", "Google RSS", "Pinterest"]

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

## 👔 `app/modes.py`
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

## 🧰 `app/utils/log.py`
```python
import logging
from logging.handlers import RotatingFileHandler
from ..config import LOG_DIR

_DEF_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

def get_logger(name: str = "ultron_eye"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(_DEF_FMT)

    fh = RotatingFileHandler(LOG_DIR / "ultron_eye.log", maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(_DEF_FMT)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
```

---

## 🗃️ `app/utils/io.py`
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

---

## 🔤 `app/utils/text.py`
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

## 🧼 `app/pipelines/clean.py`
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

from ..config import CLEANING
from ..utils import text as TX

STOP = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()

def _basic_clean(s: str, cfg=CLEANING) -> str:
    s = TX.normalize(s)
    if cfg["strip_urls"]: s = TX.strip_urls(s)
    if cfg["strip_emojis"]: s = TX.strip_emojis(s)
    if cfg["lowercase"]: s = s.lower()
    if cfg["strip_punct"]: s = TX.strip_nonletters(s, keep_spaces=True)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    return [t for t in s.split() if len(t) > 2]

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
                for t in toks: freq[t] = freq.get(t, 0) + 1
            for t, c in sorted(freq.items(), key=lambda x: x[1], reverse=True):
                rows.append({groupby: key, "token": t, "count": c})
    else:
        freq = {}
        for toks in df["tokens"]:
            for t in toks: freq[t] = freq.get(t, 0) + 1
        for t, c in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            rows.append({"token": t, "count": c})
    return pd.DataFrame(rows)
```

---

## 🧪 `app/pipelines/nlp.py` (optional sentiment for now)
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

## 🖼️ `app/pipelines/vis.py` (static & **animated** word cloud)
```python
from pathlib import Path
from typing import Iterable, List
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Static word cloud from tokens

def wordcloud_from_tokens(tokens_iter: Iterable[List[str]], width=1000, height=500, bg="white"):
    text_blob = " ".join([" ".join(toks) for toks in tokens_iter])
    if not text_blob.strip():
        return None
    wc = WordCloud(width=width, height=height, background_color=bg).generate(text_blob)
    return wc.to_array()

# Animated word cloud: generate frames over progressive windows of tokens

def animated_wordcloud(tokens_iter: Iterable[List[str]], out_path: Path, frames: int = 12,
                       step: int = 1, width: int = 900, height: int = 400, bg: str = "white", fps: int = 3):
    tokens = list(tokens_iter)
    if not tokens:
        return None
    images = []
    n = len(tokens)
    # Progressive window: 1/frames, 2/frames, ... full
    for i in range(1, frames + 1):
        end = max(1, int(i * n / frames))
        blob = " ".join([" ".join(toks) for toks in tokens[:end:step]])
        if not blob.strip():
            continue
        wc = WordCloud(width=width, height=height, background_color=bg).generate(blob)
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        fig.canvas.draw()
        # Convert to image array
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    if not images:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, images, duration=1.0/ fps)
    return out_path
```

---

## 🧱 Collectors Base & Common

### `app/collectors/base.py`
```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import List

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

### `app/collectors/common.py`
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

---

## 🌐 `app/collectors/rss.py` (Google News RSS or any RSS)
```python
import pandas as pd
import feedparser
from .base import Collector
from .common import now_iso

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

---

## 📰 `app/collectors/yahoo_news.py` (requests + BS4; Selenium fallback)
```python
import pandas as pd
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from .base import Collector
from .common import now_iso, HEADERS

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

---

## 🧵 `app/collectors/reddit.py` (requests JSON; Playwright demo optional)
```python
import pandas as pd
import requests
from .base import Collector
from .common import now_iso, HEADERS

class RedditCollector(Collector):
    name = "Reddit"

    def __init__(self, subreddit: str | None = None):
        self.subreddit = subreddit  # if None, search all

    def collect(self, term: str, limit: int = 50) -> pd.DataFrame:
        # Use Reddit JSON search endpoint (public). For heavier use, prefer official API.
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

---

## 📌 `app/collectors/pinterest.py` (Playwright headless; basic scraping)
```python
import asyncio
import pandas as pd
from playwright.sync_api import sync_playwright
from .base import Collector
from .common import now_iso

PIN_URL = "https://www.pinterest.com/search/pins/?q={query}"

class PinterestCollector(Collector):
    name = "Pinterest"

    def __init__(self, scroll_batches: int = 3, headless: bool = True):
        self.scroll_batches = scroll_batches
        self.headless = headless

    def collect(self, term: str, limit: int = 40) -> pd.DataFrame:
        rows = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(30000)
            q = term.replace(" ", "%20")
            page.goto(PIN_URL.format(query=q))

            # Scroll batches to load more pins
            for _ in range(self.scroll_batches):
                page.mouse.wheel(0, 3000)
                page.wait_for_timeout(1500)

            # Extract pins: titles, alt texts, links
            # Pinterest DOM changes often; this is a generic fallback selection
            pins = page.locator('a[href^="/pin/"]')
            count = min(pins.count(), limit)
            seen = set()
            for i in range(count):
                try:
                    el = pins.nth(i)
                    href = el.get_attribute("href")
                    if not href or href in seen:
                        continue
                    seen.add(href)
                    # try to get nearby text
                    alt = el.get_attribute("aria-label") or el.inner_text(timeout=2000) or ""
                    full_url = f"https://www.pinterest.com{href}" if href.startswith("/") else href
                    rows.append({
                        "source": self.name,
                        "fetched_at": now_iso(),
                        "term": term,
                        "url": full_url,
                        "title": alt.strip()[:200] if alt else None,
                        "text": alt.strip(),
                        "author": None,
                        "region": None,
                        "category": None,
                    })
                except Exception:
                    continue
            context.close()
            browser.close()
        return pd.DataFrame(rows)
```

> If Pinterest throttles, reduce `scroll_batches`, add delays, or consider session cookies per their terms.

---

## 🖥️ `app/app.py` (Streamlit UI + Modes + Collectors + Cleaning + Word Clouds)
```python
import streamlit as st
import pandas as pd
from pathlib import Path
from config import APP_NAME, SOURCES, RAW_DIR, PROC_DIR
from utils.log import get_logger
from utils.io import ts, write_df
from pipelines.clean import clean_frame, token_counts
from pipelines.vis import wordcloud_from_tokens, animated_wordcloud
from modes import validate_mode1, validate_mode2

# Collectors mapping
from collectors.rss import RSSCollector
from collectors.yahoo_news import YahooNewsCollector
from collectors.reddit import RedditCollector
from collectors.pinterest import PinterestCollector

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

def build_input_df() -> pd.DataFrame:
    rows = []
    # From uploaded file
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
                if line.strip(): rows.append({"text": line.strip()})
    # From paste
    if raw_text.strip():
        for line in raw_text.splitlines():
            if line.strip(): rows.append({"text": line.strip()})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["text"])

# Collector registry

def make_collectors(selected: list[str]):
    mapping = {
        "Google RSS": RSSCollector(),
        "Yahoo News": YahooNewsCollector(use_selenium=False),  # toggle to True to use Selenium
        "Reddit": RedditCollector(),
        "Pinterest": PinterestCollector(scroll_batches=3, headless=True),
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

    # 1) Collect from selected sources
    collectors = make_collectors(sources)
    all_rows = []
    for term in terms:
        for col in collectors:
            try:
                df_c = col.collect(term, limit=limit)
                all_rows.append(df_c)
                logger.info(f"Collected {len(df_c)} from {col.name} for '{term}'")
            except Exception as e:
                logger.exception(f"Collector error [{col.name}] term='{term}': {e}")
    collected_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["text"])

    # plus optional local data
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

    # For Decision mode, keep term labels as-is; for Broad mode, cycle if missing
    if "term" not in collected_df.columns or collected_df["term"].isna().all():
        cyc = []
        k = 0
        for _ in range(len(collected_df)):
            cyc.append(terms[k])
            k = (k + 1) % len(terms)
        collected_df["term"] = cyc

    # 2) Clean
    cleaned = clean_frame(collected_df.assign(text=collected_df.get("text").fillna(collected_df.get("title"))), text_col="text")
    proc_path = PROC_DIR / f"processed_{ts()}.csv"
    write_df(cleaned, proc_path)
    logger.info(f"Cleaned & saved: {proc_path.name} | rows={len(cleaned)}")

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

    # Console echo
    print(f"[UltronEye] rows={len(cleaned)} | mode={'M1' if mode.startswith('Decision') else 'M2'} | sources={sources}")
```

---

## ✅ How to Run
1. Create venv & install:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   pip install -r requirements.txt
   playwright install
   ```
2. Launch UI:
   ```bash
   streamlit run app/app.py
   ```
3. Watch terminal logs and `logs/ultron_eye.log`.

---

## 🧭 Extending Next
- **NLP (Step #3):** add topic modeling (e.g., BERTopic or sklearn LDA) into `pipelines/nlp.py`, render top topics per term.
- **Visuals (Step #4):** timelines via Altair/Matplotlib; heatmaps by category/region if sources supply metadata.
- **AI Summaries (Step #5):** call your preferred LLM to summarize `overall`, `by_term`, and anomalies; render in a right-hand panel.
- **Persistence:** add SQLite for deduplication across runs; incremental collection by timestamp.
- **Scheduling:** trigger periodic collection with `streamlit_autorefresh` or a separate cron job writing CSVs.

---

### ⚠️ Responsible Use
Always respect site Terms of Service, robots.txt, and rate limits. Use authenticated or official APIs when required and obtain consent for any non-public data. This code is for educational/demo purposes and may require adjustments to selectors and pacing to remain compliant and reliable.

