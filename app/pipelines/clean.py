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
    import html
    from bs4 import BeautifulSoup
    import re
    from app.utils import text as TX

    # Decode HTML entities (&nbsp; → space)
    s = html.unescape(s or "")

    # Remove HTML tags entirely
    s = BeautifulSoup(s, "html.parser").get_text(" ")

    # Normalize text (remove accents, etc.)
    s = TX.normalize(s)

    if cfg["strip_urls"]:
        s = TX.strip_urls(s)
    if cfg["strip_emojis"]:
        s = TX.strip_emojis(s)
    if cfg["lowercase"]:
        s = s.lower()
    if cfg["strip_punct"]:
        s = TX.strip_nonletters(s, keep_spaces=True)

    # Remove residual HTML/formatting keywords
    s = re.sub(
        r"\b(href|target|font|color|nbsp|http|https|style|align|class|id|div|span|border|background)\b",
        " ",
        s,
        flags=re.IGNORECASE,
    )

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    # drop numerics / short / hex-ish fragments
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