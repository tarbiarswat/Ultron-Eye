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
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../ultron-eye
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------------------------------------------------
def advance(progress, step, total, label=""):
    step += 1
    pct = min(int(step / total * 100), 100)
    progress.progress(pct, text=f"{pct}% {label}")
    return step

# ---------------------------------------------------------------------------

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

# …
def make_collectors(selected: list[str]):
    mapping = {
        "Google RSS": RSSCollector(),
        "Yahoo News": YahooNewsCollector(use_selenium=False),
        "Reddit": RedditCollector(),
        "Pinterest": PinterestCollector(scroll_batches=6, headless=True),  # <-- diagnose
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


    # 1) Collect from selected sources

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
            # advance after each collector attempt
            step = advance(progress, step, TOTAL_STEPS, label=f"Collected from {col.name} ({term})")

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
    # … after you write processed CSV …
    step = advance(progress, step, TOTAL_STEPS, label="Cleaned data")

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

        # static word cloud
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
            step = advance(progress, step, TOTAL_STEPS, label="Animated word cloud")
        else:
            st.info("Could not generate animated word cloud.")

    # Console echo
    print(f"[UltronEye] rows={len(cleaned)} | mode={'M1' if mode.startswith('Decision') else 'M2'} | sources={sources}")

    progress.progress(100, text="Done")
