import os
import re
import json
import glob
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "AJD Topic Explorer — Search • Compare • Suggest"
DATA_DIR = Path("data")

# -----------------------------
# Helpers
# -----------------------------

def _merge_chunked_csv(pattern: str, merged_path: Path) -> bool:
    """Merge chunked CSVs (part01, part02, ...) into a single CSV at merged_path.
    Returns True if merged, False if no parts found.
    """
    parts = sorted(glob.glob(pattern))
    if not parts:
        return False
    header_written = False
    with open(merged_path, "w", encoding="utf-8") as out:
        for idx, p in enumerate(parts):
            with open(p, "r", encoding="utf-8") as f:
                if idx == 0:
                    out.write(f.read())
                else:
                    # skip header
                    _ = f.readline()
                    out.write(f.read())
    return True

@st.cache_data(show_spinner=False)
def load_topics_df() -> pd.DataFrame:
    """Load topics CSV; if missing, try to merge chunked parts automatically."""
    topics_csv = DATA_DIR / "ajd_topics_extracted.csv"
    if not topics_csv.exists():
        _merge_chunked_csv(str(DATA_DIR / "ajd_topics_extracted.part*.csv"), topics_csv)
    if topics_csv.exists():
        df = pd.read_csv(topics_csv)
        # Normalize expected columns
        if "topic" in df.columns and "count" in df.columns:
            return df
        # Fallback: rename first two cols
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: "topic", df.columns[1]: "count"})
            return df
    return pd.DataFrame(columns=["topic", "count"])  # empty

@st.cache_data(show_spinner=False)
def load_catalogue_df() -> pd.DataFrame:
    """Load AJD catalogue CSV; if missing, try to merge chunked parts automatically."""
    cat_csv = DATA_DIR / "ajd_catalogue_raw.csv"
    if not cat_csv.exists():
        _merge_chunked_csv(str(DATA_DIR / "ajd_catalogue_raw.part*.csv"), cat_csv)
    if cat_csv.exists():
        return pd.read_csv(cat_csv)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def infer_text_columns(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"(title|synopsis|summary|topic|subject|tags|genre|theme|series|strand|category|type|desc|name|english|arabic)", re.I)
    return [c for c in df.columns if patt.search(str(c))]

@st.cache_data(show_spinner=False)
def tfidf_similarities(ajd_texts: List[str], proj_texts: List[str], min_df: int = 2) -> Tuple[pd.DataFrame, List[int]]:
    """Return cosine similarity matrix as DataFrame (proj x ajd) and row order list for ajd."""
    corpus = ajd_texts + proj_texts
    vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    n_ajd = len(ajd_texts)
    ajd_vecs = X[:n_ajd]
    proj_vecs = X[n_ajd:]
    sim = cosine_similarity(proj_vecs, ajd_vecs)
    sim_df = pd.DataFrame(sim)
    return sim_df, list(range(n_ajd))

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Search the Al Jazeera Documentary catalogue, compare your projects, and get stronger loglines.")

# Sidebar: data status
with st.sidebar:
    st.header("Dataset")
    topics_df = load_topics_df()
    cat_df = load_catalogue_df()

    st.write("**Topics**", f"{len(topics_df):,} rows" if not topics_df.empty else "— missing")
    st.write("**Catalogue**", f"{len(cat_df):,} rows" if not cat_df.empty else "— missing")

    if st.button("Reload data / clear cache"):
        load_topics_df.clear()
        load_catalogue_df.clear()
        infer_text_columns.clear()
        st.rerun()

    if topics_df.empty or cat_df.empty:
        st.warning(
            "Merged CSVs not found. If you committed chunked files, the app will auto-merge them on first run.\n"
            "Make sure your repo has `/data/ajd_topics_extracted.part01.csv…` and `/data/ajd_catalogue_raw.part01.csv…`"
        )

# Tabs
search_tab, compare_tab, similar_tab, logline_tab, diag_tab = st.tabs([
    "🔎 Search AJD Catalogue", "🔁 Topic Overlap", "🧭 Similarity Matches", "🪄 Logline Suggestions", "🧰 Diagnostics"
])

# -----------------------------
# TAB 5: Diagnostics
# -----------------------------
with diag_tab:
    st.subheader("Diagnostics")
    # Show what files exist
    data_files = sorted(glob.glob(str(DATA_DIR / "*")))
    st.write("**/data files:**", data_files)

    # If merged files exist, show heads
    tpath = DATA_DIR / "ajd_topics_extracted.csv"
    cpath = DATA_DIR / "ajd_catalogue_raw.csv"
    if tpath.exists():
        st.markdown("**Topics (head)**")
        st.dataframe(pd.read_csv(tpath).head(10))
    else:
        st.info("Topics merged CSV missing.")
    if cpath.exists():
        st.markdown("**Catalogue (head)**")
        st.dataframe(pd.read_csv(cpath).head(10))
    else:
        st.info("Catalogue merged CSV missing.")

st.markdown("---")
st.caption("© 2025 ICON Studio — AJD Topic Explorer Dashboard. Add `streamlit` to requirements.txt and run: `streamlit run app.py`") (title, angle) in enumerate(hook_angles[:n]):
            outs.append(
                f"{title}: In one line — {seed_txt} — a story where {angle}"
            )
        return outs

    if st.button("Generate loglines", type="primary"):
        if not seed:
            st.warning("Please enter a short seed description.")
        else:
            logs = propose_loglines(seed)
            for i, line in enumerate(logs, 1):
                st.write(f"**{i}.** {line}")
            # Provide a downloadable txt
            st.download_button(
                "Download loglines (.txt)",
                ("\n".join(logs)).encode("utf-8"),
                file_name=(f"loglines_{pmpt_code or 'untagged'}.txt")
            )

# Footer
st.markdown("---")
st.caption("© 2025 ICON Studio — AJD Topic Explorer Dashboard. Add `streamlit` to requirements.txt and run: `streamlit run app.py`")
