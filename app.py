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
    with open(merged_path, "w", encoding="utf-8") as out:
        for idx, p in enumerate(parts):
            with open(p, "r", encoding="utf-8") as f:
                if idx == 0:
                    out.write(f.read())
                else:
                    _ = f.readline()  # skip header
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
        if "topic" in df.columns and "count" in df.columns:
            return df
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: "topic", df.columns[1]: "count"})
            return df
    return pd.DataFrame(columns=["topic", "count"])

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
    patt = re.compile(
        r"(title|synopsis|summary|topic|subject|tags|genre|theme|series|strand|category|type|desc|name|english|arabic)",
        re.I,
    )
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
# TAB 1: Search
# -----------------------------
with search_tab:
    st.subheader("Search AJD Catalogue")
    if cat_df.empty:
        st.info("Upload/commit your catalogue CSV parts to `data/` and reload the app.")
    else:
        text_cols_default = infer_text_columns(cat_df)
        cols = st.multiselect("Columns to search in:", options=list(cat_df.columns), default=text_cols_default)
        q = st.text_input("Keyword or phrase")
        mode = st.selectbox("Match mode", ["Contains (case-insensitive)", "Exact (case-insensitive)", "Regex"], index=0)
        max_rows = st.slider("Max rows to show", 10, 2000, 200)
        if st.button("Search", type="primary"):
            if not cols or not q:
                st.warning("Provide at least one column and a query.")
            else:
                try:
                    mask = pd.Series(False, index=cat_df.index)
                    for c in cols:
                        series = cat_df[c].astype(str).fillna("")
                        if mode == "Contains (case-insensitive)":
                            mask |= series.str.contains(q, case=False, na=False, regex=False)
                        elif mode == "Exact (case-insensitive)":
                            mask |= series.str.strip().str.lower() == q.strip().lower()
                        else:  # Regex
                            mask |= series.str.contains(q, case=False, na=False, regex=True)
                    total = int(mask.sum())
                    res = cat_df[mask].head(max_rows)
                    st.success(f"Found {total:,} rows; showing {len(res):,}.")
                    if total == 0:
                        st.info("No matches found. Try switching modes, removing punctuation, or adding more columns.")
                    st.dataframe(res, use_container_width=True)
                    if not res.empty:
                        st.download_button(
                            "Download results (CSV)",
                            res.to_csv(index=False).encode("utf-8"),
                            file_name="ajd_search_results.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Search error: {e}")

# -----------------------------
# TAB 2: Topic Overlap
# -----------------------------
with compare_tab:
    st.subheader("Compare Your Project Topics vs AJD Topics")

    uploaded = st.file_uploader("Upload your project file (CSV with a topics column, or JSON list)", type=["csv", "json"])
    topics_col = st.text_input("If CSV, name of the column that contains topics (e.g., 'topics')", value="topics")

    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            p = pd.read_csv(uploaded)
            st.write("Your CSV preview:")
            st.dataframe(p.head(10), use_container_width=True)

            def explode_topics_str(text: str) -> List[str]:
                parts = re.split(r"[;,/|·•\-–—]+", str(text))
                keepout = {"and","the","for","with","from","into","about","film","doc","docs","docu","series","episode","season","ajd","al jazeera","aljazeera"}
                return [t.strip().lower() for t in parts if len(t.strip()) >= 3 and t.strip().lower() not in keepout]

            bag: List[str] = []
            if topics_col in p.columns:
                for v in p[topics_col].fillna(""):
                    bag.extend(explode_topics_str(v))
            else:
                st.error(f"Column '{topics_col}' not found in your CSV.")
                bag = []
        else:
            data = json.load(uploaded)
            items = data if isinstance(data, list) else data.get("items", [])
            def explode_topics_str(text: str) -> List[str]:
                parts = re.split(r"[;,/|·•\-–—]+", str(text))
                keepout = {"and","the","for","with","from","into","about","film","doc","docs","docu","series","episode","se
