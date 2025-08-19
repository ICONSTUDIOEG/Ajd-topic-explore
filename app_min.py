import re, json, glob
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

DATA_DIR = Path("data")

st.set_page_config(page_title="AJD Minimal", layout="wide")
st.title("AJD Minimal — Search & Diagnose")

# --- data loaders ---
@st.cache_data
def merge_if_needed(parts_glob: str, merged: Path) -> bool:
    parts = sorted(glob.glob(parts_glob))
    if not parts:
        return False
    with open(merged, "w", encoding="utf-8") as out:
        for i, p in enumerate(parts):
            with open(p, "r", encoding="utf-8") as f:
                if i == 0:
                    out.write(f.read())
                else:
                    _ = f.readline()
                    out.write(f.read())
    return True

@st.cache_data
def load_topics() -> pd.DataFrame:
    p = DATA_DIR / "ajd_topics_extracted.csv"
    if not p.exists():
        merge_if_needed(str(DATA_DIR / "ajd_topics_extracted.part*.csv"), p)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["topic","count"])

@st.cache_data
def load_catalogue() -> pd.DataFrame:
    p = DATA_DIR / "ajd_catalogue_raw.csv"
    if not p.exists():
        merge_if_needed(str(DATA_DIR / "ajd_catalogue_raw.part*.csv"), p)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

# --- sidebar ---
with st.sidebar:
    topics_df = load_topics()
    cat_df = load_catalogue()
    st.write("**Topics rows:**", len(topics_df))
    st.write("**Catalogue rows:**", len(cat_df))
    if st.button("Clear cache & reload"):
        load_topics.clear(); load_catalogue.clear()
        st.rerun()

# --- tabs (DEFINE BEFORE USING) ---
tab_search, tab_diag = st.tabs(["🔎 Search", "🧰 Diagnostics"])

# --- Search ---
with tab_search:
    st.subheader("Search AJD Catalogue")
    cat_df = load_catalogue()
    if cat_df.empty:
        st.info("Missing `data/ajd_catalogue_raw.csv` and no parts to merge.")
    else:
        # Guess text columns
        cols_guess = [c for c in cat_df.columns if re.search(r"(name|title|synopsis|series|english|arabic|desc|topic)", str(c), re.I)]
        cols = st.multiselect("Columns", options=list(cat_df.columns), default=cols_guess)
        q = st.text_input("Query (contains, case-insensitive)")
        if st.button("Search", type="primary"):
            if not cols or not q:
                st.warning("Pick columns and enter a query.")
            else:
                mask = pd.Series(False, index=cat_df.index)
                for c in cols:
                    mask |= cat_df[c].astype(str).str.contains(q, case=False, na=False)
                res = cat_df[mask].head(200)
                st.success(f"Found {int(mask.sum()):,} rows; showing {len(res):,}.")
                st.dataframe(res, use_container_width=True)

# --- Diagnostics ---
with tab_diag:
    st.subheader("Diagnostics")
    files = sorted(glob.glob(str(DATA_DIR / "*")))
    st.write("**/data files**:", files)
    p_topics = DATA_DIR / "ajd_topics_extracted.csv"
    p_cat = DATA_DIR / "ajd_catalogue_raw.csv"
    if p_topics.exists():
        st.write("Topics head:"); st.dataframe(pd.read_csv(p_topics).head(10))
    else:
        st.info("Topics merged CSV missing")
    if p_cat.exists():
        st.write("Catalogue head:"); st.dataframe(pd.read_csv(p_cat).head(10))
    else:
        st.info("Catalogue merged CSV missing")
