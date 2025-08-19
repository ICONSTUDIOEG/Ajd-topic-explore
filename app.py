import os, re, json, glob
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "AJD Topic Explorer — Search • Compare • Suggest"
DATA_DIR = Path("data")

# -----------------------------
# Helpers (no docstrings to avoid copy/paste quote issues)
# -----------------------------

def _merge_chunked_csv(pattern: str, merged_path: Path) -> bool:
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
    topics_csv = DATA_DIR / "ajd_topics_extracted.csv"
    if not topics_csv.exists():
        _merge_chunked_csv(str(DATA_DIR / "ajd_topics_extracted.part*.csv"), topics_csv)
    if topics_csv.exists():
        df = pd.read_csv(topics_csv)
        cols = [c.strip().lower() for c in df.columns]
        if "topic" in cols and "count" in cols:
            # normalize to exact names
            df = df.rename(columns={df.columns[cols.index("topic")]: "topic",
                                    df.columns[cols.index("count")]: "count"})
            return df[["topic", "count"]]
        # fallback: rename first two columns
        if len(df.columns) >= 2:
            return df.rename(columns={df.columns[0]: "topic", df.columns[1]: "count"})[["topic", "count"]]
    return pd.DataFrame(columns=["topic", "count"])

@st.cache_data(show_spinner=False)
def load_catalogue_df() -> pd.DataFrame:
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
    corpus = ajd_texts + proj_texts
    vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    n_ajd = len(ajd_texts)
    sim = cosine_similarity(X[n_ajd:], X[:n_ajd])
    return pd.DataFrame(sim), list(range(n_ajd))

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Search the Al Jazeera Documentary catalogue, compare your projects, and get stronger loglines.")

with st.sidebar:
    st.header("Dataset")
    topics_df = load_topics_df()
    cat_df = load_catalogue_df()
    st.write("**Topics**", f"{len(topics_df):,} rows" if not topics_df.empty else "— missing")
    st.write("**Catalogue**", f"{len(cat_df):,} rows" if not cat_df.empty else "— missing")

    if st.button("Reload data / clear cache"):
        load_topics_df.clear(); load_catalogue_df.clear(); infer_text_columns.clear()
        st.rerun()

    if topics_df.empty or cat_df.empty:
        st.warning(
            "Merged CSVs not found. If you committed chunked files, the app will auto-merge them on first run.\n"
            "Ensure `/data/ajd_topics_extracted.part01.csv…` and `/data/ajd_catalogue_raw.part01.csv…` exist."
        )

tabs = st.tabs([
    "🔎 Search AJD Catalogue",
    "🔁 Topic Overlap",
    "🧭 Similarity Matches",
    "🪄 Logline Suggestions",
    "🧰 Diagnostics"
])
search_tab, compare_tab, similar_tab, logline_tab, diag_tab = tabs

# -----------------------------
# TAB 1: Search
# -----------------------------
with search_tab:
    st.subheader("Search AJD Catalogue")
    if cat_df.empty:
        st.info("Upload/commit your catalogue CSV parts to `data/` and reload.")
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
                        s = cat_df[c].astype(str).fillna("")
                        if mode == "Contains (case-insensitive)":
                            mask |= s.str.contains(q, case=False, na=False, regex=False)
                        elif mode == "Exact (case-insensitive)":
                            mask |= s.str.strip().str.lower() == q.strip().lower()
                        else:
                            mask |= s.str.contains(q, case=False, na=False, regex=True)
                    total = int(mask.sum())
                    res = cat_df[mask].head(max_rows)
                    st.success(f"Found {total:,} rows; showing {len(res):,}.")
                    if total == 0:
                        st.info("No matches. Try switching modes, removing punctuation, or adding columns.")
                    st.dataframe(res, use_container_width=True)
                    if not res.empty:
                        st.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                                           file_name="ajd_search_results.csv", mime="text/csv")
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
        try:
            if uploaded.name.lower().endswith(".csv"):
                p = pd.read_csv(uploaded)
                st.write("Your CSV preview:"); st.dataframe(p.head(10), use_container_width=True)
                def explode_topics_str(text: str) -> List[str]:
                    parts = re.split(r"[;,/|·•\-–—]+", str(text))
                    keepout = {"and","the","for","with","from","into","about","film","doc","docs","docu","series","episode","season","ajd","al jazeera","aljazeera"}
                    return [t.strip().lower() for t in parts if len(t.strip())>=3 and t.strip().lower() not in keepout]
                bag: List[str] = []
                if topics_col in p.columns:
                    for v in p[topics_col].fillna(""):
                        bag.extend(explode_topics_str(v))
                else:
                    st.error(f"Column '{topics_col}' not found in your CSV."); bag = []
            else:
                data = json.load(uploaded)
                items = data if isinstance(data, list) else data.get("items", [])
                def explode_topics_str(text: str) -> List[str]:
                    parts = re.split(r"[;,/|·•\-–—]+", str(text))
                    keepout = {"and","the","for","with","from","into","about","film","doc","docs","docu","series","episode","season","ajd","al jazeera","aljazeera"}
                    return [t.strip().lower() for t in parts if len(t.strip())>=3 and t.strip().lower() not in keepout]
                bag: List[str] = []
                for item in items:
                    topics_val = item.get("topics") or item.get("tags") or ""
                    if isinstance(topics_val, list):
                        for t in topics_val: bag.extend(explode_topics_str(t))
                    else:
                        bag.extend(explode_topics_str(topics_val))

            if topics_df.empty:
                st.error("AJD topics dataset is missing.")
            else:
                ajd_topics = set(topics_df["topic"].astype(str).str.lower())
                proj_topics = set(bag)
                overlap = sorted(list(ajd_topics & proj_topics))
                only_proj = sorted(list(proj_topics - ajd_topics))
                st.markdown("### Summary")
                st.json({"ajd_unique_topics": len(ajd_topics), "project_unique_topics": len(proj_topics), "overlap_count": len(overlap)})
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Overlap topics**"); st.write(pd.DataFrame({"topic": overlap}))
                with c2:
                    st.markdown("**Project-only topics (white space)**"); st.write(pd.DataFrame({"topic": only_proj}))
                st.download_button("Download overlap (CSV)", pd.DataFrame({"topic": overlap}).to_csv(index=False).encode("utf-8"), file_name="overlap_topics.csv")
                st.download_button("Download project-only (CSV)", pd.DataFrame({"topic": only_proj}).to_csv(index=False).encode("utf-8"), file_name="project_only_topics.csv")
        except Exception as e:
            st.error(f"Error processing upload: {e}")

# -----------------------------
# TAB 3: Similarity Matches (TF-IDF)
# -----------------------------
with similar_tab:
    st.subheader("Find Closest AJD Matches for Your Films (TF-IDF)")
    if cat_df.empty:
        st.info("Catalogue missing. Add CSVs to /data and reload.")
    else:
        text_cols_default = infer_text_columns(cat_df)
        cols = st.multiselect("AJD text columns to use", options=list(cat_df.columns), default=text_cols_default)
        films = st.text_area("Paste your films as JSON list (each item: title, description, topics optional)",
                             value='[\n  {"title":"Film A","description":"Inside the rise of tech startups in MENA."},\n  {"title":"Film B","description":"Lives split across continents."}\n]')
        if st.button("Compute matches", type="primary"):
            try:
                films_data = json.loads(films)
                proj_texts = [(i.get("title","Untitled"), (i.get("title","")+" "+i.get("description"," ")).strip()) for i in films_data]
                ajd_texts = (cat_df[cols].fillna("").astype(str).agg(" ".join, axis=1)).tolist()
                sim_df, _ = tfidf_similarities(ajd_texts, [t for _, t in proj_texts])
                for row_idx, (title, _) in enumerate(proj_texts):
                    st.markdown(f"#### {title}")
                    sims = sim_df.iloc[row_idx].values
                    top_idx = sims.argsort()[::-1][:10]
                    rows = []
                    for j in top_idx:
                        score = float(sims[j]); ajd_row = cat_df.iloc[j]
                        rows.append({"score": round(score, 4), **ajd_row.to_dict()})
                    out_df = pd.DataFrame(rows)
                    st.dataframe(out_df, use_container_width=True)
                    st.download_button(f"Download matches for {title}", out_df.to_csv(index=False).encode("utf-8"),
                                       file_name=f"matches_{re.sub(r'[^A-Za-z0-9]+','_', title.lower())}.csv")
            except Exception as e:
                st.error(f"Invalid JSON or error computing similarities: {e}")

# -----------------------------
# TAB 4: Logline Suggestions
# -----------------------------
with logline_tab:
    st.subheader("Suggest Strong Loglines")
    st.caption("Craft angles that feel fresh relative to AJD coverage.")
    seed = st.text_area("Describe your film (topic, setting, main character, conflict) — a few lines.")
    pmpt_code = st.text_input("Optional prompt code (e.g., pmpt_...) to tag your session", value="")
    def propose_loglines(seed_txt: str, n: int = 6) -> List[str]:
        seed_txt = seed_txt.strip()
        angles = [
            ("Hidden History", "uncovers a buried past that reshapes the present."),
            ("Personal Lens", "tells a national story through one family or character."),
            ("System vs. Individual", "shows how policies collide with everyday survival."),
            ("Vanishing Traditions", "captures a craft or custom at the edge of extinction."),
            ("Unexpected Ally/Adversary", "pairs unlikely characters or groups in tension."),
            ("Future Stakes", "connects today's choice to tomorrow's irreversible outcome."),
        ]
        outs = []
        for _, (title, angle) in enumerate(angles[:n]):
            outs.append(f"{title}: In one line — {seed_txt} — a story where it {angle}")
        return outs
    if st.button("Generate loglines", type="primary"):
        if not seed:
            st.warning("Please enter a short seed description.")
        else:
            logs = propose_loglines(seed)
            for i, line in enumerate(logs, 1):
                st.write(f"**{i}.** {line}")
            st.download_button("Download loglines (.txt)", ("\n".join(logs)).encode("utf-8"),
                               file_name=(f"loglines_{pmpt_code or 'untagged'}.txt"))

# -----------------------------
# TAB 5: Diagnostics
# -----------------------------
with diag_tab:
    st.subheader("Diagnostics")
    data_files = sorted(glob.glob(str(DATA_DIR / "*")))
    st.write("**/data files:**", data_files)
    tpath = DATA_DIR / "ajd_topics_extracted.csv"
    cpath = DATA_DIR / "ajd_catalogue_raw.csv"
    if tpath.exists():
        st.markdown("**Topics (head)**"); st.dataframe(pd.read_csv(tpath).head(10))
    else:
        st.info("Topics merged CSV missing.")
    if cpath.exists():
        st.markdown("**Catalogue (head)**"); st.dataframe(pd.read_csv(cpath).head(10))
    else:
        st.info("Catalogue merged CSV missing.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("© 2025 ICON Studio — AJD Topic Explorer Dashboard. Add `streamlit` to requirements.txt and run: `streamlit run app.py`")
