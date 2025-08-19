import re, json, glob
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "AJD Topic Explorer — Search • Compare • Suggest"
DATA_DIR = Path("data")

# ---------------- helpers ----------------
def _merge_chunked_csv(pattern: str, merged_path: Path) -> bool:
    parts = sorted(glob.glob(pattern))
    if not parts:
        return False
    with open(merged_path, "w", encoding="utf-8") as out:
        for i, p in enumerate(parts):
            with open(p, "r", encoding="utf-8") as f:
                if i == 0:
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
            df = df.rename(columns={
                df.columns[cols.index("topic")]: "topic",
                df.columns[cols.index("count")]: "count"
            })
            return df[["topic", "count"]]
        if len(df.columns) >= 2:
            return df.rename(columns={
                df.columns[0]: "topic",
                df.columns[1]: "count"
            })[["topic", "count"]]
    return pd.DataFrame(columns=["topic", "count"])

@st.cache_data(show_spinner=False)
def load_catalogue_df() -> pd.DataFrame:
    """
    Robust loader: auto-merge parts; try multiple encodings & delimiters; clean columns/rows.
    """
    cat_csv = DATA_DIR / "ajd_catalogue_raw.csv"
    if not cat_csv.exists():
        _merge_chunked_csv(str(DATA_DIR / "ajd_catalogue_raw.part*.csv"), cat_csv)

    if not cat_csv.exists():
        return pd.DataFrame()

    encodings = ["utf-8", "utf-8-sig", "cp1256", "latin1"]
    seps = [None, ",", ";", "\t", "|"]  # None = let pandas sniff
    last_err = None
    df = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    cat_csv,
                    sep=sep,
                    encoding=enc,
                    engine="python",
                    dtype=str,
                    na_filter=False,
                    keep_default_na=False,
                    on_bad_lines="skip",
                )
                if df is not None and df.shape[0] > 0:
                    break
            except Exception as e:
                last_err = e
        if df is not None and df.shape[0] > 0:
            break

    if df is None:
        st.error(f"Could not read catalogue CSV. Last error: {last_err}")
        return pd.DataFrame()

    # Clean column names (trim / collapse spaces)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # Drop fully empty columns and rows
    df = df.dropna(axis=1, how="all")
    df = df.loc[~(df.apply(lambda r: "".join(map(str, r)).strip(), axis=1) == "")]

    # Show shape for quick verification
    st.sidebar.write(f"Loaded catalogue: {df.shape[0]:,} rows × {df.shape[1]} cols")

    return df

# ---------------- app ----------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Search the Al Jazeera Documentary catalogue, compare your projects, and get stronger loglines.")

    # Sidebar
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
            st.warning("If merged CSVs are missing, the app will merge `/data/...partNN.csv` on first run.")

    # DEFINE TABS FIRST
    (search_tab, compare_tab, similar_tab, logline_tab, diag_tab) = st.tabs([
        "🔎 Search AJD Catalogue", "🔁 Topic Overlap", "🧭 Similarity Matches", "🪄 Logline Suggestions", "🧰 Diagnostics"
    ])

    # ---------------- TAB 1: Search ----------------
    with search_tab:
        st.subheader("Search AJD Catalogue")
        @st.cache_data(show_spinner=False)
def load_catalogue_df() -> pd.DataFrame:
    """
    Robust loader: auto-merge parts; try multiple encodings & delimiters; clean columns/rows.
    """
    cat_csv = DATA_DIR / "ajd_catalogue_raw.csv"
    if not cat_csv.exists():
        _merge_chunked_csv(str(DATA_DIR / "ajd_catalogue_raw.part*.csv"), cat_csv)

    if not cat_csv.exists():
        return pd.DataFrame()

    # Try multiple encodings and delimiters
    encodings = ["utf-8", "utf-8-sig", "cp1256", "latin1"]
    seps = [None, ",", ";", "\t", "|"]  # None = let pandas sniff
    last_err = None
    df = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    cat_csv,
                    sep=sep,
                    encoding=enc,
                    engine="python",
                    dtype=str,              # keep text as strings
                    na_filter=False,        # don't auto-convert to NaN
                    keep_default_na=False,  # treat 'NA' as literal
                    on_bad_lines="skip",
                )
                if df is not None and df.shape[0] > 0:
                    break
            except Exception as e:
                last_err = e
        if df is not None and df.shape[0] > 0:
            break

    if df is None:
        st.error(f"Could not read catalogue CSV. Last error: {last_err}")
        return pd.DataFrame()

    # Clean column names (trim / collapse spaces)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # Drop fully empty columns and rows
    df = df.dropna(axis=1, how="all")
    df = df.loc[~(df.apply(lambda r: "".join(map(str, r)).strip(), axis=1) == "")]

    # Show shape for quick verification
    st.sidebar.write(f"Loaded catalogue: {df.shape[0]:,} rows × {df.shape[1]} cols")

    return df
    # ---------------- TAB 2: Topic Overlap ----------------
    with compare_tab:
        st.subheader("Compare Your Project Topics vs AJD Topics")
        topics_df = load_topics_df()
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

    # ---------------- TAB 3: Similarity Matches ----------------
    with similar_tab:
        st.subheader("Find Closest AJD Matches for Your Films (TF-IDF)")
        cat_df = load_catalogue_df()
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
                    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
                    X = vectorizer.fit_transform(ajd_texts + [t for _, t in proj_texts])
                    n_ajd = len(ajd_texts)
                    from sklearn.metrics.pairwise import cosine_similarity
                    sim = cosine_similarity(X[n_ajd:], X[:n_ajd])
                    sim_df = pd.DataFrame(sim)
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

    # ---------------- TAB 4: Logline Suggestions ----------------
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

    # ---------------- TAB 5: Diagnostics ----------------
    with diag_tab:
    st.subheader("Diagnostics")

    files = sorted(glob.glob(str(DATA_DIR / "*")))
    st.write("**/data files**:", files)

    p_cat = DATA_DIR / "ajd_catalogue_raw.csv"
    if p_cat.exists():
        st.write(f"Catalogue file size: {p_cat.stat().st_size:,} bytes")
    else:
        st.info("Catalogue merged CSV missing")

    # Reload with our robust loader and show summary
    cat_df_preview = load_catalogue_df()
    st.write(f"Catalogue DataFrame shape: {cat_df_preview.shape[0]:,} rows × {cat_df_preview.shape[1]} cols")

    if not cat_df_preview.empty:
        st.write("**Column names (first 30):**", list(cat_df_preview.columns[:30]))
        st.write("**Head(10):**")
        st.dataframe(cat_df_preview.head(10), use_container_width=True)

        # Show which columns look texty
        texty = [c for c in cat_df_preview.columns if re.search(r"(name|title|synopsis|series|english|arabic|desc|topic)", c, re.I)]
        st.write("**Guessed text columns:**", texty if texty else "(none)")
    else:
        st.warning("DataFrame is empty after loading. This usually means delimiter/encoding issues or an empty file.")
