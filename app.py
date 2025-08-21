import re, json, glob, os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st

# Optional (Similarity tab). App still runs if sklearn isn't available.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

APP_TITLE = "AJD Topic Explorer — Search • Compare • Suggest"
DATA_DIR = Path("data")

# ============================ Helpers ============================

def _merge_chunked_csv(pattern: str, merged_path: Path) -> bool:
    """Merge chunked CSVs (part01, part02, part3_01, ...) using natural sort."""
    import re as _re
    def _natsort_key(s: str):
        base = os.path.basename(s)
        return [int(t) if t.isdigit() else t.lower() for t in _re.findall(r"\d+|\D+", base)]
    parts = sorted(glob.glob(pattern), key=_natsort_key)
    if not parts:
        return False
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as out:
        for i, p in enumerate(parts):
            with open(p, "r", encoding="utf-8") as f:
                if i == 0:
                    out.write(f.read())     # header + data
                else:
                    _ = f.readline()        # skip header
                    out.write(f.read())     # data only
    return True


@st.cache_data(show_spinner=False)
def load_topics_df() -> pd.DataFrame:
    """Load topics file or auto-merge parts."""
    p = DATA_DIR / "ajd_topics_extracted.csv"
    if not p.exists():
        _merge_chunked_csv(str(DATA_DIR / "ajd_topics_extracted.part*.csv"), p)
    if not p.exists():
        return pd.DataFrame(columns=["topic", "count"])
    try:
        df = pd.read_csv(p, dtype=str, na_filter=False, keep_default_na=False,
                         engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame(columns=["topic", "count"])
    cols = [c.strip().lower() for c in df.columns]
    if "topic" in cols and "count" in cols:
        df = df.rename(columns={
            df.columns[cols.index("topic")]: "topic",
            df.columns[cols.index("count")]: "count"
        })
        return df[["topic", "count"]]
    if len(df.columns) >= 2:
        return df.rename(columns={df.columns[0]: "topic", df.columns[1]: "count"})[["topic", "count"]]
    return pd.DataFrame(columns=["topic", "count"])


@st.cache_data(show_spinner=False)
def load_catalogue_df() -> pd.DataFrame:
    """Robust loader: auto-merge parts; try several encodings & delimiters; clean columns/rows."""
    p = DATA_DIR / "ajd_catalogue_raw.csv"
    if not p.exists():
        _merge_chunked_csv(str(DATA_DIR / "ajd_catalogue_raw.part*.csv"), p)
    if not p.exists():
        return pd.DataFrame()

    encodings = ["utf-8", "utf-8-sig", "cp1256", "latin1"]
    seps = [None, ",", ";", "\t", "|"]  # None => pandas sniff
    df = None
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df_try = pd.read_csv(
                    p, sep=sep, encoding=enc, engine="python",
                    dtype=str, na_filter=False, keep_default_na=False,
                    on_bad_lines="skip"
                )
                if df_try is not None and df_try.shape[0] > 0:
                    df = df_try
                    break
            except Exception as e:
                last_err = e
        if df is not None:
            break

    if df is None:
        st.error(f"Could not read catalogue CSV. Last error: {last_err}")
        return pd.DataFrame()

    # Normalize column names
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # Drop fully empty columns and rows
    df = df.dropna(axis=1, how="all")
    if not df.empty:
        df = df.loc[~(df.apply(lambda r: "".join(map(str, r)).strip(), axis=1) == "")]

    # Quick shape
    st.sidebar.write(f"Loaded catalogue: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


@st.cache_data(show_spinner=False)
def infer_text_columns(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"(title|synopsis|summary|topic|subject|tags|genre|theme|series|strand|category|type|desc|name|english|arabic)", re.I)
    return [c for c in df.columns if patt.search(str(c))]


def _clean_hidden_series(s: pd.Series) -> pd.Series:
    """Remove zero-width/bidi chars and normalize spaces."""
    import re as _re
    ZW_PATTERN = _re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069]")
    WS_PATTERN = _re.compile(r"\s+")
    return s.astype(str).fillna("").map(lambda x: WS_PATTERN.sub(" ", ZW_PATTERN.sub("", x)).strip())


# ============================ App ============================

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Search the Al Jazeera Documentary catalogue, compare your projects, and get stronger loglines.")

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Dataset")
        topics_df = load_topics_df()
        cat_df = load_catalogue_df()
        st.write("**Topics**", f"{len(topics_df):,} rows" if not topics_df.empty else "— missing")
        st.write("**Catalogue**", f"{len(cat_df):,} rows" if not cat_df.empty else "— missing")

        # Reset UI state to avoid Streamlit KeyError on widget changes
        if st.button("Reset UI state (fix KeyError)", key="sb_reset_state"):
            st.session_state.clear()
            st.rerun()

        if st.button("Reload data / clear cache", key="sb_reload"):
            load_topics_df.clear()
            load_catalogue_df.clear()
            infer_text_columns.clear()
            st.rerun()

        if topics_df.empty or cat_df.empty:
            st.warning("If merged CSVs are missing, the app will merge `/data/...partNN.csv` on first run.")

    # ----- Tabs (define BEFORE use) -----
    search_tab, compare_tab, similar_tab, logline_tab, diag_tab = st.tabs([
        "🔎 Search AJD Catalogue",
        "🔁 Topic Overlap",
        "🧭 Similarity Matches",
        "🪄 Logline Suggestions",
        "🧰 Diagnostics"
    ])

    # ----- TAB 1: Search -----
    with search_tab:
        st.subheader("Search AJD Catalogue")
        cat_df = load_catalogue_df()
        if cat_df.empty:
            st.info("Upload/commit your catalogue CSV parts to `data/` and reload.")
        else:
            # Use preset from Diagnostics if available
            text_cols_default = infer_text_columns(cat_df)
            if "diag_cols_preset" in st.session_state and not st.session_state["diag_cols_preset"]:
                del st.session_state["diag_cols_preset"]
            preset = st.session_state.get("diag_cols_preset")
            default_cols = preset if (preset and all(c in cat_df.columns for c in preset)) else (text_cols_default or list(cat_df.columns)[:5])

            cols = st.multiselect(
                "Columns to search in:",
                options=list(cat_df.columns),
                default=default_cols,
                key="search_cols",
            )
            search_all = st.checkbox("Search ALL columns (ignore selection)", value=False, key="search_all")
            fallback_combine = st.checkbox("Fallback: combine selected columns into one field", value=True, key="search_fallback")

            q = st.text_input("Keyword or phrase", key="search_q")
            mode = st.selectbox("Match mode", ["Contains (case-insensitive)", "Exact (case-insensitive)", "Regex"], index=0, key="search_mode")
            max_rows = st.slider("Max rows to show", 10, 2000, 200, key="search_max_rows")

            # Quick Probe
            quick_probe_clicked = st.button("Quick Probe (per-column matches)", key="search_quick_probe_btn")
            if quick_probe_clicked:
                if not q:
                    st.warning("Enter a query first.")
                else:
                    use_cols = list(cat_df.columns) if search_all or not cols else cols
                    out = []
                    nonempty = []
                    for c in use_cols:
                        s = _clean_hidden_series(cat_df[c])
                        try:
                            if mode == "Contains (case-insensitive)":
                                n = s.str.contains(q, case=False, na=False, regex=False).sum()
                            elif mode == "Exact (case-insensitive)":
                                n = (s.str.strip().str.lower() == q.strip().str.lower()).sum()
                            else:
                                n = s.str.contains(q, case=False, na=False, regex=True).sum()
                        except Exception:
                            n = 0
                        out.append((c, int(n)))
                        nonempty.append((c, int((s != "").sum())))
                    st.write("**Matches per column (top 25):**")
                    st.dataframe(pd.DataFrame(out, columns=["column","matches"]).sort_values("matches", ascending=False).head(25), use_container_width=True)
                    st.write("**Non-empty cells per column (top 25):**")
                    st.dataframe(pd.DataFrame(nonempty, columns=["column","non_empty"]).sort_values("non_empty", ascending=False).head(25), use_container_width=True)

            # Search
            search_clicked = st.button("Search", type="primary", key="search_go_btn")
            if search_clicked:
                if not q:
                    st.warning("Provide a query.")
                else:
                    try:
                        use_cols = list(cat_df.columns) if search_all or not cols else cols
                        df = cat_df.copy()
                        for c in use_cols:
                            df[c] = _clean_hidden_series(df[c])

                        if fallback_combine:
                            combo = df[use_cols].agg(" ".join, axis=1)
                            if mode == "Contains (case-insensitive)":
                                mask = combo.str.contains(q, case=False, na=False, regex=False)
                            elif mode == "Exact (case-insensitive)":
                                mask = (combo.str.strip().str.lower() == q.strip().lower())
                            else:
                                mask = combo.str.contains(q, case=False, na=False, regex=True)
                        else:
                            mask = pd.Series(False, index=df.index)
                            for c in use_cols:
                                s = df[c]
                                if mode == "Contains (case-insensitive)":
                                    mask |= s.str.contains(q, case=False, na=False, regex=False)
                                elif mode == "Exact (case-insensitive)":
                                    mask |= (s.str.strip().str.lower() == q.strip().lower())
                                else:
                                    mask |= s.str.contains(q, case=False, na=False, regex=True)

                        total = int(mask.sum())
                        res = df[mask].head(max_rows)
                        st.success(f"Found {total:,} rows; showing {len(res):,}.")
                        if total == 0:
                            st.info("No matches. Use Quick Probe, try Search ALL columns, or switch modes.")
                        if not res.empty:
                            show_cols = use_cols if len(use_cols) <= 12 else use_cols[:12]
                            st.write("**Sample hits (chosen columns):**")
                            st.dataframe(res[show_cols].head(20), use_container_width=True)
                            st.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                                               file_name="ajd_search_results.csv", mime="text/csv", key="search_dl")
                        else:
                            st.dataframe(res, use_container_width=True)
                    except Exception as e:
                        st.error(f"Search error: {e}")

    # ----- TAB 2: Topic Overlap -----
    with compare_tab:
        st.subheader("Compare Your Project Topics vs AJD Topics")
        topics_df = load_topics_df()
        uploaded = st.file_uploader("Upload your project file (CSV with a topics column, or JSON list)", type=["csv", "json"], key="cmp_upload")
        topics_col = st.text_input("If CSV, name of the column that contains topics (e.g., 'topics')", value="topics", key="cmp_topics_col")

        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    p = pd.read_csv(uploaded, dtype=str, na_filter=False, keep_default_na=False,
                                    engine="python", on_bad_lines="skip")
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
                        st.error(f"Column '{topics_col}' not found in your CSV.")
                        bag = []
                else:
                    data = json.loads(uploaded.getvalue().decode("utf-8", errors="ignore"))
                    items = data if isinstance(data, list) else data.get("items", [])
                    def explode_topics_str(text: str) -> List[str]:
                        parts = re.split(r"[;,/|·•\-–—]+", str(text))
                        keepout = {"and","the","for","with","from","into","about","film","doc","docs","docu","series","episode","season","ajd","al jazeera","aljazeera"}
                        return [t.strip().lower() for t in parts if len(t.strip())>=3 and t.strip().lower() not in keepout]
                    bag: List[str] = []
                    for item in items:
                        topics_val = item.get("topics") or item.get("tags") or ""
                        if isinstance(topics_val, list):
                            for t in topics_val:
                                bag.extend(explode_topics_str(t))
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
                    st.json({"ajd_unique_topics": len(ajd_topics),
                             "project_unique_topics": len(proj_topics),
                             "overlap_count": len(overlap)})

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Overlap topics**")
                        st.dataframe(pd.DataFrame({"topic": overlap}), use_container_width=True)
                    with c2:
                        st.markdown("**Project-only topics (white space)**")
                        st.dataframe(pd.DataFrame({"topic": only_proj}), use_container_width=True)

                    st.download_button("Download overlap (CSV)",
                        pd.DataFrame({"topic": overlap}).to_csv(index=False).encode("utf-8"),
                        file_name="overlap_topics.csv", key="cmp_dl_overlap")
                    st.download_button("Download project-only (CSV)",
                        pd.DataFrame({"topic": only_proj}).to_csv(index=False).encode("utf-8"),
                        file_name="project_only_topics.csv", key="cmp_dl_projonly")
            except Exception as e:
                st.error(f"Error processing upload: {e}")

    # ----- TAB 3: Similarity Matches (TF-IDF) -----
    with similar_tab:
        st.subheader("Find Closest AJD Matches for Your Films (TF-IDF)")
        if not SKLEARN_OK:
            st.info("Install scikit-learn to enable this tab (add `scikit-learn>=1.3,<2.0` to requirements.txt).")
        else:
            cat_df = load_catalogue_df()
            if cat_df.empty:
                st.info("Catalogue missing. Add CSVs to /data and reload.")
            else:
                text_cols_default = infer_text_columns(cat_df)
                preset = st.session_state.get("diag_cols_preset")
                default_cols = preset if (preset and all(c in cat_df.columns for c in preset)) else (text_cols_default or list(cat_df.columns)[:5])

                cols = st.multiselect("AJD text columns to use", options=list(cat_df.columns), default=default_cols, key="sim_cols")
                films = st.text_area("Paste your films as JSON list (each item: title, description, topics optional)",
                    value='[\n  {"title":"Film A","description":"Inside the rise of tech startups in MENA."},\n  {"title":"Film B","description":"Lives split across continents."}\n]',
                    key="sim_films_text")
                compute_clicked = st.button("Compute matches", type="primary", key="sim_compute_btn")
                if compute_clicked:
                    try:
                        films_data = json.loads(films)
                        proj_texts = [(i.get("title","Untitled"), (i.get("title","")+" "+i.get("description"," ")).strip())
                                      for i in films_data]
                        ajd_texts = (cat_df[cols].fillna("").astype(str).agg(" ".join, axis=1)).tolist()
                        # Build TF-IDF & similarity
                        vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
                        X = vectorizer.fit_transform(ajd_texts + [t for _, t in proj_texts])
                        n_ajd = len(ajd_texts)
                        sims = cosine_similarity(X[n_ajd:], X[:n_ajd])

                        for row_idx, (title, _) in enumerate(proj_texts):
                            st.markdown(f"#### {title}")
                            row = sims[row_idx]
                            top_idx = row.argsort()[::-1][:10]
                            rows = []
                            for j in top_idx:
                                score = float(row[j]); ajd_row = cat_df.iloc[j]
                                rows.append({"score": round(score, 4), **ajd_row.to_dict()})
                            out_df = pd.DataFrame(rows)
                            st.dataframe(out_df, use_container_width=True)
                            safe_name = re.sub(r"[^A-Za-z0-9]+","_", title.lower())
                            st.download_button(f"Download matches for {title}",
                                out_df.to_csv(index=False).encode("utf-8"),
                                file_name=f"matches_{safe_name}.csv", key=f"sim_dl_{row_idx}")
                    except Exception as e:
                        st.error(f"Invalid JSON or error computing similarities: {e}")

   # ----- TAB 4: Logline Suggestions -----
# --- ensure tabs exist before using logline_tab ---
if "logline_tab" not in locals():
    search_tab, compare_tab, similar_tab, logline_tab, diag_tab = st.tabs([
        "🔎 Search AJD Catalogue",
        "🔁 Topic Overlap",
        "🧭 Similarity Matches",
        "🪄 Logline Suggestions",
        "🧰 Diagnostics",
    ])
# ----- TAB 4: Logline Suggestions -----
# (Place this exactly where your previous Loglines tab code was.)
with logline_tab:
    st.subheader("Suggest Strong Loglines")
    st.caption("Generates both a tight tagline and a detailed commissioning logline (+ an optional angle note).")

    # ---- Inputs
    seed = st.text_area(
        "Describe your film (topic, setting, main character, access, tension) — 2–5 lines.",
        key="log_seed",
        placeholder=(
            "Ex: A Cairo paramedic livestreams night shifts; the feed builds a following, "
            "but reveals a black-market oxygen ring and a choice that could cost him his license—or a life."
        ),
    )
    tone = st.selectbox("Tone / voice", ["Neutral", "Urgent", "Gritty", "Poetic"], index=1, key="log_tone")

    angle_pack = st.multiselect(
        "Angles to try",
        [
            "Character (inside-out)",
            "System vs Individual",
            "Investigation / Leak",
            "Exclusive Access",
            "Countdown / Clock",
            "Paradox / Irony",
            "Micro→Macro",
            "David vs Goliath",
            "Hidden Cost",
            "Archival Reframe",
        ],
        default=["Character (inside-out)", "System vs Individual", "Investigation / Leak", "Countdown / Clock"],
        key="log_angles",
    )

    # Length controls
    target_len_short = st.slider("Target words for TAGLINE", 8, 22, 15, key="log_len_short")
    target_len_detail = st.slider("Target words for DETAILED logline", 35, 80, 52, key="log_len_detail")
    add_angle_note = st.checkbox("Add one-line Angle Note (why this, why now)", value=True, key="log_add_note")
    n_variants = st.slider("How many variants?", 3, 12, 6, key="log_n")

    anti_cliche = st.checkbox("Remove clichés (untold, explores, journey…)", value=True, key="log_anticliche")
    freshness_nudge = st.checkbox("Nudge for freshness vs common AJD framings", value=True, key="log_freshness")

    # ---- Helpers
    import re as _re

    def _strip_cliche(text: str) -> str:
        if not anti_cliche:
            return text
        replacements = {
            r"\buntold\b": "unreported",
            r"\bnever[- ]before[- ]seen\b": "rarely seen",
            r"\bexplores\b": "cuts into",
            r"\bjourney\b": "fight",
            r"\bshines a light on\b": "exposes",
            r"\bsheds light on\b": "exposes",
            r"\bdelves into\b": "drives into",
            r"\bheart[- ]wrenching\b": "stark",
            r"\bgripping\b": "high-stakes",
            r"\bintimate portrait\b": "close-quarters view",
        }
        out = text
        for pat, rep in replacements.items():
            out = _re.sub(pat, rep, out, flags=_re.I)
        return out

    def _apply_tone(text: str, tone: str) -> str:
        t = text
        if tone == "Urgent":
            if " — " in t:
                t = t.replace(" — ", " — now, ")
            else:
                t = t.rstrip(".") + ". Now."
        elif tone == "Gritty":
            t = (
                t.replace("choice", "reckoning")
                 .replace("secret", "black-market")
                 .replace("network", "machine")
            )
        elif tone == "Poetic":
            t = (
                t.replace(" — ", ", ")
                 .replace(" vs ", " against ")
                 .replace("fight", "tremor")
                 .replace("exposes", "uncovers what we refuse to see")
            )
        return t

    def _tighten_to_words(text: str, max_words: int) -> str:
        words = text.split()
        out = " ".join(words[:max_words]).rstrip(",;:•— ")
        return (out + ".") if not out.endswith(".") else out

    def _fresh_nudge(seed_txt: str, base: str) -> str:
        if not freshness_nudge:
            return base
        hints = [
            " told through a single 36-hour window",
            " anchored in one apartment block’s WhatsApp audio",
            " using receipts, repair slips, and one leaked invoice",
            " from the perspective of the least powerful worker on shift",
            " limited to places where cameras usually aren’t allowed",
            " where every claim must be proven on camera",
        ]
        if any(k in base.lower() for k in ["36-hour", "invoice", "whatsapp", "leaked", "apartment", "proven"]):
            return base
        return base.rstrip(".") + hints[hash(seed_txt) % len(hints)] + "."

    # Angular seeds (single-sentence base ideas)
    def _angle_lines(seed_txt: str):
        s = seed_txt.strip()
        return [
            ("Character (inside-out)",
             f"When {s} collides with a rule he can no longer obey, one decision redraws the line between saving face and saving lives."),
            ("System vs Individual",
             f"A frontline worker trapped inside {s} tests how far a system will bend before it breaks—and who pays when it doesn’t."),
            ("Investigation / Leak",
             f"A leak cracks open {s}, and the paper trail forces a choice: expose the scheme or become another silent link."),
            ("Exclusive Access",
             f"Inside {s}, cameras keep rolling where they’re usually shut: what happens when the gatekeepers can’t edit the truth."),
            ("Countdown / Clock",
             f"In {s}, each hour raises the price of staying quiet—until the deadline hits and someone’s oxygen runs out."),
            ("Paradox / Irony",
             f"{s} turns attention into risk: the audience that makes the hero untouchable also makes him a target."),
            ("Micro→Macro",
             f"A tiny decision inside {s} maps the wiring of a bigger crisis—how small hands move a large machine."),
            ("David vs Goliath",
             f"Against {s}, one underpaid insider finds the only weapon Goliath can’t block: proof."),
            ("Hidden Cost",
             f"{s} looks cheaper than the fix—until the bill arrives in bodies, licenses, and sleep."),
            ("Archival Reframe",
             f"Using old calls, receipts and cached clips, {s} plays back a scandal that was hiding in plain sight."),
        ]

    # Expand a base idea into a 2–3 sentence commissioning logline (40–80 words)
    def _expand_to_detailed(seed_txt: str, base: str, tone: str, max_words: int) -> str:
        # Structure: WHO/WHERE + ACCESS → STAKES/CHOICE → ANGLE/FRESHNESS
        who_where = f"{seed_txt.strip().rstrip('.')}. " if seed_txt.strip() else ""
        stakes = (
            "As the trail sharpens, the protagonist must choose between protecting people and protecting themselves, "
            "knowing either path could end a career—or a life. "
        )
        angle = "Shot with close-quarters access and proof-led scenes, the film tests what truth looks like when the system prefers silence."
        detailed = (who_where + stakes + angle).strip()

        # Anti-cliché + tone + freshness, then tighten to target words
        detailed = _strip_cliche(detailed)
        detailed = _apply_tone(detailed, tone)
        detailed = _fresh_nudge(seed_txt, detailed)
        detailed = _tighten_to_words(detailed, max_words)
        return detailed

    # Build both: tagline (short) and detailed (expanded)
    def build_loglines(seed_txt: str, tone: str, packs: list[str], n: int, w_short: int, w_detail: int, add_note: bool):
        if not seed_txt.strip():
            return []

        pool = [p for p in _angle_lines(seed_txt) if p[0] in packs] or _angle_lines(seed_txt)
        out = []
        i = 0
        while len(out) < n:
            label, draft = pool[i % len(pool)]

            # Base → anti-cliché → tone → freshness
            tagline = _strip_cliche(draft)
            tagline = _apply_tone(tagline, tone)
            tagline = _fresh_nudge(seed_txt, tagline)
            tagline = _tighten_to_words(tagline, w_short)

            detailed = _expand_to_detailed(seed_txt, draft, tone, w_detail)

            note = ""
            if add_note:
                notes = {
                    "Character (inside-out)": "Angle: character-led, decisions on-camera, access-driven credibility.",
                    "System vs Individual": "Angle: policy-to-person pipeline; consequence lands on one body.",
                    "Investigation / Leak": "Angle: paper/voice-note trail; verifiable, document-forward scenes.",
                    "Exclusive Access": "Angle: cameras in no-go rooms; procedural realism.",
                    "Countdown / Clock": "Angle: time pressure; natural act breaks from deadlines.",
                    "Paradox / Irony": "Angle: audience attention as liability; safety vs visibility.",
                    "Micro→Macro": "Angle: one case explains many; blueprint of a wider crisis.",
                    "David vs Goliath": "Angle: asymmetry of power; the weapon is proof.",
                    "Hidden Cost": "Angle: externalities made visible; real prices paid in bodies/time.",
                    "Archival Reframe": "Angle: receipts/archives reshape the present; past in present tense.",
                }
                note = notes.get(label, "")

            out.append({
                "angle": label,
                "tagline": tagline,
                "detailed": detailed,
                "note": note,
            })
            i += 1
        return out

    # ---- Generate
    gen_clicked = st.button("Generate loglines", type="primary", key="log_gen_btn")
    if gen_clicked:
        if not seed.strip():
            st.warning("Please add a short seed.")
        else:
            rows = build_loglines(
                seed_txt=seed,
                tone=tone,
                packs=angle_pack,
                n=n_variants,
                w_short=target_len_short,
                w_detail=target_len_detail,
                add_note=add_angle_note,
            )

            if not rows:
                st.info("No output. Try different angles or increase word targets.")
            else:
                # Render
                for idx, r in enumerate(rows, 1):
                    st.markdown(f"### {idx}. {r['angle']}")
                    st.write(f"**Tagline:** {r['tagline']}")
                    st.write(f"**Detailed:** {r['detailed']}")
                    if add_angle_note and r["note"]:
                        st.caption(r["note"])
                    st.divider()

                # Download
                import io
                buf = io.StringIO()
                for idx, r in enumerate(rows, 1):
                    buf.write(f"{idx}. {r['angle']}\n")
                    buf.write(f"   Tagline: {r['tagline']}\n")
                    buf.write(f"   Detailed: {r['detailed']}\n")
                    if add_angle_note and r["note"]:
                        buf.write(f"   Note: {r['note']}\n")
                    buf.write("\n")
                st.download_button(
                    "Download all loglines (.txt)",
                    buf.getvalue().encode("utf-8"),
                    file_name="loglines_short_and_detailed.txt",
                    key="log_dl",
                )


    # ----- TAB 5: Diagnostics -----
    with diag_tab:
        st.subheader("Diagnostics")
        files = sorted(glob.glob(str(DATA_DIR / "*")))
        st.write("**/data files**:", files)

        p_cat = DATA_DIR / "ajd_catalogue_raw.csv"
        if p_cat.exists():
            st.write(f"Catalogue file size: {p_cat.stat().st_size:,} bytes")
        else:
            st.info("Catalogue merged CSV missing")

        cat_df_preview = load_catalogue_df()
        st.write(f"Catalogue DataFrame shape: {cat_df_preview.shape[0]:,} rows × {cat_df_preview.shape[1]} cols")
        if not cat_df_preview.empty:
            st.write("**Column names (first 30):**", list(cat_df_preview.columns[:30]))
            st.write("**Head(10):**")
            st.dataframe(cat_df_preview.head(10), use_container_width=True)
            texty = [c for c in cat_df_preview.columns if re.search(r"(name|title|synopsis|series|english|arabic|desc|topic)", c, re.I)]
            st.write("**Guessed text columns:**", texty if texty else "(none)")
        else:
            st.warning("DataFrame is empty after loading. This usually means delimiter/encoding issues or an empty file.")

        # Column preset for Search tab
        st.markdown("### Column preset for Search tab")
        if not cat_df_preview.empty:
            preset_cols = st.multiselect(
                "Columns to search in (preset for Search tab)",
                options=list(cat_df_preview.columns),
                default=texty or list(cat_df_preview.columns)[:5],
                key="diag_cols_preset",
            )
            st.caption("This preset is saved in session; the Search tab will use it automatically.")
        else:
            st.info("Load a catalogue first to configure the column preset.")

        # Upload/Replace catalogue CSV (session-only)
        st.markdown("### Upload / Replace catalogue CSV (session-only)")
        up = st.file_uploader("Upload a full catalogue CSV (UTF-8). Not persisted on redeploy.", type=["csv"], key="diag_upload_catalogue")
        if up is not None:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(DATA_DIR / "ajd_catalogue_raw.csv", "wb") as f:
                f.write(up.getbuffer())
            load_catalogue_df.clear()
            st.success("Uploaded. Click **Reload data / clear cache** in the sidebar, then reopen Diagnostics.")

        # Maintenance
        st.markdown("### Maintenance")
        force_merge = st.button("Force re-merge from parts (ajd_catalogue_raw.part*.csv)", key="diag_force_merge_btn")
        show_parts = st.button("Show part files", key="diag_show_parts_btn")
        if force_merge:
            merged = DATA_DIR / "ajd_catalogue_raw.csv"
            try:
                if merged.exists():
                    merged.unlink()
                ok = _merge_chunked_csv(str(DATA_DIR / "ajd_catalogue_raw.part*.csv"), merged)
                load_catalogue_df.clear()
                if ok:
                    st.success("Re-merged successfully. Click **Reload data / clear cache** in the sidebar.")
                else:
                    st.warning("No part files found matching pattern.")
            except Exception as e:
                st.error(f"Re-merge failed: {e}")
        if show_parts:
            st.write(sorted(glob.glob(str(DATA_DIR / "ajd_catalogue_raw.part*.csv"))))

    st.markdown("---")
    st.caption("© 2025 ICON Studio — AJD Topic Explorer Dashboard. Add `streamlit` to requirements.txt and run: `streamlit run app.py`")


if __name__ == "__main__":
    main()
