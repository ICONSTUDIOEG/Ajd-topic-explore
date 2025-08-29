# app.py — AJD Topic Explorer (Bilingual EN/AR) — 2025-08
# Features:
# - Robust CSV loaders + natural-sort merge of part files
# - Bilingual UI (English / العربية) with RTL when Arabic is selected
# - Tabs: Search, Compare, Similarity (optional), Loglines (strand presets + short+ detailed), Diagnostics
# - Unique widget keys to avoid Streamlit KeyError; Reset UI state; st.rerun()

import re, json, glob, os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
import os

# Optional (Similarity tab). App still runs if sklearn isn't available.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Attempt to import OpenAI if available; used for AI-powered loglines
from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)  # optional, defaults to reading from environment

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt_en}],
    max_tokens=800,
    temperature=0.7,
)
content = response.choices[0].message.content

APP_TITLE = "AJD Topic Explorer — بحث ثنائي اللغة"
DATA_DIR = Path("data")

# ============================ Bilingual helpers ============================

def get_lang() -> str:
    return st.session_state.get("sb_lang", "English")

def L(en: str, ar: str) -> str:
    return ar if get_lang() == "العربية" else en

def apply_rtl_if_arabic():
    if get_lang() == "العربية":
        st.markdown(
            """
            <style>
            html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] * {
                direction: rtl;
                text-align: right;
                font-family: "Noto Naskh Arabic", "Amiri", "Tahoma", sans-serif !important;
            }
            [data-testid="stTable"] thead tr th, [data-testid="stDataFrame"] thead tr th {
                text-align: right !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# ============================ Data helpers ============================

def _natsort_key(path: str):
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", base)]

def _merge_chunked_csv(pattern: str, merged_path: Path) -> bool:
    """Merge chunked CSVs (part01, part02, part3_01, ...) using natural sort."""
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
    """Load topics CSV (or auto-merge parts). Returns columns: topic, count (strings)."""
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
    cols_lower = [c.strip().lower() for c in df.columns]
    if "topic" in cols_lower and "count" in cols_lower:
        df = df.rename(columns={
            df.columns[cols_lower.index("topic")]: "topic",
            df.columns[cols_lower.index("count")]: "count",
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
        st.error(L(f"Could not read catalogue CSV. Last error: {last_err}",
                   f"تعذّر قراءة ملف الفهرس. آخر خطأ: {last_err}"))
        return pd.DataFrame()

    # Normalize column names
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # Drop fully empty columns and rows
    df = df.dropna(axis=1, how="all")
    if not df.empty:
        df = df.loc[~(df.apply(lambda r: "".join(map(str, r)).strip(), axis=1) == "")]

    # Quick shape (sidebar)
    st.sidebar.write(L("Loaded catalogue:","تم تحميل الفهرس:"),
                     f"{df.shape[0]:,} {L('rows','سطر')} × {df.shape[1]} {L('cols','عمود')}")
    return df

@st.cache_data(show_spinner=False)
def infer_text_columns(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"(title|synopsis|summary|topic|subject|tags|genre|theme|series|strand|category|type|desc|name|english|arabic|اسم|عنوان|ملخ|موضوع|سلسلة|عربي)", re.I)
    return [c for c in df.columns if patt.search(str(c))]

def _clean_hidden_series(s: pd.Series) -> pd.Series:
    """Remove zero-width/bidi chars and normalize spaces."""
    ZW_PATTERN = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069]")
    WS_PATTERN = re.compile(r"\s+")
    return s.astype(str).fillna("").map(lambda x: WS_PATTERN.sub(" ", ZW_PATTERN.sub("", x)).strip())

# ============================ App ============================

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title("AJD Topic Explorer — لوحة ثنائية اللغة")
    st.caption(L(
        "Search the AJD catalogue, compare topics, find matches, and craft bilingual loglines.",
        "ابحث في فهرس AJD، قارن الموضوعات، اعثر على تطابقات، وأنشئ لوجلاين ثنائي اللغة."
    ))

    # ----- Sidebar -----
    with st.sidebar:
        st.header(L("Dataset","البيانات"))
        # Language switch + RTL
        ui_lang = st.selectbox("Language / اللغة", ["English", "العربية"], key="sb_lang")
        apply_rtl_if_arabic()

        topics_df = load_topics_df()
        cat_df = load_catalogue_df()
        st.write(f"**{L('Topics','الموضوعات')}**", f"{len(topics_df):,} {L('rows','سطر')}" if not topics_df.empty else L("— missing","— غير متوفر"))
        st.write(f"**{L('Catalogue','الفهرس')}**", f"{len(cat_df):,} {L('rows','سطر')}" if not cat_df.empty else L("— missing","— غير متوفر"))

        # Reset & Reload
        if st.button(L("Reset UI state (fix KeyError)","إعادة ضبط واجهة المستخدم"), key="sb_reset_state"):
            st.session_state.clear()
            st.rerun()

        if st.button(L("Reload data / clear cache","تحديث البيانات / مسح الذاكرة المؤقتة"), key="sb_reload"):
            load_topics_df.clear(); load_catalogue_df.clear(); infer_text_columns.clear()
            st.rerun()

        if topics_df.empty or cat_df.empty:
            st.warning(L(
                "If merged CSVs are missing, the app will merge `/data/...partNN.csv` on first run.",
                "إذا كانت ملفات الدمج غير موجودة، سيقوم التطبيق بدمج `/data/...partNN.csv` في أول تشغيل."
            ))

    # ----- Tabs -----
    search_tab, compare_tab, similar_tab, logline_tab, diag_tab = st.tabs([
        L("🔎 Search AJD Catalogue","🔎 البحث في فهرس AJD"),
        L("🔁 Topic Overlap","🔁 تقاطع الموضوعات"),
        L("🧭 Similarity Matches","🧭 أقرب تطابقات"),
        L("🪄 Logline Suggestions","🪄 اقتراح لوجلاين"),
        L("🧰 Diagnostics","🧰 التشخيص"),
    ])

    # ============================ TAB 1: Search ============================
    with search_tab:
        st.subheader(L("Search AJD Catalogue","البحث في فهرس AJD"))
        cat_df = load_catalogue_df()
        if cat_df.empty:
            st.info(L("Upload/commit your catalogue CSV parts to `data/` and reload.",
                      "ارفع ملفات CSV المجزّأة داخل `data/` ثم حدّث التطبيق."))
        else:
            text_cols_default = infer_text_columns(cat_df)
            if "diag_cols_preset" in st.session_state and not st.session_state["diag_cols_preset"]:
                del st.session_state["diag_cols_preset"]
            preset = st.session_state.get("diag_cols_preset")
            default_cols = preset if (preset and all(c in cat_df.columns for c in preset)) else (text_cols_default or list(cat_df.columns)[:5])

            cols = st.multiselect(L("Columns to search in:","الأعمدة المراد البحث فيها:"),
                                  options=list(cat_df.columns), default=default_cols, key="search_cols")
            search_all = st.checkbox(L("Search ALL columns (ignore selection)","البحث في جميع الأعمدة"), value=False, key="search_all")
            fallback_combine = st.checkbox(L("Fallback: combine selected columns into one field","دمج الأعمدة المحددة في حقل واحد"), value=True, key="search_fallback")

            q = st.text_input(L("Keyword or phrase","كلمة مفتاحية أو عبارة"), key="search_q")
            mode = st.selectbox(L("Match mode","نمط المطابقة"),
                                [L("Contains (case-insensitive)","يحتوي (بدون حساسية حالة)"),
                                 L("Exact (case-insensitive)","تطابق تام (بدون حساسية حالة)"),
                                 "Regex"], index=0, key="search_mode")
            max_rows = st.slider(L("Max rows to show","الحد الأقصى للعرض"), 10, 2000, 200, key="search_max_rows")

            def _contains(series_or_text, q_, mode_):
                if isinstance(series_or_text, pd.Series):
                    s = _clean_hidden_series(series_or_text)
                    try:
                        if mode_.startswith("Contains") or mode_.startswith("يحتوي"):
                            return s.str.contains(q_, case=False, na=False, regex=False)
                        elif mode_.startswith("Exact") or mode_.startswith("تطابق"):
                            return (s.str.strip().str.lower() == str(q_).strip().lower())
                        else:
                            return s.str.contains(q_, case=False, na=False, regex=True)
                    except Exception:
                        return pd.Series(False, index=s.index)
                else:
                    s = str(series_or_text)
                    if mode_.startswith("Contains") or mode_.startswith("يحتوي"):
                        return q_.lower() in s.lower()
                    elif mode_.startswith("Exact") or mode_.startswith("تطابق"):
                        return s.strip().lower() == str(q_).strip().lower()
                    else:
                        try:
                            return re.search(q_, s, flags=re.I) is not None
                        except Exception:
                            return False

            # Quick Probe
            if st.button(L("Quick Probe (per-column matches)","فحص سريع (حسب العمود)"), key="search_quick_probe_btn"):
                if not q:
                    st.warning(L("Enter a query first.","أدخل استعلامًا أولًا."))
                else:
                    use_cols = list(cat_df.columns) if search_all or not cols else cols
                    out, nonempty = [], []
                    for c in use_cols:
                        s = _clean_hidden_series(cat_df[c])
                        try:
                            if mode.startswith("Contains") or mode.startswith("يحتوي"):
                                n = s.str.contains(q, case=False, na=False, regex=False).sum()
                            elif mode.startswith("Exact") or mode.startswith("تطابق"):
                                n = (s.str.strip().str.lower() == str(q).strip().lower()).sum()
                            else:
                                n = s.str.contains(q, case=False, na=False, regex=True).sum()
                        except Exception:
                            n = 0
                        out.append((c, int(n)))
                        nonempty.append((c, int((s != "").sum())))
                    df_out = pd.DataFrame(out, columns=[L("column","العمود"), L("matches","التطابقات")]).sort_values(L("matches","التطابقات"), ascending=False)
                    df_non = pd.DataFrame(nonempty, columns=[L("column","العمود"), L("non_empty","غير فارغ")]).sort_values(L("non_empty","غير فارغ"), ascending=False)
                    st.write(L("**Matches per column (top 25):**","**عدد التطابقات لكل عمود (أعلى 25):**"))
                    st.dataframe(df_out.head(25), use_container_width=True)
                    st.write(L("**Non-empty cells per column (top 25):**","**الخلايا غير الفارغة لكل عمود (أعلى 25):**"))
                    st.dataframe(df_non.head(25), use_container_width=True)

            # Search
            if st.button(L("Search","بحث"), type="primary", key="search_go_btn"):
                if not q:
                    st.warning(L("Provide a query.","أدخل استعلامًا."))
                else:
                    try:
                        use_cols = list(cat_df.columns) if search_all or not cols else cols
                        df2 = cat_df.copy()
                        for c in use_cols:
                            df2[c] = _clean_hidden_series(df2[c])

                        if fallback_combine:
                            combo = df2[use_cols].agg(" ".join, axis=1)
                            mask = _contains(combo, q, mode)
                        else:
                            mask = pd.Series(False, index=df2.index)
                            for c in use_cols:
                                mask |= _contains(df2[c], q, mode)

                        total = int(mask.sum())
                        res = df2[mask].head(max_rows)
                        st.success(L(f"Found {total:,} rows; showing {len(res):,}.",
                                     f"تم العثور على {total:,} صفًا؛ المعروض {len(res):,}."))
                        if total == 0:
                            st.info(L("No matches. Use Quick Probe, try Search ALL columns, or switch modes.",
                                      "لا توجد نتائج. جرّب الفحص السريع أو البحث في جميع الأعمدة أو غيّر نمط المطابقة."))
                        if not res.empty:
                            show_cols = use_cols if len(use_cols) <= 12 else use_cols[:12]
                            st.write(L("**Sample hits (chosen columns):**","**عينات نتائج (الأعمدة المختارة):**"))
                            st.dataframe(res[show_cols].head(20), use_container_width=True)
                            st.download_button(L("Download results (CSV)","تحميل النتائج (CSV)"),
                                               res.to_csv(index=False).encode("utf-8"),
                                               file_name="ajd_search_results.csv", mime="text/csv", key="search_dl")
                        else:
                            st.dataframe(res, use_container_width=True)
                    except Exception as e:
                        st.error(L(f"Search error: {e}", f"خطأ في البحث: {e}"))

    # ============================ TAB 2: Compare ============================
    with compare_tab:
        st.subheader(L("Compare Your Project Topics vs AJD Topics","مقارنة موضوعات مشروعك بموضوعات AJD"))
        topics_df = load_topics_df()
        uploaded = st.file_uploader(L("Upload your project file (CSV with a topics column, or JSON list)",
                                      "ارفع ملف مشروعك (CSV يحتوي عمود موضوعات أو JSON)"),
                                    type=["csv","json"], key="cmp_upload")
        topics_col = st.text_input(L("If CSV, name of the column that contains topics (e.g., 'topics')",
                                     "إن كان CSV، اسم العمود الذي يحتوي الموضوعات (مثل 'topics')"),
                                   value="topics", key="cmp_topics_col")

        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    p = pd.read_csv(uploaded, dtype=str, na_filter=False, keep_default_na=False, engine="python", on_bad_lines="skip")
                    st.write(L("Your CSV preview:","معاينة CSV:")); st.dataframe(p.head(10), use_container_width=True)
                    def explode_topics_str(text: str) -> List[str]:
                        parts = re.split(r"[;,/|·•\-–—]+", str(text))
                        keepout = {"and","the","for","with","from","into","about","film","doc","docs","docu","series","episode","season","ajd","al jazeera","aljazeera",
                                   "و","في","من","عن","إلى","على","فيلم","وثائقي","حلقة","سلسلة","الجزيرة","AJD"}
                        return [t.strip().lower() for t in parts if len(t.strip())>=3 and t.strip().lower() not in keepout]
                    bag: List[str] = []
                    if topics_col in p.columns:
                        for v in p[topics_col].fillna(""):
                            bag.extend(explode_topics_str(v))
                    else:
                        st.error(L(f"Column '{topics_col}' not found in your CSV.",
                                   f"العمود '{topics_col}' غير موجود في CSV."))
                        bag = []
                else:
                    data = json.loads(uploaded.getvalue().decode("utf-8", errors="ignore"))
                    items = data if isinstance(data, list) else data.get("items", [])
                    def explode_topics_str(text: str) -> List[str]:
                        parts = re.split(r"[;,/|·•\-–—]+", str(text))
                        keepout = {"and","the","for","with","from","into","about","film","doc","docs","docu","series","episode","season","ajd","al jazeera","aljazeera",
                                   "و","في","من","عن","إلى","على","فيلم","وثائقي","حلقة","سلسلة","الجزيرة","AJD"}
                        return [t.strip().lower() for t in parts if len(t.strip())>=3 and t.strip().lower() not in keepout]
                    bag: List[str] = []
                    for item in items:
                        topics_val = item.get("topics") or item.get("tags") or ""
                        if isinstance(topics_val, list):
                            for t in topics_val: bag.extend(explode_topics_str(t))
                        else:
                            bag.extend(explode_topics_str(topics_val))

                if topics_df.empty:
                    st.error(L("AJD topics dataset is missing.","بيانات موضوعات AJD غير متوفرة."))
                else:
                    ajd_topics = set(topics_df["topic"].astype(str).str.lower())
                    proj_topics = set(bag)
                    overlap = sorted(list(ajd_topics & proj_topics))
                    only_proj = sorted(list(proj_topics - ajd_topics))

                    st.markdown(L("### Summary","### الملخّص"))
                    st.json({L("ajd_unique_topics","عدد موضوعات AJD الفريدة"): len(ajd_topics),
                             L("project_unique_topics","عدد موضوعات مشروعك"): len(proj_topics),
                             L("overlap_count","عدد المتقاطعات"): len(overlap)})

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(L("**Overlap topics**","**موضوعات متقاطعة**"))
                        st.dataframe(pd.DataFrame({"topic": overlap}), use_container_width=True)
                    with c2:
                        st.markdown(L("**Project-only topics (white space)**","**موضوعات متاحة لمشروعك فقط**"))
                        st.dataframe(pd.DataFrame({"topic": only_proj}), use_container_width=True)

                    st.download_button(L("Download overlap (CSV)","تحميل المتقاطع (CSV)"),
                        pd.DataFrame({"topic": overlap}).to_csv(index=False).encode("utf-8"),
                        file_name="overlap_topics.csv", key="cmp_dl_overlap")
                    st.download_button(L("Download project-only (CSV)","تحميل (موضوعات مشروعك فقط)"),
                        pd.DataFrame({"topic": only_proj}).to_csv(index=False).encode("utf-8"),
                        file_name="project_only_topics.csv", key="cmp_dl_projonly")
            except Exception as e:
                st.error(L(f"Error processing upload: {e}", f"خطأ في معالجة التحميل: {e}"))

    # ============================ TAB 3: Similarity (optional) ============================
    with similar_tab:
        st.subheader(L("Find Closest AJD Matches for Your Films (TF-IDF)","العثور على أقرب تطابقات لأفلامك (TF-IDF)"))
        if not SKLEARN_OK:
            st.info(L("Install scikit-learn to enable this tab (add `scikit-learn>=1.3,<2.0` to requirements.txt).",
                      "ثبّت scikit-learn لتفعيل هذه الصفحة (أضِف `scikit-learn>=1.3,<2.0`)."))
        else:
            cat_df = load_catalogue_df()
            if cat_df.empty:
                st.info(L("Catalogue missing. Add CSVs to /data and reload.",
                          "الفهرس غير متوفر. أضِف ملفات CSV إلى /data ثم حدّث."))
            else:
                text_cols_default = infer_text_columns(cat_df)
                preset = st.session_state.get("diag_cols_preset")
                default_cols = preset if (preset and all(c in cat_df.columns for c in preset)) else (text_cols_default or list(cat_df.columns)[:5])

                cols = st.multiselect(L("AJD text columns to use","أعمدة نصية لاستخدامها"), options=list(cat_df.columns), default=default_cols, key="sim_cols")
                films = st.text_area(L("Paste your films as JSON list (title, description, topics optional)",
                                       "ألصق أفلامك كقائمة JSON (عنوان، وصف، موضوعات اختياري)"),
                    value='[\n  {"title":"Film A","description":"Inside the rise of tech startups in MENA."},\n  {"title":"Film B","description":"Lives split across continents."}\n]',
                    key="sim_films_text")
                compute_clicked = st.button(L("Compute matches","احسب التطابقات"), type="primary", key="sim_compute_btn")
                if compute_clicked:
                    try:
                        films_data = json.loads(films)
                        proj_texts = [(i.get("title","Untitled"), (i.get("title","")+" "+i.get("description"," ")).strip())
                                      for i in films_data]
                        ajd_texts = (cat_df[cols].fillna("").astype(str).agg(" ".join, axis=1)).tolist()
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
                            safe_name = re.sub(r"[^A-Za-z0-9]+","_", str(title).lower())
                            st.download_button(L(f"Download matches for {title}","تحميل النتائج"),
                                out_df.to_csv(index=False).encode("utf-8"),
                                file_name=f"matches_{safe_name}.csv", key=f"sim_dl_{row_idx}")
                    except Exception as e:
                        st.error(L(f"Invalid JSON or error computing similarities: {e}",
                                   f"JSON غير صالح أو خطأ أثناء الحساب: {e}"))

    # ============================ TAB 4: Loglines (Bilingual + Presets) ============================
    with logline_tab:
        # UI labels depend on language
        st.subheader(L("Suggest Strong Loglines","إنشاء لوجلاين قوي"))
        st.caption(L(
            "Generates a tight tagline + a detailed commissioning logline, with presets & anti-cliché.",
            "ينتج شعارًا قصيرًا جذابًا + لوجلاين تفصيلي للعرض، مع قوالب جاهزة ومعالجة للكليشيهات."
        ))

        seed = st.text_area(
            L("Describe your film (topic, access, character, tension) — 2–5 lines.",
              "صف فيلمك (الموضوع، الوصول، الشخصية، التوتر) — سطران إلى خمسة."),
            key="log_seed",
            placeholder=L(
                "Ex: A Cairo paramedic livestreams night shifts; the feed exposes a black-market oxygen ring and a choice that could cost him his license—or a life.",
                "مثال: مسعف في القاهرة يبث نوبات الليل مباشرة؛ يكشف البث شبكة أكسجين سوداء وخيارًا قد يكلّفه رخصته — أو حياة."
            )
        )

        preset = st.selectbox(
            L("Strand preset (optional)","نمط البرنامج/السلسلة (اختياري)"),
            [
                L("Custom","مخصص"),
                L("Witness / Observational","ويتنس / رصدي"),
                L("Investigative (doc-forward)","تحقيقي (يعتمد الوثائق)"),
                L("Current Affairs (system vs individual)","شؤون جارية (النظام والفرد)"),
                L("Archive-led / Past in Present","أرشيفي / الماضي في الحاضر"),
                L("Exclusive Access","وصول حصري"),
                L("Countdown / Clock","عدّ تنازلي / سباق مع الوقت"),
            ],
            key="log_preset"
        )

        tone = st.selectbox(
            L("Tone / voice","النبرة / الأسلوب"),
            ["Neutral", "Urgent", "Gritty", "Poetic"] if get_lang()=="English" else ["محايد", "عاجل", "خشن", "شعري"],
            index=1, key="log_tone"
        )

        angle_pack = st.multiselect(
            L("Angles to try","زوايا السرد المقترحة"),
            (
                ["Character (inside-out)", "System vs Individual", "Investigation / Leak", "Exclusive Access",
                 "Countdown / Clock", "Paradox / Irony", "Micro→Macro", "David vs Goliath",
                 "Hidden Cost", "Archival Reframe"]
                if get_lang()=="English" else
                ["شخصية (من الداخل للخارج)", "النظام مقابل الفرد", "تحقيق / تسريب", "وصول حصري",
                 "عدّ تنازلي / الساعة", "مفارقة / سخرية", "من الصغير إلى الكبير", "داود وجالوت",
                 "الكلفة الخفية", "إعادة تأطير أرشيفي"]
            ),
            default=None, key="log_angles"
        )

        target_len_short = st.slider(L("Target words for TAGLINE","عدد كلمات الشعار القصير"), 8, 22, 15, key="log_len_short")
        target_len_detail = st.slider(L("Target words for DETAILED logline","عدد كلمات اللوجلاين التفصيلي"), 35, 90, 55, key="log_len_detail")
        add_angle_note = st.checkbox(L("Add one-line Angle Note (why this, why now)","إضافة سطر توضيحي عن الزاوية (لماذا هذا ولماذا الآن)"), value=True, key="log_add_note")
        n_variants = st.slider(L("How many variants?","عدد الصيغ؟"), 3, 12, 6, key="log_n")

        anti_cliche = st.checkbox(L("Remove clichés (untold, explores, journey…)","إزالة الكليشيهات (غير مسبوق، يستكشف، رحلة...)"), value=True, key="log_anticliche")
        freshness_nudge = st.checkbox(L("Nudge for freshness vs common AJD framings","تعزيز الجِدة مقابل القوالب الشائعة"), value=True, key="log_freshness")

        # Optionally use ChatGPT/OpenAI to generate loglines instead of the template-based method.
        use_ai = st.checkbox(
            L(
                "Use ChatGPT for loglines (requires OPENAI_API_KEY)",
                "استخدم ChatGPT لإنشاء اللوجلاين (يتطلب OPENAI_API_KEY)"
            ),
            value=False,
            key="log_use_ai"
        )

        # Preset application (evaluated at generation time)
        def preset_defaults(name: str):
            if get_lang() == "العربية":
                if name == "ويتنس / رصدي":
                    return {"tone": "محايد", "angles": ["شخصية (من الداخل للخارج)", "وصول حصري", "من الصغير إلى الكبير"], "short": 14, "detail": 60}
                if name == "تحقيقي (يعتمد الوثائق)":
                    return {"tone": "عاجل", "angles": ["تحقيق / تسريب", "الكلفة الخفية", "النظام مقابل الفرد", "إعادة تأطير أرشيفي"], "short": 14, "detail": 60}
                if name == "شؤون جارية (النظام والفرد)":
                    return {"tone": "عاجل", "angles": ["النظام مقابل الفرد", "عدّ تنازلي / الساعة", "داود وجالوت"], "short": 15, "detail": 55}
                if name == "أرشيفي / الماضي في الحاضر":
                    return {"tone": "محايد", "angles": ["إعادة تأطير أرشيفي", "من الصغير إلى الكبير"], "short": 15, "detail": 58}
                if name == "وصول حصري":
                    return {"tone": "محايد", "angles": ["وصول حصري", "شخصية (من الداخل للخارج)"], "short": 14, "detail": 55}
                if name == "عدّ تنازلي / سباق مع الوقت":
                    return {"tone": "عاجل", "angles": ["عدّ تنازلي / الساعة", "النظام مقابل الفرد"], "short": 13, "detail": 50}
            else:
                if name == "Witness / Observational":
                    return {"tone": "Neutral", "angles": ["Character (inside-out)", "Exclusive Access", "Micro→Macro"], "short": 14, "detail": 60}
                if name == "Investigative (doc-forward)":
                    return {"tone": "Urgent", "angles": ["Investigation / Leak", "Hidden Cost", "System vs Individual", "Archival Reframe"], "short": 14, "detail": 60}
                if name == "Current Affairs (system vs individual)":
                    return {"tone": "Urgent", "angles": ["System vs Individual", "Countdown / Clock", "David vs Goliath"], "short": 15, "detail": 55}
                if name == "Archive-led / Past in Present":
                    return {"tone": "Neutral", "angles": ["Archival Reframe", "Micro→Macro"], "short": 15, "detail": 58}
                if name == "Exclusive Access":
                    return {"tone": "Neutral", "angles": ["Exclusive Access", "Character (inside-out)"], "short": 14, "detail": 55}
                if name == "Countdown / Clock":
                    return {"tone": "Urgent", "angles": ["Countdown / Clock", "System vs Individual"], "short": 13, "detail": 50}
            return None

        _pd = preset_defaults(preset)
        tone_used = _pd["tone"] if _pd else tone
        angles_used = _pd["angles"] if _pd else (angle_pack or [])
        short_used = _pd["short"] if _pd else target_len_short
        detail_used = _pd["detail"] if _pd else target_len_detail

        # ---- Helpers for loglines ----
        def _strip_cliche(text: str) -> str:
            if not anti_cliche:
                return text
            if get_lang() == "العربية":
                repl = {
                    r"\bغير مسبوق\b": "نادر",
                    r"\bيسلط الضوء\b": "يكشف",
                    r"\bيسبر أغوار\b": "يغوص مباشرة في",
                    r"\bرحلة\b": "صراع",
                    r"\bحميمي\b": "قريب الملامس",
                    r"\bمؤثر\b": "عالي المخاطر",
                }
            else:
                repl = {
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
            for pat, rep in repl.items():
                out = re.sub(pat, rep, out, flags=re.I)
            return out

        def _apply_tone(text: str, t: str) -> str:
            s = text
            if get_lang() == "العربية":
                if t == "عاجل":
                    s = s.rstrip(".!؟") + " الآن."
                elif t == "خشن":
                    s = s.replace("اختيار", "محاسبة").replace("سر", "سوق سوداء")
                elif t == "شعري":
                    s = s.replace(" — ", "، ").replace(" ضد ", " أمام ")
            else:
                if t == "Urgent":
                    s = s.rstrip(".") + ". Now."
                elif t == "Gritty":
                    s = s.replace("choice", "reckoning").replace("secret", "black-market")
                elif t == "Poetic":
                    s = s.replace(" — ", ", ").replace(" vs ", " against ")
            return s

        def _tighten_to_words(text: str, max_words: int) -> str:
            words = text.split()
            out = " ".join(words[:max_words]).rstrip(",;:•— ")
            return out + "." if not out.endswith((".", "؟")) else out

        def _fresh_nudge(seed_txt: str, base: str) -> str:
            if not freshness_nudge:
                return base
            hints_en = [
                " told through a single 36-hour window",
                " anchored in one apartment block’s WhatsApp audio",
                " using receipts, repair slips, and one leaked invoice",
                " from the perspective of the least powerful worker on shift",
                " limited to places where cameras usually aren’t allowed",
                " where every claim must be proven on camera",
            ]
            hints_ar = [
                " تُروى ضمن نافذة زمنية واحدة مدتها ٣٦ ساعة",
                " ترتكز على مذكرات صوتية في واتساب لعمارة واحدة",
                " بوثائق: إيصالات، فواتير صيانة، وفاتورة مسرّبة",
                " من منظور أضعف عامل في الوردية",
                " بحدود أماكن لا يُسمح عادة بالتصوير فيها",
                " حيث يجب إثبات كل ادّعاء على الكاميرا",
            ]
            hints = hints_ar if get_lang()=="العربية" else hints_en
            low = base.lower()
            triggers = ["36-hour","whatsapp","leaked","invoice","proven","عمارة","مسرّبة","٣٦"]
            if any(t in low for t in triggers):
                return base
            return base.rstrip(".؟") + hints[hash(seed_txt) % len(hints)] + "."

        def _angle_lines(seed_txt: str):
            s = seed_txt.strip()
            if get_lang() == "العربية":
                return [
                    ("شخصية (من الداخل للخارج)", f"عندما يتقاطع {s} مع قاعدة لم يعد يحتملها، قرارٌ واحد يعيد رسم الحدّ بين حفظ الوجه وإنقاذ الأرواح."),
                    ("النظام مقابل الفرد", f"عامل في الصفّ الأول عالق داخل {s} يختبر إلى أي حدّ ينثني النظام قبل أن ينكسر — ومن يدفع حين لا ينثني."),
                    ("تحقيق / تسريب", f"تسريبٌ يفتح {s}، والسلسلة الورقية تفرض خيارًا: كشف الشبكة أم البقاء حلقة صامتة."),
                    ("وصول حصري", f"داخل {s} تبقى الكاميرات دائرة حيث تُطفأ عادة: ماذا يحدث حين لا يستطيع الحُرّاس تعديل الحقيقة."),
                    ("عدّ تنازلي / الساعة", f"في {s} كل ساعة ترفع ثمن الصمت — حتى يحين الموعد ويغيب الأكسجين."),
                    ("مفارقة / سخرية", f"{s} يحوّل الانتباه إلى خطر: الجمهور الذي يحمي بطله يجعله هدفًا."),
                    ("من الصغير إلى الكبير", f"تفصيل صغير داخل {s} يرسم خرائط أزمة أكبر — كيف تحرّك الأيدي الصغيرة آلةً كبيرة."),
                    ("داود وجالوت", f"في مواجهة {s} يجد عامل مُنهك السلاح الوحيد الذي لا يصدّه جالوت: الدليل."),
                    ("الكلفة الخفية", f"{s} يبدو أرخص من الحل — حتى تصل الفاتورة بالأجساد والرخصة والنوم."),
                    ("إعادة تأطير أرشيفي", f"بالمكالمات القديمة والإيصالات واللقطات المخبّأة يعيد {s} تشغيل فضيحة كانت أمام العيون."),
                ]
            else:
                return [
                    ("Character (inside-out)", f"When {s} collides with a rule he can no longer obey, one decision redraws the line between saving face and saving lives."),
                    ("System vs Individual", f"A frontline worker trapped inside {s} tests how far a system will bend before it breaks—and who pays when it doesn’t."),
                    ("Investigation / Leak", f"A leak cracks open {s}, and the paper trail forces a choice: expose the scheme or become another silent link."),
                    ("Exclusive Access", f"Inside {s}, cameras keep rolling where they’re usually shut: what happens when the gatekeepers can’t edit the truth."),
                    ("Countdown / Clock", f"In {s}, each hour raises the price of staying quiet—until the deadline hits and someone’s oxygen runs out."),
                    ("Paradox / Irony", f"{s} turns attention into risk: the audience that makes the hero untouchable also makes him a target."),
                    ("Micro→Macro", f"A tiny decision inside {s} maps the wiring of a bigger crisis—how small hands move a large machine."),
                    ("David vs Goliath", f"Against {s}, one underpaid insider finds the only weapon Goliath can’t block: proof."),
                    ("Hidden Cost", f"{s} looks cheaper than the fix—until the bill arrives in bodies, licenses, and sleep."),
                    ("Archival Reframe", f"Using old calls, receipts and cached clips, {s} plays back a scandal that was hiding in plain sight."),
                ]

        def _expand_to_detailed(seed_txt: str, tone_used: str, max_words: int) -> str:
            if get_lang() == "العربية":
                who = (seed_txt.strip().rstrip(".!؟") + ". ") if seed_txt.strip() else ""
                stakes = "ومع اقتراب الخيط، يجد البطل نفسه بين حماية الناس أو حماية نفسه — وكِلاهما ثمنه كارثة مهنية أو إنسانية. "
                angle = "بلقطات قريبة ودليل على الشاشة، يختبر الفيلم كيف تبدو الحقيقة حين يفضّل النظام الصمت."
                out = (who + stakes + angle).strip()
            else:
                who = (seed_txt.strip().rstrip(".") + ". ") if seed_txt.strip() else ""
                stakes = "As the trail sharpens, the protagonist must choose between protecting people and protecting themselves — either path could end a career or a life. "
                angle  = "Shot with close-quarters access and proof-led scenes, the film tests what truth looks like when the system prefers silence."
                out = (who + stakes + angle).strip()
            out = _strip_cliche(out)
            out = _apply_tone(out, tone_used)
            out = _fresh_nudge(seed_txt, out)
            out = _tighten_to_words(out, max_words)
            return out

        def build_loglines(seed_txt: str, tone_used: str, angles: List[str], n: int, w_short: int, w_detail: int, add_note: bool):
            pool = [p for p in _angle_lines(seed_txt) if not angles or p[0] in angles]
            if not pool:
                pool = _angle_lines(seed_txt)
            out_rows = []
            i = 0
            while len(out_rows) < n:
                label, draft = pool[i % len(pool)]
                tagline = _tighten_to_words(_fresh_nudge(seed_txt, _apply_tone(_strip_cliche(draft), tone_used)), w_short)
                detailed = _expand_to_detailed(seed_txt, tone_used, w_detail)

                if add_note:
                    if get_lang() == "العربية":
                        notes = {
                            "شخصية (من الداخل للخارج)": "زاوية: قيادة شخصية، قرارات على الكاميرا، مصداقية من الوصول.",
                            "النظام مقابل الفرد": "زاوية: من القرار السياسي إلى الجسد؛ العاقبة تقع على فرد.",
                            "تحقيق / تسريب": "زاوية: أثر وثائقي ومقاطع صوت؛ مشاهد قابلة للتحقق.",
                            "وصول حصري": "زاوية: كاميرات في غرف ممنوعة؛ واقعية إجرائية.",
                            "عدّ تنازلي / الساعة": "زاوية: ضغط زمني؛ فواصل طبيعية مع المواعيد.",
                            "مفارقة / سخرية": "زاوية: الانتباه كعبء أمني؛ الأمان مقابل الظهور.",
                            "من الصغير إلى الكبير": "زاوية: حالة واحدة تشرح بنية أزمة أوسع.",
                            "داود وجالوت": "زاوية: لا تكافؤ قوة؛ السلاح هو الدليل.",
                            "الكلفة الخفية": "زاوية: تكاليف غير مرئية تصبح محسوسة.",
                            "إعادة تأطير أرشيفي": "زاوية: الماضي يعيد تعريف الحاضر.",
                        }
                    else:
                        notes = {
                            "Character (inside-out)": "Angle: character-led; decisions on camera; access-driven credibility.",
                            "System vs Individual": "Angle: policy-to-person pipeline; consequence on one body.",
                            "Investigation / Leak": "Angle: document/voice-note trail; verifiable scenes.",
                            "Exclusive Access": "Angle: cameras in no-go rooms; procedural realism.",
                            "Countdown / Clock": "Angle: time pressure with natural act breaks.",
                            "Paradox / Irony": "Angle: visibility as liability; safety vs fame.",
                            "Micro→Macro": "Angle: one case explains the system; blueprint of a wider crisis.",
                            "David vs Goliath": "Angle: power asymmetry; weapon is proof.",
                            "Hidden Cost": "Angle: externalities made visible; bodies/time as currency.",
                            "Archival Reframe": "Angle: past re-arranges the present; archive-forward.",
                        }
                    note = notes.get(label, "")
                else:
                    note = ""

                out_rows.append({"angle": label, "tagline": tagline, "detailed": detailed, "note": note})
                i += 1
            return out_rows

        if st.button(L("Generate loglines","إنشاء اللوجلاين"), type="primary", key="log_gen_btn"):
            if not seed.strip():
                st.warning(L("Please add a short seed.","الرجاء إدخال وصف قصير أولاً."))
            else:
                # If the user opted to use AI and prerequisites are met, try to generate loglines via OpenAI
                if use_ai:
                    if not OPENAI_AVAILABLE:
                        st.error(L(
                            "The openai package is not installed. Please ensure it is listed in requirements.txt.",
                            "حزمة openai غير مثبتة. يرجى التأكد من إضافتها في ملف المتطلبات requirements.txt."
                        ))
                    else:
                        api_key = os.getenv("OPENAI_API_KEY")
                        if not api_key:
                            st.error(L(
                                "Environment variable OPENAI_API_KEY is not set. Please set it before using ChatGPT.",
                                "متغير البيئة OPENAI_API_KEY غير معرّف. الرجاء ضبطه قبل استخدام ChatGPT."
                            ))
                        else:
                            angles_str = ", ".join(angles_used) if angles_used else ""
                            prompt_en = (
                                "You are a creative writer who crafts short, compelling loglines for documentary film proposals "
                                "in both English and Arabic. "
                                f"Generate {n_variants} variants for the following film description. "
                                "Each variant should include an English tagline (" 
                                f"<= {short_used} words), an Arabic tagline (<= {short_used} words), "
                                "an English detailed logline (" 
                                f"<= {detail_used} words), and an Arabic detailed logline "
                                f"(<= {detail_used} words). "
                                f"Film description: {seed.strip()}. Tone: {tone_used}. Angles: {angles_str}. "
                                "Output each variant as four lines in the order: English tagline, Arabic tagline, "
                                "English detailed, Arabic detailed, separated by newlines. Separate variants with a blank line."
                            )
                            try:
                                openai.api_key = api_key
                                resp = openai.ChatCompletion.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "system",
                                         "content": "You are an assistant that writes bilingual loglines for documentary proposals."},
                                        {"role": "user", "content": prompt_en},
                                    ],
                                    max_tokens=800,
                                    temperature=0.7,
                                )
                                content = resp.choices[0].message.get("content", "").strip()
                                if not content:
                                    st.info(L(
                                        "No output returned from OpenAI.",
                                        "لم يرجع OpenAI أي مخرجات."
                                    ))
                                else:
                                    st.markdown(L("### AI-generated loglines","### لوجلاين منشأ بالذكاء الاصطناعي"))
                                    variants = [block.strip() for block in content.split("\n\n") if block.strip()]
                                    for v in variants:
                                        lines = [ln.strip() for ln in v.split("\n") if ln.strip()]
                                        st.divider()
                                        if len(lines) >= 1:
                                            st.write(f"**{L('English Tagline','الشعار القصير بالإنجليزية')}**: {lines[0]}")
                                        if len(lines) >= 2:
                                            st.write(f"**{L('Arabic Tagline','الشعار القصير بالعربية')}**: {lines[1]}")
                                        if len(lines) >= 3:
                                            st.write(f"**{L('English Detailed','اللوجلاين التفصيلي بالإنجليزية')}**: {lines[2]}")
                                        if len(lines) >= 4:
                                            st.write(f"**{L('Arabic Detailed','اللوجلاين التفصيلي بالعربية')}**: {lines[3]}")
                            except Exception as e:
                                st.error(L(
                                    f"Error from OpenAI: {e}",
                                    f"خطأ من OpenAI: {e}"
                                ))
                # If AI not used, fall back to the template-based logline generation
                else:
                    rows = build_loglines(seed, tone_used, angles_used, n_variants, short_used, detail_used, add_angle_note)
                    if not rows:
                        st.info(L("No output. Try different angles or increase word targets.","لا يوجد ناتج. جرّب زوايا أخرى أو زد عدد الكلمات."))
                    else:
                        for idx, r in enumerate(rows, 1):
                            st.markdown(f"### {idx}. {r['angle']}")
                            st.write(f"**{L('Tagline','الشعار القصير')}:** {r['tagline']}")
                            st.write(f"**{L('Detailed','اللوجلاين التفصيلي')}:** {r['detailed']}")
                            if add_angle_note and r["note"]:
                                st.caption(r["note"])
                            st.divider()
                        import io
                        buf = io.StringIO()
                        for idx, r in enumerate(rows, 1):
                            buf.write(f"{idx}. {r['angle']}\n")
                            buf.write(f"   {L('Tagline','الشعار')}: {r['tagline']}\n")
                            buf.write(f"   {L('Detailed','تفصيلي')}: {r['detailed']}\n")
                            if add_angle_note and r["note"]:
                                buf.write(f"   {L('Note','ملاحظة')}: {r['note']}\n\n")
                        st.download_button(L("Download all loglines (.txt)","تحميل جميع اللوجلاينات (.txt)"),
                                           buf.getvalue().encode("utf-8"),
                                           file_name="loglines_bilingual.txt", key="log_dl_bi")

    # ============================ TAB 5: Diagnostics ============================
    with diag_tab:
        st.subheader(L("Diagnostics","التشخيص"))
        files = sorted(glob.glob(str(DATA_DIR / "*")))
        st.write(L("**/data files**:","**ملفات /data**:"), files)

        p_cat = DATA_DIR / "ajd_catalogue_raw.csv"
        if p_cat.exists():
            st.write(L("Catalogue file size:","حجم ملف الفهرس:"), f"{p_cat.stat().st_size:,} {L('bytes','بايت')}")
        else:
            st.info(L("Catalogue merged CSV missing","ملف الفهرس الموحّد غير موجود"))

        cat_df_preview = load_catalogue_df()
        st.write(L("Catalogue DataFrame shape:","أبعاد بيانات الفهرس:"), f"{cat_df_preview.shape[0]:,} × {cat_df_preview.shape[1]}")
        if not cat_df_preview.empty:
            st.write(L("**Column names (first 30):**","**أسماء الأعمدة (أول 30):**"), list(cat_df_preview.columns[:30]))
            st.write(L("**Head(10):**","**أول 10 صفوف:**"))
            st.dataframe(cat_df_preview.head(10), use_container_width=True)
            texty = [c for c in cat_df_preview.columns if re.search(r"(name|title|synopsis|series|english|arabic|desc|topic|اسم|عنوان|ملخ|عربي|سلسلة|موضوع)", c, re.I)]
            st.write(L("**Guessed text columns:**","**أعمدة نصية محتملة:**"), texty if texty else L("(none)","(لا يوجد)"))
        else:
            st.warning(L("DataFrame is empty after loading. This usually means delimiter/encoding issues or an empty file.",
                         "البيانات فارغة بعد التحميل—غالبًا مشكلة فاصل/ترميز أو ملف فارغ."))

        st.markdown(L("### Column preset for Search tab","### ضبط أعمدة البحث"))
        if not cat_df_preview.empty:
            _ = st.multiselect(
                L("Columns to search in (preset for Search tab)","الأعمدة المراد البحث فيها (يُستخدم تلقائيًا)"),
                options=list(cat_df_preview.columns),
                default=texty or list(cat_df_preview.columns)[:5],
                key="diag_cols_preset",
            )
            st.caption(L("This preset is saved in session; the Search tab will use it automatically.",
                         "سيُحفظ هذا الإعداد في الجلسة؛ وسيستخدمه تبويب البحث تلقائيًا."))
        else:
            st.info(L("Load a catalogue first to configure the column preset.","حمّل الفهرس أولًا لضبط أعمدة البحث."))

        st.markdown(L("### Upload / Replace catalogue CSV (session-only)","### رفع/استبدال ملف الفهرس (جلسة فقط)"))
        up = st.file_uploader(L("Upload a full catalogue CSV (UTF-8). Not persisted on redeploy.",
                                "ارفع ملف CSV كامل (UTF-8). لن يُحفظ بعد إعادة النشر."),
                              type=["csv"], key="diag_upload_catalogue")
        if up is not None:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(DATA_DIR / "ajd_catalogue_raw.csv", "wb") as f:
                f.write(up.getbuffer())
            load_catalogue_df.clear()
            st.success(L("Uploaded. Click **Reload data / clear cache** in the sidebar, then reopen Diagnostics.",
                         "تم الرفع. اضغط **تحديث البيانات / مسح الذاكرة المؤقتة** من الشريط الجانبي، ثم افتح التشخيص."))

        st.markdown(L("### Maintenance","### صيانة"))
        force_merge = st.button(L("Force re-merge from parts (ajd_catalogue_raw.part*.csv)","إعادة الدمج من الأجزاء"), key="diag_force_merge_btn")
        show_parts = st.button(L("Show part files","عرض ملفات الأجزاء"), key="diag_show_parts_btn")
        if force_merge:
            merged = DATA_DIR / "ajd_catalogue_raw.csv"
            try:
                if merged.exists(): merged.unlink()
                ok = _merge_chunked_csv(str(DATA_DIR / "ajd_catalogue_raw.part*.csv"), merged)
                load_catalogue_df.clear()
                if ok:
                    st.success(L("Re-merged successfully. Click **Reload data / clear cache** in the sidebar.",
                                 "تم الدمج بنجاح. اضغط **تحديث البيانات / مسح الذاكرة المؤقتة** من الشريط الجانبي."))
                else:
                    st.warning(L("No part files found matching pattern.","لا توجد ملفات أجزاء مطابقة للنمط."))
            except Exception as e:
                st.error(L(f"Re-merge failed: {e}", f"فشل الدمج: {e}"))
        if show_parts:
            st.write(sorted(glob.glob(str(DATA_DIR / "ajd_catalogue_raw.part*.csv"))))

    st.markdown("---")
    st.caption(L(
        "© 2025 ICON Studio — AJD Topic Explorer Dashboard.",
        "© 2025 ICON Studio — لوحة استكشاف موضوعات AJD."
    ))

if __name__ == "__main__":
    main()
