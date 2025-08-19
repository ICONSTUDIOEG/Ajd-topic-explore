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
        hook_angles = [
            ("Hidden History", "uncovers a buried past that reshapes the present."),
            ("Personal Lens", "tells a national story through one family or character."),
            ("System vs. Individual", "shows how policies collide with everyday survival."),
            ("Vanishing Traditions", "captures a craft or custom at the edge of extinction."),
            ("Unexpected Ally/Adversary", "pairs unlikely characters or groups in tension."),
            ("Future Stakes", "connects today's choice to tomorrow's irreversible outcome."),
        ]
        outs = []
        for k, (title, angle) in enumerate(hook_angles[:n]):
            outs.append(
                f"{title}: In one line — {seed_txt} — a story where it {angle}"
            )
        return outs

    if st.button("Generate loglines", type="primary"):
        if not seed:
            st.warning("Please enter a short seed description.")
        else:
            logs = propose_loglines(seed)
            for i, line in enumerate(logs, 1):
                st.write(f"**{i}.** {line}")
            st.download_button(
                "Download loglines (.txt)",
                ("\n".join(logs)).encode("utf-8"),
                file_name=(f"loglines_{pmpt_code or 'untagged'}.txt")
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "© 2025 ICON Studio — AJD Topic Explorer Dashboard. "
    "Add `streamlit` to requirements.txt and run: `streamlit run app.py`"
)
