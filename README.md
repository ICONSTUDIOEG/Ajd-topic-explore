# Ajd-topic-explore
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ajd-topic-explore.streamlit.app/)
# AJD Topic Explorer

Tools to examine the Al Jazeera Documentary (AJD) catalogue, extract topic clusters, and compare them with your project's topics and films.

## What's inside
- `data/ajd_catalogue_raw.csv` — Cleaned export of the uploaded AJD Excel sheet.
- `data/ajd_topics_extracted.csv` — Unique topics with frequencies, auto-extracted from text columns.
- `scripts/compare_topics.py` — Compare overlap between AJD topics and your project topics (CSV/JSON).
- `scripts/similarity_matches.py` — TF-IDF similarity between your film loglines and AJD entries to find nearest neighbours.
- `examples/project_topics.csv` — Example CSV structure for your project.
- `examples/project_films.json` — Example JSON structure for your film list.

## Quickstart

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Compare your project topics (CSV) with AJD topics
python scripts/compare_topics.py --project-file examples/project_topics.csv --project-topics-col topics --out comparison_report.json

# 4) Or compare JSON films + loglines against the catalogue using TF-IDF
python scripts/similarity_matches.py --project-json examples/project_films.json --out similarity_matches.json
Designed and Powered by Dr.Ashraf Ahmed - ICON STUDIO 2025 www.icon-studios.com
