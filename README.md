# Ajd-topic-explore
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ajd-topic-explore.streamlit.app/)
# AJD Topic Explorer

## Why These Tools Matter for Documentary Producers and Filmmakers

Documentary filmmaking begins long before the camera starts rolling. The most difficult stage is often finding the right story, proving its originality, and transforming a broad topic into a compelling cinematic narrative. These tools are designed to support that creative development process.

### 1. Accelerating Story Discovery

Filmmakers often spend weeks researching potential subjects and exploring whether a topic can sustain a full documentary. The platform helps identify story opportunities quickly by connecting themes, keywords, and historical references across the Al Jazeera Documentary catalogue.

Instead of searching manually through hundreds of titles, producers can instantly discover related films, recurring themes, and gaps that may represent new storytelling opportunities.

### 2. Verifying Originality Before Development

One of the biggest risks in documentary development is investing time and resources in a project that has already been produced.

The tool allows users to compare new ideas against existing documentary catalogues and archives, helping teams determine:

* Whether a topic has already been covered.
* Which angles have already been explored.
* Where fresh perspectives may still exist.
* How to position a project uniquely for commissioners and broadcasters.

### 3. Generating Stronger Editorial Concepts

A documentary idea often starts as a broad subject such as migration, water scarcity, artificial intelligence, heritage preservation, or social change.

The AI module helps transform these broad themes into:

* Clear documentary concepts.
* Strong loglines.
* Possible narrative approaches.
* Investigative angles.
* Human-centered story entry points.

This provides filmmakers with a stronger foundation before writing treatments or pitching editors.

### 4. Supporting Creative Development

The platform functions as a research and development partner.

It can help creative teams explore:

* Alternative story structures.
* Character-driven approaches.
* Investigative frameworks.
* Historical timelines.
* Geographic and cultural dimensions of a topic.

Rather than replacing creative decision-making, it expands the range of possibilities available to filmmakers.

### 5. Reducing Development Time

Documentary development can take months of preliminary research before a proposal is ready.

By combining catalogue research, topic exploration, originality checking, and AI-assisted ideation in one workflow, the platform significantly reduces the time required to move from an initial idea to a pitch-ready concept.

### 6. Improving Pitch Quality

Commissioning editors and broadcasters evaluate hundreds of proposals every year.

Projects developed using structured research and originality verification are more likely to demonstrate:

* Editorial clarity.
* Market relevance.
* Distinctiveness.
* Strong narrative potential.

This gives producers greater confidence when preparing proposals for broadcasters, foundations, and funding bodies.

### 7. Building Better Documentary Proposals

The platform supports the development of essential proposal components, including:

* Title generation.
* Logline creation.
* Synopsis development.
* Treatment structure.
* Audience identification.
* Editorial positioning.

These elements can then be refined by the filmmaker and adapted to the requirements of specific broadcasters such as Al Jazeera Documentary Channel.

### Conclusion

The purpose of the platform is not to replace filmmakers, researchers, or producers. Its purpose is to augment their creative process by reducing repetitive research tasks, revealing hidden connections, verifying originality, and helping transform raw ideas into stronger documentary projects.

It acts as a creative research assistant, development consultant, and catalogue intelligence system designed specifically for documentary storytelling.


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
