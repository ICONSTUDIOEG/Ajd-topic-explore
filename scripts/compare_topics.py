import argparse
import pandas as pd
from collections import Counter
import re
import json

def explode_topics_str(text):
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r"[;,/|·•\-–—]+", text)
    bag = []
    for p in parts:
        token = p.strip().lower()
        if len(token) >= 3 and token not in {"and","the","for","with","from","into","about","film","doc","docs","docu","series","episode","season","ajd","al jazeera","aljazeera"}:
            bag.append(token)
    return bag

def main():
    parser = argparse.ArgumentParser(description="Compare your project topics against Al Jazeera Documentary topics.")
    parser.add_argument("--ajd-topics", default="data/ajd_topics_extracted.csv")
    parser.add_argument("--project-file", required=True)
    parser.add_argument("--project-topics-col", default=None)
    parser.add_argument("--out", default="comparison_report.json")
    args = parser.parse_args()

    ajd = pd.read_csv(args.ajd_topics)
    ajd_topics = set(ajd['topic'].astype(str).str.lower())

    # Load project
    if args.project_file.lower().endswith(".csv"):
        p = pd.read_csv(args.project_file)
        col = args.project_topics_col or next((c for c in p.columns if re.search(r"(topic|tags|keywords|themes)", c, re.I)), None)
        if not col:
            raise ValueError("Specify --project-topics-col")
        proj_bag = []
        for v in p[col].fillna("").astype(str):
            proj_bag.extend(explode_topics_str(v))
    elif args.project_file.lower().endswith(".json"):
        with open(args.project_file,"r",encoding="utf-8") as f:
            data=json.load(f)
        proj_bag=[]
        for item in data if isinstance(data,list) else data.get("items",[]):
            topics_val=item.get("topics") or item.get("tags") or ""
            if isinstance(topics_val,list):
                for t in topics_val: proj_bag.extend(explode_topics_str(str(t)))
            else: proj_bag.extend(explode_topics_str(str(topics_val)))
    else:
        raise ValueError("CSV or JSON only")

    proj_counts=Counter(proj_bag)
    proj_topics=set(proj_counts.keys())
    overlap=sorted(list(ajd_topics & proj_topics))
    only_project=sorted(list(proj_topics - ajd_topics))
    only_ajd=sorted(list(ajd_topics - proj_topics))[:2000]

    report={
        "summary":{"ajd_unique_topics":len(ajd_topics),"project_unique_topics":len(proj_topics),"overlap_count":len(overlap)},
        "overlap_topics":overlap,
        "project_only_topics":only_project,
        "ajd_only_topics_sample":only_ajd,
        "project_topic_frequencies":proj_counts.most_common(200)
    }
    with open(args.out,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report["summary"],indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
