import argparse, json, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    parser = argparse.ArgumentParser(description="Compute TF-IDF similarities between AJD entries and your film loglines.")
    parser.add_argument("--ajd-catalogue", default="data/ajd_catalogue_raw.csv")
    parser.add_argument("--ajd-text-cols", default="")
    parser.add_argument("--project-json", required=True)
    parser.add_argument("--out", default="similarity_matches.json")
    args = parser.parse_args()

    df = pd.read_csv(args.ajd_catalogue)
    cols = [c.strip() for c in args.ajd_text_cols.split(",") if c.strip()] or [c for c in df.columns if re.search(r"(title|synopsis|summary|desc|topic|tags)", c, re.I)]
    df["__text"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)

    with open(args.project_json,"r",encoding="utf-8") as f: films=json.load(f)
    project_texts=[(i.get("title","Untitled"), (i.get("title","")+ " "+ i.get("description","")).strip()) for i in films]

    corpus=list(df["__text"].astype(str).values)+[t for _,t in project_texts]
    vectorizer=TfidfVectorizer(min_df=2,ngram_range=(1,2))
    X=vectorizer.fit_transform(corpus)

    n_ajd=len(df)
    ajd_vecs=X[:n_ajd]
    proj_vecs=X[n_ajd:]

    sim=cosine_similarity(proj_vecs,ajd_vecs)
    results=[]
    for idx,(title,text) in enumerate(project_texts):
        sims=sim[idx]
        top_idx=sims.argsort()[::-1][:10]
        matches=[]
        for j in top_idx:
            row=df.iloc[j].to_dict()
            matches.append({"ajd_index":int(j),"score":float(sims[j]),"ajd_row":row})
        results.append({"project_title":title,"matches":matches})

    with open(args.out,"w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
    print(json.dumps({"films":len(project_texts),"ajd_rows":n_ajd},indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
