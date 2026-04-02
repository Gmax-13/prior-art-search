import pandas as pd

def load_keywords(input_csv):
    df = pd.read_csv(input_csv)
    keywords = df.loc[0, "keywords"]

    if pd.isna(keywords):
        return []

    return [k.strip().lower() for k in keywords.split(",")]


def keyword_search(
    input_csv,
    classified_csv,
    output_csv,
    min_score=3,
    top_n=20
):
    keywords = load_keywords(input_csv)
    df = pd.read_csv(classified_csv, dtype=str)

    def keyword_score(row):
        score    = 0
        # Null safety: fillna before str conversion to avoid "nan" matching keywords
        title    = "" if (not row.get("title")    or str(row["title"]).lower()    == "nan") else str(row["title"]).lower()
        abstract = "" if (not row.get("abstract") or str(row["abstract"]).lower() == "nan") else str(row["abstract"]).lower()

        for kw in keywords:
            score += title.count(kw) * 3
            score += abstract.count(kw) * 1

        return score

    df["keyword_score"] = df.apply(keyword_score, axis=1)

    # Precision filter
    df = df[df["keyword_score"] >= min_score]

    # Sort by relevance
    df = df.sort_values(by="keyword_score", ascending=False)

    # Keep top-N results only
    df = df.head(top_n)

    df.to_csv(output_csv, index=False)

    print("Keyword Search Completed")
    print("Keywords:", keywords)
    print("Minimum Score Threshold:", min_score)
    print("Results Returned:", len(df))


if __name__ == "__main__":
    keyword_search(
        input_csv="input.csv",
        classified_csv="classified_patents.csv",
        output_csv="keyword_filtered_patents.csv",
        min_score=3,
        top_n=20
    )