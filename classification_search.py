import pandas as pd


def safe_split(value, delimiter=";") -> list:
    """Split a delimited string safely, handling nulls and empty values."""
    if not value or (isinstance(value, float)):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    return [tok.strip() for tok in s.split(delimiter) if tok.strip()]


def load_classification_codes(input_csv):
    df = pd.read_csv(input_csv, dtype=str)
    ipc_codes = df.loc[0, "ipc_codes"] if "ipc_codes" in df.columns else ""
    cpc_codes = df.loc[0, "cpc_codes"] if "cpc_codes" in df.columns else ""

    codes = []
    # Input CSV uses comma separator; corpus uses semicolon
    for raw in [ipc_codes, cpc_codes]:
        for delim in [";", ","]:
            parts = safe_split(raw, delim)
            if parts:
                codes.extend([c.upper() for c in parts])
                break

    return list(set(codes))


def classification_search(
    input_csv,
    patents_csv,
    output_csv,
    ipc_column="ipc_codes",
    cpc_column="cpc_codes"
):
    input_codes = load_classification_codes(input_csv)

    patents_df = pd.read_csv(patents_csv, dtype=str)
    patents_df[ipc_column] = patents_df[ipc_column].fillna("")
    patents_df[cpc_column] = patents_df[cpc_column].fillna("")

    def match_classification(row):
        # Corpus uses semicolon-separated codes
        all_codes = (
            safe_split(row[ipc_column], ";") +
            safe_split(row[cpc_column], ";")
        )
        for patent_code in all_codes:
            patent_code = patent_code.strip().upper()
            for input_code in input_codes:
                if patent_code.startswith(input_code):
                    return True
        return False

    filtered_df = patents_df[patents_df.apply(match_classification, axis=1)]
    filtered_df.to_csv(output_csv, index=False)

    print("Classification Search Completed")
    print("Input IPC/CPC Codes:", input_codes)
    print("Total Patents:", len(patents_df))
    print("Matched Patents:", len(filtered_df))


if __name__ == "__main__":
    classification_search(
        input_csv="input.csv",
        patents_csv="all_scrapped_patents.csv",
        output_csv="Output/classified_patents.csv"
    )