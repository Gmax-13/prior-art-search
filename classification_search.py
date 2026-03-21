import pandas as pd

def load_classification_codes(input_csv):
    df = pd.read_csv(input_csv)
    ipc_codes = df.loc[0, "ipc_codes"]
    cpc_codes = df.loc[0, "cpc_codes"]

    codes = []

    if pd.notna(ipc_codes):
        codes.extend([c.strip().upper() for c in ipc_codes.split(",")])

    if pd.notna(cpc_codes):
        codes.extend([c.strip().upper() for c in cpc_codes.split(",")])

    return list(set(codes))


def classification_search(
    input_csv,
    patents_csv,
    output_csv,
    ipc_column="ipc_codes",
    cpc_column="cpc_codes"
):
    input_codes = load_classification_codes(input_csv)

    patents_df = pd.read_csv(patents_csv)
    patents_df[ipc_column] = patents_df[ipc_column].fillna("")
    patents_df[cpc_column] = patents_df[cpc_column].fillna("")

    def match_classification(row):
        all_codes = (
            row[ipc_column].upper().split(",") +
            row[cpc_column].upper().split(",")
        )

        for patent_code in all_codes:
            patent_code = patent_code.strip()
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
        patents_csv="all_scraped_patents.csv",
        output_csv="classified_patents.csv"
    )
