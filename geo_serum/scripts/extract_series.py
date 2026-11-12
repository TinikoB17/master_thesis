"""Splits one GEO series matrix into an expression table (rows = samples, columns = miRBase
accessions, values as deposited) and a metadata table (Title, Disease_State, Sex, Age), the
same layout as the GSE164174 tables made by extract_series_matrix.py."""
import re

import pandas as pd

from geo_series import read_series_matrix

SERIES = snakemake.wildcards.series
DISEASE_FIELD = snakemake.params.disease_field
CONTROLS = snakemake.params.controls  # {sample field: [values that mean healthy control]}

SEX = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}


def disease_state(metadata: pd.DataFrame) -> pd.Series:
    """Healthy controls keep their control label (CONTROLS); every other sample gets the
    series' disease field (None: no field), or its title without the bracketed sample code
    where that is empty. A title that is only a sample code (no bracketed code, e.g. "XA0020"
    in GSE113740) carries no label, so those samples get their source name instead."""
    state = pd.Series(None, index=metadata.index, dtype=object)
    for field, labels in CONTROLS.items():
        state = state.fillna(metadata[field].where(metadata[field].isin(labels)))
    if DISEASE_FIELD is not None and DISEASE_FIELD in metadata.columns:
        state = state.fillna(metadata[DISEASE_FIELD].replace("", None))
    labelled = metadata["Sample_title"].str.contains(r"\[[^\]]*\]\s*$", regex=True)
    return state.fillna(metadata["title_label"].where(labelled, metadata["Sample_source_name_ch1"]))


def main():
    metadata, expression = read_series_matrix(snakemake.input.series_matrix)

    sex_field = next(field for field in ["Sex", "gender"] if field in metadata.columns)
    table = pd.DataFrame({
        "Title": metadata["Sample_title"],
        "Disease_State": disease_state(metadata),
        # unknown / uncertain become empty
        "Sex": metadata[sex_field].str.strip().str.lower().map(SEX),
        "Age": pd.to_numeric(metadata["age"], errors="coerce").astype("Int64"),
    }, index=metadata.index)
    table.index.name = "Sample"

    expression = expression.T
    expression.index.name = "Sample"
    expression.columns.name = None
    if not expression.index.equals(table.index):
        raise ValueError(f"{SERIES}: samples in the expression table and the metadata do not match")

    expression.to_csv(snakemake.output.expression, sep="\t")
    table.to_csv(snakemake.output.metadata, sep="\t")
    print(f"{SERIES}: {expression.shape[0]} samples x {expression.shape[1]} probes; "
          f"sex missing {table['Sex'].isna().sum()}, age missing {table['Age'].isna().sum()}")
    print(table["Disease_State"].value_counts().to_string())


if __name__ == "__main__":
    main()
