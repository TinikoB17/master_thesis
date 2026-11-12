"""Reads GEO series matrix files: sample metadata (with the characteristics as columns) and the
expression table (rows = probes, columns = GSM samples)."""
import io

import pandas as pd


def read_series_matrix(path):
    """Returns (metadata, expression). metadata has one row per sample (index = GSM accession),
    with the "!Sample_*" fields, one column per "key: value" characteristic, and title_label
    (the title without its bracketed sample code)."""
    sample_lines, table_lines = [], []
    in_table = False
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                table_lines.append(line)
            elif line.startswith("!Sample_"):
                sample_lines.append(line)

    columns = {}
    for line in sample_lines:
        parts = [part.strip('"') for part in line.rstrip("\n").split("\t")]
        key, values = parts[0].lstrip("!"), parts[1:]
        if key == "Sample_characteristics_ch1":
            # One line per characteristic; a sample without it has an empty value
            names = {value.split(":", 1)[0].strip() for value in values if ":" in value}
            for name in names:
                columns[name] = [value.split(":", 1)[1].strip()
                                 if value.split(":", 1)[0].strip() == name else None
                                 for value in values]
        elif key not in columns:  # keep the first line of repeated fields
            columns[key] = values
    metadata = pd.DataFrame(columns).set_index("Sample_geo_accession")
    # Title without a trailing bracketed sample code: "Non-cancer 1 [SH4YL42_Cell3]" -> "Non-cancer 1"
    metadata["title_label"] = (metadata["Sample_title"]
                               .str.replace(r"\s*\[[^\]]*\]\s*$", "", regex=True).str.strip())

    expression = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)
    return metadata, expression
