import io
import pandas as pd

path = "original_input/GSE164174_series_matrix.txt"

meta_lines, table_lines = [], []
in_table = False
with open(path) as f:
    for line in f:
        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if line.startswith("!series_matrix_table_end"):
            break
        if in_table:
            table_lines.append(line)
        elif line.startswith("!Sample_"):
            meta_lines.append(line)

# Expression: rows = probes/miRNAs, columns = GSM sample IDs
expr = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)

# Metadata: one row per sample
rows = {}
for line in meta_lines:
    parts = line.rstrip("\n").split("\t")
    key = parts[0].lstrip("!")
    vals = [p.strip('"') for p in parts[1:]]
    if key == "Sample_characteristics_ch1":
        key = vals[0].split(":")[0].strip()
        vals = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
    rows[key] = vals

meta = pd.DataFrame(rows).set_index("Sample_geo_accession")

print(expr.shape)
print(expr.index[:5].tolist())
print(meta.columns.tolist())
