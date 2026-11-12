import pandas as pd

EXPRESSION = snakemake.input.expression
NAME_MAP = snakemake.input.name_map
OUTPUT = snakemake.output.expression_named

expression = pd.read_csv(EXPRESSION, sep="\t", index_col=0)
name_map = pd.read_csv(NAME_MAP, sep="\t").set_index("Accession")["Name"]

# Multi-accession probes ("MIMAT0000232, MIMAT0004563") have no name and are dropped
multi = [column for column in expression.columns if "," in column]
expression = expression.drop(columns=multi)
missing = expression.columns.difference(name_map.index)
if not missing.empty:
    raise ValueError(f"Accessions without a name: {missing.tolist()}")
print(f"Dropped {len(multi)} multi-accession probes")

expression.columns = expression.columns.map(name_map)
expression.to_csv(OUTPUT, sep="\t")
print(f"{expression.shape[0]} samples x {expression.shape[1]} miRNAs -> {OUTPUT}")
