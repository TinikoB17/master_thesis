import pandas as pd
from sklearn.model_selection import train_test_split

EXPRESSION = snakemake.input.expression
METADATA = snakemake.input.metadata
HOLDOUT_FRACTION = snakemake.params.holdout_fraction
SEED = snakemake.params.seed


def model_table(expression: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Age, Sex and the miRNAs, one row per sample (same layout as the miRNATissueAtlas pipeline)."""
    return metadata[["Age", "Sex"]].join(expression)


def main():
    expression = pd.read_csv(EXPRESSION, sep="\t", index_col=0)
    metadata = pd.read_csv(METADATA, sep="\t", index_col=0).loc[expression.index]
    data = model_table(expression, metadata)

    # Stratified by Disease_State and decade of age, so the holdout set has the same mix of
    # groups (e.g. the three control groups) and covers the same age range
    strata = metadata["Disease_State"] + "_" + (data["Age"] // 10).astype(int).astype(str)
    rare = strata.map(strata.value_counts()) < 2
    strata[rare] = "rare"  # strata with a single sample cannot be split
    if (strata == "rare").sum() == 1:
        strata[rare] = strata[~rare].iloc[0]
    train, holdout = train_test_split(data, test_size=HOLDOUT_FRACTION, stratify=strata,
                                      random_state=SEED)

    train.to_csv(snakemake.output.train, sep="\t")
    holdout.to_csv(snakemake.output.holdout, sep="\t")
    for name, part in [("Train", train), ("Holdout", holdout)]:
        print(f"{name}: {len(part)} samples, age {part['Age'].min()}-{part['Age'].max()}, "
              f"median {part['Age'].median()}; "
              + ", ".join(f"{group} {n}" for group, n in
                          metadata.loc[part.index, "Disease_State"].value_counts().items()))


if __name__ == "__main__":
    main()
