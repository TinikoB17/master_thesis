"""Adds the cohort (control group, Disease_State) as 0/1 features to the train and holdout
tables: one column per cohort except the first in COHORT_ORDER (the reference). The model
scripts always keep these columns (model_utils.feature_selector)."""
import pandas as pd

from model_utils import COHORT_PREFIX

COHORT_ORDER = snakemake.params.cohort_order


def add_cohorts(table: pd.DataFrame, cohorts: pd.Series) -> pd.DataFrame:
    unknown = set(cohorts) - set(COHORT_ORDER)
    if unknown:
        raise ValueError(f"Cohorts not in the cohort order: {unknown}")
    columns = pd.DataFrame({f"{COHORT_PREFIX}{cohort.replace(' ', '_')}": (cohorts == cohort).astype(int)
                            for cohort in COHORT_ORDER[1:]}, index=table.index)
    # After Age and Sex, before the miRNAs
    return pd.concat([table[["Age", "Sex"]], columns, table.drop(columns=["Age", "Sex"])], axis=1)


def main():
    metadata = pd.read_csv(snakemake.input.metadata, sep="\t", index_col=0)
    for name in ["train", "holdout"]:
        table = pd.read_csv(snakemake.input[name], sep="\t", index_col=0)
        add_cohorts(table, metadata.loc[table.index, "Disease_State"]).to_csv(
            snakemake.output[name], sep="\t")
    metadata.to_csv(snakemake.output.metadata, sep="\t")
    print(f"Cohort features: {[COHORT_PREFIX + c.replace(' ', '_') for c in COHORT_ORDER[1:]]} "
          f"(reference {COHORT_ORDER[0]})")


if __name__ == "__main__":
    main()
