"""Healthy-control dataset of one series: the control groups, known age and sex, optionally an
age range; samples with an identical expression profile (deposited twice) are kept once.

metadata has Group (the control group) and Disease_State (= Group), which the split and model
rules use for stratification and per-group errors, plus each sample's original floor value."""
import hashlib

import pandas as pd

GROUPS = snakemake.params.groups
AGE_RANGE = snakemake.params.age_range  # (min, max), both inclusive, or None for all ages


def fingerprint(values: pd.Series) -> str:
    return hashlib.md5(values.round(4).to_numpy().tobytes()).hexdigest()


def main():
    expression = pd.read_csv(snakemake.input.expression, sep="\t", index_col=0)
    metadata = pd.read_csv(snakemake.input.metadata, sep="\t", index_col=0).loc[expression.index]
    floors = pd.read_csv(snakemake.input.floors, sep="\t", index_col=0)

    steps = {"samples": len(metadata)}
    keep = metadata["Disease_State"].isin(GROUPS)
    steps["controls"] = int(keep.sum())
    keep &= metadata["Age"].notna() & metadata["Sex"].notna()
    steps["with age and sex"] = int(keep.sum())
    if AGE_RANGE is not None:
        keep &= metadata["Age"].between(*AGE_RANGE)
        steps[f"aged {AGE_RANGE[0]}-{AGE_RANGE[1]}"] = int(keep.sum())
    profiles = expression.loc[keep].apply(fingerprint, axis=1)
    duplicates = profiles.index[profiles.duplicated(keep="first")]
    keep[duplicates] = False
    steps["unique profiles"] = int(keep.sum())

    controls = metadata.loc[keep, ["Title", "Disease_State", "Sex", "Age"]].copy()
    controls.insert(0, "Group", controls["Disease_State"])
    controls["Age"] = controls["Age"].astype(int)
    controls["Original_Floor"] = floors.loc[controls.index, "original_floor"]
    controls["Undetected_Fraction"] = floors.loc[controls.index, "undetected_fraction"]

    expression.loc[controls.index].to_csv(snakemake.output.expression, sep="\t")
    controls.to_csv(snakemake.output.metadata, sep="\t")
    print("  ->  ".join(f"{step}: {n}" for step, n in steps.items()))
    print(controls.groupby("Group")["Age"].describe()[["count", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
