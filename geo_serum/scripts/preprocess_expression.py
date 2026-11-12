"""Expression values ready for analysis: one common floor for undetected values (unless
COMMON_FLOOR is None), quantile normalisation across samples (if QUANTILE_NORMALIZE), then only
the miRBase high-confidence miRNAs.

Undetected miRNAs were set to log2(0.1) by the authors and every sample was then scaled by its
own factor, so each sample has its own floor value. They all get COMMON_FLOOR instead
(scripts/undetected.py). Floors are found on the whole profile (2,550 miRNAs, hundreds of
repeats per sample) before the miRNA filter.

Quantile normalisation (as preprocessCore::normalize.quantiles) gives every sample the same
distribution: the reference is the mean of the sorted profiles, and each value is replaced by
the reference value at its rank. Tied values (above all each sample's undetected floor) get the
mean of the reference over their ranks, so they stay tied. It runs on the whole profile before
the miRNA filter and uses all samples of the series (no ages or labels, but the holdout and
cancer samples do shape the reference)."""
import numpy as np
import pandas as pd

from undetected import apply_common_floor, undetected_values

COMMON_FLOOR = snakemake.params.common_floor  # None: keep the values as deposited
QUANTILE_NORMALIZE = snakemake.params.quantile_normalize


def quantile_normalize(expression: pd.DataFrame) -> pd.DataFrame:
    """Rows = samples. Each sample's values get the reference distribution, ties averaged."""
    values = expression.to_numpy(dtype=float)
    n_samples, n_features = values.shape
    reference = np.sort(values, axis=1).mean(axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(reference)])
    normalized = np.empty_like(values)
    for i in range(n_samples):
        row = values[i]
        order = np.argsort(row, kind="stable")
        sorted_row = row[order]
        # Runs of equal values: [start, end) positions in the sorted row
        starts = np.flatnonzero(np.r_[True, sorted_row[1:] != sorted_row[:-1]])
        ends = np.r_[starts[1:], n_features]
        run_means = (cumulative[ends] - cumulative[starts]) / (ends - starts)
        normalized[i, order] = np.repeat(run_means, ends - starts)
    return pd.DataFrame(normalized, index=expression.index, columns=expression.columns)


def main():
    expression = pd.read_csv(snakemake.input.expression, sep="\t", index_col=0)
    with open(snakemake.input.high_confidence_mirnas) as f:
        high_confidence = {line.strip() for line in f if line.strip()}

    if COMMON_FLOOR is None:
        # Values as deposited; the per-sample floors are only recorded
        floored, floors = expression, undetected_values(expression)[1]
    else:
        floored, floors = apply_common_floor(expression, COMMON_FLOOR)
    if QUANTILE_NORMALIZE:
        floored = quantile_normalize(floored)
    keep = [mirna for mirna in floored.columns if mirna in high_confidence]
    floored[keep].to_csv(snakemake.output.expression, sep="\t")
    floors.to_csv(snakemake.output.floors, sep="\t")

    print(f"Original floors: {floors['original_floor'].min():.2f} to "
          f"{floors['original_floor'].max():.2f}; lowest detected value "
          f"{floors['lowest_detected'].min():.3f}; {floors['undetected_fraction'].mean():.1%} "
          f"of the values undetected, "
          + ("kept at each sample's own floor" if COMMON_FLOOR is None else f"set to {COMMON_FLOOR}"))
    if QUANTILE_NORMALIZE:
        print(f"Quantile normalised {floored.shape[0]} samples x {floored.shape[1]} miRNAs")
    print(f"Kept {len(keep)} of {expression.shape[1]} miRNAs (high-confidence; "
          f"{len(high_confidence) - len(keep)} high-confidence miRNAs not on the array)")


if __name__ == "__main__":
    main()
