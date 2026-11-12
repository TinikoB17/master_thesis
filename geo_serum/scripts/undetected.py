"""Undetected values in the 3D-Gene series: the authors set undetected miRNAs to log2(0.1) and
then scaled each sample, so every sample has its own floor value. These helpers find that floor
and replace it with one common value (see set_common_floor.py for the reasoning)."""
import pandas as pd


def sample_floors(expression: pd.DataFrame) -> pd.Series:
    """The floor is each sample's most frequent value (hundreds of repeats). It is not always
    the minimum: in some samples 1-4 values lie slightly below it."""
    return expression.round(9).apply(lambda row: row.value_counts().index[0], axis=1)


def undetected_values(expression: pd.DataFrame):
    """Undetected values (at or below each sample's floor) and a per-sample summary (original
    floor, number and fraction of undetected values, lowest detected value)."""
    floors = sample_floors(expression)
    undetected = expression.round(9).le(floors, axis=0)
    summary = pd.DataFrame({
        "original_floor": floors,
        "n_undetected": undetected.sum(axis=1),
        "undetected_fraction": undetected.mean(axis=1),
        "lowest_detected": expression.mask(undetected).min(axis=1),
    })
    return undetected, summary


def apply_common_floor(expression: pd.DataFrame, common_floor: float):
    """Values at or below each sample's floor become common_floor. Returns the new table and the
    per-sample summary of undetected_values()."""
    undetected, summary = undetected_values(expression)
    lowest_detected = summary["lowest_detected"].min()
    if common_floor >= lowest_detected:
        raise ValueError(f"Common floor {common_floor} is not below the lowest detected value "
                         f"({lowest_detected:.3f}); undetected would rank above detected values")
    return expression.mask(undetected, common_floor), summary
