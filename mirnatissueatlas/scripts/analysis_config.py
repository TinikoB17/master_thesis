import os
from pathlib import Path

import yaml

# Reads config.yaml from the repository root, for modules that are imported by the
# Snakemake scripts (metrics.py) and so have no snakemake object.
# A separate workflow (e.g. plasma/Snakefile) points ANALYSIS_CONFIG to its own config.

CONFIG_PATH = Path(os.environ.get("ANALYSIS_CONFIG", Path(__file__).resolve().parents[1] / "config.yaml"))

with open(CONFIG_PATH) as file:
    CONFIG = yaml.safe_load(file)


def age_bins(min_age=None, max_age=None, width=None):
    """Bin edges from min to max age in steps of `width`. A last bin of at most half the
    width is merged into the one before it, e.g. 30-85 by 10 -> [30, 40, 50, 60, 70, 85]."""
    min_age = CONFIG["age"]["min"] if min_age is None else min_age
    max_age = CONFIG["age"]["max"] if max_age is None else max_age
    width = CONFIG["age"]["bin_width"] if width is None else width
    edges = list(range(min_age, max_age, width)) + [max_age]
    if len(edges) > 2 and edges[-1] - edges[-2] <= width / 2:
        del edges[-2]
    return edges
