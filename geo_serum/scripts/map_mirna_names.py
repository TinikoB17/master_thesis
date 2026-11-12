import time
import urllib.request
import re

import pandas as pd

MATURE_FA = snakemake.input.mature_fa
EXPRESSIONS = snakemake.input.expressions  # expression tables of all series
NAME_MAP = snakemake.output.name_map

# miRBase entry page, used for accessions missing from the mature.fa download
ENTRY_URL = "https://mirbase.org/mature/{accession}"
REQUEST_DELAY = 0.5  # seconds between page requests


def read_mature_fa(path):
    """Accession -> mature miRNA name from the miRBase (hsa) mature.fa headers ("> name accession ...")."""
    names = {}
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                name, accession = line[1:].split()[:2]
                names[accession] = name
    return names


def fetch_page(url, attempts=4):
    """Fetches the page, retrying with a growing wait when the request fails or times out."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read().decode()
        except OSError as error:
            if attempt == attempts:
                raise
            print(f"{url}: {error}, retrying ({attempt}/{attempts - 1})")
            time.sleep(5 * attempt)


def fetch_name(accession):
    """Reads the mature miRNA name from the miRBase entry page."""
    url = ENTRY_URL.format(accession=accession)
    page = fetch_page(url)
    text = re.sub(r"<[^>]+>", " ", page)
    match = re.search(r"miRBase entry:\s*(\S+)", text)
    if match is None:
        raise ValueError(f"No miRNA name found on {url}")
    time.sleep(REQUEST_DELAY)
    return match.group(1)


def main():
    fa_names = read_mature_fa(MATURE_FA)
    # Accessions of all series, in the order they first appear
    columns = pd.Index(list(dict.fromkeys(
        column for path in EXPRESSIONS
        for column in pd.read_csv(path, sep="\t", nrows=0, index_col=0).columns)))

    # Probes that match several miRNAs with the same mature sequence
    # ("MIMAT0000232, MIMAT0004563") are left out of the mapping and so dropped
    multi = [column for column in columns if "," in column]
    print(f"Dropping {len(multi)} multi-accession probes: {multi}")

    rows = []
    for accession in columns.difference(multi, sort=False):
        if accession in fa_names:
            name, source = fa_names[accession], "mature.fa"
        else:
            name, source = fetch_name(accession), "mirbase.org"
        rows.append({"Accession": accession, "Name": name, "Source": source})

    name_map = pd.DataFrame(rows)
    if not name_map["Name"].str.startswith("hsa-").all():
        raise ValueError("Non-human miRNA names in the mapping")
    duplicates = name_map.loc[name_map["Name"].duplicated(keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Several probes map to the same name:\n{duplicates}")

    name_map.to_csv(NAME_MAP, sep="\t", index=False)
    print(name_map["Source"].value_counts())
    print(f"{len(name_map)} accessions mapped -> {NAME_MAP}")


if __name__ == "__main__":
    main()
