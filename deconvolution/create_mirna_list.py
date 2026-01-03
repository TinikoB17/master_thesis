import pandas as pd

tsi_path = snakemake.input.tsi
mirna_list_path = snakemake.output.mirnas

tsi = pd.read_csv(tsi_path, sep='\t')
high_tsi_miRNAs = tsi[tsi["TSI"] >= 0.8]["miRNA"].to_list()

with open(mirna_list_path, 'w') as file:
    for mirna in high_tsi_miRNAs:
        file.write(f'{mirna}\n')
