import pandas as pd

tsi_path = snakemake.input.tsi
mirna_list_path = snakemake.output.mirnas

tsi = pd.read_csv(tsi_path, sep='\t')
high_tsi_miRNAs = tsi[tsi["TSI"] >= 0.6]["miRNA"].to_list()

with open(mirna_list_path, 'w') as file:
    file.write("miRNA\n")
    for mirna in high_tsi_miRNAs:
        file.write(f'{mirna}\n')
