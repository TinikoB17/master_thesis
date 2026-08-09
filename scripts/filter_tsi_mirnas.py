import pandas as pd

mirnas = pd.read_csv("deconvolution/dataframes/mirna_list.txt")

model_inp = pd.read_csv("model_input_data/blood_healthy_age_scaled.csv", sep='\t').set_index("Sample")

features = ["Age", "Project", "Sequencing_Method", "Sex"]

features_to_keep = [c for c in model_inp.columns.to_list() if c in features or c in mirnas["miRNA"].to_list()]

tsi_df = model_inp[features_to_keep]

tsi_df.to_csv("model_input_data/filtered_tsi.csv", sep='\t')
