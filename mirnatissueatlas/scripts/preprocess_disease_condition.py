import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

def alter_disease_condition(annotated_human_miRNA: sc.AnnData):

    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"].str.contains("Healthy"), "Disease_Condition"] = "Healthy"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"].str.contains("healthy"), "Disease_Condition"] = "Healthy"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"] == "Parkinson’s disease", "Disease_Condition"] = "PD"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"].str.contains("normal"), "Disease_Condition"] = "Normal"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"].str.contains("adenocarcinoma", case=False), "Disease_Condition"] = "adenocarcinoma"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"].str.contains(" carcinoma"), "Disease_Condition"] = "carcinoma"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"].str.contains("cancer"), "Disease_Condition"] = "Cancer"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"].isin(["myeloma", "Multiple myeloma"]), "Disease_Condition"] = "carcinoma"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"] == "benign", "Disease_Condition"] = "tumor"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"] == "control", "Disease_Condition"] = "Healthy"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"] == "C", "Disease_Condition"] = "Healthy"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"].str.contains("Control"), "Disease_Condition"] = "Healthy"
    annotated_human_miRNA.obs.loc[annotated_human_miRNA.obs["Disease_Condition"].isin(["unknown", "Not available", "Not known"]), "Disease_Condition"] = "unknown"
    return annotated_human_miRNA

