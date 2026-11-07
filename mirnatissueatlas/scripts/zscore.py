import pandas as pd
import scanpy as sc

# Per-project z-scoring with statistics from the (healthy) training set.
# Used by model_preprocessing_train_stats.py.


def add_obs_columns(annotated_object: sc.AnnData, columns_to_add: list) -> pd.DataFrame:
    mirna_expression = annotated_object.to_df()
    obs_dataframe = sc.get.obs_df(annotated_object, keys=columns_to_add)
    return mirna_expression.join(obs_dataframe, how="left")


def compute_train_stats(train_df: pd.DataFrame, mirna_cols: list):
    grouped = train_df.groupby("Project", observed=True)[mirna_cols]
    project_means = grouped.mean()
    project_stds = grouped.std(ddof=1)

    global_mean = train_df[mirna_cols].mean()
    global_std = train_df[mirna_cols].std(ddof=1)

    return project_means, project_stds, global_mean, global_std


def apply_zscore(df: pd.DataFrame, mirna_cols: list, train_stats) -> pd.DataFrame:
    project_means, project_stds, global_mean, global_std = train_stats
    df = df.copy()
    projects = df["Project"].astype(str)

    known = projects.isin(project_means.index.astype(str))
    if (~known).any():
        print(f"Projects not in training set, using global train stats: {projects[~known].unique().tolist()}")

    means = project_means.reindex(projects).set_index(df.index)
    stds = project_stds.reindex(projects).set_index(df.index)
    means = means.fillna(global_mean)
    stds = stds.fillna(global_std)

    # Constant miRNAs within a project: centre only, so they become 0
    stds = stds.mask(stds == 0, 1.0).fillna(1.0)

    df[mirna_cols] = (df[mirna_cols] - means[mirna_cols]) / stds[mirna_cols]
    df[mirna_cols] = df[mirna_cols].fillna(0)
    return df
