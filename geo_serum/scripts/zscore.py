"""Per-project z-scoring with statistics from the training set, as in the miRNATissueAtlas pipeline
(mirnatissueatlas/scripts/zscore.py). Here a "project" is a control group (Disease_State),
the unit that separates in the UMAP.

Each miRNA is centred and scaled within its project with the project's training mean and SD;
the test set uses the same training stats (a project missing from training gets the global
training stats). This removes differences between projects, including any real age difference
between them, and keeps the age signal within each project."""
import pandas as pd

TRAIN, HOLDOUT, METADATA = snakemake.input.train, snakemake.input.holdout, snakemake.input.metadata
PROJECT = "Disease_State"


def compute_train_stats(train: pd.DataFrame, projects: pd.Series, mirnas: list):
    grouped = train[mirnas].groupby(projects)
    return grouped.mean(), grouped.std(ddof=1), train[mirnas].mean(), train[mirnas].std(ddof=1)


def apply_zscore(data: pd.DataFrame, projects: pd.Series, mirnas: list, train_stats) -> pd.DataFrame:
    project_means, project_stds, global_mean, global_std = train_stats
    known = projects.isin(project_means.index)
    if (~known).any():
        print(f"Projects not in the training set, using global train stats: "
              f"{projects[~known].unique().tolist()}")
    means = project_means.reindex(projects).set_index(data.index).fillna(global_mean)
    stds = project_stds.reindex(projects).set_index(data.index).fillna(global_std)
    # Constant miRNAs within a project (e.g. never detected): centre only, so they become 0
    stds = stds.mask(stds == 0, 1.0).fillna(1.0)
    data = data.copy()
    data[mirnas] = ((data[mirnas] - means[mirnas]) / stds[mirnas]).fillna(0)
    return data


def main():
    train = pd.read_csv(TRAIN, sep="\t", index_col=0)
    holdout = pd.read_csv(HOLDOUT, sep="\t", index_col=0)
    metadata = pd.read_csv(METADATA, sep="\t", index_col=0)
    mirnas = [column for column in train.columns if column not in ("Age", "Sex")]

    train_projects = metadata.loc[train.index, PROJECT]
    stats = compute_train_stats(train, train_projects, mirnas)
    apply_zscore(train, train_projects, mirnas, stats).to_csv(snakemake.output.train, sep="\t")
    apply_zscore(holdout, metadata.loc[holdout.index, PROJECT], mirnas, stats).to_csv(
        snakemake.output.holdout, sep="\t")
    metadata.to_csv(snakemake.output.metadata, sep="\t")

    sizes = train_projects.value_counts()
    print(f"z-scored {len(mirnas)} miRNAs within {len(sizes)} projects "
          f"(training sizes {sizes.min()}-{sizes.max()})")


if __name__ == "__main__":
    main()
