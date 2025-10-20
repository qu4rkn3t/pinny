import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


def plot_hyperparameter_correlation_matrix(df: pd.DataFrame, save_dir: str):
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()

    metric_correlations = corr[["RMSE", "R2"]].drop(["RMSE", "R2"])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        metric_correlations,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Hyperparameter Correlation with Performance", weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hyperparameter_correlation_matrix.png"))
    plt.close()
