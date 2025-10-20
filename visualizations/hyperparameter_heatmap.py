import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_hyperparameter_heatmap(
    df: pd.DataFrame, x_param: str, y_param: str, metric: str, title: str, save_dir: str
):
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df_clean = df.dropna(subset=[metric])
    if df_clean.empty:
        return

    pivot_table = df_clean.pivot_table(
        values=metric, index=y_param, columns=x_param, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_table, annot=True, fmt=".4f", cmap="viridis", linewidths=0.5, ax=ax
    )

    ax.set_title(title, weight="bold", pad=20)
    ax.set_xlabel(x_param.replace("_", " ").title())
    ax.set_ylabel(y_param.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"heatmap_{metric}.png"))
    plt.close()
