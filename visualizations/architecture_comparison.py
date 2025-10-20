import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_architecture_comparison(
    df: pd.DataFrame, metric: str, title: str, save_dir: str
):
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df_clean = df.dropna(subset=[metric])
    if df_clean.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(
        x="model_class",
        y=metric,
        data=df_clean,
        ax=ax,
        palette="muted",
        inner="quartile",
        cut=0,
    )

    ax.set_title(title, weight="bold")
    ax.set_xlabel("Model Architecture")
    if metric in ["RMSE", "MAE", "MSE"]:
        ax.set_yscale("log")
        ax.set_ylabel(f"{metric} (log scale)")
    else:
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"arch_comparison_{metric}.png"))
    plt.close()
