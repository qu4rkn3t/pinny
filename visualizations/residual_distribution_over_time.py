import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any


def plot_residual_distribution_over_time(
    residual_history: np.ndarray, params: Dict[str, Any], title: str, save_dir: str
):
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs = np.arange(1, residual_history.shape[0] + 1) * 100

    mean_residuals = np.mean(np.abs(residual_history), axis=1)
    std_residuals = np.std(residual_history, axis=1)

    ax.plot(epochs, mean_residuals, label="Mean Absolute Residual", color="green")
    ax.fill_between(
        epochs,
        mean_residuals - std_residuals,
        mean_residuals + std_residuals,
        color="green",
        alpha=0.2,
        label="Residual Std. Dev.",
    )

    ax.set_yscale("log")
    ax.set_title(f"Residual Distribution vs. Epoch for {title}")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("ODE Residual (log scale)")
    ax.legend()
    ax.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_distribution.png"))
    plt.close()
