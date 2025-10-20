import matplotlib.pyplot as plt
import numpy as np
import os


def plot_solution_comparison(
    t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, title: str, save_dir: str
):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(
        t,
        y_true,
        label="Exact Solution",
        color="black",
        ls="--",
    )
    ax1.plot(t, y_pred, label="PINN Solution", color="red")
    ax1.set_title(f"Solution Comparison for {title}")
    ax1.set_ylabel("y(t)")
    ax1.legend()
    ax1.grid(True, which="both")

    error = np.abs(y_true - y_pred)
    ax2.plot(t, error, label="Absolute Error", color="orange")
    ax2.fill_between(t.flatten(), error.flatten(), color="orange", alpha=0.2)
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Absolute Error")
    ax2.set_yscale("log")
    ax2.grid(True, which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "solution_and_error.png"))
    plt.close()
