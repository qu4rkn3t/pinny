import matplotlib.pyplot as plt
import numpy as np
import os


def plot_loss_curve(loss_history: list, title: str, save_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(len(loss_history))

    ax.plot(epochs, loss_history, color="blue")
    ax.set_yscale("log")
    ax.set_title(f"Training Loss for {title}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
