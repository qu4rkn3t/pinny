from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np

import torch
from torch.optim import Adam


from problems.base import BaseProblem
from utils.metrics import compute_metrics


@dataclass
class PinnResult:
    model: torch.nn.Module
    metrics: Dict[str, float]
    loss_history: List[float]
    y_pred: np.ndarray
    residual_history: np.ndarray | None = None


def train_pinn(
    model: torch.nn.Module,
    problem: BaseProblem,
    params: Dict[str, Any],
    track_residuals: bool = False,
) -> PinnResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model.to(device=device, dtype=dtype)

    t_train = torch.linspace(
        params["start"], params["stop"], 100, device=device, dtype=dtype
    ).reshape(-1, 1)
    t_train.requires_grad_(True)

    t_eval_res = torch.linspace(
        params["start"], params["stop"], 200, device=device, dtype=dtype
    ).reshape(-1, 1)

    optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    loss_history = []
    residual_history_list = []

    epochs = params.get("epochs", 1000)

    pbar = tqdm(
        range(1, epochs + 1),
        desc="Training",
        leave=False,
        disable=params.get("silent", False),
    )

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        res = problem.residual(model, t_train)
        ode_loss = torch.mean(res**2)
        ic_loss = problem.ic_loss(model)

        loss = params["ode_loss_weight"] * ode_loss + params["ic_loss_weight"] * ic_loss

        if torch.isnan(loss):
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 10 == 0:
            pbar.set_postfix({"loss": loss.item()})

        if track_residuals and epoch % 100 == 0:
            model.eval()
            t_eval_res.requires_grad_(True)
            res_eval = problem.residual(model, t_eval_res)
            residual_history_list.append(res_eval.detach().cpu().numpy().flatten())
            t_eval_res.requires_grad_(False)

    model.eval()
    t_eval = torch.linspace(
        params["start"], params["stop"], 500, device=device, dtype=dtype
    ).reshape(-1, 1)

    with torch.no_grad():
        y_pred_np = model(t_eval).cpu().numpy()

    y_true = problem.exact(t_eval.cpu()).detach().numpy()
    metrics = compute_metrics(y_true, y_pred_np)

    if np.isnan(y_pred_np).any():
        metrics = {k: np.nan for k in metrics}

    residual_history_np = (
        np.stack(residual_history_list) if residual_history_list else None
    )

    return PinnResult(
        model=model,
        metrics=metrics,
        loss_history=loss_history,
        y_pred=y_pred_np,
        residual_history=residual_history_np,
    )
