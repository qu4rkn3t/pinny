import torch
from dataclasses import dataclass

from problems.base import BaseProblem


@dataclass(frozen=True)
class ExpDecay(BaseProblem):
    name: str = "Exponential Decay"

    k: float = 1.0

    def residual(self, model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
        y = model(t)

        dy_dt = torch.autograd.grad(
            outputs=y,
            inputs=t,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
        )[0]

        return dy_dt + self.k * y

    def ic_loss(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        t0 = torch.zeros((1, 1), device=device, dtype=dtype)

        y0_true = torch.ones((1, 1), device=device, dtype=dtype)
        y0_pred = model(t0)

        return torch.nn.functional.mse_loss(y0_pred, y0_true)

    def exact(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.k * t)


@dataclass(frozen=True)
class LogisticGrowth(BaseProblem):
    name: str = "Logistic Growth"

    r: float = 1.2
    K: float = 1.0
    y0: float = 0.1

    def residual(self, model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
        y = model(t)

        dy_dt = torch.autograd.grad(
            outputs=y,
            inputs=t,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
        )[0]

        return dy_dt - self.r * y * (1.0 - y / self.K)

    def ic_loss(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        t0 = torch.zeros((1, 1), device=device, dtype=dtype)

        y0_true = torch.tensor([[self.y0]], device=device, dtype=dtype)
        y0_pred = model(t0)

        return torch.nn.functional.mse_loss(y0_pred, y0_true)

    def exact(self, t: torch.Tensor) -> torch.Tensor:
        return self.K / (1.0 + ((self.K - self.y0) / self.y0) * torch.exp(-self.r * t))


@dataclass(frozen=True)
class HarmonicOscillator(BaseProblem):
    name: str = "Harmonic Oscillator"

    w: float = 2.0
    y0: float = 1.0
    v0: float = 0.0

    def residual(self, model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
        y = model(t)

        dy_dt = torch.autograd.grad(
            outputs=y,
            inputs=t,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
        )[0]
        d2y_dt2 = torch.autograd.grad(
            outputs=dy_dt,
            inputs=t,
            grad_outputs=torch.ones_like(dy_dt),
            create_graph=True,
        )[0]

        return d2y_dt2 + (self.w**2) * y

    def ic_loss(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        t0 = torch.zeros((1, 1), device=device, dtype=dtype, requires_grad=True)

        y0_true = torch.tensor([[self.y0]], device=device, dtype=dtype)
        v0_true = torch.tensor([[self.v0]], device=device, dtype=dtype)

        y0_pred = model(t0)
        v0_pred = torch.autograd.grad(
            outputs=y0_pred,
            inputs=t0,
            grad_outputs=torch.ones_like(y0_pred),
            create_graph=True,
        )[0]

        position_loss = torch.nn.functional.mse_loss(y0_pred, y0_true)
        velocity_loss = torch.nn.functional.mse_loss(v0_pred, v0_true)

        return position_loss + velocity_loss

    def exact(self, t: torch.Tensor) -> torch.Tensor:
        return self.y0 * torch.cos(self.w * t) + (self.v0 / self.w) * torch.sin(
            self.w * t
        )
