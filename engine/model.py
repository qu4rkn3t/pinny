import torch
import torch.nn as nn
from typing import Callable


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        hidden_dim: int,
        activation: Callable,
    ) -> None:
        super().__init__()
        self.activation = activation()

        layers = [nn.Linear(input_dim, width), self.activation]
        for _ in range(hidden_dim - 1):
            layers.extend([nn.Linear(width, width), self.activation])
        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: nn.Module):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out += identity
        return self.activation(out)


class ResNetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        hidden_dim: int,
        activation: Callable,
    ) -> None:
        super().__init__()
        act_func = activation()

        self.input_layer = nn.Linear(input_dim, width)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(width, act_func) for _ in range(hidden_dim)]
        )
        self.output_layer = nn.Linear(width, output_dim)
        self.activation = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x
