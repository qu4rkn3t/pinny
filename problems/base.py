import abc
import torch


class BaseProblem(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def residual(self, model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def ic_loss(self, model: torch.nn.Module) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def exact(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
