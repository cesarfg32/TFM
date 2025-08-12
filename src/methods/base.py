from abc import ABC, abstractmethod
import torch
class CLMethod(ABC):
    def init(self, model: torch.nn.Module): pass
    def before_task(self, task_id: int, model: torch.nn.Module): pass
    @abstractmethod
    def loss(self, batch, model: torch.nn.Module) -> torch.Tensor: ...
    def after_batch(self, model: torch.nn.Module): pass
    def after_task(self, task_id: int, model: torch.nn.Module): pass
