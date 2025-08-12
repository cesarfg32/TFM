import torch
from .base import CLMethod
class As_Snn(CLMethod):
    def __init__(self): self.mse = torch.nn.MSELoss()
    def loss(self, batch, model): x,y = batch; return self.mse(model(x), y)
