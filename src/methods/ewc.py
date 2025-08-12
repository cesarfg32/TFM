import torch
from .base import CLMethod
class EWC(CLMethod):
    def __init__(self, lam=0.2, fisher_batches=256, criterion=None):
        self.lam = lam; self.fisher_batches = fisher_batches
        self.criterion = criterion or torch.nn.MSELoss()
        self.prev_params = {}; self.fisher = {}
    def _ewc_penalty(self, model):
        pen = 0.0
        for n,p in model.named_parameters():
            if p.requires_grad and (n in self.prev_params):
                F = self.fisher.get(n, torch.zeros_like(p)); pen = pen + (F * (p - self.prev_params[n]).pow(2)).sum()
        return pen
    def loss(self, batch, model):
        x,y = batch; y_hat = model(x); base = self.criterion(y_hat, y)
        return base + (self.lam * self._ewc_penalty(model) if self.prev_params else 0.0)
    def estimate_and_snapshot(self, loader, model, device):
        fisher = {n: torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
        used = 0
        for batch in loader:
            if used >= self.fisher_batches: break
            if isinstance(batch,(list,tuple)):
                batch = [b.to(device) if hasattr(b,"to") else b for b in batch]
            x,y = batch; model.zero_grad(set_to_none=True)
            loss = self.criterion(model(x), y); loss.backward()
            for (n,p) in model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += (p.grad.detach() ** 2)
            used += len(x)
        for n in fisher: fisher[n] = fisher[n] / max(1, used)
        self.fisher = fisher
        self.prev_params = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
