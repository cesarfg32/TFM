import torch
from contextlib import nullcontext
class Trainer:
    def __init__(self, model, method, datastream, optimizer, callbacks=None, device=None, amp='off'):
        self.model, self.method, self.ds = model, method, datastream
        self.optimizer, self.cbs = optimizer, (callbacks or [])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp
    def fit(self):
        self.model.to(self.device)
        if hasattr(self.method, "init"): self.method.init(self.model)
        use_cuda = (self.device == "cuda" and torch.cuda.is_available())
        if self.amp == 'bf16' and use_cuda:
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16); scaler = None
        elif self.amp == 'fp16' and use_cuda:
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16); scaler = torch.cuda.amp.GradScaler()
        else:
            autocast_ctx = nullcontext(); scaler = None
        try: torch.set_float32_matmul_precision('high')
        except Exception: pass
        for t, loader in enumerate(self.ds.tasks()):
            for cb in self.cbs:
                if hasattr(cb,"before_task"): cb.before_task(t)
            if hasattr(self.method,"before_task"): self.method.before_task(t, self.model)
            for epoch in range(self.ds.epochs_per_task(t)):
                for batch in loader:
                    if isinstance(batch,(list,tuple)):
                        batch = [b.to(self.device) if hasattr(b,"to") else b for b in batch]
                    else:
                        batch = batch.to(self.device) if hasattr(batch,"to") else batch
                    with autocast_ctx:
                        loss = self.method.loss(batch, self.model)
                    self.optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward(); scaler.step(self.optimizer); scaler.update()
                    else:
                        loss.backward(); self.optimizer.step()
                    for cb in self.cbs:
                        if hasattr(cb,"after_batch"): cb.after_batch(loss)
            if hasattr(self.method,"estimate_and_snapshot"):
                self.method.estimate_and_snapshot(loader, self.model, self.device)
            if hasattr(self.method,"after_task"): self.method.after_task(t, self.model)
            for cb in self.cbs:
                if hasattr(cb,"after_task"): cb.after_task(t)
        for cb in self.cbs:
            if hasattr(cb,"after_fit"): cb.after_fit()
