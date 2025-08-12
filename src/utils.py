import torch, random, numpy as np
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
REG = {"method": {}, "model": {}, "dataset": {}}
def register(kind, name):
    def deco(cls):
        REG[kind][name] = cls
        return cls
    return deco
