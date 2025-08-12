import os, h5py, torch, numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

class UdacityH5(Dataset):
    def __init__(self, h5_path: str):
        if not os.path.exists(h5_path): raise FileNotFoundError(h5_path)
        self.h5 = h5py.File(h5_path, 'r'); self.images = self.h5['images']; self.steering = self.h5['steering']
    def __len__(self): return self.images.shape[0]
    def __getitem__(self, idx): return torch.tensor(self.images[idx]), torch.tensor(self.steering[idx]).float()

def _rate_spike(frames: np.ndarray, T: int, gain: float, rng: np.random.Generator):
    rates = np.clip(frames * gain, 0.0, 1.0)
    return rng.binomial(1, rates[None, ...], size=(T,) + rates.shape).astype(np.float32)
def _latency_spike(frames: np.ndarray, T: int):
    t_first = np.rint((T-1) * (1.0 - np.clip(frames, 0.0, 1.0))).astype(np.int32)
    spikes = np.zeros((T,) + frames.shape, dtype=np.float32)
    for t in range(T): spikes[t] = (t_first == t).astype(np.float32)
    return spikes

class OnTheFlySpiking(Dataset):
    def __init__(self, base_ds: UdacityH5, indices, encode='rate', T=20, gain=0.5, seed=42):
        self.base, self.indices, self.encode, self.T, self.gain, self.seed = base_ds, list(indices), encode, T, gain, int(seed)
        sample = self.base.images[0]; self._has_time = (sample.ndim == 4)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx_abs = self.indices[i]; x = self.base.images[idx_abs]; y = self.base.steering[idx_abs]
        if self._has_time: X = torch.tensor(x)
        else:
            frames = np.asarray(x, dtype=np.float32); rng = np.random.default_rng(self.seed + int(idx_abs))
            if self.encode == 'rate': spikes = _rate_spike(frames, self.T, self.gain, rng)
            elif self.encode == 'latency': spikes = _latency_spike(frames, self.T)
            else: spikes = np.repeat(frames[None, ...], self.T, axis=0).astype(np.float32)
            X = torch.tensor(spikes)
        return X, torch.tensor(y).float()

def random_split_indices(n, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed); idx = np.arange(n); rng.shuffle(idx)
    n_val = int(round(n * val_ratio)); return idx[n_val:].tolist(), idx[:n_val].tolist()
def stratified_split_indices(steering, val_ratio=0.2, bins=15, seed=42):
    steering = np.asarray(steering); hist, edges = np.histogram(steering, bins=bins); rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for i in range(bins):
        mask = (steering >= edges[i]) & (steering < edges[i+1]); bidx = np.where(mask)[0]
        if bidx.size == 0: continue; rng.shuffle(bidx); n_val = int(round(bidx.size * val_ratio))
        val_idx.extend(bidx[:n_val].tolist()); train_idx.extend(bidx[n_val:].tolist())
    return train_idx, val_idx

def get_train_val_loaders(h5_path, batch_size=64, num_workers=0, val_ratio=0.2, split='stratified', bins=15, seed=42, encode=None, T=20, gain=0.5):
    base = UdacityH5(h5_path)
    if val_ratio <= 0.0:
        sample = base.images[0]; has_time = (sample.ndim == 4)
        ds = OnTheFlySpiking(base, range(len(base)), encode=encode, T=T, gain=gain, seed=seed) if (encode is not None and not has_time) else base
        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0), prefetch_factor=2 if num_workers>0 else None)
        return train_loader, None
    if split == 'random': train_idx, val_idx = random_split_indices(len(base), val_ratio, seed)
    else: train_idx, val_idx = stratified_split_indices(base.steering[:], val_ratio, bins, seed)
    sample = base.images[0]; has_time = (sample.ndim == 4)
    if encode is not None and not has_time:
        ds_train = OnTheFlySpiking(base, train_idx, encode=encode, T=T, gain=gain, seed=seed); ds_val = OnTheFlySpiking(base, val_idx, encode=encode, T=T, gain=gain, seed=seed)
    else:
        ds_train = Subset(base, train_idx); ds_val = Subset(base, val_idx)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0), prefetch_factor=2 if num_workers>0 else None)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0), prefetch_factor=2 if num_workers>0 else None)
    return train_loader, val_loader

class SimpleStream:
    def __init__(self, loaders, epochs_per_task=3): self._loaders = loaders; self._epochs = epochs_per_task
    def tasks(self): 
        for ld in self._loaders: yield ld
    def epochs_per_task(self, t: int) -> int: return self._epochs

class UdacityH5Stream:
    def __init__(self, root: str, tasks, batch_size=64, num_workers=0, epochs_per_task=3, encode=None, T=20, gain=0.5, seed=42):
        self.root, self.task_ids, self.bs, self.nw, self._epochs = root, tasks, batch_size, num_workers, epochs_per_task
        self.encode, self.T, self.gain, self.seed = encode, T, gain, seed
    def tasks(self):
        for t in self.task_ids:
            path = os.path.join(self.root, f"{t}.h5"); base = UdacityH5(path); sample = base.images[0]; has_time = (sample.ndim == 4)
            ds = OnTheFlySpiking(base, range(len(base)), encode=self.encode, T=self.T, gain=self.gain, seed=self.seed) if (self.encode is not None and not has_time) else base
            yield DataLoader(ds, batch_size=self.bs, shuffle=True, num_workers=self.nw, pin_memory=True, persistent_workers=(self.nw>0), prefetch_factor=2 if self.nw>0 else None)
    def epochs_per_task(self, t: int) -> int: return self._epochs
