import os, csv, h5py, cv2, numpy as np
from typing import Tuple
from tqdm import tqdm
def load_udacity_csv(csv_path: str) -> list:
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 7: continue
            rows.append({"center": r[0].strip(), "left": r[1].strip(), "right": r[2].strip(),
                         "steering": float(r[3]), "throttle": float(r[4]), "brake": float(r[5]), "speed": float(r[6])})
    return rows
def balance_steering_data(rows: list, num_bins: int = 25, samples_per_bin: int = 200) -> list:
    steer = np.array([r["steering"] for r in rows]); hist, bins = np.histogram(steer, num_bins); keep = []
    for i in range(num_bins):
        bin_idx = np.where((steer >= bins[i]) & (steer < bins[i+1]))[0]
        if len(bin_idx) > samples_per_bin: bin_idx = np.random.choice(bin_idx, samples_per_bin, replace=False)
        keep.extend(bin_idx.tolist())
    keep = sorted(keep); return [rows[i] for i in keep]
def preprocess_image(path: str, to_gray: bool = True, resize: Tuple[int,int] = (160,320)) -> np.ndarray:
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); img = cv2.resize(img, (resize[1], resize[0]))
        img = img.astype(np.float32) / 255.0; img = np.expand_dims(img, axis=0)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); img = cv2.resize(img, (resize[1], resize[0]))
        img = img.astype(np.float32) / 255.0; img = np.transpose(img, (2,0,1))
    return img
def encode_rate(frames: np.ndarray, num_steps: int = 8, gain: float = 1.0) -> np.ndarray:
    C,H,W = frames.shape; T = num_steps; rates = np.clip(frames * gain, 0.0, 1.0)
    return np.random.binomial(1, rates[None, ...], size=(T, C, H, W)).astype(np.float32)
def encode_latency(frames: np.ndarray, num_steps: int = 8) -> np.ndarray:
    C,H,W = frames.shape; T = num_steps; t_first = np.rint((T-1) * (1.0 - np.clip(frames, 0.0, 1.0))).astype(np.int32)
    spikes = np.zeros((T,C,H,W), dtype=np.float32)
    for t in range(T): spikes[t] = (t_first == t).astype(np.float32)
    return spikes
def build_h5_from_csv(csv_path: str, out_h5: str, image_root: str = None, use_camera: str = "center",
                      balance: bool = True, num_bins: int = 25, samples_per_bin: int = 200,
                      to_gray: bool = True, resize: Tuple[int,int] = (160,320),
                      encode: str = None, num_steps: int = 8, gain: float = 1.0):
    rows = load_udacity_csv(csv_path)
    if balance: rows = balance_steering_data(rows, num_bins=num_bins, samples_per_bin=samples_per_bin)
    def resolve(p): return os.path.join(image_root, os.path.basename(p)) if image_root and not os.path.isabs(p) else p
    X, y = [], []
    for r in tqdm(rows, desc="Procesando"):
        img_path = resolve(r[use_camera]); img = preprocess_image(img_path, to_gray=to_gray, resize=resize)
        if encode == 'rate': img = encode_rate(img, num_steps=num_steps, gain=gain)
        elif encode == 'latency': img = encode_latency(img, num_steps=num_steps)
        X.append(img); y.append(r["steering"])
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, "w") as h5:
        h5.create_dataset("images", data=X, compression="gzip", compression_opts=4)
        h5.create_dataset("steering", data=y)
        h5.attrs["encoding"] = encode or "none"; h5.attrs["num_steps"] = int(num_steps); h5.attrs["gain"] = float(gain)
        h5.attrs["to_gray"] = bool(to_gray); h5.attrs["resize_h"] = int(resize[0]); h5.attrs["resize_w"] = int(resize[1])
        h5.attrs["use_camera"] = (use_camera or "center"); h5.attrs["balanced"] = bool(balance)
        h5.attrs["num_bins"] = int(num_bins); h5.attrs["samples_per_bin"] = int(samples_per_bin)
    return out_h5
