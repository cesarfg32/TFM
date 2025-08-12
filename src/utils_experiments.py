import os, json, hashlib, time, platform, torch
def _short_hash(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:6]
def build_out_dir(base_dir="outputs/runs", method="method", cfg: dict=None, encoder_meta: dict=None):
    cfg = cfg or {}; encoder_meta = encoder_meta or {}
    parts = [method]
    enc = encoder_meta.get("encoding", None)
    if enc and enc != "none": parts.append(str(enc))
    T = encoder_meta.get("num_steps", None); g = encoder_meta.get("gain", None)
    if T is not None: parts.append(f"T{T}")
    if g is not None: parts.append(f"g{g}")
    if 'lr' in cfg: parts.append(f"lr{cfg['lr']}")
    if 'seed' in cfg: parts.append(f"seed{cfg['seed']}")
    slug = "_".join(str(p) for p in parts)
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + slug + "__" + _short_hash({**cfg, **encoder_meta})
    out_dir = os.path.join(base_dir, run_id); os.makedirs(out_dir, exist_ok=True); return out_dir
def write_manifest(out_dir, method, tasks, encoder_meta, cfg):
    mani = {"method": method, "tasks": tasks, "encoder": encoder_meta, "config": {k:v for k,v in cfg.items() if k not in ("out","out_base")},
            "system": {"python": platform.python_version(), "torch": torch.__version__,
                       "cuda_available": torch.cuda.is_available(),
                       "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(mani, f, indent=2, ensure_ascii=False)
