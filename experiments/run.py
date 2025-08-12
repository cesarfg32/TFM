import argparse, yaml, os, torch, h5py
from src.utils import set_seed
from src.datasets import UdacityH5Stream, get_train_val_loaders, SimpleStream
from src.model import PilotNetSNN
from src.callbacks import CSVLogger, TimingLogger
from src.trainer import Trainer
from src.utils_experiments import build_out_dir, write_manifest
import src.methods.ewc, src.methods.as_snn, src.methods.sa_snn, src.methods.sca_snn, src.methods.colanet

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--data-root', default='data/udacity')
    ap.add_argument('--tasks', default='circuit1,circuit2')
    ap.add_argument('--method', default='ewc')
    ap.add_argument('--mode', choices=['continual','supervised'], default='continual')
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--split', choices=['stratified','random'], default='stratified')
    ap.add_argument('--bins', type=int, default=15)
    ap.add_argument('--preset', choices=['fast','std','accurate','offline'], default='std')
    ap.add_argument('--encode', choices=['none','rate','latency'], default=None)
    ap.add_argument('--T', type=int, default=None)
    ap.add_argument('--gain', type=float, default=None)
    ap.add_argument('--amp', choices=['off','fp16','bf16'], default=None)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_base', default='outputs/runs')
    ap.add_argument('--use-codecarbon', action='store_true')
    return ap.parse_args()

def read_h5_attrs(h5_path):
    if not os.path.exists(h5_path): return {}
    with h5py.File(h5_path, 'r') as h5:
        return {k: h5.attrs[k] for k in h5.attrs.keys()}

def detect_in_channels(h5_path):
    if not os.path.exists(h5_path): return 3
    with h5py.File(h5_path, 'r') as h5:
        shp = h5['images'].shape
    return shp[2] if len(shp)==5 else shp[1]

def apply_preset(args, encoder_meta):
    presets = {'fast':{'encode':'rate','T':12,'gain':0.5,'amp':'bf16'},
               'std':{'encode':'rate','T':20,'gain':0.5,'amp':'bf16'},
               'accurate':{'encode':'rate','T':32,'gain':0.6,'amp':'bf16'},
               'offline':{'encode':None,'T':None,'gain':None,'amp':'bf16'}}
    p = presets[args.preset]
    tasks = args.tasks.split(','); first_h5 = os.path.join(args.data_root, f"{tasks[0]}.h5")
    has_time = False
    if os.path.exists(first_h5):
        with h5py.File(first_h5,'r') as h5:
            has_time = (h5['images'].ndim == 5)
    encode = args.encode if args.encode is not None else p['encode']
    T      = args.T if args.T is not None else p['T']
    gain   = args.gain if args.gain is not None else p['gain']
    amp    = args.amp if args.amp is not None else p['amp']
    if has_time: encode = None
    return encode, T, gain, amp

if __name__ == '__main__':
    args = parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config,'r') as f: cfg = yaml.safe_load(f)
        for k,v in cfg.items():
            if hasattr(args,k): setattr(args,k,v)

    set_seed(args.seed)
    tasks = args.tasks.split(',')
    first_h5 = os.path.join(args.data_root, f"{tasks[0]}.h5")
    encoder_meta = read_h5_attrs(first_h5)
    encode, T, gain, amp = apply_preset(args, encoder_meta)

    cfg_dict = vars(args).copy(); cfg_dict.update({'encode':encode,'T':T,'gain':gain,'amp':amp})
    out_dir = build_out_dir(base_dir=args.out_base, method=args.method, cfg=cfg_dict, encoder_meta=encoder_meta)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f: yaml.safe_dump(cfg_dict, f, sort_keys=False)
    write_manifest(out_dir, args.method, tasks, encoder_meta, cfg_dict)

    in_ch = detect_in_channels(first_h5)
    model = PilotNetSNN(in_channels=in_ch)

    from src.methods.ewc import EWC
    method = EWC()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    cbs = [CSVLogger(out_dir), TimingLogger(out_dir)]

    tracker = None
    if args.use_codecarbon:
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(output_dir=out_dir); tracker.start()
        except Exception as e:
            print("[WARN] CodeCarbon no disponible:", e)

    if args.mode == 'continual':
        ds = UdacityH5Stream(root=args.data_root, tasks=tasks, batch_size=args.batch_size, num_workers=args.num_workers,
                             epochs_per_task=args.epochs, encode=encode, T=T or 1, gain=gain or 1.0, seed=args.seed)
    else:
        h5_path = first_h5
        train_loader, val_loader = get_train_val_loaders(h5_path, batch_size=args.batch_size, num_workers=args.num_workers,
                                                         val_ratio=args.val_ratio, split=args.split, bins=args.bins, seed=args.seed,
                                                         encode=encode, T=T or 1, gain=gain or 1.0)
        loaders = [train_loader] + ([val_loader] if val_loader is not None else [])
        ds = SimpleStream(loaders, epochs_per_task=args.epochs)

    Trainer(model, method, ds, optim, callbacks=cbs, amp=amp).fit()
    if tracker is not None:
        try: tracker.stop()
        except Exception: pass
