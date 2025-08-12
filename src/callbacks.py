import csv, os, time, json, torch
class CSVLogger:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, 'metrics.csv')
        self._init = False; self.epoch = 0; self.task = 0
    def before_task(self, t): self.task = t
    def after_batch(self, loss):
        if not self._init:
            with open(self.path, 'w', newline='') as f: csv.writer(f).writerow(['task','epoch','loss'])
            self._init = True
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow([self.task, self.epoch, float(loss.detach().cpu())])
    def after_epoch(self, epoch): self.epoch = epoch + 1
class TimingLogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir; self.timings = {"tasks": []}; self._task_start=None; self._current=None
    def before_task(self, t):
        self._task_start = time.perf_counter(); self._current = {"task": t, "epochs": [], "cuda_max_mem": None}
    def after_task(self, t):
        dur = time.perf_counter() - self._task_start if self._task_start else None
        if torch.cuda.is_available():
            mem = int(torch.cuda.max_memory_allocated()); torch.cuda.reset_peak_memory_stats()
        else: mem = None
        self._current["duration_sec"] = dur; self._current["cuda_max_mem"] = mem
        self.timings["tasks"].append(self._current); self._current=None
    def after_fit(self):
        with open(os.path.join(self.out_dir, "timings.json"), "w", encoding="utf-8") as f:
            json.dump(self.timings, f, indent=2)
