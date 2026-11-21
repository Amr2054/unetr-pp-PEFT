"""Time optimizer steps or CT predictions and report peak allocated CUDA memory."""
import json
from statistics import median
from time import perf_counter

import torch


class ComputeProfile:
    def __init__(self, device):
        self.device = device
        self.cuda = device.type == "cuda"
        self.samples = []
        self.timings = []

    def start(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.begin = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.begin.record()
        else:
            self.begin = perf_counter()

    def stop(self, **sample):
        if self.cuda:
            self.end.record()
            self.timings.append((self.begin, self.end))
            peak = torch.cuda.max_memory_allocated(self.device) / 2**30
        else:
            self.timings.append((perf_counter() - self.begin) * 1000)
            peak = None
        self.samples.append(dict(sample, peak_allocated_gpu_gib=peak))

    def save(self, directory, phase, **settings):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            durations_ms = [start.elapsed_time(end) for start, end in self.timings]
            peak = max(s["peak_allocated_gpu_gib"] for s in self.samples)
        else:
            durations_ms = self.timings
            peak = None
        training = phase == "training"
        metric = "median_ms_per_step" if training else "median_seconds_per_ct"
        unit = "ms/step" if training else "s/CT"
        durations = durations_ms if training else [ms / 1000 for ms in durations_ms]
        for sample, duration in zip(self.samples, durations):
            sample["ms" if training else "seconds"] = duration
        report = dict(settings, phase=phase, device=str(self.device), torch_version=torch.__version__,
                      gpu=torch.cuda.get_device_name(self.device) if self.cuda else None,
                      peak_allocated_gpu_gib=peak, samples=self.samples)
        report[metric] = median(durations)
        path = directory / "compute_metrics.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        memory = f"{peak:.3f} GiB peak allocated" if self.cuda else "CPU timing"
        return f"{phase} profile | median {report[metric]:.3f} {unit} | {memory} | {path}"
