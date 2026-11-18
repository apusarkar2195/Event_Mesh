import time
import torch
from collections import defaultdict

class SLAMProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}

    def start(self, name):
        # Synchronize to get accurate GPU timings
        torch.cuda.synchronize()
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        torch.cuda.synchronize()
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timings[name].append(elapsed)
            del self.start_times[name]

    def total_seconds(self, name):
        """Total elapsed time (seconds) for a given timer name."""
        return sum(self.timings[name]) if name in self.timings else 0.0

    def summary(self):
        print(f"\n{'='*10} SLAM PERFORMANCE SUMMARY {'='*10}")
        print(f"{'Task':<25} | {'Avg (ms)':<10} | {'Total (s)':<10}")
        print("-" * 50)
        for name, values in self.timings.items():
            avg_ms = (sum(values) / len(values)) * 1000
            total_s = sum(values)
            print(f"{name:<25} | {avg_ms:>10.2f} | {total_s:>10.3f}")
        print(f"{'='*46}\n")