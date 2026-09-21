"""Real CPU-bound task kernels with a heavy-tailed size distribution."""
import hashlib
import time
from dataclasses import dataclass

import numpy as np


def _matmul(n):
    a = np.random.default_rng(n).random((n, n))
    return float((a @ a).sum())


def _fft(n):
    x = np.random.default_rng(n).random(n)
    for _ in range(4):
        x = np.fft.irfft(np.fft.rfft(x), n)
    return float(x.sum())


def _sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return int(s.sum())


def _hash(n):
    h = b"qsched"
    for _ in range(n):
        h = hashlib.sha256(h).digest()
    return h.hex()


KERNELS = {"matmul": (_matmul, 60, 420), "fft": (_fft, 1 << 16, 1 << 21),
           "sieve": (_sieve, 200_000, 8_000_000), "hash": (_hash, 20_000, 600_000)}


@dataclass
class Task:
    kind: str
    size: int
    cost: float = 0.0  # measured seconds, filled by calibrate()

    def run(self):
        return KERNELS[self.kind][0](self.size)


def make_taskset(n, seed=0):
    """n tasks, sizes log-uniform so a few are much heavier than the rest."""
    rng = np.random.default_rng(seed)
    kinds = list(KERNELS)
    tasks = []
    for _ in range(n):
        k = kinds[rng.integers(len(kinds))]
        _, lo, hi = KERNELS[k]
        tasks.append(Task(k, int(np.exp(rng.uniform(np.log(lo), np.log(hi))))))
    return tasks


def calibrate(tasks, repeats=2):
    """Measure each task's runtime on this machine (best of `repeats`)."""
    for t in tasks:
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            t.run()
            best = min(best, time.perf_counter() - t0)
        t.cost = best
    return tasks
