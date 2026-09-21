"""python -m qcmp.sched {info|bench|offload} ..."""
import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

from . import solvers as S
from .monitor import snapshot
from .runner import execute
from .tasks import calibrate, make_taskset


def cmd_info(_):
    print(json.dumps(snapshot(), indent=2))


def cmd_bench(a):
    m = a.workers or max(2, (os.cpu_count() or 2) // 2)
    print(f"Calibrating {a.tasks} tasks ...")
    tasks = calibrate(make_taskset(a.tasks, a.seed))
    costs = np.array([t.cost for t in tasks])
    speeds = [1.0] * m
    print(f"total work {costs.sum():.2f}s, heaviest task {costs.max():.2f}s, "
          f"ideal makespan on {m} workers {max(costs.sum() / m, costs.max()):.2f}s")
    sched = {"round_robin": S.round_robin(costs, speeds),
             "lpt": S.lpt_greedy(costs, speeds),
             "anneal": S.anneal(costs, speeds, seed=a.seed)}
    rows = []
    for name, assign in sched.items():
        pred = S.makespan(costs, assign, speeds)
        walls = [execute(tasks, assign, m) for _ in range(a.repeats)]
        rows.append({"scheduler": name, "predicted_s": pred,
                     "measured_s": statistics.median(walls), "runs_s": walls})
    base = rows[0]["measured_s"]
    print(f"\n{'scheduler':<12}{'predicted':>11}{'measured':>11}{'speedup':>9}")
    for r in rows:
        r["speedup_vs_round_robin"] = base / r["measured_s"]
        print(f"{r['scheduler']:<12}{r['predicted_s']:>10.3f}s{r['measured_s']:>10.3f}s"
              f"{r['speedup_vs_round_robin']:>8.2f}x")
    out = {"host": platform.node(), "system": snapshot(), "workers": m, "tasks": a.tasks,
           "seed": a.seed, "total_work_s": float(costs.sum()), "rows": rows}
    os.makedirs("results", exist_ok=True)
    path = f"results/bench_{platform.node() or 'host'}_{int(time.time())}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}")


def cmd_qaoa(a):
    rng = np.random.default_rng(a.seed)
    print(f"{'n':>3}{'optimal':>9}{'lpt':>9}{'qaoa':>9}{'anneal':>9}   (2 machines, relative to optimal)")
    for n in range(6, a.max_tasks + 1, 2):
        costs = rng.lognormal(0, 1.0, n)
        opt = S.makespan(costs, S.exact_optimum(costs, [1, 1], limit=2 ** 20), [1, 1])
        lpt = S.makespan(costs, S.lpt_greedy(costs, [1, 1]), [1, 1])
        qa, _ = S.qaoa_partition(costs, layers=a.layers, seed=a.seed)
        qa = S.makespan(costs, qa, [1, 1])
        an = S.makespan(costs, S.anneal(costs, [1, 1], steps=5000), [1, 1])
        print(f"{n:>3}{opt:>9.3f}{lpt / opt:>9.3f}{qa / opt:>9.3f}{an / opt:>9.3f}")


def cmd_offload(a):
    """Model-based phone<->PC offload: NOT a measurement of two real devices."""
    tasks = calibrate(make_taskset(a.tasks, a.seed), repeats=1)
    costs = np.array([t.cost for t in tasks])
    speeds = [1.0, a.phone_speed]  # PC=1.0, phone relative speed
    mb = np.full(len(costs), a.mb_per_task)
    def plan_time(assign):
        loads = np.zeros(2)
        for c, m, x in zip(costs, mb, assign):
            loads[x] += c / speeds[x] + (m / a.mbps if x == 1 else 0.0)
        return loads.max()
    plans = {"all on PC": np.zeros(len(costs), int), "round robin": S.round_robin(costs, speeds),
             "lpt (speed-aware)": S.lpt_greedy(costs, speeds)}
    best, best_v = None, float("inf")
    for seed in range(5):
        cand = S.anneal(costs, speeds, seed=seed, start=plans["lpt (speed-aware)"])
        v = plan_time(cand)
        if v < best_v:
            best, best_v = cand, v
    plans["anneal"] = best
    print(f"phone speed {a.phone_speed}x of PC, link {a.mbps} MB/s, {a.mb_per_task} MB/task (MODEL)")
    for k, v in plans.items():
        print(f"{k:<20}{plan_time(v):8.3f}s  ({(np.asarray(v) == 1).sum()} tasks on phone)")


def main():
    ap = argparse.ArgumentParser(prog="qcmp.sched")
    sub = ap.add_subparsers(required=True)
    sub.add_parser("info").set_defaults(fn=cmd_info)
    b = sub.add_parser("bench")
    b.add_argument("--tasks", type=int, default=32)
    b.add_argument("--workers", type=int)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_bench)
    q = sub.add_parser("qaoa")
    q.add_argument("--max-tasks", type=int, default=14)
    q.add_argument("--layers", type=int, default=2)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(fn=cmd_qaoa)
    o = sub.add_parser("offload")
    o.add_argument("--tasks", type=int, default=32)
    o.add_argument("--phone-speed", type=float, default=0.35)
    o.add_argument("--mbps", type=float, default=20.0)
    o.add_argument("--mb-per-task", type=float, default=2.0)
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(fn=cmd_offload)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
