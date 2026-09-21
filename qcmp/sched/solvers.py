"""Schedulers: assign n tasks (costs c_i) to m machines (speeds v_j) to minimise makespan.

Assignments are integer arrays a[i] in [0, m). Machine j's time is sum(c_i for a_i=j)/v_j.
"""
import itertools
import math

import numpy as np


def makespan(costs, assign, speeds):
    loads = np.zeros(len(speeds))
    for c, a in zip(costs, assign):
        loads[a] += c
    return float((loads / np.asarray(speeds, float)).max())


def round_robin(costs, speeds):
    """What naive static partitioning does: task i -> machine i mod m."""
    return np.arange(len(costs)) % len(speeds)


def lpt_greedy(costs, speeds):
    """Longest-processing-time-first: strong classical baseline (4/3 bound, equal speeds)."""
    speeds = np.asarray(speeds, float)
    loads = np.zeros(len(speeds))
    assign = np.zeros(len(costs), dtype=int)
    for i in np.argsort(costs)[::-1]:
        j = int(np.argmin((loads + costs[i]) / speeds))
        assign[i] = j
        loads[j] += costs[i]
    return assign


def exact_optimum(costs, speeds, limit=1_000_000):
    """Brute force; only for tiny instances."""
    n, m = len(costs), len(speeds)
    if m ** n > limit:
        raise ValueError("instance too large for brute force")
    best, best_a = math.inf, None
    for a in itertools.product(range(m), repeat=n):
        v = makespan(costs, a, speeds)
        if v < best:
            best, best_a = v, np.array(a)
    return best_a


def anneal(costs, speeds, steps=20000, seed=0, start=None):
    """Simulated annealing on the scheduling energy (a QUBO/Ising-style objective).

    Energy = makespan + 0.05 * (sum of squared machine times)/makespan, which smooths
    the landscape. Classical algorithm; 'quantum-inspired' only in its formulation.
    """
    rng = np.random.default_rng(seed)
    costs = np.asarray(costs, float)
    speeds = np.asarray(speeds, float)
    n, m = len(costs), len(speeds)
    a = lpt_greedy(costs, speeds) if start is None else np.array(start)
    loads = np.zeros(m)
    for i in range(n):
        loads[a[i]] += costs[i]

    def energy(l):
        t = l / speeds
        mk = t.max()
        return mk + 0.05 * (t ** 2).sum() / max(mk, 1e-12)

    e = energy(loads)
    best_a, best_mk = a.copy(), (loads / speeds).max()
    t0 = 0.1 * best_mk
    for s in range(steps):
        temp = t0 * (1e-3) ** (s / steps)
        i = rng.integers(n)
        old = a[i]
        if rng.random() < 0.5 or n < 2:
            new = rng.integers(m)
            if new == old:
                continue
            loads[old] -= costs[i]; loads[new] += costs[i]
            e2 = energy(loads)
            if e2 <= e or rng.random() < math.exp((e - e2) / temp):
                a[i], e = new, e2
            else:
                loads[old] += costs[i]; loads[new] -= costs[i]
        else:  # swap two tasks on different machines
            k = rng.integers(n)
            if a[k] == old:
                continue
            ok = a[k]
            loads[old] += costs[k] - costs[i]; loads[ok] += costs[i] - costs[k]
            e2 = energy(loads)
            if e2 <= e or rng.random() < math.exp((e - e2) / temp):
                a[i], a[k], e = ok, old, e2
            else:
                loads[old] -= costs[k] - costs[i]; loads[ok] -= costs[i] - costs[k]
        mk = (loads / speeds).max()
        if mk < best_mk:
            best_mk, best_a = mk, a.copy()
    return best_a


# ---- QAOA for 2-machine balanced partition (simulated state vector) ----------------

def _rx_all(psi, n, beta):
    """Apply exp(-i beta X) on every qubit of an n-qubit state vector."""
    c, s = math.cos(beta), -1j * math.sin(beta)
    psi = psi.reshape([2] * n)
    for q in range(n):
        p0 = np.take(psi, 0, axis=q)
        p1 = np.take(psi, 1, axis=q)
        psi = np.stack([c * p0 + s * p1, s * p0 + c * p1], axis=q)
    return psi.reshape(-1)


def qaoa_partition(costs, layers=2, iters=120, seed=0, top=32):
    """QAOA (state-vector simulation) for the two-way number partition problem.

    Ising energy E(z) = (sum_i c_i z_i)^2 with z_i = +/-1, whose minimum is the
    balanced split. Parameters are tuned by random search + local refinement on <E>;
    the best classical evaluation among the `top` most probable bitstrings is returned.
    Practical up to ~16 tasks (2^n amplitudes). Returns (assignment, info dict).
    """
    c = np.asarray(costs, float)
    n = len(c)
    if n > 18:
        raise ValueError("QAOA simulation limited to <=18 tasks")
    c = c / c.sum()
    idx = np.arange(2 ** n)
    z = 1 - 2 * ((idx[:, None] >> np.arange(n)[::-1]) & 1)  # (2^n, n) of +/-1
    energy = (z @ c) ** 2

    def run(params):
        psi = np.full(2 ** n, 1 / math.sqrt(2 ** n), dtype=complex)
        for g, b in zip(params[:layers], params[layers:]):
            psi = psi * np.exp(-1j * g * energy * 4)
            psi = _rx_all(psi, n, b)
        return np.abs(psi) ** 2

    rng = np.random.default_rng(seed)
    obj = lambda p: float(run(p) @ energy)
    best_p = rng.uniform(0, math.pi, 2 * layers)
    best_v = obj(best_p)
    for _ in range(iters):
        cand = rng.uniform(0, math.pi, 2 * layers)
        v = obj(cand)
        if v < best_v:
            best_p, best_v = cand, v
    step = 0.3
    for _ in range(iters):  # local refinement
        cand = best_p + rng.normal(0, step, 2 * layers)
        v = obj(cand)
        if v < best_v:
            best_p, best_v = cand, v
        else:
            step = max(step * 0.98, 0.02)
    prob = run(best_p)
    cand_idx = np.argsort(prob)[::-1][:top]
    best_a, best_mk = None, math.inf
    for k in cand_idx:
        a = ((k >> np.arange(n)[::-1]) & 1).astype(int)
        mk = makespan(costs, a, [1.0, 1.0])
        if mk < best_mk:
            best_mk, best_a = mk, a
    return best_a, {"params": best_p.tolist(), "expected_energy": best_v,
                    "uniform_energy": float(energy.mean()),
                    "top_prob_mass": float(prob[cand_idx].sum())}
