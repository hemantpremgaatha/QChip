import numpy as np

from qcmp.sched import solvers as S


def test_lpt_beats_round_robin_on_skewed_costs():
    costs = np.array([9.0, 1, 1, 1, 8, 1, 1, 1])
    sp = [1.0, 1.0]
    assert S.makespan(costs, S.lpt_greedy(costs, sp), sp) < S.makespan(costs, S.round_robin(costs, sp), sp)


def test_anneal_matches_optimum_small():
    rng = np.random.default_rng(3)
    costs = rng.lognormal(0, 1, 10)
    sp = [1.0, 1.0]
    opt = S.makespan(costs, S.exact_optimum(costs, sp), sp)
    an = S.makespan(costs, S.anneal(costs, sp, steps=8000), sp)
    assert an <= opt * 1.02


def test_qaoa_valid_and_concentrates_on_low_energy():
    rng = np.random.default_rng(5)
    costs = rng.lognormal(0, 1, 8)
    a, info = S.qaoa_partition(costs, layers=2, iters=60)
    assert len(a) == 8 and set(a) <= {0, 1}
    assert info["expected_energy"] < info["uniform_energy"]
    opt = S.makespan(costs, S.exact_optimum(costs, [1, 1]), [1, 1])
    assert S.makespan(costs, a, [1, 1]) <= opt * 1.15


def test_speed_aware_assignment():
    costs = np.ones(8)
    a = S.lpt_greedy(costs, [3.0, 1.0])
    assert (a == 0).sum() > (a == 1).sum()
