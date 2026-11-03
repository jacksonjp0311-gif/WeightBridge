from experiments.phase_iiie.cached_diagnostics import _add_curvature
from experiments.phase_iiie.synthetic_tasks import features, make_tasks


def test_cached_curvature_reconstructs_d_and_charges_only_new_work():
    tasks = make_tasks("test", seed=81041, per_mechanism=2)
    for task in tasks:
        c, c_cost = features(task, level="C")
        d, d_cost = features(task, level="D")
        incremental = d_cost - c_cost
        assert _add_curvature(task, c, incremental.units - 1) is None
        assert _add_curvature(task, c, incremental.units) == (d, incremental)
        assert "interaction_signal" not in c
