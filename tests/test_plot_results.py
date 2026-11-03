from benchmarks.runners.plot_results import canonical_experiments, paired_cluster_bootstrap


def test_ledger_plot_selection_ignores_identity_backfill_and_superseded_replay():
    original = {"schema": "weightbridge.experiment.v1", "experiment_id": "run-a", "measurement_key": "a"}
    canonical = {"schema": "weightbridge.experiment.v1", "experiment_id": "run-b", "measurement_key": "b"}
    duplicate = {"schema": "weightbridge.experiment.v1", "experiment_id": "run-c", "measurement_key": "b"}
    correction = {"schema": "weightbridge.ledger_correction.v1", "experiment_id": "duplicate-correction",
                  "disposition": "DUPLICATE_REPLAY_NOT_INDEPENDENT", "supersedes": "run-a"}
    identity = {"schema": "weightbridge.ledger_correction.v1", "experiment_id": "identity-backfill",
                "disposition": "IDENTITY_METADATA_BACKFILLED", "supersedes": "run-b",
                "canonical_experiment_id": "run-b"}
    selected = canonical_experiments([original, canonical, duplicate, correction, identity])
    assert [item["experiment_id"] for item in selected] == ["run-b"]


def test_metadata_addendum_is_resolved_without_mutating_historical_record():
    original = {"schema": "weightbridge.experiment.v1", "experiment_id": "run-a",
                "measurement_key": "a", "evaluation": {"dataset": "toy"}}
    addendum = {"schema": "weightbridge.ledger_correction.v1", "experiment_id": "run-a-metadata",
                "disposition": "METADATA_BACKFILLED", "references_experiment_id": "run-a",
                "record_addendum": {"evaluation": {"split": "practice"}, "affected_parameters": {"count": 0}}}
    resolved = canonical_experiments([original, addendum])[0]
    assert resolved["evaluation"] == {"dataset": "toy", "split": "practice"}
    assert resolved["affected_parameters"] == {"count": 0}
    assert original["evaluation"] == {"dataset": "toy"}


def test_seed_cluster_bootstrap_is_paired_and_reproducible():
    baseline = [[False, True], [False, False], [True, True]]
    candidate = [[True, True], [True, False], [True, True]]
    first = paired_cluster_bootstrap(baseline, candidate, seed=8, resamples=1000)
    second = paired_cluster_bootstrap(baseline, candidate, seed=8, resamples=1000)
    assert first == second
    assert first["number_of_independent_seeds"] == 3
    assert first["mean_difference"] == 1 / 3
    assert first["ci95_clustered_by_seed"][0] <= first["mean_difference"] <= first["ci95_clustered_by_seed"][1]
