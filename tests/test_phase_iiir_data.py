"""Exact-data isolation tests; fixtures are not neural-model measurements."""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import json

import pytest

from experiments.phase_iiiq_baseline import data as q
from experiments.phase_iiir_replication import data, initialize
from weightbridge.core import EditRejected, file_hash, json_dump_atomic


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("r-data")
    # A complete synthetic historical Q archive lets the production host check
    # every old private population against separately committed public hashes.
    q_protocol = write(root/"q-private/protocol.json", {"data": {"seed_prefix": "history-test",
        "populations": {n: asdict(s) for n, s in q.DEFAULT_POPULATIONS.items()}}})
    q_manifest = q.freeze(root/"q-private/material", root/"q-host/hidden.json", protocol=q_protocol,
        host_seeds={"selection": "q-selection-test", "hidden": "q-hidden-test"})
    public = root/"published/results/phase-iii-q-baseline/baseline-v1"
    write(public/"protocol.json", json.loads(q_protocol.read_bytes()))
    # File identities, not only parsed documents, must agree.
    (public/"protocol.json").write_bytes(q_protocol.read_bytes())
    (public/"material").mkdir(parents=True)
    (public/"material/manifest.json").write_bytes((root/"q-private/material/manifest.json").read_bytes())
    blocked, anchors, history = data.historical_exclusions(root/"q-private", root/"q-host/hidden.json",
                                                         published_root=root/"published")
    protocol = write(root/"master.json", {"source_sha256": {}, "data": {"seed_prefix": "r-data-test",
        "populations": {n: asdict(s) for n, s in data.DEFAULT_POPULATIONS.items()}}})
    # Freeze uses the same checked historical sets, avoiding a production path
    # in deterministic tests. The separate test below checks mismatch refusal.
    original = data.historical_exclusions
    try:
        data.historical_exclusions = lambda *a, **kw: (set(blocked), set(anchors), deepcopy(history))
        manifest = data.freeze(root/"material", root/"banks", protocol=protocol,
            q_study=root/"q-private", q_hidden=root/"q-host/hidden.json",
            host_seeds={n: f"r-private-test/{n}" for n in data.HOST_SPLITS})
    finally:
        data.historical_exclusions = original
    populations = {n: data.load_split(root/"material", n, manifest, protocol=protocol,
        host_authorized=n in data.HOST_SPLITS, host_root=root/"banks" if n in data.HOST_SPLITS else None)
        for n in data.SPLITS}
    return root, protocol, manifest, populations, blocked, anchors, q_manifest


def test_counts_target_balance_and_confirmation_banks(frozen):
    root, protocol, manifest, populations, *_ = frozen
    assert data.read_manifest(root/"material", protocol=protocol) == manifest
    assert len(populations["train"]) == len(populations["replay"]) == 1024
    assert len(populations["development"]) == 256
    for bank in ("bank-a", "bank-b"):
        assert len(populations[bank+"-selection"]) == len(populations[bank+"-hidden"]) == 256
        assert len(populations[bank+"-preservation-validation"]) == 320
    assert len(populations["smoke-r8"]) == len(populations["smoke-r4"]) == 32
    for name, rows in populations.items():
        q.validate(rows, data.CANONICAL[name], data.DEFAULT_POPULATIONS[name])
        assert all(row["completion"] == " "+q.oracle(row["family"], row["operands"])+"\n" for row in rows)


def test_all_numeric_equivalent_tasks_and_truth_flip_anchors_stay_disjoint(frozen):
    _, _, _, populations, blocked, anchors, _ = frozen
    seen_tasks, seen_families, seen_anchors = set(blocked), set(), set(anchors)
    for rows in populations.values():
        tasks = {q.semantic_key(r["family"], r["operands"]) for r in rows}
        families = {r["family_id"] for r in rows}
        original = [r for r in rows if r["group"] == "target" and r["counterfactual"] == r["order"] == r["wording"] == 0]
        fresh_anchors = {Decimal(r["operands"][0]) for r in original}
        assert not tasks & seen_tasks and not families & seen_families and not fresh_anchors & seen_anchors
        seen_tasks |= tasks; seen_families |= families; seen_anchors |= fresh_anchors


def test_bank_seeds_and_raw_files_never_enter_shared_material(frozen):
    root, _, manifest, *_ = frozen
    assert not any((root/"material"/f"{n}.json").exists() for n in data.HOST_SPLITS)
    assert "r-private-test" not in json.dumps(manifest)
    assert manifest["historical_exclusion"]["q_all_seven_populations_verified"]
    assert all(row["published_commitment_verified"] for row in manifest["historical_exclusion"]["private_sources"].values())


@pytest.mark.parametrize("name", sorted(data.HOST_SPLITS))
def test_each_bank_population_needs_host_access_and_distinct_private_root(frozen, name):
    root, protocol, manifest, *_ = frozen
    with pytest.raises(ValueError, match="staged host"):
        data.load_split(root/"material", name, manifest, protocol=protocol, host_root=root/"banks")
    with pytest.raises(ValueError, match="staged host"):
        data.load_split(root/"material", name, manifest, protocol=protocol, host_authorized=True)
    with pytest.raises(ValueError, match="separate private"):
        data.load_split(root/"material", name, manifest, protocol=protocol, host_authorized=True, host_root=root/"material")


def test_source_tampering_and_write_once(frozen, tmp_path):
    root, protocol, manifest, populations, *_ = frozen
    with pytest.raises(FileExistsError):
        data.freeze(root/"material", root/"banks", protocol=protocol, q_study=root/"q-private", q_hidden=root/"q-host/hidden.json")
    damaged = deepcopy(populations["train"])
    damaged[0]["answer"] = "0"
    write(tmp_path/"train.json", damaged)
    with pytest.raises(ValueError, match="identity differs"):
        data.load_split(tmp_path, "train", manifest, protocol=protocol)


def test_history_cannot_be_declared_fresh_with_a_missing_or_changed_q_roster(frozen, tmp_path):
    root, *_ = frozen
    private = tmp_path/"q-private"
    (private/"material").mkdir(parents=True)
    (private/"protocol.json").write_bytes((root/"q-private/protocol.json").read_bytes())
    (private/"material/manifest.json").write_bytes((root/"q-private/material/manifest.json").read_bytes())
    with pytest.raises(FileNotFoundError):
        data.historical_exclusions(private, root/"q-host/hidden.json", published_root=root/"published")
    write(private/"material/smoke.json", [])
    with pytest.raises(ValueError, match="commitment"):
        data.historical_exclusions(private, root/"q-host/hidden.json", published_root=root/"published")


def test_host_nested_archive_scan_extracts_only_semantic_task_identity(tmp_path):
    row = {"family": "larger-decimal", "operands": ["-1.200", "-2.10000"], "answer": "private"}
    value = {"measurement": {"rows": [row]}, "weights": {"shape": [5, 3]}, "numbers": [1, 2]}
    rows = list(data._identity_rows(value))
    used, anchors = set(), set()
    assert data._block_rows(rows, used, anchors) == 1
    assert used == {q.semantic_key("larger-decimal", ["-2.1", "-1.2"])}
    assert anchors == {Decimal("-1.200")}
    with pytest.raises(ValueError):
        data._block_rows([{"family": "unknown", "operands": ["1"]}], set(), set())


def test_q_views_bind_shared_data_and_distinct_confirmation_banks(frozen):
    root, master_path, manifest, *_ = frozen
    paths = {}
    for arm in ("rank8", "rank4"):
        mapping = data.arm_mapping(arm)
        config = {"data": {"seed_prefix": "r-view-test", "populations": {
            canonical: manifest["splits"][actual]["population"] for canonical, actual in mapping.items()}}}
        protocol = write(root/arm/"protocol.json", config)
        view = data.create_q_view(root/"material", root/"banks", manifest, root/arm/"material",
            arm=arm, arm_protocol=protocol, master_protocol=master_path)
        assert q.read_manifest(root/arm/"material", protocol=protocol) == view
        assert not (root/arm/"material/hidden.json").exists()
        paths[arm] = root/arm/"material"
        for split in ("train", "replay", "development", "smoke"):
            assert q.load_split(paths[arm], split, view, protocol=protocol)
    for split in ("train", "replay", "development"):
        assert file_hash(paths["rank8"]/f"{split}.json") == file_hash(paths["rank4"]/f"{split}.json")
    for split in ("smoke", "selection", "preservation-validation"):
        assert file_hash(paths["rank8"]/f"{split}.json") != file_hash(paths["rank4"]/f"{split}.json")


def test_arm_protocols_change_only_preregistered_rank_and_bank_binding():
    master = json.loads(initialize.PROTOCOL.read_bytes())
    a = initialize.arm_protocol(master, "rank8", master_sha256="a"*64, global_manifest_sha256="b"*64)
    b = initialize.arm_protocol(master, "rank4", master_sha256="a"*64, global_manifest_sha256="b"*64)
    assert a["training"] == b["training"] and a["inference"] == b["inference"]
    assert a["adapter"]["rank"] == 8 and b["adapter"]["rank"] == 4
    assert a["adapter"]["alpha"] / a["adapter"]["rank"] == b["adapter"]["alpha"] / b["adapter"]["rank"] == 2
    assert set(a["data"]["populations"]) == set(b["data"]["populations"]) == set(q.SPLITS)
    assert a["master_protocol_sha256"] == b["master_protocol_sha256"] == "a"*64
    with pytest.raises(ValueError):
        initialize.arm_protocol(master, "rank2", master_sha256="a"*64, global_manifest_sha256="b"*64)


def test_full_write_once_initialization_and_smoke_bound_admission(frozen, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.phase_iiir_replication import smoke, study
    from experiments.phase_iiir_replication.training import data_exposure_receipt
    old_root, _, _, _, blocked, anchors, _ = frozen
    inventory = initialize.ROOT/"results/phase-iii-q-baseline/preflight/inventory-v1/architecture-inventory.json"
    native = initialize._inventory(inventory)
    monkeypatch.setattr(initialize, "load_checkpoint", lambda path: SimpleNamespace(checkpoint_sha256=lambda: initialize.PARENT))
    monkeypatch.setattr(initialize, "_inventory", lambda path, checkpoint=None: native)
    monkeypatch.setattr(data, "historical_exclusions", lambda *a, **kw: (set(blocked), set(anchors), {"fixture": True}))
    protocol = write(tmp_path/"master.json", json.loads(initialize.PROTOCOL.read_bytes()))
    base, out = tmp_path/"base", tmp_path/"study"
    receipt = initialize.freeze(base, inventory, out, protocol=protocol,
        q_study=old_root/"q-private", q_hidden=old_root/"q-host/hidden.json")
    assert receipt["status"] == "FROZEN" and not receipt["inference_performed"]
    assert receipt["matching_training_rows"] and receipt["matching_data_order"]
    assert receipt["arms"]["rank8"]["trainable_parameters"] == 5411328
    assert receipt["arms"]["rank4"]["trainable_parameters"] == 2705664
    for arm, rank in (("rank8", 8), ("rank4", 4)):
        root = out/arm
        config = json.loads((root/"protocol.json").read_bytes())
        view = q.read_manifest(root/"material", protocol=root/"protocol.json")
        assert len(view["splits"]) == 7
        assert not (root/"main-authorization.json").exists()
        assert not (root/"material/hidden.json").exists()
        assert config["master_protocol_sha256"] == file_hash(out/"protocol.json")
        assert config["global_data_manifest_sha256"] == file_hash(out/"material/manifest.json")
        order = data_exposure_receipt(root/"training-bundle.json", config)
        assert order["examples_seen"] == 2048
        gate = {"passed": True, "protocol_sha256": file_hash(root/"protocol.json"), "rank": rank,
            "alpha": rank*2, "authorization_sha256": file_hash(root/"smoke-authorization.json"),
            "training_bundle_sha256": file_hash(root/"smoke-bundle.json")}
        gate_path = write(root/"gate.json", gate)
        monkeypatch.setattr(smoke, "validate_gate", lambda *a, **kw: gate)
        checked = []
        monkeypatch.setattr(study, "validate_replication_gate", lambda cfg, path: checked.append(path))
        admitted = initialize.authorize_main(out, arm, gate_path,
            replication_gate=tmp_path/"replication.json" if rank == 4 else None)
        assert admitted["rank"] == rank and admitted["max_runs"] == 1
        assert admitted["smoke_gate_sha256"] == file_hash(gate_path)
        assert bool(checked) == (rank == 4)
        with pytest.raises(EditRejected):
            initialize.authorize_main(out, arm, gate_path,
                replication_gate=tmp_path/"replication.json" if rank == 4 else None)
    with pytest.raises(FileExistsError):
        initialize.freeze(base, inventory, out, protocol=protocol,
            q_study=old_root/"q-private", q_hidden=old_root/"q-host/hidden.json")
