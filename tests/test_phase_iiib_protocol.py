import json
from collections import defaultdict
from types import SimpleNamespace

from experiments.phase_iiib import qwen_material
from weightbridge.adaptation.awl_v2 import parse_awl_v2
from weightbridge.core import file_hash
from weightbridge.evaluation.hf_edit import _load_suite


def test_frozen_qwen_material_balances_formats_and_separates_items(tmp_path, monkeypatch):
    shard = tmp_path / "test.safetensors"
    shard.write_bytes(b"fixture identity")
    design = json.loads(qwen_material.DESIGN.read_bytes())
    design.update(shard_sha256=file_hash(shard))
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    monkeypatch.setattr(qwen_material, "DESIGN", design_path)
    checkpoint = SimpleNamespace(checkpoint_sha256=lambda: design["model_sha256"], shards=[shard])
    monkeypatch.setattr(qwen_material, "load_checkpoint", lambda _: checkpoint)
    seen_rosters = set()

    for index, is_pilot in enumerate((True, False, False)):
        output = tmp_path / f"material-{index}"
        frozen = qwen_material.materialize(tmp_path, output, pilot=is_pilot, roster_index=max(0, index - 1))
        practice_keys, holdout_keys = set(), set()

        def read(relative):
            assert file_hash(output / relative) == frozen["artifacts"][relative]
            return _load_suite((output / relative).read_bytes())

        def keys(suite, group):
            by_item = defaultdict(list)
            for row in suite[group]:
                by_item[row["item_key"]].append(row)
            for paired in by_item.values():
                assert {row["answer_position"] for row in paired} == {0, 1}
                assert {row["label"] for row in paired} == set(suite["choices"])
                assert len(paired) == 2
            return set(by_item)

        for cycle in frozen["cycles"]:
            generic, aware, screen = (read(cycle["practice"][method]) for method in ("generic", "aware", "screen"))
            assert generic["target"] == aware["target"] == screen["target"]
            assert len(generic["sentinel"]) == len(aware["sentinel"])
            assert set(row["family"] for row in generic["sentinel"]) == {"generic"}
            assert set(row["family"] for row in aware["sentinel"]) == {"generic", *(
                family for family in design["families"] if family != cycle["family"]
            )}
            current_practice = keys(screen, "target") | keys(screen, "sentinel")
            assert not practice_keys & current_practice
            practice_keys.update(current_practice)
            for arm, relative in cycle["intents"].items():
                intent = parse_awl_v2(json.loads((output / relative).read_bytes()))
                fitting = generic if arm == "A" else aware
                assert set(intent.preserve) == {row["family"] for row in fitting["sentinel"]}

            ab, xy = (read(cycle["holdouts"][name]) for name in ("ab", "xy"))
            assert ab["choices"] != xy["choices"]
            assert keys(ab, "target") == keys(xy, "target")
            assert keys(ab, "sentinel") == keys(xy, "sentinel")
            current_holdout = keys(ab, "target") | keys(ab, "sentinel")
            assert not holdout_keys & current_holdout
            holdout_keys.update(current_holdout)

        final_ab, final_xy = (read(frozen["final"][name]) for name in ("ab", "xy"))
        assert keys(final_ab, "target") == keys(final_xy, "target")
        final_keys = keys(final_ab, "target") | keys(final_ab, "sentinel")
        assert not final_keys & (practice_keys | holdout_keys)
        assert not practice_keys & holdout_keys
        roster_keys = practice_keys | holdout_keys | final_keys
        assert len(roster_keys) == frozen["item_count"]
        assert not seen_rosters & roster_keys
        seen_rosters.update(roster_keys)
