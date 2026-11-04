import json

import pytest

from experiments.phase_iiin import execute as driver
from weightbridge.core import json_dump_atomic


def test_interruption_is_retained_and_restart_uses_new_evidence_paths(tmp_path, monkeypatch):
    attempts = []

    def failed(base, material, out):
        out.mkdir()
        json_dump_atomic(out / "partial.json", {"immutable": True})
        attempts.append(out)
        raise RuntimeError("strict replay refusal")

    monkeypatch.setattr(driver, "run", failed)
    first, receipt = tmp_path / "first", tmp_path / "first-receipt.json"
    with pytest.raises(RuntimeError, match="strict replay"):
        driver.execute(tmp_path, tmp_path, first, receipt)
    original = receipt.read_bytes()
    assert json.loads(original)["status"] == "INTERRUPTED"
    assert json.loads(original)["resources"]["cpu_seconds"] >= 0
    with pytest.raises(FileExistsError):
        driver.execute(tmp_path, tmp_path, first, tmp_path / "replacement.json")
    with pytest.raises(FileExistsError):
        driver.execute(tmp_path, tmp_path, tmp_path / "fresh", receipt)

    def restarted(base, material, out):
        out.mkdir()
        value = {"detail": {"reason": "fixture_completed"}}
        json_dump_atomic(out / "fit-report.json", value)
        return value

    monkeypatch.setattr(driver, "run", restarted)
    driver.execute(tmp_path, tmp_path, tmp_path / "second", tmp_path / "second-receipt.json")
    assert receipt.read_bytes() == original
    assert json.loads((first / "partial.json").read_bytes()) == {"immutable": True}
    assert json.loads((tmp_path / "second-receipt.json").read_bytes())["status"] == "COMPLETED"
    assert len(attempts) == 1
