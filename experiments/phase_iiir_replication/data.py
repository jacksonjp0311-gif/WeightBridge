"""Host-only fresh families and two independent confirmation banks.

The proven Q renderer and Decimal oracle are reused unchanged. R partition
names live in the manifest; each file retains its canonical Q row schema so
the existing training, measurement and independent-verification code replays it.
Only commitments and aggregate exclusion counts may be published.
"""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import secrets

from experiments.phase_iiiq_baseline import data as q
from weightbridge.core import file_hash, json_dump_atomic
from .integrity import ROOT, read

SCHEMA = "weightbridge.phase_iiir.data.v1"
PROTOCOL = Path(__file__).with_name("protocol.json")
SPLITS = ("smoke-r8", "smoke-r4", "train", "replay", "development",
          "bank-a-preservation-validation", "bank-a-selection", "bank-a-hidden",
          "bank-b-preservation-validation", "bank-b-selection", "bank-b-hidden")
HOST_SPLITS = frozenset(name for name in SPLITS if name.startswith("bank-"))
CANONICAL = {name: ("smoke" if name.startswith("smoke-") else name[7:]
                    if name.startswith("bank-") else name) for name in SPLITS}
DEFAULT_POPULATIONS = {
    name: q.DEFAULT_POPULATIONS[CANONICAL[name]] for name in SPLITS}


def _plan(protocol):
    config = read(protocol)["data"]
    if (set(config["populations"]) != set(SPLITS)
            or not isinstance(config.get("seed_prefix"), str) or not config["seed_prefix"]):
        raise ValueError("Replication must freeze all eleven distinct populations")
    return config["seed_prefix"], {
        name: q.PopulationSpec(**config["populations"][name]) for name in SPLITS}


def _block_rows(rows, used, anchors):
    """Extract only task identities, never return protected answers/observations."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Historical roster must contain a nonempty row list")
    count = 0
    for row in rows:
        if not isinstance(row, dict) or "family" not in row or "operands" not in row:
            raise ValueError("Historical roster lacks typed task identities")
        family, operands = row["family"], row["operands"]
        # Earlier decimal rosters use the same family identities. Refuse
        # ambiguous unexpected task types rather than claim full coverage.
        if family not in {"larger-decimal", "copy", "integer-add", "integer-successor"}:
            raise ValueError("Unsupported historical task identity")
        values = tuple(Decimal(value) for value in operands)
        if not values or not all(value.is_finite() for value in values):
            raise ValueError("Nonfinite historical task operand")
        used.add(q.semantic_key(family, operands))
        if family == "larger-decimal" and all(value < 0 for value in values):
            sizes = [len(value.partition(".")[2]) for value in operands]
            if len(set(sizes)) > 1:
                anchors.add(values[sizes.index(min(sizes))])
        count += 1
    return count


def _identity_rows(value):
    """Read-only traversal of host archives, yielding only recognized task rows."""
    if isinstance(value, dict):
        if (value.get("family") in {"larger-decimal", "copy", "integer-add", "integer-successor"}
                and isinstance(value.get("operands"), list)):
            yield value
        for child in value.values():
            if isinstance(child, (list, dict)):
                yield from _identity_rows(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (list, dict)):
                yield from _identity_rows(child)


def historical_exclusions(q_study, q_hidden, *, additional_rosters=(), historical_roots=(), published_root=ROOT):
    """Read historical private data only in the host, verify Q commitments.

    Additional explicitly named private decimal rosters are hashed and checked
    for numeric identity. Their completeness is not inferred from filenames.
    """
    q_study, q_hidden, published_root = map(Path, (q_study, q_hidden, published_root))
    manifest_path = q_study / "material/manifest.json"
    published = published_root / "results/phase-iii-q-baseline/baseline-v1"
    if (file_hash(manifest_path) != file_hash(published / "material/manifest.json")
            or file_hash(q_study / "protocol.json") != file_hash(published / "protocol.json")):
        raise ValueError("Historical Q manifest/protocol differ from published commitments")
    manifest = read(manifest_path)
    if set(manifest["splits"]) != set(q.SPLITS):
        raise ValueError("Historical Q population coverage is incomplete")
    used, anchors = q._blocked()
    sources = {}
    for split in q.SPLITS:
        path = q_hidden if split == "hidden" else q_study / "material" / f"{split}.json"
        expected = manifest["splits"][split]
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"Historical Q {split} roster does not match its commitment")
        rows = read(path)
        q.validate(rows, split, q.PopulationSpec(**expected["population"]))
        sources[f"phase-q/{split}"] = {"sha256": file_hash(path), "rows": _block_rows(rows, used, anchors),
                                      "published_commitment_verified": True}
    for index, path in enumerate(additional_rosters):
        path = Path(path)
        content = read(path)
        rows = content.get("rows") if isinstance(content, dict) else content
        sources[f"additional-private-{index:03d}"] = {
            "sha256": file_hash(path), "rows": _block_rows(rows, used, anchors),
            "published_commitment_verified": False}
    archive_scan = []
    seen_digests = {entry["sha256"] for entry in sources.values()}
    for index, root in enumerate(historical_roots):
        root = Path(root)
        if not root.is_dir() or root.resolve().is_relative_to(ROOT):
            raise ValueError("Historical archive scan requires an existing private directory")
        files = sorted(root.rglob("*.json"))
        if len(files) > 10000:
            raise ValueError("Historical archive file bound exceeded")
        checked = []
        for path in files:
            if path.stat().st_size > 64*1024**2:
                raise ValueError("Historical JSON exceeds the bounded host scan size")
            digest = file_hash(path)
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
            rows = list(_identity_rows(read(path)))
            # Many archived receipts contain no raw task identities. Their
            # hashes are still committed so the scope is independently auditable.
            count = _block_rows(rows, used, anchors) if rows else 0
            checked.append({"sha256": digest, "recognized_row_occurrences": count})
        archive_scan.append({"archive": root.name, "json_files": len(files),
            "unique_files_checked": len(checked), "files": checked,
            "recognized_row_occurrences": sum(row["recognized_row_occurrences"] for row in checked)})
    receipt = {"schema": "weightbridge.phase_iiir.historical_exclusion.v1",
        "q_manifest_sha256": file_hash(manifest_path),
        "q_protocol_sha256": file_hash(q_study / "protocol.json"), "private_sources": sources,
        "disclosed_source_sha256": q._sources(q.PROTOCOL), "additional_archive_scan": archive_scan,
        "blocked_numeric_tasks": len(used), "blocked_negative_family_anchors": len(anchors),
        "q_all_seven_populations_verified": True,
        "coverage": "All published decimal sources enumerated by Q, all seven hash-verified Q populations, explicitly listed additional private rosters and recognized task rows in the enumerated private JSON archive trees.",
        "limitation": "Does not assert access to unlisted/unavailable older private rosters or task identities stored outside recognized JSON family/operands objects.",
        "protected_rows_exposed": False}
    return used, anchors, receipt


def _private_paths(material, host_root):
    material, host_root = Path(material).resolve(), Path(host_root).resolve()
    if (material.is_relative_to(ROOT) or host_root.is_relative_to(ROOT)
            or material.is_relative_to(host_root) or host_root.is_relative_to(material)):
        raise ValueError("Material and confirmation banks require separate private host paths")
    return material, host_root


def freeze(material, host_root, *, protocol=PROTOCOL, q_study, q_hidden,
           additional_rosters=(), historical_roots=(), host_seeds=None):
    material, host_root = _private_paths(material, host_root)
    if material.exists() or host_root.exists():
        raise FileExistsError("Replication populations are write-once")
    prefix, populations = _plan(protocol)
    supplied = dict(host_seeds or {})
    if set(supplied)-HOST_SPLITS or any(not isinstance(s, str) or not s for s in supplied.values()):
        raise ValueError("Only private confirmation-bank seeds may be supplied")
    seeds = {name: supplied.get(name, secrets.token_hex(32)) if name in HOST_SPLITS
             else f"{prefix}/{name}" for name in SPLITS}
    if len(set(seeds.values())) != len(SPLITS):
        raise ValueError("Every population requires its own seed")
    used, anchors, history = historical_exclusions(q_study, q_hidden, additional_rosters=additional_rosters,
                                                historical_roots=historical_roots)
    manifest = {"schema": SCHEMA, "protocol_sha256": file_hash(protocol),
        "historical_exclusion": history, "domain": q.DOMAIN, "selection_is_error_based": False,
        "splits": {}, "source_sha256": read(protocol)["source_sha256"]}
    material.mkdir(parents=True)
    host_root.mkdir(parents=True)
    seen_families = set()
    for name in SPLITS:
        rows = q.generate(CANONICAL[name], populations[name], seeds[name], used, anchors)
        families = {row["family_id"] for row in rows}
        if seen_families & families:
            raise RuntimeError("A semantic family leaked across study partitions")
        seen_families.update(families)
        destination = (host_root if name in HOST_SPLITS else material) / f"{name}.json"
        json_dump_atomic(destination, rows)
        manifest["splits"][name] = {"sha256": file_hash(destination), "rows": len(rows),
            "population": asdict(populations[name]), "canonical_split": CANONICAL[name],
            "semantic_families": len(families), "underlying_tasks": len({r["task_id"] for r in rows}),
            "seed_commitment": hashlib.sha256(seeds[name].encode()).hexdigest(),
            "host_private": name in HOST_SPLITS}
    json_dump_atomic(material / "manifest.json", manifest)
    return manifest


def read_manifest(material, *, protocol=PROTOCOL):
    manifest = read(Path(material) / "manifest.json")
    _, populations = _plan(protocol)
    if (manifest.get("schema") != SCHEMA or manifest.get("protocol_sha256") != file_hash(protocol)
            or manifest.get("source_sha256") != read(protocol)["source_sha256"]
            or manifest.get("domain") != q.DOMAIN or manifest.get("selection_is_error_based") is not False
            or set(manifest.get("splits", {})) != set(SPLITS)):
        raise ValueError("Replication manifest differs from its frozen study")
    for name, spec in populations.items():
        row = manifest["splits"][name]
        if (row["population"] != asdict(spec) or row["rows"] != spec.row_count
                or row["canonical_split"] != CANONICAL[name] or row["host_private"] != (name in HOST_SPLITS)):
            raise ValueError("Replication split differs from its frozen counts or access role")
    return manifest


def load_split(material, split, manifest, *, protocol=PROTOCOL, host_authorized=False, host_root=None):
    if split not in SPLITS or (split in HOST_SPLITS and (not host_authorized or host_root is None)):
        raise ValueError("Confirmation banks require explicit staged host authorization")
    if split in HOST_SPLITS:
        _, root = _private_paths(material, host_root)
    else:
        root = Path(material)
    path = root / f"{split}.json"
    expected = manifest["splits"][split]
    if file_hash(path) != expected["sha256"]:
        raise ValueError("Replication roster identity differs")
    rows = read(path)
    q.validate(rows, CANONICAL[split], q.PopulationSpec(**expected["population"]))
    return rows


def arm_mapping(arm):
    if arm not in ("rank8", "rank4"):
        raise ValueError("Unknown prospectively authorized rank arm")
    bank, smoke = ("bank-a", "smoke-r8") if arm == "rank8" else ("bank-b", "smoke-r4")
    return {name: smoke if name == "smoke" else f"{bank}-{name}" if name in
            ("preservation-validation", "selection", "hidden") else name for name in q.SPLITS}


def create_q_view(material, host_root, manifest, out, *, arm, arm_protocol, master_protocol):
    """Prospectively create immutable Q-compatible host views; no model scoring.

    The copied fit/development files are byte-identical across arms. Selection
    and preservation files stay host-private; hidden never enters the view.
    Training workers receive only the separate two-split training bundle.
    """
    out = Path(out)
    if out.exists() or out.resolve().is_relative_to(ROOT):
        raise ValueError("Arm material requires a new private view")
    mapping = arm_mapping(arm)
    value = {"schema": q.SCHEMA, "protocol_sha256": file_hash(arm_protocol),
        "sources": q._sources(arm_protocol), "domain": q.DOMAIN,
        "selection_is_error_based": False, "splits": {}, "artifacts": {}}
    out.mkdir(parents=True)
    for canonical, actual in mapping.items():
        entry = manifest["splits"][actual]
        source = (Path(host_root) if actual in HOST_SPLITS else Path(material)) / f"{actual}.json"
        if file_hash(source) != entry["sha256"]:
            raise ValueError("Global source changed while creating an arm view")
        # Do not copy sealed content into the model-study material directory.
        if canonical != "hidden":
            with (out / f"{canonical}.json").open("xb") as stream:
                stream.write(source.read_bytes())
        value["splits"][canonical] = {key: entry[key] for key in
            ("sha256", "rows", "population", "underlying_tasks", "semantic_families", "seed_commitment")}
        value["artifacts"][f"{canonical}.json"] = entry["sha256"]
    json_dump_atomic(out / "manifest.json", value)
    q.read_manifest(out, protocol=arm_protocol)
    return value
