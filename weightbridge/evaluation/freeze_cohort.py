"""Freeze a host identity receipt into a new MCP cohort protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .mcp_cohort import ROOT, CohortProtocolError, _canonical_hash, _read_json, validate_protocol


_EXPECTED_TEMPLATE_HOLDS = {
    "LLM identity has no host verification",
    "LLM sampling configuration is missing",
    "LLM identity evidence path is missing",
    "protocol is a template and has not been frozen for a specific LLM",
}


def freeze_protocol(
    template_path: str | Path,
    identity_receipt_path: str | Path,
    output_path: str | Path,
    *,
    cohort_id: str,
    identity_provenance: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Create a new protocol and colocated receipt without overwriting files.

    The provenance string is the host operator's account of the identity
    check. It is not a cryptographic provider attestation.
    """
    root = Path(root).resolve()
    template_file = Path(template_path).resolve(strict=True)
    receipt_file = Path(identity_receipt_path).resolve(strict=True)
    output_file = Path(output_path).resolve()
    if not output_file.is_relative_to(root):
        raise CohortProtocolError("frozen protocol output must be inside the project root")
    if not output_file.parent.is_dir():
        raise CohortProtocolError("frozen protocol output directory must already exist")
    receipt_output = output_file.with_name(f"{output_file.stem}.identity_receipt.json")
    if output_file.exists() or receipt_output.exists():
        raise FileExistsError("protocol and receipt outputs are immutable; choose new output paths")

    template = _read_json(template_file)
    if template.get("status") != "TEMPLATE_HOLD":
        raise CohortProtocolError("input must be an unfrozen TEMPLATE_HOLD protocol")
    template_errors = validate_protocol(template, root=root)
    unexpected = [
        error for error in template_errors
        if not error.startswith("LLM identity is not pinned:")
        and error not in _EXPECTED_TEMPLATE_HOLDS
    ]
    if unexpected:
        raise CohortProtocolError("template is not intact: " + "; ".join(unexpected))

    if not isinstance(cohort_id, str) or not cohort_id.strip():
        raise CohortProtocolError("cohort_id must be a nonempty string")
    if not isinstance(identity_provenance, str) or not identity_provenance.strip():
        raise CohortProtocolError("identity_provenance must describe the host verification")

    receipt_bytes = receipt_file.read_bytes()
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CohortProtocolError("identity receipt must be valid UTF-8 JSON") from exc
    required_text = ("provider", "model_id", "version", "host_call_id")
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "weightbridge.llm_identity_receipt.v1"
            or any(not isinstance(receipt.get(key), str) or not receipt[key].strip()
                   for key in required_text)
            or not isinstance(receipt.get("sampling"), dict)
            or not receipt["sampling"]):
        raise CohortProtocolError("identity receipt is incomplete or has an unsupported schema")

    model_binding = template.get("model_binding")
    if not isinstance(model_binding, dict):
        raise CohortProtocolError("template model_binding must be an object")
    evidence_relative = receipt_output.relative_to(root).as_posix()
    model_binding.update({
        "provider": receipt["provider"],
        "model_id": receipt["model_id"],
        "version": receipt["version"],
        "sampling": receipt["sampling"],
        "identity_provenance": identity_provenance.strip(),
        "identity_verified": True,
        "identity_evidence_path": evidence_relative,
        "identity_evidence_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    })
    protocol = dict(template)
    protocol["cohort_id"] = cohort_id.strip()
    protocol["model_binding"] = model_binding
    protocol["status"] = "FROZEN"
    protocol["protocol_sha256"] = _canonical_hash(protocol)

    created_receipt = False
    created_protocol = False
    try:
        with receipt_output.open("xb") as stream:
            stream.write(receipt_bytes)
        created_receipt = True
        errors = validate_protocol(protocol, root=root)
        if errors:
            raise CohortProtocolError("frozen protocol failed validation: " + "; ".join(errors))
        with output_file.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n")
        created_protocol = True
    except Exception:
        if created_protocol and output_file.exists():
            output_file.unlink()
        if created_receipt and receipt_output.exists():
            receipt_output.unlink()
        raise
    return {
        "protocol_path": output_file,
        "identity_receipt_path": receipt_output,
        "protocol_sha256": protocol["protocol_sha256"],
        "identity_evidence_sha256": model_binding["identity_evidence_sha256"],
        "cohort_id": protocol["cohort_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a new MCP cohort protocol from a host identity receipt")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--identity-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="new protocol file inside the WeightBridge project")
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--identity-provenance", required=True,
                        help="host/operator description of the identity check")
    args = parser.parse_args()
    try:
        result = freeze_protocol(
            args.template, args.identity_receipt, args.out,
            cohort_id=args.cohort_id,
            identity_provenance=args.identity_provenance,
        )
    except (OSError, CohortProtocolError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
