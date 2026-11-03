"""Compare completed decimal values using exact, bounded decimal parsing."""
from __future__ import annotations

from decimal import Decimal
import re
import time

from experiments.phase_iiig.endpoint import bare_runner
from weightbridge.adaptation.content_scoring import generate_responses
from weightbridge.provenance import peak_rss_bytes


_DECIMAL = re.compile(r"[ \t]*(-?(?:[0-9]{1,9}(?:\.[0-9]{1,9})?|\.[0-9]{1,9}))[ \t]*")


def canonical_decimal(text):
    match = _DECIMAL.fullmatch(text)
    if match is None:
        return None
    value = Decimal(match[1])
    return "0" if not value else format(value.normalize(), "f")


def parse_decimal(text, stop_reason):
    first_line, newline, _ = text.partition("\n")
    if not newline and stop_reason != "eos":
        return None
    return canonical_decimal(first_line.removesuffix("\r") if newline else first_line)


def measure_decimal(runner, rows, settings):
    wall, cpu = time.perf_counter(), time.process_time()
    outputs, resources = generate_responses(
        bare_runner(runner), [row["prompt"] for row in rows],
        max_input_tokens=settings["max_input_tokens"], max_new_tokens=settings["max_new_tokens"],
        batch_size=settings["batch_size"],
    )
    measured = [{**row, **output, "exact": parse_decimal(output["text"], output["stop_reason"])}
                for row, output in zip(rows, outputs, strict=True)]
    return {"rows": measured, "resources": {**resources,
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes()}}
