"""Frozen generative behavior contracts and endpoint-aligned candidate guards."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from decimal import Decimal
import re
import time
from typing import Literal

from weightbridge.adaptation.content_scoring import generate_responses, prepare_continuations
from weightbridge.provenance import peak_rss_bytes


_GROUPS = ("target", "positive", "preserve")
_STOP_REASONS = ("eos", "token_budget")


@dataclass(frozen=True)
class ParsedOutput:
    """A completed output may still violate the grammar or task options."""

    completed: bool
    answer: str | None


@dataclass(frozen=True)
class Question:
    answer: str
    allowed_answers: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BehavioralEndpointContract:
    """Host-owned endpoint semantics; adapters provide decoding, never promotion."""

    endpoint_id: str
    prompt_representation: Literal["bare", "native"]
    output_grammar: str
    completion_conditions: str
    parse: Callable[[str, str], ParsedOutput]
    question: Callable[[Mapping], Question]
    max_input_tokens: int
    max_new_tokens: int
    batch_size: int
    required_wordings: tuple[int, ...]
    max_candidate_evaluations: int

    def __post_init__(self) -> None:
        if self.prompt_representation not in ("bare", "native"):
            raise ValueError("Unknown endpoint prompt representation")
        if any(not isinstance(value, str) or not value.strip() for value in
               (self.endpoint_id, self.output_grammar, self.completion_conditions)):
            raise ValueError("Endpoint identity, grammar and completion rules are required")
        if not callable(self.parse) or not callable(self.question):
            raise ValueError("Endpoint requires host parser and question validator")
        for name in ("max_input_tokens", "max_new_tokens", "batch_size", "max_candidate_evaluations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(self.required_wordings, tuple) or not self.required_wordings
                or any(type(value) is not int or value < 0 for value in self.required_wordings)
                or len(set(self.required_wordings)) != len(self.required_wordings)):
            raise ValueError("Endpoint wording IDs must be a nonempty tuple of distinct nonnegative integers")

    def manifest(self) -> dict:
        return {
            "schema": "weightbridge.behavioral_endpoint.v1",
            "id": self.endpoint_id,
            "prompt_representation": self.prompt_representation,
            "output_grammar": self.output_grammar,
            "completion_conditions": self.completion_conditions,
            "required_wordings": list(self.required_wordings),
            "max_candidate_evaluations": self.max_candidate_evaluations,
            "decode": {"do_sample": False, "num_beams": 1,
                       "max_input_tokens": self.max_input_tokens, "max_new_tokens": self.max_new_tokens,
                       "batch_size": self.batch_size},
            "preservation": "no lost correctness or validity on any parent-correct or parent-valid presentation",
        }


class CandidateEvaluationBudget:
    """One shared counter for candidate calls; parent/independent endpoint calls are separate."""

    def __init__(self, contract: BehavioralEndpointContract) -> None:
        self.limit = contract.max_candidate_evaluations
        self.used = 0

    def charge(self) -> None:
        if self.used >= self.limit:
            raise ValueError("Endpoint candidate evaluation budget exhausted")
        self.used += 1


class _BareTokenizer:
    chat_template = None

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def endpoint_runner(runner, contract: BehavioralEndpointContract):
    """Apply only the frozen prefix representation, leaving the model and native IDs unchanged."""
    if contract.prompt_representation == "native":
        return runner
    view = copy(runner)
    view.tokenizer = _BareTokenizer(runner.tokenizer)
    return view


def _roster(rows: Sequence[Mapping], contract: BehavioralEndpointContract):
    if not rows:
        raise ValueError("Endpoint roster is empty")
    items = defaultdict(dict)
    parsed = {}
    for row in rows:
        if (not isinstance(row, Mapping) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                                for key in ("id", "item", "prompt"))
                or row.get("group") not in _GROUPS or type(row.get("wording")) is not int
                or row["wording"] not in contract.required_wordings):
            raise ValueError("Malformed frozen endpoint presentation")
        if row["id"] in parsed or row["wording"] in items[row["item"]]:
            raise ValueError("Duplicate endpoint presentation or wording")
        question = contract.question(row)
        if (not isinstance(question, Question) or not isinstance(question.answer, str) or not question.answer
                or (question.allowed_answers is not None and question.answer not in question.allowed_answers)):
            raise ValueError("Endpoint question validator returned invalid semantics")
        parsed[row["id"]] = question
        items[row["item"]][row["wording"]] = row
    for forms in items.values():
        if set(forms) != set(contract.required_wordings):
            raise ValueError("Frozen endpoint item is missing required wordings")
        semantics = {(row["group"], parsed[row["id"]].answer,
                      frozenset(parsed[row["id"]].allowed_answers or ())) for row in forms.values()}
        if len(semantics) != 1 or len({row["prompt"] for row in forms.values()}) != len(forms):
            raise ValueError("Endpoint wordings must have identical semantics and distinct prompts")
    return parsed, items


def _observation(row: Mapping, question: Question, contract: BehavioralEndpointContract) -> dict:
    text, reason = row.get("text"), row.get("stop_reason")
    if not isinstance(text, str) or reason not in _STOP_REASONS:
        raise ValueError("Malformed generation result")
    parsed = contract.parse(text, reason)
    allowed = question.allowed_answers
    if not parsed.completed:
        status = "incomplete"
    elif parsed.answer is None:
        status = "invalid_grammar"
    elif allowed is not None and parsed.answer not in allowed:
        status = "out_of_option"
    else:
        status = "correct" if parsed.answer == question.answer else "incorrect"
    return {"exact": parsed.answer, "completed": parsed.completed, "grammar_valid": parsed.answer is not None,
            "valid": status in ("correct", "incorrect"), "correct": status == "correct", "status": status}


def measure_endpoint(runner, rows: Sequence[Mapping], contract: BehavioralEndpointContract, *,
                     decoder: Callable = generate_responses,
                     candidate_budget: CandidateEvaluationBudget | None = None) -> dict:
    """Execute real greedy decoding and classify all frozen presentations in order."""
    questions, _ = _roster(rows, contract)
    if candidate_budget is not None:
        if candidate_budget.limit != contract.max_candidate_evaluations:
            raise ValueError("Candidate budget differs from the endpoint contract")
        candidate_budget.charge()
    wall, cpu = time.perf_counter(), time.process_time()
    outputs, resources = decoder(endpoint_runner(runner, contract), [row["prompt"] for row in rows],
                                 max_input_tokens=contract.max_input_tokens, max_new_tokens=contract.max_new_tokens,
                                 batch_size=contract.batch_size)
    measured = []
    for row, output in zip(rows, outputs, strict=True):
        # Keep decoder metadata separate from frozen question metadata.
        if any(key in row for key in output):
            raise ValueError("Decoder attempted to replace frozen endpoint metadata")
        merged = {**row, **output}
        measured.append({**merged, **_observation(merged, questions[row["id"]], contract)})
    return {"endpoint": contract.manifest(), "rows": measured, "resources": {**resources,
            "wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
            "peak_rss_bytes": peak_rss_bytes()}}


def compare_endpoint(parent: Mapping, candidate: Mapping, contract: BehavioralEndpointContract, *,
                     min_target_gain: int = 1, min_repeatable_corrections: int = 0) -> dict:
    """Pass only real target gains across every required wording and no prohibited regressions."""
    if (type(min_target_gain) is not int or min_target_gain < 1
            or type(min_repeatable_corrections) is not int or min_repeatable_corrections < 0):
        raise ValueError("Endpoint verification needs a positive target gain and a nonnegative repeatable count")
    if parent.get("endpoint") != contract.manifest() or candidate.get("endpoint") != contract.manifest():
        raise ValueError("Parent and candidate must use the same frozen endpoint")
    prior_questions, prior_items = _roster(parent["rows"], contract)
    post_questions, post_items = _roster(candidate["rows"], contract)
    keys = ("id", "item", "wording", "prompt", "group")
    if (prior_questions != post_questions or [tuple(row[key] for key in keys) for row in parent["rows"]]
            != [tuple(row[key] for key in keys) for row in candidate["rows"]]):
        raise ValueError("Parent and candidate must use the same frozen roster and row order")
    before = {row["id"]: _observation(row, prior_questions[row["id"]], contract) for row in parent["rows"]}
    after = {row["id"]: _observation(row, post_questions[row["id"]], contract) for row in candidate["rows"]}

    def solved(forms, observations):
        return all(observations[row["id"]]["correct"] for row in forms.values())

    def metrics(rows, items, observations):
        groups = {}
        for group in _GROUPS:
            presentations = [observations[row["id"]] for row in rows if row["group"] == group]
            group_items = [forms for forms in items.values() if next(iter(forms.values()))["group"] == group]
            groups[group] = {"items": len(group_items), "items_correct": sum(solved(forms, observations)
                            for forms in group_items), "presentations": len(presentations),
                            **{key: sum(row[key] for row in presentations)
                               for key in ("completed", "grammar_valid", "valid", "correct")},
                            "out_of_option": sum(row["status"] == "out_of_option" for row in presentations)}
        return groups

    pm = metrics(parent["rows"], prior_items, before)
    cm = metrics(candidate["rows"], post_items, after)
    corrected, repeated = [], []
    for item, forms in prior_items.items():
        if (next(iter(forms.values()))["group"] != "target" or solved(forms, before)
                or not solved(post_items[item], after)):
            continue
        corrected.append(item)
        observations = [before[row["id"]] for row in forms.values()]
        if (all(observation["status"] == "incorrect" for observation in observations)
                and len({observation["exact"] for observation in observations}) == 1):
            repeated.append(item)
    regressions = [row_id for row_id in before if before[row_id]["correct"] and not after[row_id]["correct"]]
    lost_valid = [row_id for row_id in before if before[row_id]["valid"] and not after[row_id]["valid"]]
    gain = cm["target"]["items_correct"] - pm["target"]["items_correct"]
    return {"passed": gain >= min_target_gain and len(repeated) >= min_repeatable_corrections
                      and not regressions and not lost_valid,
            "target_item_gain": gain, "target_presentation_gain": cm["target"]["correct"] - pm["target"]["correct"],
            "corrected_target_items": corrected, "corrected_repeatable_target_items": repeated,
            "prohibited_regressions": regressions, "lost_valid_presentations": lost_valid,
            "changed_presentations": [row_id for row_id in before if before[row_id] != after[row_id]],
            "parent": pm, "candidate": cm}


_DECIMAL = re.compile(r"[ \t]*(-?(?:[0-9]{1,9}(?:\.[0-9]{1,9})?|\.[0-9]{1,9}))[ \t]*")
_VALUES = re.compile(r"Values: (\S+) or (\S+)\nAnswer:[ \t]*\Z")
_COPY = re.compile(r"Decimal: (\S+)\nAnswer:[ \t]*\Z")


def canonical_decimal(text: str) -> str | None:
    match = _DECIMAL.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        return None
    value = Decimal(match[1])
    return "0" if not value else format(value.normalize(), "f")


def parse_decimal_output(text: str, stop_reason: str) -> ParsedOutput:
    first, newline, _ = text.partition("\n")
    if not newline and stop_reason != "eos":
        return ParsedOutput(False, None)
    return ParsedOutput(True, canonical_decimal(first.removesuffix("\r") if newline else first))


def decimal_question(row: Mapping) -> Question:
    """Check the semantic answer and operands against the actual final displayed question."""
    family = row.get("family")
    if family == "larger-decimal":
        pattern, groups = _VALUES, ("target", "positive", "preserve")
    elif family == "copy":
        pattern, groups = _COPY, ("preserve",)
    else:
        raise ValueError("Unsupported decimal task family")
    if row.get("group") not in groups:
        raise ValueError("Decimal task family does not match its roster group")
    prompt = row.get("prompt")
    tail = "\n".join(prompt.split("\n")[-2:]) if isinstance(prompt, str) else ""
    match = pattern.fullmatch(tail)
    operands = row.get("operands")
    if (match is None or not isinstance(operands, (tuple, list)) or tuple(operands) != match.groups()):
        raise ValueError("Declared operands do not match the displayed decimal question")
    options = tuple(canonical_decimal(value) for value in operands)
    if None in options or len(set(options)) != len(options):
        raise ValueError("Decimal question has malformed or equal operands")
    correct = max(options, key=Decimal) if family == "larger-decimal" else options[0]
    if canonical_decimal(row.get("answer")) != correct:
        raise ValueError("Declared answer does not match the semantic decimal answer")
    return Question(correct, options if family == "larger-decimal" else None)


def decimal_endpoint(settings: Mapping, *, required_wordings: tuple[int, ...] = (0, 1),
                     max_candidate_evaluations: int = 1) -> BehavioralEndpointContract:
    return BehavioralEndpointContract(
        "weightbridge.decimal_first_line.v1", "bare",
        "One exact bounded decimal on the first line; comparison output must be a displayed operand",
        "Observed newline or EOS; reaching the cap without either is incomplete",
        parse_decimal_output, decimal_question, settings["max_input_tokens"], settings["max_new_tokens"],
        settings["batch_size"], required_wordings, max_candidate_evaluations,
    )


@dataclass(frozen=True)
class FitContrast:
    id: str
    prompt: str
    answers: tuple[str, str]
    preserve: bool
    kind: Literal["task_choice", "auxiliary_copy_decoy", "legacy_comparison_decoy"]

    def to_solver_row(self) -> dict:
        return {"id": self.id, "prompt": self.prompt, "answers": list(self.answers), "preserve": self.preserve}

    as_fit_row = to_solver_row


def validate_decimal_fit_row(presentation: Mapping, fit: Mapping, *,
                             mode: Literal["semantic", "legacy_decoy_control"] = "semantic") -> FitContrast:
    """Refuse a non-displayed rival for task-choice fitting; legacy decoys are diagnostic only."""
    if mode not in ("semantic", "legacy_decoy_control"):
        raise ValueError("Unknown contrast fitting mode")
    question = decimal_question(presentation)
    if (not isinstance(fit.get("id"), str) or fit["id"] != presentation.get("id")
            or fit.get("prompt") != presentation.get("prompt") or type(fit.get("preserve")) is not bool
            or (not fit["preserve"] and presentation.get("group") != "target")):
        raise ValueError("Fit row does not match the frozen question or preservation role")
    answers = fit.get("answers")
    if not isinstance(answers, (tuple, list)) or len(answers) != 2:
        raise ValueError("Fit row requires two answer surfaces")
    correct, rival = map(canonical_decimal, answers)
    if correct != question.answer or rival is None or rival == correct:
        raise ValueError("Fit row has an invalid correct answer or rival")
    if question.allowed_answers is None:
        kind = "auxiliary_copy_decoy"
    elif mode == "legacy_decoy_control" and fit["preserve"]:
        kind = "legacy_comparison_decoy"
    else:
        kind = "task_choice"
    if kind == "task_choice":
        other = next(option for option in question.allowed_answers if option != question.answer)
        if rival != other:
            raise ValueError("Comparison contrast rival must be the other displayed operand")
    elif Decimal(rival) != Decimal(correct) + 1:
        raise ValueError("Declared diagnostic decoy does not match the frozen +1 rule")
    return FitContrast(fit["id"], fit["prompt"], tuple(answers), fit["preserve"], kind)


def build_decimal_fit_rows(runner, measured: Mapping, contract: BehavioralEndpointContract, *,
                           target_items: Sequence[str], preservation_items: Sequence[str], max_rows: int,
                           mode: Literal["semantic", "legacy_decoy_control"] = "semantic") -> tuple[FitContrast, ...]:
    """Compile only authorized repeatable errors and previously solved preservation questions."""
    if measured.get("endpoint") != contract.manifest():
        raise ValueError("Fit evidence does not use the frozen endpoint")
    questions, items = _roster(measured["rows"], contract)
    target, preserved = set(target_items), set(preservation_items)
    if not target or target & preserved or (target | preserved) - items.keys():
        raise ValueError("Malformed or unauthorized fit item scope")
    if type(max_rows) is not int or max_rows < 1 or (len(target) + len(preserved)) * len(contract.required_wordings) > max_rows:
        raise ValueError("Fit scope exceeds the frozen row budget")
    if mode not in ("semantic", "legacy_decoy_control"):
        raise ValueError("Unknown contrast fitting mode")
    for item in target | preserved:
        forms = items[item]
        observations = [_observation(row, questions[row["id"]], contract) for row in forms.values()]
        if item in target:
            if (next(iter(forms.values()))["group"] != "target" or
                    any(result["status"] != "incorrect" for result in observations)
                    or len({result["exact"] for result in observations}) != 1):
                raise ValueError("Fit target must be a repeatable completed wrong-option choice")
        elif any(not result["correct"] for result in observations):
            raise ValueError("Fit preservation examples must be solved across every wording")
    tokenizer = endpoint_runner(runner, contract).tokenizer
    fit = []
    for row in measured["rows"]:
        item, prompt = row["item"], row["prompt"]
        if item not in target | preserved:
            continue
        question = questions[row["id"]]
        if question.allowed_answers is None or (item in preserved and mode == "legacy_decoy_control"):
            decoy = Decimal(question.answer) + 1
            rival = canonical_decimal(format(decoy, "f"))
        else:
            rival = next(option for option in question.allowed_answers if option != question.answer)
        if rival is None:
            raise ValueError("Diagnostic decoy is outside the decimal grammar")
        # Prefer the displayed spelling rather than removing significant tokenization differences.
        spellings = {canonical_decimal(value): value for value in row["operands"]}
        values = (spellings[question.answer], spellings.get(rival, rival))
        spaces = ("", " ") if prompt[-1:].isspace() else (" ", "")
        for space in spaces:
            answers = tuple(space + value for value in values)
            try:
                prepare_continuations(tokenizer, [prompt], [answers], max_input_tokens=contract.max_input_tokens)
            except ValueError:
                continue
            fit.append(validate_decimal_fit_row(row, {"id": row["id"], "prompt": prompt, "answers": answers,
                                                     "preserve": item in preserved}, mode=mode))
            break
        else:
            raise ValueError("Decimal contrast changes the frozen prompt token boundary")
    return tuple(fit)
