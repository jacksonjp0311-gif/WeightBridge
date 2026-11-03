"""Replicate interaction mechanisms using independent, frozen NumPy fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from weightbridge.adaptation import bic_math


def mse(prediction, target):
    return float(np.mean(np.square(prediction - target)))


def finite_derivatives(loss, point, step):
    """Central finite differences for one explicitly bounded two-dimensional fixture."""
    first, second = step * np.eye(2)
    value = loss(point)
    plus = np.array([loss(point + e) for e in (first, second)])
    minus = np.array([loss(point - e) for e in (first, second)])
    grad = (plus - minus) / (2 * step)
    diagonal = (plus - 2 * value + minus) / step**2
    cross = (loss(point + first + second) - loss(point + first - second)
             - loss(point - first + second) + loss(point - first - second)) / (4 * step**2)
    return grad, np.array([[diagonal[0], cross], [cross, diagonal[1]]])


def product_derivatives(point, target, weight):
    a, b = point
    residual = a * b - target
    gradient = 2 * weight * residual * np.array([b, a])
    mixed = 2 * weight * (2 * a * b - target)
    hessian = np.array([[2 * weight * b**2, mixed], [mixed, 2 * weight * a**2]])
    return gradient, hessian


def compare_local_models(loss, radius):
    origin_loss = loss(np.zeros(2))
    measured = {}
    for name, interaction in (("first_order", False), ("interaction", True)):
        local = bic_math.fit_local_quadratic(loss, origin_loss, radius=radius, interaction=interaction)
        candidate = bic_math.select_coefficients(local, radius=radius)
        actual = loss(candidate)
        measured[name] = {"coefficients": candidate.tolist(), "predicted_loss": local.predict(candidate),
                          "actual_loss": actual, "accepted": actual < origin_loss,
                          "probe_calls": len(local.probes), "verification_calls": 1,
                          "gradient": local.gradient.tolist(), "hessian": local.hessian.tolist()}
    return measured


def scalar_fixtures(step):
    target, weight = 0.4, 0.506

    def loss(point):
        return float(weight * (point[0] * point[1] - target) ** 2)

    origin, active = np.zeros(2), np.array([1.0, 0.0])
    derivative_errors = []
    for point in (origin, active, np.array([0.31, -0.17])):
        analytical = product_derivatives(point, target, weight)
        numerical = finite_derivatives(loss, point, step)
        derivative_errors.extend(float(np.max(np.abs(a - b))) for a, b in zip(analytical, numerical))

    gradient, hessian = product_derivatives(origin, target, weight)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    direction = eigenvectors[:, 0]
    # Solve the actual line loss in the chosen direction, including the quartic term.
    radius = np.sqrt(target / np.prod(direction))
    candidate = radius * direction
    active_gradient, _ = product_derivatives(active, target, weight)
    first_order_direction = -active_gradient / np.linalg.norm(active_gradient)
    line_candidates = [active + scale * first_order_direction for scale in np.linspace(0.0, 2 * target, 17)]
    active_final = min(loss(proposal) for proposal in line_candidates)

    def harmful_loss(point):
        product = float(point[0] * point[1])
        return (product - 8 * product**2 - target) ** 2

    harmful_gradient, harmful_hessian = finite_derivatives(harmful_loss, origin, step)
    harmful_candidate = 0.6 * np.sign(direction)
    harmful_baseline = harmful_loss(origin)
    harmful_local = float(harmful_baseline + harmful_gradient @ harmful_candidate
                          + 0.5 * harmful_candidate @ harmful_hessian @ harmful_candidate)
    harmful_actual = harmful_loss(harmful_candidate)
    dormant_compiler = compare_local_models(loss, float(np.sqrt(2 * target)))
    active_compiler = compare_local_models(lambda update: loss(active + update), target)
    harmful_compiler = compare_local_models(harmful_loss, float(np.linalg.norm(harmful_candidate)))
    return {
        "two_weight": {
            "target": target, "weight": weight, "initial_loss": loss(origin),
            "gradient": gradient.tolist(), "hessian": hessian.tolist(),
            "minimum_curvature": float(eigenvalues[0]), "first_order_final_loss": loss(origin),
            "interaction_final_loss": loss(candidate), "exact_final_loss": loss([np.sqrt(target)] * 2),
            "analytical_direction_proposals": 1, "shared_compiler": dormant_compiler,
        },
        "active_first_order": {
            "initial_loss": loss(active), "gradient_norm": float(np.linalg.norm(active_gradient)),
            "first_order_final_loss": active_final, "exact_final_loss": loss([1.0, target]),
            "candidate_evaluations": len(line_candidates), "shared_compiler": active_compiler,
        },
        "harmful_interaction": {
            "initial_loss": harmful_baseline, "local_predicted_loss": harmful_local,
            "actual_loss": harmful_actual, "rejected": harmful_local < harmful_baseline < harmful_actual,
            "failure": "misleading_local_approximation", "shared_compiler": harmful_compiler,
        },
        "derivative_max_absolute_error": max(derivative_errors),
    }


def dormant_case(seed, cfg, step):
    rng = np.random.default_rng(seed)
    width, count = cfg["features"], cfg["candidate_evaluations_per_arm"]
    limit = cfg["maximum_absolute_factor_product"]
    designated = int(rng.integers(width))
    coefficient = rng.uniform(0.2, 0.6) * rng.choice([-1.0, 1.0])
    practice = rng.normal(size=(cfg["practice_rows"], width))
    test = rng.normal(size=(cfg["test_rows"], width))
    target = practice[:, designated] * coefficient + rng.normal(scale=cfg["noise_sd"], size=len(practice))
    expected = test[:, designated] * coefficient + rng.normal(scale=cfg["noise_sd"], size=len(test))

    # At a=b=0, d²L/(da_j db_j) = -2 E[x_j y] and all first derivatives vanish.
    # Compute only these width diagonal mixed blocks. One contraction covers all pairs.
    mixed = (-2.0 / len(practice)) * (practice.T @ target)
    selected = int(np.argmax(np.abs(mixed)))

    def prediction(features, index, product):
        a = np.sqrt(abs(product))
        b = np.copysign(a, product)
        return features[:, index] * (a * b)

    def fitted_product(index):
        values = practice[:, index]
        return float(np.clip(values @ target / (values @ values), -limit, limit))

    def best(candidates):
        return min(((choice, mse(prediction(practice, *choice), target)) for choice in candidates),
                   key=lambda pair: pair[1])

    guided_scale = fitted_product(selected)
    strengths = np.linspace(0.25, 1.75, count - 1, endpoint=False)
    guided, guided_loss = best([(selected, 0.0)] + [(selected, float(product))
                               for product in np.clip(guided_scale * strengths, -limit, limit)])
    random_indices = rng.choice(width, size=count - 1, replace=False)
    random, random_loss = best([(int(random_indices[0]), 0.0)]
                               + [(int(index), fitted_product(index)) for index in random_indices])
    projections = practice.T @ target
    energies = np.sum(practice**2, axis=0)
    products = np.clip(projections / energies, -limit, limit)
    exact_index = int(np.argmax(2 * projections * products - energies * products**2))
    exact = exact_index, float(products[exact_index])
    numerical = finite_derivatives(
        lambda point: mse(practice[:, selected] * (point[0] * point[1]), target), np.zeros(2), step,
    )
    derivative_error = abs(float(numerical[1][0, 1]) - float(mixed[selected]))

    def result(choice, practice_loss=None):
        index, product = choice
        return {"selected_feature": index, "factor_product": product, "hit": index == designated and product != 0.0,
                "update_norm": float(np.sqrt(2 * abs(product))),
                "practice_mse": mse(prediction(practice, *choice), target) if practice_loss is None else practice_loss,
                "test_mse": mse(prediction(test, *choice), expected)}

    return {"seed": seed, "designated_feature": designated, "mixed_finite_difference_error": derivative_error,
            "first_order": result((0, 0.0)), "mixed_guided": result(guided, guided_loss),
            "random_paired": result(random, random_loss), "exact_one_feature": result(exact)}


def permutation_fixture(seed):
    rng = np.random.default_rng(seed)
    hidden_weights = rng.normal(size=(4, 4))
    output_weights = rng.normal(size=4)
    practice, test = rng.normal(size=(96, 4)), rng.normal(size=(256, 4))
    permutation = np.array([2, 3, 1, 0])
    raw_update = np.array([0.8, 0.0, 0.0, 0.0])

    def hidden(inputs):
        return np.tanh(inputs @ hidden_weights.T)

    source_practice, source_test = hidden(practice), hidden(test)
    permuted_practice, permuted_test = source_practice[:, permutation], source_test[:, permutation]
    permuted_output = output_weights[permutation]
    baseline = source_test @ output_weights
    expected = source_test @ (output_weights + raw_update)
    residual = source_practice @ (output_weights + raw_update) - permuted_practice @ permuted_output
    column_energy = np.sum(permuted_practice**2, axis=0)
    correlations = permuted_practice.T @ residual
    feature = int(np.argmax(np.square(correlations) / column_energy))
    recompiled = np.zeros(4)
    recompiled[feature] = correlations[feature] / column_energy[feature]
    return {
        "equivalent_baseline_max_absolute_difference": float(np.max(np.abs(baseline - permuted_test @ permuted_output))),
        "unadapted_test_mse": mse(baseline, expected),
        "raw_transfer_test_mse": mse(permuted_test @ (permuted_output + raw_update), expected),
        "recompiled_test_mse": mse(permuted_test @ (permuted_output + recompiled), expected),
        "permutation": permutation.tolist(), "raw_update": raw_update.tolist(), "recompiled_update": recompiled.tolist(),
    }


def sequential_case(seed, cfg):
    """Five conflicting tasks on a shared linear bottleneck with independent examples."""
    rng = np.random.default_rng(seed)
    dimensions = cfg["shared_dimensions"]
    common = rng.normal(size=dimensions)
    directions = np.eye(dimensions) + 0.35 * common
    directions = np.vstack((directions, directions[0] - 0.3 * directions[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    desired = rng.uniform(0.4, 0.8, cfg["cycles"]) * np.array([1, -1, 1, -1, -1])
    initial = rng.normal(scale=0.1, size=dimensions)
    practices = [rng.normal(size=(cfg["practice_rows"], 1)) * direction for direction in directions]
    tests = [rng.normal(size=(cfg["test_rows"], 1)) * direction for direction in directions]

    def labels(matrix, i):
        return desired[i] * (matrix @ directions[i])

    def train(arm):
        weights, events, states = initial.copy(), [], []
        for i, practice in enumerate(practices):
            expected = labels(practice, i)
            residual = expected - practice @ weights
            targets = practice / np.sqrt(len(practice))
            residuals = residual / np.sqrt(len(practice))
            prior = np.vstack(practices[:i]) if i else np.empty((0, dimensions))
            _, singular, right = np.linalg.svd(prior, full_matrices=True)
            rank = int(np.sum(singular > 1e-10))
            basis = right[rank:].T if arm == "strict_nullspace" else np.eye(dimensions)
            if arm == "soft_retention" and i:
                targets = np.vstack((targets, np.sqrt(cfg["retention_penalty"] / len(prior)) * prior))
                residuals = np.concatenate((residuals, np.zeros(len(prior))))
            targets = np.vstack((targets, np.sqrt(cfg["update_penalty"]) * np.eye(dimensions)))
            residuals = np.concatenate((residuals, np.zeros(dimensions)))
            coefficients, *_ = np.linalg.lstsq(targets @ basis, residuals, rcond=None)
            update = basis @ coefficients
            candidate = weights + update
            before, after = mse(practice @ weights, expected), mse(practice @ candidate, expected)
            accepted = before - after > cfg["minimum_target_improvement"]
            events.append({"cycle": i + 1, "nullspace_dimension": dimensions - rank,
                           "target_practice_before": before, "target_practice_candidate": after,
                           "update_norm": float(np.linalg.norm(update)), "accepted": accepted,
                           "prior_practice_drift": mse(prior @ candidate, prior @ weights) if i else 0.0})
            if accepted:
                weights = candidate
            states.append(weights.copy())

        # Score hidden rows only after the whole sequence has committed or rejected each update.
        final_errors = [mse(test @ weights, labels(test, i)) for i, test in enumerate(tests)]
        retention = [mse(test @ weights, test @ states[i]) for i, test in enumerate(tests[:-1])]
        for i, test in enumerate(tests):
            parent = initial if i == 0 else states[i - 1]
            events[i]["target_test_parent"] = mse(test @ parent, labels(test, i))
            events[i]["target_test_retained"] = mse(test @ states[i], labels(test, i))
        return {"events": events, "final_target_test_mse": float(np.mean(final_errors)),
                "final_prior_target_test_mse": float(np.mean(final_errors[:-1])),
                "final_prior_behavior_drift": float(np.mean(retention)),
                "accepted_cycles": sum(event["accepted"] for event in events)}

    return {"seed": seed, **{arm: train(arm) for arm in ("target_only", "soft_retention", "strict_nullspace")}}


def aggregate(dormant, sequential, protocol):
    dormant_summary = {}
    for arm in ("first_order", "mixed_guided", "random_paired", "exact_one_feature"):
        rows = [case[arm] for case in dormant]
        dormant_summary[arm] = {"feature_hits": sum(row["hit"] for row in rows),
                                "mean_test_mse": float(np.mean([row["test_mse"] for row in rows]))}
    budget = protocol["dormant"]["candidate_evaluations_per_arm"]
    return {
        "dormant": {"cases": len(dormant), "nonlinear_evaluations_per_case": {
            "first_order": 1, "mixed_guided": budget, "random_paired": budget, "exact_one_feature": 1,
        }, "mixed_contractions_per_guided_case": 1,
            "guided_scalar_fits_per_case": 1, "random_scalar_fits_per_case": budget - 1,
            "derivative_audit_calls_per_case": 9, "arms": dormant_summary},
        "sequential": {"networks": len(sequential), "cycles_per_network": protocol["sequential"]["cycles"],
                       "arms": {arm: {field: float(np.mean([case[arm][field] for case in sequential]))
                                      for field in ("final_target_test_mse", "final_prior_target_test_mse",
                                                    "final_prior_behavior_drift", "accepted_cycles")}
                                for arm in ("target_only", "soft_retention", "strict_nullspace")}},
    }


def evaluate_gates(scalar, permuted, dormant, sequential, summary, gates):
    estimates = summary["dormant"]["arms"]
    guided, random = estimates["mixed_guided"], estimates["random_paired"]
    derivative_error = max(scalar["derivative_max_absolute_error"],
                           max(case["mixed_finite_difference_error"] for case in dormant))
    measured = summary["sequential"]["arms"]
    return {
        "finite_difference_agreement": derivative_error <= gates["derivative_max_absolute_error"],
        "dormant_scalar_escape": (scalar["two_weight"]["interaction_final_loss"] <= gates["exact_repair_max_loss"]
                                  and scalar["two_weight"]["shared_compiler"]["interaction"]["actual_loss"]
                                  <= gates["exact_repair_max_loss"]
                                  and not scalar["two_weight"]["shared_compiler"]["first_order"]["accepted"]
                                  and scalar["two_weight"]["gradient"] == [0.0, 0.0]),
        "dormant_family_selection": (guided["feature_hits"] / len(dormant) >= gates["guided_dormant_min_repair_rate"]
                                     and (guided["feature_hits"] - random["feature_hits"]) / len(dormant)
                                     >= gates["guided_dormant_min_advantage"]),
        "dormant_family_generalization": guided["mean_test_mse"] <= gates["guided_dormant_max_mse_ratio"] * random["mean_test_mse"],
        "active_first_order_suffices": (scalar["active_first_order"]["first_order_final_loss"] <= gates["exact_repair_max_loss"]
                                        and scalar["active_first_order"]["shared_compiler"]["first_order"]["actual_loss"]
                                        <= gates["exact_repair_max_loss"]),
        "harmful_candidate_rejected": (not gates["require_harmful_proposal_rejection"]
                                       or (scalar["harmful_interaction"]["rejected"]
                                           and not scalar["harmful_interaction"]["shared_compiler"]["interaction"]["accepted"])),
        "model_coordinate_recompilation": (permuted["raw_transfer_test_mse"] >= gates["permutation_raw_min_test_mse"]
                                           and permuted["recompiled_test_mse"] <= gates["exact_repair_max_loss"]),
        "soft_preservation_reduces_drift": (not gates["require_sequential_soft_drift_reduction"]
                                          or measured["soft_retention"]["final_prior_behavior_drift"]
                                          < measured["target_only"]["final_prior_behavior_drift"]),
        "strict_nullspace_exhaustion": (not gates["require_strict_nullspace_exhaustion"]
                                       or all(case["strict_nullspace"]["events"][-1]["nullspace_dimension"] == 0
                                              and not case["strict_nullspace"]["events"][-1]["accepted"]
                                              for case in sequential)),
    }


def run(protocol):
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    step = protocol["finite_difference_step"]
    stage = time.perf_counter()
    scalar = scalar_fixtures(step)
    scalar_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    permutation = permutation_fixture(protocol["permutation_seed"])
    permutation_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    dormant_config = protocol["dormant"]
    dormant = [dormant_case(dormant_config["seed_start"] + i, dormant_config, step)
               for i in range(dormant_config["cases"])]
    dormant_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    sequential_config = protocol["sequential"]
    sequential = [sequential_case(sequential_config["seed_start"] + i, sequential_config)
                  for i in range(sequential_config["networks"])]
    sequential_seconds = time.perf_counter() - stage
    summary = aggregate(dormant, sequential, protocol)
    gates = evaluate_gates(scalar, permutation, dormant, sequential, summary, protocol["gates"])
    return {"scalar": scalar, "permutation": permutation, "summary": summary, "gates": gates,
            "exploratory_gate_passed": all(gates.values()), "dormant_cases": dormant, "sequential_cases": sequential,
            "resources": {"wall_seconds": time.perf_counter() - start_wall,
                          "process_cpu_seconds": time.process_time() - start_cpu, "numpy": np.__version__,
                          "stage_wall_seconds": {"scalar_fixtures": scalar_seconds, "permutation": permutation_seconds,
                                                 "dormant": dormant_seconds, "sequential": sequential_seconds}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("synthetic_protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["schema"] != "weightbridge.phase_iiib.synthetic_protocol.v1":
        parser.error("Unsupported synthetic protocol schema")
    result = {"schema": "weightbridge.phase_iiib.synthetic_report.v1",
              "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
              "shared_compiler_source_sha256": hashlib.sha256(Path(bic_math.__file__).read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **run(protocol)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"summary": result["summary"], "gates": result["gates"],
                      "resources": result["resources"]}, indent=2))
    return 0 if result["exploratory_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
