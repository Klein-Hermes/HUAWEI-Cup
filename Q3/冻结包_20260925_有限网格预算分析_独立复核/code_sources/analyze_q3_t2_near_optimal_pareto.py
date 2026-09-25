#!/usr/bin/env python3
"""Exact finite-grid Q3-T2 analysis: near-optimal sets, Pareto front, returns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_REL = Path("Q3/03_代码/analyze_q3_t2_near_optimal_pareto.py")
OUT_REL = Path("Q3/04_结果/Q3_T2_近最优_Pareto_边际收益")
FIG_REL = OUT_REL / "figures"
Q2_GRID_REL = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q2_BOOT_REL = Path("Q2/03_结果/经典ScalingLaw基线/v1/cluster_bootstrap_estimates.csv")
M0_FRONTIER_REL = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
M0_BOOT_REL = Path("Q3/04_结果/M0_ND_cluster_bootstrap_frontier.csv")
M3_REL = Path("Q3/04_结果/m3_q3_support_grid.csv")
M4_REL = Path("Q3/04_结果/m4_budget_feasible_mask.csv")
M1_REL = Path("Q3/04_结果/m1_optimum_boundary_audit.csv")
THRESHOLDS = (Decimal("0.01"), Decimal("0.03"), Decimal("0.05"))
BOOTSTRAP_COUNT = 1000

sys.path.insert(0, str(ROOT / "Q3/03_代码"))
import solve_q3_m0_reference_grid as m0_solver  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def input_paths() -> dict[str, Path]:
    return {
        "b1_predictions": ROOT / Q2_GRID_REL,
        "q2_cluster_bootstrap_estimates": ROOT / Q2_BOOT_REL,
        "m0_reference_frontier": ROOT / M0_FRONTIER_REL,
        "m0_bootstrap_frontier": ROOT / M0_BOOT_REL,
        "m3_support_grid": ROOT / M3_REL,
        "m4_budget_mask": ROOT / M4_REL,
        "m1_selected_points": ROOT / M1_REL,
        "q2_fit_parameters": ROOT / "Q2/03_结果/经典ScalingLaw基线/v1/fit_parameters.json",
    }


def exact_cost(point: dict[str, Any], context: int) -> tuple[Fraction, Fraction, Fraction]:
    n_b = Fraction(Decimal(str(point["N_params_B"])))
    d_b = Fraction(Decimal(str(point["D_tokens_B"])))
    n_abs = n_b * 10**9
    d_abs = d_b * 10**9
    require(n_abs.denominator == 1 and d_abs.denominator == 1, "N/D do not convert to integer absolute units")
    nd = int(n_abs) * int(d_abs)
    training = Fraction(6 * nd)
    attention = Fraction(nd * context, 5000)
    return training, attention, training + attention


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile from an empty list")
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_inputs() -> dict[str, Any]:
    paths = input_paths()
    missing = [rel(path) for path in paths.values() if not path.is_file()]
    require(not missing, f"Missing required Q3-T2 input(s): {missing}")

    b1_rows = read_csv(paths["b1_predictions"])
    m3_rows = read_csv(paths["m3_support_grid"])
    m4_rows = read_csv(paths["m4_budget_mask"])
    base_rows = read_csv(paths["m0_reference_frontier"])
    base_boot_rows = read_csv(paths["m0_bootstrap_frontier"])
    m1_rows = read_csv(paths["m1_selected_points"])
    bootstrap_rows = read_csv(paths["q2_cluster_bootstrap_estimates"])

    verified_points, fit_parameters, fit_metadata = m0_solver.verify_b1_grid()
    require(len(b1_rows) == 1176, f"Expected 1,176 B1 rows, found {len(b1_rows)}")
    require(len(verified_points) == len(b1_rows), "Canonical M0 verification count differs from B1 table")
    require(len(m3_rows) == 1176, f"Expected 1,176 M3 support rows, found {len(m3_rows)}")
    require(len(m4_rows) == 17640, f"Expected 17,640 M4 mask rows, found {len(m4_rows)}")
    require(len(base_rows) == 15 and len(base_boot_rows) == 15 and len(m1_rows) == 15,
            "Expected 15 reference, bootstrap, and M1 scenarios")
    require(len(bootstrap_rows) == BOOTSTRAP_COUNT,
            f"Expected {BOOTSTRAP_COUNT} existing Q2 bootstrap fits, found {len(bootstrap_rows)}")

    verified_by_id = {int(row["run_id"]): row for row in verified_points}
    raw_by_id = {int(row["run_id"]): row for row in b1_rows}
    require(len(raw_by_id) == len(b1_rows), "Duplicate run_id in B1 predictions")
    require(set(raw_by_id) == set(verified_by_id), "B1 and canonical M0 run_id sets differ")

    points: list[dict[str, Any]] = []
    for run_id, raw in raw_by_id.items():
        verified = verified_by_id[run_id]
        require(math.isclose(float(raw["predicted_val_loss"]), float(verified["predicted_val_loss"]),
                             rel_tol=0, abs_tol=1e-8),
                f"Stored M0 prediction does not reproduce at run_id={run_id}")
        point = {
            "grid_id": run_id,
            "run_id": run_id,
            "N_params_B": Decimal(raw["N_params_B"]),
            "D_tokens_B": Decimal(raw["D_tokens_B"]),
            "predicted_loss": Decimal(raw["predicted_val_loss"]),
            "observed_loss": Decimal(raw["observed_val_loss"]),
            "raw": raw,
        }
        points.append(point)

    m3_by_id = {int(row["grid_id"]): row for row in m3_rows}
    require(len(m3_by_id) == len(m3_rows), "Duplicate grid_id in M3 support grid")
    require(set(m3_by_id) == set(raw_by_id), "M3 support IDs differ from B1 prediction IDs")
    for point in points:
        support = m3_by_id[point["grid_id"]]
        require(Decimal(support["N_params_B"]) == point["N_params_B"]
                and Decimal(support["D_tokens_B"]) == point["D_tokens_B"],
                f"N/D mismatch between M3 and B1 at grid_id={point['grid_id']}")
        point["m3_allowed"] = parse_bool(support["m3_search_allowed"])

    budgets = sorted({int(row["budget_flops"]) for row in base_rows})
    contexts = sorted({int(row["context_length_tokens"]) for row in base_rows})
    require(tuple(budgets) == tuple(m0_solver.BUDGETS_FLOPS), "Budget tiers differ from frozen Q3 contract")
    require(len(contexts) == 5, f"Expected five C7 context scenarios, found {len(contexts)}")
    q_versions = {row["q1_q_version"] for row in base_rows}
    q0_values = {Decimal(row["q0_reference_a1_macro"]) for row in base_rows}
    require(q_versions == {"Q1-q_huber-v1"}, "M0 frontier does not use the frozen q_huber interface")
    require(len(q0_values) == 1 and math.isclose(float(next(iter(q0_values))), 0.49847810484764643,
                                                 rel_tol=0, abs_tol=1e-12),
            "M0 frontier Q0 differs from the frozen q_huber baseline")
    base_by_key = {
        (int(row["budget_flops"]), int(row["context_length_tokens"])): row for row in base_rows
    }
    m1_by_key = {
        (int(row["budget_flops"]), int(row["context_length_tokens"])): row for row in m1_rows
    }
    boot_frontier_by_key = {
        (int(row["budget_flops"]), int(row["context_length_tokens"])): row for row in base_boot_rows
    }
    require(len(base_by_key) == len(base_rows) and len(m1_by_key) == len(m1_rows)
            and len(boot_frontier_by_key) == len(base_boot_rows), "Duplicate budget/context scenario")

    m4_by_key: dict[tuple[int, int, int], dict[str, str]] = {}
    for row in m4_rows:
        key = (int(row["budget_flops"]), int(row["context_length_tokens"]), int(row["grid_id"]))
        require(key not in m4_by_key, f"Duplicate M4 scenario/grid key: {key}")
        m4_by_key[key] = row
    require(len(m4_by_key) == len(budgets) * len(contexts) * len(points),
            "M4 budget mask does not cover all 15 × 1,176 keys")

    cost_by_context_grid: dict[tuple[int, int], Fraction] = {}
    m4_cost_checks = 0
    m4_mask_checks = 0
    for context in contexts:
        for point in points:
            training, attention, total = exact_cost(point, context)
            cost_by_context_grid[(context, point["grid_id"])] = total
            for budget in budgets:
                row = m4_by_key[(budget, context, point["grid_id"])]
                m4_training = float(row["C_train_flops"])
                m4_quality = float(row["C_Q_flops"])
                m4_attention = float(row["C_attn_flops"])
                m4_total = float(row["C_total_flops"])
                require(math.isclose(m4_training, float(training), rel_tol=1e-12, abs_tol=1.0),
                        f"M4 training-cost mismatch at C={budget}, L={context}, grid_id={point['grid_id']}")
                require(math.isclose(m4_quality, 0.0, rel_tol=0, abs_tol=1e-12),
                        f"M4 quality uplift cost is nonzero at fixed Q0: {point['grid_id']}")
                require(math.isclose(m4_attention, float(attention), rel_tol=1e-12, abs_tol=1.0),
                        f"M4 attention-cost mismatch at C={budget}, L={context}, grid_id={point['grid_id']}")
                require(math.isclose(m4_total, float(total), rel_tol=1e-12, abs_tol=1.0),
                        f"M4 total cost mismatch at C={budget}, L={context}, grid_id={point['grid_id']}")
                expected_mask = total <= budget
                require(parse_bool(row["budget_feasible"]) == expected_mask,
                        f"M4 budget mask mismatch at C={budget}, L={context}, grid_id={point['grid_id']}")
                m4_cost_checks += 1
                m4_mask_checks += 1
    require(m4_cost_checks == 17640 and m4_mask_checks == 17640, "Incomplete M4 verification")

    scenario_candidates: dict[tuple[int, int], list[dict[str, Any]]] = {}
    scenario_best: dict[tuple[int, int], dict[str, Any]] = {}
    scenario_summaries: dict[tuple[int, int], dict[str, Any]] = {}
    near_rows: list[dict[str, Any]] = []
    for context in contexts:
        for budget in budgets:
            key = (budget, context)
            eligible = [
                point for point in points
                if point["m3_allowed"]
                and parse_bool(m4_by_key[(budget, context, point["grid_id"])]["budget_feasible"])
            ]
            require(eligible, f"No M3/M4 feasible candidate at budget={budget}, context={context}")
            selected = min(
                eligible,
                key=lambda point: (
                    point["predicted_loss"],
                    cost_by_context_grid[(context, point["grid_id"])],
                    point["N_params_B"],
                    point["D_tokens_B"],
                    point["run_id"],
                ),
            )
            scenario_candidates[key] = eligible
            scenario_best[key] = selected
            base = base_by_key[key]
            m1 = m1_by_key[key]
            require(int(base["selected_run_id"]) == selected["run_id"],
                    f"Recomputed optimum differs from existing M0 frontier at {key}")
            require(int(m1["m1_selected_grid_id"]) == selected["grid_id"],
                    f"Recomputed optimum differs from existing M1 at {key}")
            require(int(m1["m1_selected_run_id"]) == selected["run_id"],
                    f"M1 selected run_id differs from B1/M0 at {key}")
            require(Decimal(m1["selected_N_B"]) == selected["N_params_B"]
                    and Decimal(m1["selected_D_B"]) == selected["D_tokens_B"],
                    f"M1 selected N/D differs from B1/M0 at {key}")
            require(math.isclose(float(m1["m0_predicted_val_loss"]), float(selected["predicted_loss"]),
                                 rel_tol=0, abs_tol=1e-8),
                    f"M1 selected Loss differs from B1/M0 at {key}")
            require(parse_bool(m1["final_feasible"]) and parse_bool(m1["m3_search_allowed"])
                    and parse_bool(m1["budget_feasible"]),
                    f"Existing M1 selected point is not marked fully feasible at {key}")
            best_loss = selected["predicted_loss"]
            require(best_loss > 0, f"Expected positive Loss for relative gaps at {key}")
            scenario_summaries[key] = {
                "budget_flops": budget,
                "context_length_tokens": context,
                "scenario_id": m4_by_key[(budget, context, points[0]["grid_id"])]["scenario_id"],
                "best": selected,
                "best_loss": best_loss,
                "feasible_candidate_count": len(eligible),
                "actual_cost": cost_by_context_grid[(context, selected["grid_id"])],
                "bootstrap_expected": boot_frontier_by_key[key],
            }
            for point in points:
                mask_row = m4_by_key[(budget, context, point["grid_id"])]
                feasible = point["m3_allowed"] and parse_bool(mask_row["budget_feasible"])
                gap_abs = point["predicted_loss"] - best_loss if feasible else None
                gap_rel = gap_abs / abs(best_loss) if feasible else None
                near_rows.append({
                    "scenario_id": mask_row["scenario_id"],
                    "budget_flops": budget,
                    "context_length_tokens": context,
                    "grid_id": point["grid_id"],
                    "run_id": point["run_id"],
                    "N_params_B": str(point["N_params_B"]),
                    "D_tokens_B": str(point["D_tokens_B"]),
                    "m0_predicted_val_loss": str(point["predicted_loss"]),
                    "b1_observed_val_loss": str(point["observed_loss"]),
                    "m3_search_allowed": point["m3_allowed"],
                    "budget_feasible": parse_bool(mask_row["budget_feasible"]),
                    "final_feasible": feasible,
                    "C_total_flops": float(cost_by_context_grid[(context, point["grid_id"])]),
                    "scenario_best_loss": str(best_loss),
                    "loss_gap_absolute": "" if gap_abs is None else str(gap_abs),
                    "loss_gap_relative": "" if gap_rel is None else str(gap_rel),
                    "near_optimal_1pct": bool(feasible and gap_rel <= THRESHOLDS[0]),
                    "near_optimal_3pct": bool(feasible and gap_rel <= THRESHOLDS[1]),
                    "near_optimal_5pct": bool(feasible and gap_rel <= THRESHOLDS[2]),
                    "bootstrap_optimal_selection_frequency": 0.0,
                    "bootstrap_near_optimal_1pct_frequency": 0.0,
                    "bootstrap_near_optimal_3pct_frequency": 0.0,
                    "bootstrap_near_optimal_5pct_frequency": 0.0,
                })

    for context in contexts:
        eligible_sets = [
            {point["grid_id"] for point in scenario_candidates[(budget, context)]}
            for budget in budgets
        ]
        require(eligible_sets[0] <= eligible_sets[1] <= eligible_sets[2],
                f"Feasible sets are not nested across budget tiers at context={context}")

    pareto_rows: list[dict[str, Any]] = []
    pareto_counts: dict[int, tuple[int, int]] = {}
    supported = [point for point in points if point["m3_allowed"]]
    for context in contexts:
        candidates = sorted(
            supported,
            key=lambda point: (
                cost_by_context_grid[(context, point["grid_id"])],
                point["predicted_loss"],
                point["run_id"],
            ),
        )
        front: list[dict[str, Any]] = []
        best_loss_at_lower_cost: Decimal | None = None
        index = 0
        while index < len(candidates):
            end = index + 1
            cost = cost_by_context_grid[(context, candidates[index]["grid_id"])]
            while end < len(candidates) and cost_by_context_grid[(context, candidates[end]["grid_id"])] == cost:
                end += 1
            cost_group = candidates[index:end]
            group_best = min(point["predicted_loss"] for point in cost_group)
            if best_loss_at_lower_cost is None or group_best < best_loss_at_lower_cost:
                front.extend(point for point in cost_group if point["predicted_loss"] == group_best)
            best_loss_at_lower_cost = (
                group_best if best_loss_at_lower_cost is None
                else min(best_loss_at_lower_cost, group_best)
            )
            index = end

        front_ids = {point["grid_id"] for point in front}
        for point in candidates:
            dominated = any(
                other["grid_id"] != point["grid_id"]
                and cost_by_context_grid[(context, other["grid_id"])]
                <= cost_by_context_grid[(context, point["grid_id"])]
                and other["predicted_loss"] <= point["predicted_loss"]
                and (
                    cost_by_context_grid[(context, other["grid_id"])]
                    < cost_by_context_grid[(context, point["grid_id"])]
                    or other["predicted_loss"] < point["predicted_loss"]
                )
                for other in candidates
            )
            require((point["grid_id"] in front_ids) != dominated,
                    f"Pareto classification failed at context={context}, grid_id={point['grid_id']}")

        front.sort(key=lambda point: (
            cost_by_context_grid[(context, point["grid_id"])],
            point["predicted_loss"],
            point["run_id"],
        ))
        pareto_counts[context] = (len(front), len(candidates))
        for rank, point in enumerate(front, 1):
            total = cost_by_context_grid[(context, point["grid_id"])]
            row: dict[str, Any] = {
                "context_length_tokens": context,
                "pareto_rank_by_cost": rank,
                "grid_id": point["grid_id"],
                "run_id": point["run_id"],
                "N_params_B": str(point["N_params_B"]),
                "D_tokens_B": str(point["D_tokens_B"]),
                "m0_predicted_val_loss": str(point["predicted_loss"]),
                "b1_observed_val_loss": str(point["observed_loss"]),
                "C_total_flops": float(total),
                "pareto_scope": "all M3-allowed observed B1 points at fixed context; fixed Q0/M0",
            }
            for budget in budgets:
                row[f"budget_feasible_{budget}"] = total <= budget
            pareto_rows.append(row)

    bootstrap_rows = sorted(bootstrap_rows, key=lambda row: int(row["replicate"]))
    require([int(row["replicate"]) for row in bootstrap_rows] == list(range(1, BOOTSTRAP_COUNT + 1)),
            "Q2 bootstrap replicate IDs must be exactly 1..1000")
    min_observed = min(float(point["observed_loss"]) for point in points)
    for fit in bootstrap_rows:
        sample = fit["sampled_cluster_sequence"].split(";")
        require(len(sample) == 8, f"Bootstrap replicate {fit['replicate']} does not sample eight trajectories")
        values = {name: float(fit[name]) for name in ("E", "A", "alpha", "B", "beta")}
        require(all(math.isfinite(value) for value in values.values()), "Nonfinite bootstrap parameter")
        require(0 <= values["E"] < min_observed and all(values[name] > 0 for name in ("A", "alpha", "B", "beta")),
                f"Bootstrap parameter bounds fail at replicate {fit['replicate']}")

    return {
        "paths": paths,
        "points": points,
        "point_by_id": {point["grid_id"]: point for point in points},
        "fit_parameters": fit_parameters,
        "fit_metadata": fit_metadata,
        "m3_by_id": m3_by_id,
        "m4_by_key": m4_by_key,
        "m0_by_key": base_by_key,
        "m1_by_key": m1_by_key,
        "bootstrap_frontier_by_key": boot_frontier_by_key,
        "bootstrap_rows": bootstrap_rows,
        "budgets": budgets,
        "contexts": contexts,
        "costs": cost_by_context_grid,
        "scenario_candidates": scenario_candidates,
        "scenario_best": scenario_best,
        "scenario_summaries": scenario_summaries,
        "near_rows": near_rows,
        "pareto_rows": pareto_rows,
        "pareto_counts": pareto_counts,
        "input_row_counts": {
            "b1_predictions": len(b1_rows),
            "m3_support_grid": len(m3_rows),
            "m4_budget_mask": len(m4_rows),
            "m0_reference_frontier": len(base_rows),
            "m0_bootstrap_frontier": len(base_boot_rows),
            "m1_selected_points": len(m1_rows),
            "q2_cluster_bootstrap_estimates": len(bootstrap_rows),
        },
        "verification_counts": {
            "m0_prediction_rows_reproduced": len(points),
            "m3_nd_alignment_rows": len(points),
            "m4_cost_rows": m4_cost_checks,
            "m4_budget_flag_rows": m4_mask_checks,
            "m0_optimum_reconciliations": len(base_rows),
            "m1_optimum_reconciliations": len(m1_rows),
        },
    }


def compute_bootstrap(inputs: dict[str, Any], contexts: list[int]) -> dict[str, Any]:
    budgets: list[int] = inputs["budgets"]
    points: list[dict[str, Any]] = inputs["points"]
    selected_draws: dict[tuple[int, int], list[dict[str, Any]]] = {
        (budget, context): [] for context in contexts for budget in budgets
    }
    near_counts: dict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    selection_counts: dict[tuple[int, int], dict[int, int]] = {
        key: defaultdict(int) for key in selected_draws
    }

    for fit in inputs["bootstrap_rows"]:
        parameters = {name: float(fit[name]) for name in ("E", "A", "alpha", "B", "beta")}
        predicted = {
            point["grid_id"]: (
                parameters["E"]
                + parameters["A"] * float(point["N_params_B"]) ** (-parameters["alpha"])
                + parameters["B"] * float(point["D_tokens_B"]) ** (-parameters["beta"])
            )
            for point in points
        }
        require(all(math.isfinite(value) for value in predicted.values()), "Nonfinite bootstrap M0 prediction")
        for context in contexts:
            for budget in budgets:
                key = (budget, context)
                feasible = inputs["scenario_candidates"][key]
                selected = min(
                    feasible,
                    key=lambda point: (
                        predicted[point["grid_id"]],
                        inputs["costs"][(context, point["grid_id"])],
                        point["N_params_B"],
                        point["D_tokens_B"],
                        point["run_id"],
                    ),
                )
                best_loss = predicted[selected["grid_id"]]
                selection_counts[key][selected["grid_id"]] += 1
                for point in feasible:
                    gap = (predicted[point["grid_id"]] - best_loss) / abs(best_loss)
                    for slot, threshold in enumerate((0.01, 0.03, 0.05)):
                        if gap <= threshold + 1e-12:
                            near_counts[(budget, context, point["grid_id"])][slot] += 1
                selected_draws[key].append({
                    "replicate": int(fit["replicate"]),
                    "grid_id": selected["grid_id"],
                    "loss": best_loss,
                    "cost": float(inputs["costs"][(context, selected["grid_id"])]),
                })

    frontier_checks = 0
    for (budget, context), draws in selected_draws.items():
        expected = inputs["bootstrap_frontier_by_key"][(budget, context)]
        baseline_id = int(inputs["m0_by_key"][(budget, context)]["selected_run_id"])
        baseline_frequency = selection_counts[(budget, context)].get(baseline_id, 0) / BOOTSTRAP_COUNT
        require(math.isclose(
            baseline_frequency, float(expected["baseline_selection_frequency"]), rel_tol=0, abs_tol=1e-12
        ), f"Bootstrap baseline selection frequency mismatch at {(budget, context)}")
        losses = [draw["loss"] for draw in draws]
        for key_name, quantile in (
            ("predicted_loss_p2_5", 0.025),
            ("predicted_loss_median", 0.5),
            ("predicted_loss_p97_5", 0.975),
        ):
            require(math.isclose(
                percentile(losses, quantile), float(expected[key_name]), rel_tol=0, abs_tol=1e-9
            ), f"Bootstrap Loss quantile mismatch at {(budget, context)}: {key_name}")
        frontier_checks += 1

    return {
        "selected_draws": selected_draws,
        "near_counts": near_counts,
        "selection_counts": selection_counts,
        "frontier_reconciliations": frontier_checks,
    }


def attach_bootstrap_frequencies(
    near_rows: list[dict[str, Any]], bootstrap: dict[str, Any], contexts: list[int]
) -> None:
    total = BOOTSTRAP_COUNT
    for row in near_rows:
        context = int(row["context_length_tokens"])
        if context not in contexts:
            continue
        budget = int(row["budget_flops"])
        grid_id = int(row["grid_id"])
        key = (budget, context, grid_id)
        row["bootstrap_optimal_selection_frequency"] = (
            bootstrap["selection_counts"][(budget, context)].get(grid_id, 0) / total
        )
        counts = bootstrap["near_counts"].get(key, [0, 0, 0])
        row["bootstrap_near_optimal_1pct_frequency"] = counts[0] / total
        row["bootstrap_near_optimal_3pct_frequency"] = counts[1] / total
        row["bootstrap_near_optimal_5pct_frequency"] = counts[2] / total


def build_marginal_rows(inputs: dict[str, Any], bootstrap: dict[str, Any], contexts: list[int]) -> list[dict[str, Any]]:
    budgets: list[int] = inputs["budgets"]
    outputs: list[dict[str, Any]] = []
    for context in contexts:
        for index, (from_budget, to_budget) in enumerate(zip(budgets, budgets[1:])):
            before = inputs["scenario_summaries"][(from_budget, context)]
            after = inputs["scenario_summaries"][(to_budget, context)]
            before_loss = float(before["best_loss"])
            after_loss = float(after["best_loss"])
            improvement = before_loss - after_loss
            require(improvement >= -1e-12, f"Optimal Loss worsens as budget grows at context={context}")
            improvement = max(0.0, improvement)
            budget_decades = math.log10(to_budget / from_budget)
            actual_increment = float(after["actual_cost"] - before["actual_cost"])
            require(actual_increment >= -1.0, f"Selected actual cost falls across nested budgets at context={context}")
            actual_increment = max(0.0, actual_increment)
            paired_draws = [
                left["loss"] - right["loss"]
                for left, right in zip(
                    bootstrap["selected_draws"][(from_budget, context)],
                    bootstrap["selected_draws"][(to_budget, context)],
                )
            ]
            before_selected_id = inputs["scenario_best"][(from_budget, context)]["grid_id"]
            after_selected_id = inputs["scenario_best"][(to_budget, context)]["grid_id"]
            max_n = max(point["N_params_B"] for point in inputs["points"])
            max_d = max(point["D_tokens_B"] for point in inputs["points"])
            row = {
                "context_length_tokens": context,
                "transition": "low_to_mid" if index == 0 else "mid_to_high",
                "from_budget_flops": from_budget,
                "to_budget_flops": to_budget,
                "budget_increment_flops": to_budget - from_budget,
                "budget_multiple": to_budget / from_budget,
                "budget_increase_decades": budget_decades,
                "from_grid_id": before_selected_id,
                "to_grid_id": after_selected_id,
                "from_N_params_B": str(before["best"]["N_params_B"]),
                "from_D_tokens_B": str(before["best"]["D_tokens_B"]),
                "to_N_params_B": str(after["best"]["N_params_B"]),
                "to_D_tokens_B": str(after["best"]["D_tokens_B"]),
                "from_m0_predicted_val_loss": before_loss,
                "to_m0_predicted_val_loss": after_loss,
                "loss_reduction": improvement,
                "relative_loss_reduction": improvement / abs(before_loss),
                "loss_reduction_per_budget_decade": improvement / budget_decades,
                "from_actual_cost_flops": float(before["actual_cost"]),
                "to_actual_cost_flops": float(after["actual_cost"]),
                "actual_cost_increment_flops": actual_increment,
                "loss_reduction_per_1e21_added_actual_flops": (
                    improvement / (actual_increment / 1e21) if actual_increment > 0 else ""
                ),
                "from_optimum_bootstrap_selection_frequency": (
                    bootstrap["selection_counts"][(from_budget, context)].get(before_selected_id, 0)
                    / BOOTSTRAP_COUNT
                ),
                "to_optimum_bootstrap_selection_frequency": (
                    bootstrap["selection_counts"][(to_budget, context)].get(after_selected_id, 0)
                    / BOOTSTRAP_COUNT
                ),
                "bootstrap_loss_reduction_p2_5": percentile(paired_draws, 0.025),
                "bootstrap_loss_reduction_median": percentile(paired_draws, 0.5),
                "bootstrap_loss_reduction_p97_5": percentile(paired_draws, 0.975),
                "bootstrap_positive_gain_fraction": sum(value > 0 for value in paired_draws) / len(paired_draws),
                "bootstrap_replicates": len(paired_draws),
                "from_at_b1_N_upper": before["best"]["N_params_B"] == max_n,
                "from_at_b1_D_upper": before["best"]["D_tokens_B"] == max_d,
                "to_at_b1_N_upper": after["best"]["N_params_B"] == max_n,
                "to_at_b1_D_upper": after["best"]["D_tokens_B"] == max_d,
                "interpretation": "fixed-Q0 B1/M0 in-sample support-grid sensitivity; not causal or full Q3 Q/p optimum",
            }
            outputs.append(row)
    require(len(outputs) == len(contexts) * 2, "Expected two adjacent-budget comparisons per context")
    return outputs


def summarize_near_counts(near_rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[int]]:
    summary: dict[tuple[int, int], list[int]] = {}
    for row in near_rows:
        key = (int(row["budget_flops"]), int(row["context_length_tokens"]))
        counts = summary.setdefault(key, [0, 0, 0])
        for index, column in enumerate(("near_optimal_1pct", "near_optimal_3pct", "near_optimal_5pct")):
            counts[index] += int(bool(row[column]))
    return summary


def write_figure_contract(path: Path) -> None:
    text = """# Q3-T2 结果图图表契约

- 论点：在固定 Q0 的 B1/M0 有限支持域内，展示成本—Loss 非支配方案、预算情景近优规模，以及预算升级对应的 Loss 边际下降及其轨迹 Bootstrap 区间。
- 面板 A：Cost–Loss Pareto 散点；点是离散的 B1 实测配置，不以折线暗示连续解。灰点为全部支持点，彩色点为各上下文的精确非支配解。
- 面板 B：15 个预算×上下文情景的 1%、3%、5% 近优方案数量热图；数字为可行候选点计数，不是概率。
- 面板 C：5 个上下文下低→中和中→高预算的每预算数量级 Loss 下降；误差线为使用相同 replicate 配对的 1,000 次 B1 轨迹 Bootstrap 2.5%–97.5% 区间。
- 数据源：B1/M0 预测（1,176 行）、M3 支持（1,176 行）、M4 掩码（17,640 行）、已有 Q2 cluster-bootstrap 参数（1,000 组）。
- 口径：Q=Q0=0.4984781048，p 不入模；总成本用绝对参数与 Token 计数；M0 Loss 使用 Q2 既有拟合预测。
- 解释限制：只覆盖 8 条 B1 轨迹和已观测支持域；Bootstrap 区间是既定 M0 参数敏感性，不代表模型形式误差或域外迁移。高预算结果可能触及 B1 网格上界。
- 导出：SVG 可编辑文本；彩色 PNG 300 DPI；灰度图由同尺寸 PNG 转换，保留相同像素尺寸。
"""
    path.write_text(text, encoding="utf-8")


def make_figure(
    inputs: dict[str, Any],
    near_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    marginal_rows: list[dict[str, Any]],
    output_base: Path,
) -> dict[str, Any]:
    skill_root = Path("C:/Users/86147/.codex/skills/math-modeling")
    sys.path.insert(0, str(ROOT / "utils"))
    sys.path.insert(0, str(skill_root / "tools/figure/scripts"))
    import matplotlib

    matplotlib.use("Agg")
    from plot_style import apply_publication_style, audit_design, audit_layout
    from export_figure import export_figure
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    style_report = apply_publication_style(language="en", width="report")
    contexts = inputs["contexts"]
    budgets = inputs["budgets"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    marker_cycle = ["o", "s", "^", "D", "P"]

    fig, axes = plt.subplot_mosaic(
        [["pareto", "near"], ["returns", "returns"]],
        figsize=(11.0, 7.1),
        gridspec_kw={"width_ratios": [1.3, 1.0], "height_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )
    ax = axes["pareto"]
    all_points = inputs["points"]
    for context in contexts:
        x_all = [
            float(inputs["costs"][(context, point["grid_id"])])
            for point in all_points if point["m3_allowed"]
        ]
        y_all = [
            float(point["predicted_loss"])
            for point in all_points if point["m3_allowed"]
        ]
        ax.scatter(x_all, y_all, s=8, color="#A8AFB7", alpha=0.12, linewidths=0, rasterized=True)
    for index, context in enumerate(contexts):
        subset = [row for row in pareto_rows if int(row["context_length_tokens"]) == context]
        ax.scatter(
            [float(row["C_total_flops"]) for row in subset],
            [float(row["m0_predicted_val_loss"]) for row in subset],
            s=24,
            color=colors[index],
            marker=marker_cycle[index],
            linewidths=0.35,
            edgecolors="white",
            label=f"{context:,}",
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total compute cost (FLOPs)")
    ax.set_ylabel("M0 predicted validation Loss")
    ax.set_title("A  Cost–Loss Pareto sets")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Context", ncol=2, frameon=False, fontsize=7)

    ax = axes["near"]
    near_summary = summarize_near_counts(near_rows)
    matrix: list[list[int]] = []
    labels: list[str] = []
    for budget in budgets:
        for context in contexts:
            matrix.append(near_summary[(budget, context)])
            labels.append(f"{budget:.0e} / {context:,}")
    image = ax.imshow(np.asarray(matrix), aspect="auto", cmap="Blues")
    ax.set_xticks(range(3), ["1%", "3%", "5%"])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Relative Loss tolerance")
    ax.set_title("B  Near-optimal counts")
    ax.tick_params(axis="y", labelsize=6.5)
    peak = max(max(row) for row in matrix)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center",
                    fontsize=6.5, color="#111827" if value < peak * 0.75 else "white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Candidate count")

    ax = axes["returns"]
    contexts_pos = list(range(len(contexts)))
    transition_style = {
        "low_to_mid": {"label": "Low → mid", "color": "#0072B2", "marker": "o", "offset": -0.10},
        "mid_to_high": {"label": "Mid → high", "color": "#D55E00", "marker": "s", "offset": 0.10},
    }
    context_pos = {context: i for i, context in enumerate(contexts)}
    for transition, style in transition_style.items():
        subset = [row for row in marginal_rows if row["transition"] == transition]
        xs, ys, lower, upper = [], [], [], []
        for row in subset:
            x = context_pos[int(row["context_length_tokens"])] + style["offset"]
            decades = float(row["budget_increase_decades"])
            median = float(row["bootstrap_loss_reduction_median"]) / decades
            lo = float(row["bootstrap_loss_reduction_p2_5"]) / decades
            hi = float(row["bootstrap_loss_reduction_p97_5"]) / decades
            xs.append(x)
            ys.append(median)
            lower.append(max(0.0, median - lo))
            upper.append(max(0.0, hi - median))
        ax.errorbar(
            xs, ys, yerr=[lower, upper], linestyle="none", marker=style["marker"],
            markersize=4.8, capsize=2.2, color=style["color"], label=style["label"],
        )
    ax.set_xticks(contexts_pos, [f"{context:,}" for context in contexts])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Loss reduction per budget decade")
    ax.set_title("C  Return per budget decade")
    ax.axhline(0, color="#6B7280", linewidth=0.7)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.canvas.draw()
    layout_warnings = audit_layout(fig)
    design_warnings = audit_design(fig)
    base = output_base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    exported = export_figure(
        fig,
        basename=str(base),
        formats=["svg", "png"],
        dpi=300,
        size_inches=(11.0, 7.1),
        grayscale_preview=False,
        tight=False,
    )
    png_path = base.with_suffix(".png")
    gray_path = base.with_name(base.name + "_grayscale.png")
    with Image.open(png_path) as source:
        gray = source.convert("L")
        gray.save(gray_path, dpi=(300, 300))
        gray.close()
        pixel_size = source.size
        dpi_metadata = source.info.get("dpi")
    plt.close(fig)
    return {
        "exported": [str(Path(path).resolve().relative_to(ROOT)) for path in exported],
        "grayscale_preview": gray_path.resolve().relative_to(ROOT).as_posix(),
        "pixel_size": list(pixel_size),
        "png_dpi_metadata": list(dpi_metadata) if dpi_metadata else None,
        "svg_editable_text_count": Path(base.with_suffix(".svg")).read_text(encoding="utf-8").count("<text"),
        "layout_warnings": layout_warnings,
        "design_warnings": design_warnings,
        "style_report": style_report if isinstance(style_report, (str, int, float, bool, list, dict)) else str(style_report),
    }


def input_profile(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "path": rel(path),
            "format": "json",
            "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    rows = read_csv(path)
    fieldnames = list(rows[0]) if rows else []
    key_candidates = ("grid_id", "run_id", "replicate", "scenario_id")
    unique_counts = {}
    for key in key_candidates:
        if key in fieldnames:
            unique_counts[key] = len({row[key] for row in rows})
    missing = {field: sum(not row.get(field, "").strip() for row in rows) for field in fieldnames}
    numeric_summary: dict[str, dict[str, Any]] = {}
    for field in fieldnames:
        values: list[float] = []
        numeric = True
        for row in rows:
            value = row.get(field, "").strip()
            if not value:
                continue
            try:
                parsed = float(value)
            except ValueError:
                numeric = False
                break
            if not math.isfinite(parsed):
                numeric = False
                break
            values.append(parsed)
        if not numeric or not values:
            continue
        ordered = sorted(values)
        def qtile(q: float) -> float:
            position = (len(ordered) - 1) * q
            lower, upper = math.floor(position), math.ceil(position)
            if lower == upper:
                return ordered[lower]
            weight = position - lower
            return ordered[lower] * (1 - weight) + ordered[upper] * weight
        q1, median, q3 = qtile(0.25), qtile(0.5), qtile(0.75)
        iqr = q3 - q1
        outliers = sum(value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr for value in values)
        numeric_summary[field] = {
            "count": len(values),
            "min": min(values),
            "q1": q1,
            "median": median,
            "mean": statistics.fmean(values),
            "q3": q3,
            "max": max(values),
            "iqr_outlier_count": outliers,
        }
    return {
        "path": rel(path),
        "format": "csv",
        "rows": len(rows),
        "columns": fieldnames,
        "unique_key_counts": unique_counts,
        "missing_by_column": missing,
        "numeric_summary": numeric_summary,
        "sha256": sha256_file(path),
    }


def write_report(
    path: Path,
    inputs: dict[str, Any],
    near_rows: list[dict[str, Any]],
    marginal_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    figure_info: dict[str, Any],
) -> None:
    near_summary = summarize_near_counts(near_rows)
    lines = [
        "# Q3-T2 近最优集、Cost–Loss Pareto 与边际收益",
        "",
        "## 方法与范围",
        "",
        "本分析固定 Q=Q0=0.4984781048，不优化 p；只在 M3 允许的 B1 1,176 个实测 N/D 组合内枚举。候选可行性使用现有 M4 掩码，并用绝对 N、D 与成本公式逐行复核。M0 Loss 来自 Q2 B1 同域拟合预测，因此是样本内参考，不是独立验证。",
        "",
        "近最优阈值按每个预算×上下文情景各自的最小 M0 Loss 计算，满足 (Loss−Loss*)/|Loss*| ≤ 1%、3% 或 5%。Cost–Loss Pareto 按上下文对全部 M3 允许支持点最小化总成本与预测 Loss，再列明每档预算是否可行。边际收益比较同一上下文中的低→中与中→高预算，以每增加一个预算数量级的 Loss 下降为主指标。",
        "",
        f"参数敏感性复用既有 {BOOTSTRAP_COUNT} 次、8 条 B1 轨迹的 cluster-bootstrap 拟合；配对区间不重新抽样或重拟合。",
        "",
        "## 近最优集合规模",
        "",
        "| 预算 FLOPs | 上下文 | 可行候选 | 1% 近优 | 3% 近优 | 5% 近优 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in inputs["budgets"]:
        for context in inputs["contexts"]:
            summary = inputs["scenario_summaries"][(budget, context)]
            counts = near_summary[(budget, context)]
            lines.append(
                f"| {budget:.0e} | {context:,} | {summary['feasible_candidate_count']:,} "
                f"| {counts[0]:,} | {counts[1]:,} | {counts[2]:,} |"
            )

    lines.extend([
        "",
        "候选级 Bootstrap 频率见 q3_near_optimal.csv；频率表示在既有轨迹参数重抽样下仍落入对应阈值的比例，不是候选成为真实最优的概率。",
        "",
        "## Pareto 前沿",
        "",
        "| 上下文 | 前沿点数 | M3 支持点数 |",
        "|---:|---:|---:|",
    ])
    for context in inputs["contexts"]:
        front, total = inputs["pareto_counts"][context]
        lines.append(f"| {context:,} | {front:,} | {total:,} |")

    lines.extend([
        "",
        "非支配判定使用精确有限候选域；所有排除点至少被一个成本不高且 Loss 不高、并至少一项严格更优的点支配。",
        "",
        "## 预算边际收益",
        "",
        "| 上下文 | 区间 | Loss 下降 | 相对下降 | 每预算数量级下降 | Bootstrap 95% 区间 |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in marginal_rows:
        lines.append(
            f"| {int(row['context_length_tokens']):,} | {row['transition']} "
            f"| {row['loss_reduction']:.6g} | {row['relative_loss_reduction']:.3%} "
            f"| {row['loss_reduction_per_budget_decade']:.6g} "
            f"| [{row['bootstrap_loss_reduction_p2_5']:.6g}, {row['bootstrap_loss_reduction_p97_5']:.6g}] |"
        )

    lines.extend([
        "",
        "## 核验",
        "",
        f"- 复现 M0 原最优点：{inputs['verification_counts']['m0_optimum_reconciliations']}/15。",
        f"- 对齐 M1 选点与三类可行标记：{inputs['verification_counts']['m1_optimum_reconciliations']}/15。",
        f"- 独立复算并核对 M4 成本与预算标记：{inputs['verification_counts']['m4_cost_rows']}/17,640。",
        f"- 逐情景复现已有 Bootstrap Loss 分位数与原最优点频率：{bootstrap['frontier_reconciliations']}/15。",
        f"- 图布局警告：{len(figure_info['layout_warnings'])}；图设计警告：{len(figure_info['design_warnings'])}。",
        "",
        "## 解释边界",
        "",
        "- 结果是固定 Q0、M0、M3/M4 与 B1 实测 N/D 支持域内的离散分析，不是完整 Q3 联合最优。",
        "- Q2 的 p Gate 仍未通过；Q→Loss 响应也未识别，所以 p 与质量增益不进入这些目标。",
        "- Bootstrap 仅传播 8 条 B1 轨迹下的 M0 参数变化，不覆盖模型形式误差、域外迁移或 B1 支持网格之外的配置。",
        "- 若高预算下所有候选均可行或选点触及 N/D 支持上界，边际收益受当前 B1 网格边界约束，不应外推到更大模型或数据量。",
        "",
        "复现命令：python Q3/03_代码/analyze_q3_t2_near_optimal_pareto.py",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    inputs: dict[str, Any],
    output_paths: list[Path],
    figure_info: dict[str, Any],
    bootstrap: dict[str, Any],
    run_mode: str,
) -> dict[str, Any]:
    package_versions = {}
    for package in ("numpy", "pandas", "matplotlib", "Pillow"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    input_records = {
        name: input_profile(path) for name, path in inputs["paths"].items()
    }
    output_records = {
        rel(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in output_paths if path.is_file()
    }
    return {
        "artifact": "Q3-T2 exact finite-grid near-optimal, Cost-Loss Pareto, marginal-return analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": run_mode,
        "command": "python Q3/03_代码/analyze_q3_t2_near_optimal_pareto.py",
        "p1_command": "python Q3/03_代码/analyze_q3_t2_near_optimal_pareto.py --p1",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions,
        "random_seed": None,
        "randomness": "No new sampling; reuses the frozen 1,000-row Q2 trajectory-bootstrap parameter table.",
        "contract": {
            "quality": "Q=Q0=0.49847810484764643; CQ=0",
            "mixture": "p excluded; Q2 p response is not identified",
            "candidate_domain": "M3-allowed observed B1 N/D support points only; no interpolation or extrapolation",
            "loss": "Q2 B1/M0 predicted validation Loss from the existing fitted predictions",
            "cost_flops": "6*N_abs*D_abs + (N_abs*D_abs*context_length)/5000",
            "near_optimal_relative_gap": "(candidate_loss - scenario_min_loss) / abs(scenario_min_loss) <= 0.01, 0.03, 0.05",
            "pareto_objectives": ["minimize total compute cost", "minimize M0 predicted Loss"],
            "marginal_main_metric": "paired scenario Loss reduction per added decade of budget cap",
            "bootstrap": "paired by the same replicate ID across budgets; percentile interval 2.5% to 97.5%; 8 B1 trajectory clusters",
        },
        "dimensions": {
            "b1_grid_points": len(inputs["points"]),
            "supported_grid_points": sum(point["m3_allowed"] for point in inputs["points"]),
            "budget_tiers": inputs["budgets"],
            "context_lengths": inputs["contexts"],
            "scenarios": len(inputs["scenario_summaries"]),
            "bootstrap_replicates": BOOTSTRAP_COUNT,
            "input_rows": inputs["input_row_counts"],
            "verification_counts": inputs["verification_counts"],
        },
        "input_files": input_records,
        "output_files": output_records,
        "figure": figure_info,
        "reconciliation": {
            "m0_selected_points": "All 15 recomputed from the B1/M0 candidate table with the existing tie-break rule.",
            "m1_selected_points": "All 15 grid IDs match, and all selected points are marked feasible.",
            "m4_masks": "All 17,640 exact costs and budget-feasible flags independently reconciled.",
            "existing_bootstrap": "All 15 baseline selection frequencies and predicted-Loss percentiles reconciled.",
            "pareto": "Exact two-objective non-dominance; checked against all supported candidates.",
            "budget_nesting": "Verified for each of the five context lengths.",
        },
        "claim_boundary": "Fixed-Q0 B1/M0 finite-support sensitivity; not causal, not out-of-domain validation, and not a full Q/p joint optimum.",
    }


def run_p1() -> dict[str, Any]:
    inputs = load_inputs()
    contexts = [inputs["contexts"][0]]
    bootstrap = compute_bootstrap(inputs, contexts)
    attach_bootstrap_frequencies(inputs["near_rows"], bootstrap, contexts)
    marginal_rows = build_marginal_rows(inputs, bootstrap, contexts)
    near_counts = summarize_near_counts(inputs["near_rows"])
    pareto_rows = [row for row in inputs["pareto_rows"] if int(row["context_length_tokens"]) in contexts]
    expected_cost = float(inputs["scenario_summaries"][
        (inputs["budgets"][0], contexts[0])
    ]["actual_cost"])
    receipt = {
        "gate": "P1 minimum runnable result",
        "command": "python Q3/03_代码/analyze_q3_t2_near_optimal_pareto.py --p1",
        "status": "PASS",
        "scope": "One real context (the shortest C7 length), all three budgets, the complete B1 support grid, and all 1,000 existing bootstrap parameter replicates.",
        "checks": {
            "b1_rows": inputs["input_row_counts"]["b1_predictions"] == 1176,
            "m3_alignment": inputs["verification_counts"]["m3_nd_alignment_rows"] == 1176,
            "m4_cost_and_mask": inputs["verification_counts"]["m4_cost_rows"] == 17640
            and inputs["verification_counts"]["m4_budget_flag_rows"] == 17640,
            "m0_and_m1_selected_points": inputs["verification_counts"]["m0_optimum_reconciliations"] == 15
            and inputs["verification_counts"]["m1_optimum_reconciliations"] == 15,
            "bootstrap_reconciliation_for_context": bootstrap["frontier_reconciliations"] == len(inputs["budgets"]),
            "pareto_context_nonempty": bool(pareto_rows),
            "two_marginal_intervals": len(marginal_rows) == 2,
            "near_optimal_sets_nested": all(
                near_counts[(budget, contexts[0])][0] <= near_counts[(budget, contexts[0])][1]
                <= near_counts[(budget, contexts[0])][2]
                for budget in inputs["budgets"]
            ),
        },
        "sample_context_length_tokens": contexts[0],
        "sample_low_budget_flops": inputs["budgets"][0],
        "sample_low_budget_feasible_candidates": inputs["scenario_summaries"][
            (inputs["budgets"][0], contexts[0])
        ]["feasible_candidate_count"],
        "sample_low_budget_selected_grid_id": inputs["scenario_best"][
            (inputs["budgets"][0], contexts[0])
        ]["grid_id"],
        "sample_low_budget_selected_cost_flops": expected_cost,
        "sample_near_optimal_counts_by_budget": {
            str(budget): near_counts[(budget, contexts[0])] for budget in inputs["budgets"]
        },
        "sample_pareto_front_count": len(pareto_rows),
        "sample_marginal_returns": marginal_rows,
        "input_sha256": {name: sha256_file(path) for name, path in inputs["paths"].items()},
        "claim_boundary": "Fixed-Q0 B1/M0 support-grid smoke slice; not full Q3 Q/p optimization.",
    }
    receipt["status"] = "PASS" if all(receipt["checks"].values()) else "FAIL"
    path = ROOT / OUT_REL / "p1_smoke_receipt.json"
    write_json(path, receipt)
    return receipt


def run_full() -> dict[str, Any]:
    inputs = load_inputs()
    bootstrap = compute_bootstrap(inputs, inputs["contexts"])
    attach_bootstrap_frequencies(inputs["near_rows"], bootstrap, inputs["contexts"])
    marginal_rows = build_marginal_rows(inputs, bootstrap, inputs["contexts"])

    out_dir = ROOT / OUT_REL
    fig_dir = ROOT / FIG_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_base = ROOT / FIG_REL / "result_q3_t2_near_optimal_pareto_marginal_return"

    near_path = out_dir / "q3_near_optimal.csv"
    pareto_path = out_dir / "q3_pareto.csv"
    marginal_path = out_dir / "q3_marginal_return.csv"
    contract_path = out_dir / "figure_contract.md"
    report_path = out_dir / "report.md"
    receipt_path = out_dir / "p1_smoke_receipt.json"
    manifest_path = out_dir / "reproduction_manifest.json"
    write_csv(near_path, inputs["near_rows"])
    write_csv(pareto_path, inputs["pareto_rows"])
    write_csv(marginal_path, marginal_rows)
    write_figure_contract(contract_path)
    if not receipt_path.is_file():
        run_p1()
    figure_info = make_figure(inputs, inputs["near_rows"], inputs["pareto_rows"], marginal_rows, fig_base)
    write_report(report_path, inputs, inputs["near_rows"], marginal_rows, bootstrap, figure_info)
    outputs = [
        ROOT / SCRIPT_REL,
        near_path,
        pareto_path,
        marginal_path,
        contract_path,
        report_path,
        receipt_path,
        fig_base.with_suffix(".svg"),
        fig_base.with_suffix(".png"),
        fig_base.with_name(fig_base.name + "_grayscale.png"),
    ]
    manifest = build_manifest(inputs, outputs, figure_info, bootstrap, "full")
    write_json(manifest_path, manifest)
    return {
        "near_rows": len(inputs["near_rows"]),
        "pareto_rows": len(inputs["pareto_rows"]),
        "marginal_rows": len(marginal_rows),
        "scenario_count": len(inputs["scenario_summaries"]),
        "bootstrap_reconciliations": bootstrap["frontier_reconciliations"],
        "figure": figure_info,
        "manifest": rel(manifest_path),
        "output_directory": rel(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", action="store_true", help="Run the real-input vertical P1 smoke slice only.")
    args = parser.parse_args()
    result = run_p1() if args.p1 else run_full()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.p1 and result["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
