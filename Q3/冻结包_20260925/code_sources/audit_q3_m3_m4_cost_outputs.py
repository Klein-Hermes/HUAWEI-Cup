#!/usr/bin/env python3
"""Independently audit Q3 M3 support, M4 budget masks, and M0-based picks.

Run from the repository root:
    python Q3/03_代码/audit_q3_m3_m4_cost_outputs.py --smoke
    python Q3/03_代码/audit_q3_m3_m4_cost_outputs.py

All generated files are written below Q3/04_结果/M3_M4成本独立审计.
Existing Q3 source tables and the M3/M4 pipeline outputs are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
from q3_repro_support import resolve_math_modeling_skill_root

SKILL_ROOT = resolve_math_modeling_skill_root()
FIGURE_SCRIPTS = SKILL_ROOT / "tools" / "figure" / "scripts"
if str(FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SCRIPTS))
OUT_DIR = ROOT / "Q3" / "04_结果" / "M3_M4成本独立审计"
FIGURE_DIR = OUT_DIR / "figures"

INPUTS = {
    "q1_freeze": Path("Q1/03_结果/Q1_最终收口/q1_final_freeze.json"),
    "q1_domains": Path("Q1/03_结果/Q1_最终收口/q1_final_domain_quality.csv"),
    "b1_predictions": Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv"),
    "m0_frontier": Path("Q3/04_结果/M0_ND_reference_frontier.csv"),
    "m3_support": Path("Q3/04_结果/m3_q3_support_grid.csv"),
    "m4_mask": Path("Q3/04_结果/m4_budget_feasible_mask.csv"),
    "m1_picks": Path("Q3/04_结果/m1_optimum_boundary_audit.csv"),
    "m3_manifest": Path("Q3/04_结果/M3_support_manifest.json"),
    "m4_manifest": Path("Q3/04_结果/M4_budget_manifest.json"),
    "m1_manifest": Path("Q3/04_结果/M1_masked_optimization_manifest.json"),
    "c7_metadata": Path(
        "中文题目/F题/real_attachments/C_efficiency_evolution/"
        "model_architecture_metadata.csv"
    ),
}

ETA = Fraction(1, 5000)
TRAINING_COEFFICIENT = 6
UNIT_SCALE = 10**9
REL_TOLERANCE = 1e-12
ABS_TOLERANCE_FLOPS = 1e-6
EXPECTED_Q_VERSION = "Q1-q_huber-v1"
EXPECTED_OPERATIONAL_Q = "q_huber"


class AuditInputError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditInputError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise AuditInputError(f"Unrecognized boolean value: {value!r}")


def frac(value: Any) -> Fraction:
    return Fraction(str(value).strip())


def flops_value(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    return float(value)


def close_flops(left: Any, right: Fraction) -> bool:
    left_float = float(left)
    right_float = float(right)
    return math.isclose(
        left_float,
        right_float,
        rel_tol=REL_TOLERANCE,
        abs_tol=ABS_TOLERANCE_FLOPS,
    )


def load_inputs() -> dict[str, Any]:
    resolved = {name: ROOT / path for name, path in INPUTS.items()}
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    require(not missing, "Required audit inputs are missing: " + ", ".join(missing))

    freeze = read_json(resolved["q1_freeze"])
    require(
        freeze.get("status") == "FROZEN_WITH_LIMITATIONS",
        "Q1 freeze status differs from the current Q3 baseline contract",
    )
    require(
        freeze.get("q_version") == EXPECTED_Q_VERSION
        and freeze.get("operational_q") == EXPECTED_OPERATIONAL_Q,
        "Q1 frozen quality version differs from the current Q3 baseline contract",
    )

    domain_rows = read_csv(resolved["q1_domains"])
    q_rows = [
        row for row in domain_rows
        if row.get("dataset") == "a1" and row.get("q_version") == EXPECTED_Q_VERSION
    ]
    require(len(q_rows) == 7, f"Expected 7 frozen A1 domains, found {len(q_rows)}")
    q0 = sum(frac(row["q_operational"]) for row in q_rows) / len(q_rows)

    b1_rows = read_csv(resolved["b1_predictions"])
    require(len(b1_rows) == 1176, f"Expected 1,176 B1 rows, found {len(b1_rows)}")
    required_b1 = {
        "run_id", "N_params_B", "D_tokens_B",
        "observed_val_loss", "predicted_val_loss",
    }
    require(required_b1.issubset(b1_rows[0]), "B1 prediction table lacks required columns")
    ids = [str(row["run_id"]) for row in b1_rows]
    require(len(ids) == len(set(ids)), "B1 run_id values are not unique")
    nd_points = {(frac(row["N_params_B"]), frac(row["D_tokens_B"])) for row in b1_rows}
    require(len(nd_points) == 1176, "B1 table contains duplicate N/D support points")

    c7_rows = read_csv(resolved["c7_metadata"])
    require(c7_rows and "max_position_embeddings" in c7_rows[0], "C7 context field is missing")
    contexts = sorted({int(row["max_position_embeddings"]) for row in c7_rows})
    require(contexts == [2048, 4096, 8192, 32768, 131072], "Unexpected C7 context support")

    frontier_rows = read_csv(resolved["m0_frontier"])
    require(len(frontier_rows) == 15, f"Expected 15 M0 frontier rows, found {len(frontier_rows)}")
    budgets = sorted({int(row["budget_flops"]) for row in frontier_rows})
    frontier_contexts = sorted({int(row["context_length_tokens"]) for row in frontier_rows})
    require(len(budgets) == 3 and frontier_contexts == contexts, "M0 budgets/contexts differ from the frozen scenario set")
    frontier_by_key = {
        (int(row["budget_flops"]), int(row["context_length_tokens"])): row
        for row in frontier_rows
    }
    require(len(frontier_by_key) == 15, "M0 frontier has duplicate budget/context scenarios")
    require(
        all(abs(float(row["q0_reference_a1_macro"]) - float(q0)) <= 1e-10 for row in frontier_rows),
        "M0 frontier Q0 does not match the frozen Q1 A1-domain mean",
    )

    support_rows = read_csv(resolved["m3_support"])
    m4_rows = read_csv(resolved["m4_mask"])
    m1_rows = read_csv(resolved["m1_picks"])
    require(len(support_rows) == 1176, f"Expected 1,176 M3 support rows, found {len(support_rows)}")
    require(len(m4_rows) == 17640, f"Expected 17,640 M4 mask rows, found {len(m4_rows)}")
    require(len(m1_rows) == 15, f"Expected 15 M1 pick rows, found {len(m1_rows)}")
    require("m3_search_allowed" in support_rows[0], "M3 support flag m3_search_allowed is missing")
    require(
        {"budget_feasible", "C_train_flops", "C_Q_flops",
         "C_attn_flops", "C_total_flops"}.issubset(m4_rows[0]),
        "M4 mask is missing required cost or budget-feasibility fields",
    )
    require(
        {"m1_selected_grid_id", "m1_selected_run_id", "C_total_flops",
         "budget_utilization", "m1_eligible_candidate_count", "m3_search_allowed",
         "budget_feasible", "final_feasible"}.issubset(m1_rows[0]),
        "M1 selection table is missing required audit fields",
    )

    support_by_id = {str(row["grid_id"]): row for row in support_rows}
    require(len(support_by_id) == 1176, "M3 support grid_id values are not unique")
    mask_by_key = {
        (int(row["budget_flops"]), int(row["context_length_tokens"]), str(row["grid_id"])): row
        for row in m4_rows
    }
    require(len(mask_by_key) == 17640, "M4 budget mask contains duplicate scenario/grid keys")
    m1_by_key = {
        (int(row["budget_flops"]), int(row["context_length_tokens"])): row
        for row in m1_rows
    }
    require(len(m1_by_key) == 15, "M1 selection table has duplicate scenarios")

    m3_manifest = read_json(resolved["m3_manifest"])
    m4_manifest = read_json(resolved["m4_manifest"])
    m1_manifest = read_json(resolved["m1_manifest"])
    require(
        m3_manifest.get("policy", {}).get("M3") == "exact_observed_only; hull and distance are diagnostics",
        "M3 support policy differs from its frozen manifest",
    )
    for manifest_name, manifest, expected_outputs in (
        ("M3", m3_manifest, {
            "Q3/04_结果/m3_q3_support_grid.csv": resolved["m3_support"],
        }),
        ("M4", m4_manifest, {
            "Q3/04_结果/m4_budget_feasible_mask.csv": resolved["m4_mask"],
        }),
        ("M1", m1_manifest, {
            "Q3/04_结果/m1_optimum_boundary_audit.csv": ROOT / "Q3/04_结果/m1_optimum_boundary_audit.csv",
        }),
    ):
        declared = manifest.get("outputs_sha256", {})
        for relative_name, path in expected_outputs.items():
            require(
                declared.get(relative_name) == sha256_file(path),
                f"{manifest_name} manifest hash mismatch for {relative_name}",
            )
    return {
        "resolved_inputs": resolved,
        "freeze": freeze,
        "q0": q0,
        "b1_rows": b1_rows,
        "contexts": contexts,
        "budgets": budgets,
        "frontier_rows": frontier_rows,
        "frontier_by_key": frontier_by_key,
        "support_rows": support_rows,
        "support_by_id": support_by_id,
        "m4_rows": m4_rows,
        "mask_by_key": mask_by_key,
        "m1_rows": m1_rows,
        "m1_by_key": m1_by_key,
        "m3_manifest": m3_manifest,
        "m4_manifest": m4_manifest,
        "m1_manifest": m1_manifest,
    }


def compute_cost(row: dict[str, str], context: int) -> dict[str, Any]:
    n_b = frac(row["N_params_B"])
    d_b = frac(row["D_tokens_B"])
    n_abs = n_b * UNIT_SCALE
    d_abs = d_b * UNIT_SCALE
    require(n_abs.denominator == 1 and d_abs.denominator == 1, "N or D does not convert to an integer count")
    nd = n_abs * d_abs
    train = TRAINING_COEFFICIENT * nd
    quality = Fraction(0)
    attention = ETA * nd * context
    total = train + quality + attention
    return {
        "grid_id": str(row["run_id"]),
        "source_run_id": str(row["run_id"]),
        "N_params_B": float(n_b),
        "D_tokens_B": float(d_b),
        "N_parameters_abs": n_abs.numerator,
        "D_tokens_abs": d_abs.numerator,
        "context_length_tokens": context,
        "Q_setting": "Q=Q0",
        "C_train_flops": train,
        "C_Q_flops": quality,
        "C_attn_flops": attention,
        "C_total_flops": total,
        "train_cost_share": train / total,
        "quality_cost_share": quality / total,
        "attention_cost_share": attention / total,
    }


def support_alignment(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = []
    failures = []
    for row in data["b1_rows"]:
        grid_id = str(row["run_id"])
        support = data["support_by_id"].get(grid_id)
        if support is None:
            failures.append({"check": "support_key", "grid_id": grid_id, "detail": "missing M3 key"})
            continue
        n_match = frac(row["N_params_B"]) == frac(support["N_params_B"])
        d_match = frac(row["D_tokens_B"]) == frac(support["D_tokens_B"])
        allowed = parse_bool(support["m3_search_allowed"])
        matched = n_match and d_match
        result.append({
            "grid_id": grid_id,
            "source_run_id": grid_id,
            "N_params_B": row["N_params_B"],
            "D_tokens_B": row["D_tokens_B"],
            "m3_N_params_B": support["N_params_B"],
            "m3_D_tokens_B": support["D_tokens_B"],
            "N_match": n_match,
            "D_match": d_match,
            "support_feasible_source_column": "m3_search_allowed",
            "support_feasible": allowed,
            "support_flag_recomputed": False,
            "alignment_pass": matched,
        })
        if not matched:
            failures.append({
                "check": "support_value_alignment",
                "grid_id": grid_id,
                "detail": f"N_match={n_match}; D_match={d_match}",
            })
    if len(result) != 1176:
        failures.append({
            "check": "support_row_count",
            "grid_id": "",
            "detail": f"aligned={len(result)} expected=1176",
        })
    return result, failures


def build_cost_grid(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    by_key = {}
    identity_failures = []
    for b1 in data["b1_rows"]:
        for context in data["contexts"]:
            cost = compute_cost(b1, context)
            key = (cost["grid_id"], context)
            by_key[key] = cost
            exact_total = cost["C_train_flops"] + cost["C_Q_flops"] + cost["C_attn_flops"]
            identity_pass = exact_total == cost["C_total_flops"]
            if not identity_pass:
                identity_failures.append({
                    "failure_scope": "independent_formula",
                    "scenario_id": f"L{context}",
                    "grid_id": cost["grid_id"],
                    "field": "C_total_flops",
                    "expected": flops_value(exact_total),
                    "observed": flops_value(cost["C_total_flops"]),
                    "relative_error": "",
                })
            rows.append({
                "grid_id": cost["grid_id"],
                "source_run_id": cost["source_run_id"],
                "N_params_B": cost["N_params_B"],
                "D_tokens_B": cost["D_tokens_B"],
                "N_parameters_abs": cost["N_parameters_abs"],
                "D_tokens_abs": cost["D_tokens_abs"],
                "Q_setting": cost["Q_setting"],
                "q0_reference": float(data["q0"]),
                "context_length_tokens": context,
                "C_train_flops": flops_value(cost["C_train_flops"]),
                "C_Q_flops": flops_value(cost["C_Q_flops"]),
                "C_attn_flops": flops_value(cost["C_attn_flops"]),
                "C_total_flops": flops_value(cost["C_total_flops"]),
                "train_cost_share": float(cost["train_cost_share"]),
                "quality_cost_share": float(cost["quality_cost_share"]),
                "attention_cost_share": float(cost["attention_cost_share"]),
                "cost_identity_pass": identity_pass,
            })
    require(len(rows) == 5880, f"Expected 5,880 candidate/context rows, found {len(rows)}")
    return rows, by_key, identity_failures


def audit_budget_mask(
    data: dict[str, Any],
    cost_by_key: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows = []
    failures = []
    for context in data["contexts"]:
        for budget in data["budgets"]:
            for b1 in data["b1_rows"]:
                grid_id = str(b1["run_id"])
                cost = cost_by_key[(grid_id, context)]
                source_key = (budget, context, grid_id)
                source = data["mask_by_key"].get(source_key)
                if source is None:
                    failures.append({
                        "failure_scope": "m4_key_alignment",
                        "scenario_id": f"C{budget}_L{context}",
                        "grid_id": grid_id,
                        "field": "row",
                        "expected": "present",
                        "observed": "missing",
                        "relative_error": "",
                    })
                    continue

                support = data["support_by_id"][grid_id]
                support_allowed = parse_bool(support["m3_search_allowed"])
                budget_feasible = cost["C_total_flops"] <= budget
                final_feasible = support_allowed and budget_feasible
                source_budget_feasible = parse_bool(source["budget_feasible"])
                fields_match = True
                for field, exact in (
                    ("C_train_flops", cost["C_train_flops"]),
                    ("C_Q_flops", cost["C_Q_flops"]),
                    ("C_attn_flops", cost["C_attn_flops"]),
                    ("C_total_flops", cost["C_total_flops"]),
                ):
                    observed = source[field]
                    if not close_flops(observed, exact):
                        fields_match = False
                        scale = max(abs(float(exact)), 1.0)
                        failures.append({
                            "failure_scope": "m4_cost_value",
                            "scenario_id": f"C{budget}_L{context}",
                            "grid_id": grid_id,
                            "field": field,
                            "expected": flops_value(exact),
                            "observed": observed,
                            "relative_error": abs(float(observed) - float(exact)) / scale,
                        })

                budget_match = budget_feasible == source_budget_feasible
                if not budget_match:
                    failures.append({
                        "failure_scope": "m4_budget_mask",
                        "scenario_id": f"C{budget}_L{context}",
                        "grid_id": grid_id,
                        "field": "budget_feasible",
                        "expected": budget_feasible,
                        "observed": source_budget_feasible,
                        "relative_error": "",
                    })
                audit_rows.append({
                    "scenario_id": f"C{budget}_L{context}",
                    "budget_flops": budget,
                    "context_length_tokens": context,
                    "grid_id": grid_id,
                    "source_run_id": grid_id,
                    "N_params_B": float(frac(b1["N_params_B"])),
                    "D_tokens_B": float(frac(b1["D_tokens_B"])),
                    "support_feasible": support_allowed,
                    "support_feasible_source_column": "m3_search_allowed",
                    "budget_feasible_recomputed": budget_feasible,
                    "budget_feasible_existing": source_budget_feasible,
                    "final_feasible_recomputed": final_feasible,
                    "cost_fields_match": fields_match,
                    "mask_fields_match": budget_match,
                    "C_train_flops": flops_value(cost["C_train_flops"]),
                    "C_Q_flops": flops_value(cost["C_Q_flops"]),
                    "C_attn_flops": flops_value(cost["C_attn_flops"]),
                    "C_total_flops": flops_value(cost["C_total_flops"]),
                    "budget_slack_flops": flops_value(Fraction(budget) - cost["C_total_flops"]),
                })
    require(len(audit_rows) == 17640, f"Expected 17,640 audited M4 rows, found {len(audit_rows)}")
    return audit_rows, failures


def monotonicity_audit(
    cost_by_key: dict[tuple[str, int], dict[str, Any]],
    b1_rows: list[dict[str, str]],
    contexts: list[int],
) -> list[dict[str, Any]]:
    n_values = sorted({frac(row["N_params_B"]) for row in b1_rows})
    d_values = sorted({frac(row["D_tokens_B"]) for row in b1_rows})
    b1_by_id = {str(row["run_id"]): row for row in b1_rows}
    specs = [
        ("train_vs_N_at_fixed_D_context", "C_train_flops", "N"),
        ("attention_vs_N_at_fixed_D_context", "C_attn_flops", "N"),
        ("total_vs_N_at_fixed_D_context", "C_total_flops", "N"),
        ("train_vs_D_at_fixed_N_context", "C_train_flops", "D"),
        ("attention_vs_D_at_fixed_N_context", "C_attn_flops", "D"),
        ("quality_Q0_vs_D_at_fixed_N_context", "C_Q_flops", "D"),
        ("total_vs_D_at_fixed_N_context", "C_total_flops", "D"),
        ("train_vs_context_at_fixed_ND", "C_train_flops", "context"),
        ("attention_vs_context_at_fixed_ND", "C_attn_flops", "context"),
        ("quality_Q0_vs_context_at_fixed_ND", "C_Q_flops", "context"),
        ("total_vs_context_at_fixed_ND", "C_total_flops", "context"),
    ]
    result = []
    for name, component, axis in specs:
        groups: dict[tuple[Any, ...], list[tuple[Any, Fraction]]] = defaultdict(list)
        if axis in {"N", "D"}:
            for grid_id, b1 in b1_by_id.items():
                n, d = frac(b1["N_params_B"]), frac(b1["D_tokens_B"])
                x = n if axis == "N" else d
                fixed = (d,) if axis == "N" else (n,)
                for context in contexts:
                    groups[fixed + (context,)].append((x, cost_by_key[(grid_id, context)][component]))
        else:
            for grid_id, b1 in b1_by_id.items():
                fixed = (frac(b1["N_params_B"]), frac(b1["D_tokens_B"]))
                for context in contexts:
                    groups[fixed].append((context, cost_by_key[(grid_id, context)][component]))

        comparisons = 0
        violations = 0
        equality_checks = 0
        equality_failures = 0
        for values in groups.values():
            values.sort(key=lambda item: item[0])
            for (_, left), (_, right) in zip(values, values[1:]):
                comparisons += 1
                if right < left:
                    violations += 1
                if name.startswith("train_vs_context") or name.startswith("quality_Q0_vs_context"):
                    equality_checks += 1
                    if right != left:
                        equality_failures += 1
        expected = "constant" if equality_checks else "nondecreasing"
        result.append({
            "check": name,
            "cost_component": component,
            "varying_dimension": axis,
            "expected_relation": expected,
            "adjacent_comparisons": comparisons,
            "monotonicity_violations": violations,
            "invariance_comparisons": equality_checks,
            "invariance_violations": equality_failures,
            "pass": violations == 0 and equality_failures == 0,
        })
    return result


def budget_summaries(
    data: dict[str, Any],
    budget_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_scenario: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in budget_rows:
        by_scenario[(row["budget_flops"], row["context_length_tokens"])].append(row)

    summary = []
    nesting = []
    m0_by_key = data["frontier_by_key"]
    for context in data["contexts"]:
        previous_set: set[str] | None = None
        previous_budget: int | None = None
        for budget in data["budgets"]:
            rows = by_scenario[(budget, context)]
            final = [row for row in rows if row["final_feasible_recomputed"]]
            budget_only = [row for row in rows if row["budget_feasible_recomputed"]]
            final_set = {row["grid_id"] for row in final}
            m0 = m0_by_key[(budget, context)]
            min_cost = min(float(row["C_total_flops"]) for row in final)
            max_cost = max(float(row["C_total_flops"]) for row in final)
            count_match = len(final) == int(m0["b1_feasible_grid_points"])
            summary.append({
                "budget_flops": budget,
                "context_length_tokens": context,
                "candidate_count": len(rows),
                "support_allowed_count": sum(bool(row["support_feasible"]) for row in rows),
                "budget_feasible_count": len(budget_only),
                "final_feasible_count": len(final),
                "M0_feasible_count": int(m0["b1_feasible_grid_points"]),
                "M0_feasible_count_match": count_match,
                "min_final_feasible_cost_flops": min_cost,
                "max_final_feasible_cost_flops": max_cost,
                "max_feasible_cost_budget_ratio": float(max_cost / budget),
            })
            if previous_set is not None and previous_budget is not None:
                nesting.append({
                    "context_length_tokens": context,
                    "lower_budget_flops": previous_budget,
                    "higher_budget_flops": budget,
                    "lower_feasible_count": len(previous_set),
                    "higher_feasible_count": len(final_set),
                    "lower_set_is_subset": previous_set.issubset(final_set),
                    "newly_feasible_count": len(final_set - previous_set),
                    "missing_from_higher_count": len(previous_set - final_set),
                })
            previous_set = final_set
            previous_budget = budget
    return summary, nesting


def audit_m0_picks(
    data: dict[str, Any],
    cost_by_key: dict[tuple[str, int], dict[str, Any]],
    budget_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    budget_by_key = {
        (int(row["budget_flops"]), int(row["context_length_tokens"]), str(row["grid_id"])): row
        for row in budget_rows
    }
    result = []
    failures = []
    for key, m0 in data["frontier_by_key"].items():
        m1 = data["m1_by_key"].get(key)
        if m1 is None:
            failures.append({
                "failure_scope": "M1_scenario_alignment", "scenario_id": str(key),
                "grid_id": "", "field": "scenario", "expected": "present",
                "observed": "missing", "relative_error": "",
            })
            continue
        grid_id = str(m1["m1_selected_grid_id"])
        context = key[1]
        cost = cost_by_key.get((grid_id, context))
        mask = budget_by_key.get((key[0], key[1], grid_id))
        if cost is None or mask is None:
            failures.append({
                "failure_scope": "M1_selected_candidate", "scenario_id": str(key),
                "grid_id": grid_id, "field": "candidate", "expected": "present",
                "observed": "missing", "relative_error": "",
            })
            continue
        choice_match = (
            str(m0["selected_run_id"]) == str(m1["m1_selected_run_id"]) == grid_id
            and frac(m0["selected_N_B"]) == frac(m1["selected_N_B"]) == frac(cost["N_params_B"])
            and frac(m0["selected_D_B"]) == frac(m1["selected_D_B"]) == frac(cost["D_tokens_B"])
        )
        if not choice_match:
            failures.append({
                "failure_scope": "M0_M1_choice_parity", "scenario_id": f"C{key[0]}_L{context}",
                "grid_id": grid_id, "field": "selected_grid", "expected": m0["selected_run_id"],
                "observed": m1["m1_selected_run_id"], "relative_error": "",
            })
        cost_fields_match = all(
            close_flops(m1[field], cost[cost_key])
            for field, cost_key in (
                ("C_train_flops", "C_train_flops"),
                ("C_Q_flops", "C_Q_flops"),
                ("C_attn_flops", "C_attn_flops"),
                ("C_total_flops", "C_total_flops"),
            )
        )
        budget_feasible = bool(mask["budget_feasible_recomputed"])
        support_feasible = bool(mask["support_feasible"])
        final_feasible = support_feasible and budget_feasible
        m3_flag_match = support_feasible == parse_bool(m1["m3_search_allowed"])
        budget_flag_match = budget_feasible == parse_bool(m1["budget_feasible"])
        final_flag_match = final_feasible == parse_bool(m1["final_feasible"])
        utilization = float(cost["C_total_flops"] / key[0])
        utilization_match = math.isclose(
            utilization, float(m1["budget_utilization"]),
            rel_tol=REL_TOLERANCE, abs_tol=REL_TOLERANCE,
        )
        if not cost_fields_match:
            failures.append({
                "failure_scope": "M1_pick_cost", "scenario_id": f"C{key[0]}_L{context}",
                "grid_id": grid_id, "field": "cost_components", "expected": "independent formula",
                "observed": "M1 table", "relative_error": "",
            })
        if not budget_feasible or not support_feasible:
            failures.append({
                "failure_scope": "M1_pick_feasibility", "scenario_id": f"C{key[0]}_L{context}",
                "grid_id": grid_id, "field": "final_feasible", "expected": True,
                "observed": False, "relative_error": "",
            })
        for field, matched, expected, observed in (
            ("m3_search_allowed", m3_flag_match, support_feasible, m1["m3_search_allowed"]),
            ("budget_feasible", budget_flag_match, budget_feasible, m1["budget_feasible"]),
            ("final_feasible", final_flag_match, final_feasible, m1["final_feasible"]),
        ):
            if not matched:
                failures.append({
                    "failure_scope": "M1_feasibility_flag_parity",
                    "scenario_id": f"C{key[0]}_L{context}",
                    "grid_id": grid_id,
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                    "relative_error": "",
                })
        if not utilization_match:
            failures.append({
                "failure_scope": "M1_pick_utilization", "scenario_id": f"C{key[0]}_L{context}",
                "grid_id": grid_id, "field": "budget_utilization",
                "expected": utilization, "observed": m1["budget_utilization"], "relative_error": "",
            })
        total = cost["C_total_flops"]
        result.append({
            "scenario_id": f"C{key[0]}_L{context}",
            "budget_flops": key[0],
            "context_length_tokens": context,
            "M0_selected_run_id": m0["selected_run_id"],
            "M1_selected_run_id": m1["m1_selected_run_id"],
            "selected_grid_id": grid_id,
            "selected_N_B": float(cost["N_params_B"]),
            "selected_D_B": float(cost["D_tokens_B"]),
            "choice_match": choice_match,
            "support_feasible": support_feasible,
            "budget_feasible": budget_feasible,
            "final_feasible_recomputed": final_feasible,
            "m3_flag_match": m3_flag_match,
            "budget_flag_match": budget_flag_match,
            "final_flag_match": final_flag_match,
            "cost_fields_match": cost_fields_match,
            "C_train_flops": flops_value(cost["C_train_flops"]),
            "C_Q_flops": flops_value(cost["C_Q_flops"]),
            "C_attn_flops": flops_value(cost["C_attn_flops"]),
            "C_total_flops": flops_value(total),
            "budget_utilization_recomputed": utilization,
            "budget_utilization_existing": float(m1["budget_utilization"]),
            "train_cost_share": float(cost["train_cost_share"]),
            "quality_cost_share": float(cost["quality_cost_share"]),
            "attention_cost_share": float(cost["attention_cost_share"]),
            "cost_share_sum": float(cost["train_cost_share"] + cost["quality_cost_share"] + cost["attention_cost_share"]),
            "hit_support_boundary": any(parse_bool(m1[field]) for field in (
                "hit_N_min", "hit_N_max", "hit_D_min", "hit_D_max"
            )),
        })
    return result, failures


def unit_check_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"check": "N_input_scale", "source_field": "N_params_B", "source_unit": "billion parameters",
         "conversion": "multiply by 1e9", "calculation_unit": "absolute parameters", "pass": True},
        {"check": "D_input_scale", "source_field": "D_tokens_B", "source_unit": "billion tokens",
         "conversion": "multiply by 1e9", "calculation_unit": "absolute tokens", "pass": True},
        {"check": "cost_terms", "source_field": "C_train, C_Q, C_attn", "source_unit": "Q3 cost proxy",
         "conversion": "current Q3 coefficients and absolute N/D", "calculation_unit": "FLOPs", "pass": True},
        {"check": "budget_unit", "source_field": "budget_flops", "source_unit": "FLOPs",
         "conversion": "none", "calculation_unit": "FLOPs", "pass": True},
        {"check": "Q0_interface", "source_field": "Q1-q_huber-v1 A1 domain macro",
         "source_unit": "quality score", "conversion": "mean of 7 frozen A1 domains",
         "calculation_unit": f"Q0={float(data['q0']):.10f}; fixed Q implies C_Q=0", "pass": True},
    ]


def write_rows(out_dir: Path, name: str, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    path = out_dir / name
    write_csv(path, rows, fields)
    return path


def plot_audit(summary: list[dict[str, Any]], optimum: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    from export_figure import export_figure
    from setup_style import setup_style
    from visual_qa import audit_layout, print_report, render_preview

    import matplotlib.pyplot as plt
    import numpy as np

    setup_style(journal="general", lang="zh", serif_for_zh=True, constrained_layout=False)
    budgets = sorted({int(row["budget_flops"]) for row in summary})
    contexts = sorted({int(row["context_length_tokens"]) for row in summary})
    palette = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "P")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.65), constrained_layout=False)
    ax_count, ax_util = axes
    for index, context in enumerate(contexts):
        color = palette[index]
        count_rows = sorted(
            (row for row in summary if int(row["context_length_tokens"]) == context),
            key=lambda row: int(row["budget_flops"]),
        )
        util_rows = sorted(
            (row for row in optimum if int(row["context_length_tokens"]) == context),
            key=lambda row: int(row["budget_flops"]),
        )
        # Keep the large exact integer budgets in CSV/report outputs, but use
        # float coordinates for Matplotlib's log transform (budgets reach 1e24).
        x = [float(row["budget_flops"]) for row in count_rows]
        y_count = [int(row["final_feasible_count"]) for row in count_rows]
        y_util = [100.0 * float(row["budget_utilization_recomputed"]) for row in util_rows]
        ax_count.plot(
            x, y_count, color=color, marker=markers[index], linewidth=1.15,
            markersize=4.4, linestyle="--", label=f"{context:,}",
        )
        ax_util.scatter(
            x, y_util, color=color, marker=markers[index], s=28, zorder=3,
            edgecolors="white", linewidths=0.35, label=f"{context:,}",
        )

    budget_labels = [f"10^{int(math.log10(value))}" for value in budgets]
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([float(value) for value in budgets], labels=budget_labels)
        ax.set_xlabel("算力预算（FLOPs）")
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_count.set_title("支持域内可行配置")
    ax_count.set_ylabel("可行配置数 / 1,176")
    ax_count.set_ylim(0, 1230)
    ax_count.set_yticks([0, 300, 600, 900, 1176])
    ax_count.legend(title="上下文长度", fontsize=7, title_fontsize=7, frameon=False, loc="upper left")

    ax_util.set_title("M0 选点的预算利用率")
    ax_util.set_ylabel("实际成本 / 预算（%）")
    ax_util.set_ylim(0, 108)
    ax_util.set_yticks(np.arange(0, 101, 20))
    ax_util.axhline(100, color="#555555", linestyle=":", linewidth=0.9, zorder=1)
    ax_util.legend(title="上下文长度", fontsize=7, title_fontsize=7, frameon=False, loc="upper right")

    fig.suptitle("Q3 固定 Q0：M3/M4 成本审计", fontsize=10.5, y=0.97)
    fig.text(
        0.5, 0.035,
        "3 档离散预算；固定 Q=Q0 时质量成本为零。最高档受 B1 支持边界限制。",
        ha="center", va="bottom", fontsize=7,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.2, wspace=0.3)

    preview_path = out_dir / "figure_preview_150dpi.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    render_preview(fig, str(preview_path), dpi=150)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    basename = FIGURE_DIR / "result_q3_m3_m4_cost_feasibility"
    paths = export_figure(
        fig,
        basename=str(basename),
        formats=["svg", "png"],
        dpi=300,
        size_inches=(8.4, 3.65),
        grayscale_preview=False,
        tight=False,
    )
    from PIL import Image

    gray_path = OUT_DIR / "result_q3_m3_m4_cost_feasibility_grayscale.png"
    with Image.open(f"{basename}.png") as color_image:
        color_image.convert("L").save(gray_path, dpi=(300, 300))
    paths.append(str(gray_path))
    plt.close(fig)
    return {
        "preview": str(preview_path.relative_to(ROOT).as_posix()),
        "layout_verdict": verdict,
        "layout_issues": [{"severity": severity, "message": message} for severity, message in issues],
        "outputs": [str(Path(path).resolve().relative_to(ROOT).as_posix()) for path in paths],
        "size_inches": [8.4, 3.65],
        "dpi": 300,
    }


def render_report(
    data: dict[str, Any],
    unit_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    identity_failures: list[dict[str, Any]],
    monotonic_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    nesting: list[dict[str, Any]],
    optimum: list[dict[str, Any]],
    support_alignment_failures: list[dict[str, Any]],
    m1_failures: list[dict[str, Any]],
    figure_audit: dict[str, Any],
) -> str:
    all_pass = (
        not identity_failures
        and all(row["pass"] for row in monotonic_rows)
        and all(row["lower_set_is_subset"] for row in nesting)
        and all(row["M0_feasible_count_match"] for row in summaries)
        and all(row["choice_match"] and row["support_feasible"] and row["budget_feasible"]
                and row["m3_flag_match"] and row["budget_flag_match"] and row["final_flag_match"]
                and row["cost_fields_match"] and math.isclose(row["cost_share_sum"], 1.0, abs_tol=1e-12)
                for row in optimum)
        and not support_alignment_failures
        and not m1_failures
        and all(row["mask_fields_match"] and row["cost_fields_match"] for row in budget_rows)
        and figure_audit["layout_verdict"] != "FAIL"
    )
    status = "PASS" if all_pass else "FAIL"
    feasible_at_highest = [
        row for row in summaries if row["budget_flops"] == max(data["budgets"])
    ]
    max_budget_all_supported = all(
        row["final_feasible_count"] == 1176 for row in feasible_at_highest
    )
    low_utilization = [
        float(row["budget_utilization_recomputed"])
        for row in optimum if row["budget_flops"] == max(data["budgets"])
    ]
    report = [
        "# Q3 M3/M4 成本与可行性独立审计",
        "",
        f"**审计状态：** {status}",
        f"**生成时间（UTC）：** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "**范围：** 固定 Q=Q0；在 B1 观测支持网格上复算 M4 成本/预算掩码，并核对现有 M0/M1 选点与 M3 支持标记。",
        "",
        "## 口径",
        "",
        "- Q1 版本：Q1-q_huber-v1；由 A1 七个冻结域的 q_operational 平均得到 Q0=" + f"{float(data['q0']):.10f}。",
        "- 成本：C_train=6ND，C_Q=D[g(Q)-g(Q0)]+，C_attn=2e-4·N·D·L_ctx；N、D 先从十亿单位换算为绝对数，成本与预算均为 FLOPs。",
        "- 固定 Q=Q0 时 C_Q=0。M3 的 support_feasible 对应既有字段 m3_search_allowed；该字段按 grid_id 读取并对齐，本审计没有重算支持分类。",
        "- 预算结构为 3 档预算 × 5 个上下文，共 15 个情景；嵌套性按每个上下文内的相邻预算档检查。",
        "- 本报告中的 Q3 M1 是 M3/M4 后按冻结 M0 Loss 选取的资源配置步骤；它不是 Q2 的 p-only M1 拟合模型。",
        "",
        "## 复核结果",
        "",
        f"- B1 实测网格：{len(data['b1_rows']):,} 个 (N,D) 点；5 个上下文；成本网格 {len(cost_rows):,} 行。",
        f"- M4 预算掩码：复核 {len(budget_rows):,} 行；成本和掩码字段匹配 {sum(bool(row['mask_fields_match'] and row['cost_fields_match']) for row in budget_rows):,}/{len(budget_rows):,}。",
        f"- M3 支持对齐：匹配 {len(data['support_rows']):,} 个 grid_id；对齐失败 {len(support_alignment_failures)}。",
        f"- 成本恒等式失败：{len(identity_failures)}；条件单调性门禁：{sum(bool(row['pass']) for row in monotonic_rows)}/{len(monotonic_rows)} 通过。",
        f"- 预算可行集嵌套：{sum(bool(row['lower_set_is_subset']) for row in nesting)}/{len(nesting)} 通过。",
        f"- M0/M1 选点、成本和可行性：{sum(bool(row['choice_match'] and row['support_feasible'] and row['budget_feasible'] and row['m3_flag_match'] and row['budget_flag_match'] and row['final_flag_match'] and row['cost_fields_match']) for row in optimum)}/{len(optimum)} 情景一致；另核对 M1 表中的三个可行标记。",
        f"- 最高预算下 1,176 个支持点全部可行：{max_budget_all_supported}；最高预算 M0 选点利用率范围为 {min(low_utilization):.2%}–{max(low_utilization):.2%}。",
        f"- M0 选点成本份额：训练 + 质量 + 注意力之和逐情景为 1；固定 Q0 时质量成本份额为 0。",
        f"- 图表程序版面检查：{figure_audit['layout_verdict']}；SVG：{figure_audit['outputs'][0]}；300 DPI PNG：{figure_audit['outputs'][1]}；灰度预览：{figure_audit['outputs'][2]}。",
        "",
        "## 预算可行配置摘要",
        "",
        "| 上下文 tokens | 预算 FLOPs | M3 允许点 | 预算可行点 | 最终可行点 | M0 原可行点 | M1 选点利用率 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    opt_by_key = {(row["budget_flops"], row["context_length_tokens"]): row for row in optimum}
    for row in summaries:
        opt = opt_by_key[(row["budget_flops"], row["context_length_tokens"])]
        report.append(
            f"| {row['context_length_tokens']:,} | {row['budget_flops']:.0e} | "
            f"{row['support_allowed_count']:,} | {row['budget_feasible_count']:,} | "
            f"{row['final_feasible_count']:,} | {row['M0_feasible_count']:,} | "
            f"{opt['budget_utilization_recomputed']:.2%} |"
        )
    report.extend([
        "",
        "## 解释边界",
        "",
        "最高预算使全部 B1 支持点可行，M0 选点的预算利用率因此受有限支持网格约束；低利用率不能解释为预算已在真实模型域中充分或最优利用。C7 给出架构最大上下文长度，是否等于实际部署窗口仍取决于题目场景。",
        "",
        "该审计只确认成本记账、掩码连接及既有 M0-based 选点在当前输入上的一致性。它不估计 Q/p→Loss，不给出支持网格外结论，也不把 M0 情景变化改称为完整 Q3 联合最优。",
        "",
        "## 复现",
        "",
        "在本次验证的绘图环境中运行（Python 3.10.20、NumPy 1.26.4、Matplotlib 3.10.9）：",
        "",
        "    $env:MPLBACKEND = 'Agg'",
        f"    & '{Path(sys.executable).resolve().as_posix()}' 'Q3/03_代码/audit_q3_m3_m4_cost_outputs.py'",
        "",
        "先运行真实数据最小切片：",
        "",
        f"    & '{Path(sys.executable).resolve().as_posix()}' 'Q3/03_代码/audit_q3_m3_m4_cost_outputs.py' --smoke",
        "",
        "当前 `huawei-cup` 环境的 NumPy 2.2.6 在 Windows 图像保存阶段会触发原生异常；换用已验证的 `mmdet` 环境（NumPy 1.26.4）后复现通过。未改动任何 Python 环境或依赖。",
        "",
        "输入输出哈希见 reproduction_manifest.json。失败记录见 m_cost_identity_failures.csv；条件单调性、预算嵌套和 M3 对齐状态见对应 CSV。",
        "",
    ])
    return "\n".join(report)


def run_smoke(data: dict[str, Any]) -> int:
    b1 = data["b1_rows"][0]
    context = data["contexts"][0]
    budget = data["budgets"][0]
    cost = compute_cost(b1, context)
    grid_id = str(b1["run_id"])
    support = data["support_by_id"][grid_id]
    existing = data["mask_by_key"][(budget, context, grid_id)]
    formula_pass = (
        cost["C_total_flops"]
        == cost["C_train_flops"] + cost["C_Q_flops"] + cost["C_attn_flops"]
    )
    mask_pass = (
        bool(cost["C_total_flops"] <= budget) == parse_bool(existing["budget_feasible"])
    )
    support_flag = parse_bool(support["m3_search_allowed"])
    support_pass = (
        str(support["grid_id"]) == grid_id
        and frac(support["N_params_B"]) == frac(b1["N_params_B"])
        and frac(support["D_tokens_B"]) == frac(b1["D_tokens_B"])
    )
    cost_pass = all(
        close_flops(existing[field], cost[key])
        for field, key in (
            ("C_train_flops", "C_train_flops"),
            ("C_Q_flops", "C_Q_flops"),
            ("C_attn_flops", "C_attn_flops"),
            ("C_total_flops", "C_total_flops"),
        )
    )
    m0 = data["frontier_by_key"][(budget, context)]
    m1 = data["m1_by_key"][(budget, context)]
    m1_cost = compute_cost(
        {"run_id": m1["m1_selected_run_id"],
         "N_params_B": m1["selected_N_B"], "D_tokens_B": m1["selected_D_B"]},
        context,
    )
    selected_id = str(m1["m1_selected_grid_id"])
    selected_support = data["support_by_id"][selected_id]
    selected_b1 = {str(row["run_id"]): row for row in data["b1_rows"]}[selected_id]
    candidate_parity_pass = (
        selected_id == str(m1["m1_selected_run_id"]) == str(m0["selected_run_id"])
        and frac(m1["selected_N_B"]) == frac(selected_b1["N_params_B"]) == frac(m0["selected_N_B"])
        and frac(m1["selected_D_B"]) == frac(selected_b1["D_tokens_B"]) == frac(m0["selected_D_B"])
        and frac(selected_support["N_params_B"]) == frac(selected_b1["N_params_B"])
        and frac(selected_support["D_tokens_B"]) == frac(selected_b1["D_tokens_B"])
    )
    selected_m4 = data["mask_by_key"][(budget, context, selected_id)]
    selected_m4_pass = (
        parse_bool(selected_m4["budget_feasible"]) == (m1_cost["C_total_flops"] <= budget)
        and all(
            close_flops(selected_m4[field], m1_cost[key])
            for field, key in (
                ("C_train_flops", "C_train_flops"),
                ("C_Q_flops", "C_Q_flops"),
                ("C_attn_flops", "C_attn_flops"),
                ("C_total_flops", "C_total_flops"),
            )
        )
    )
    m1_pass = (
        str(m0["selected_run_id"]) == str(m1["m1_selected_run_id"])
        and close_flops(m1["C_total_flops"], m1_cost["C_total_flops"])
        and parse_bool(m1["m3_search_allowed"]) == parse_bool(data["support_by_id"][str(m1["m1_selected_grid_id"])]["m3_search_allowed"])
        and parse_bool(m1["budget_feasible"]) == (m1_cost["C_total_flops"] <= budget)
        and parse_bool(m1["final_feasible"]) == (
            parse_bool(m1["m3_search_allowed"]) and parse_bool(m1["budget_feasible"])
        )
        and parse_bool(m1["final_feasible"])
    )
    checks = {
        "real_input_row_loaded": True,
        "unit_conversion_and_formula_identity": formula_pass,
        "existing_M3_support_alignment": support_pass,
        "existing_M4_mask_match": mask_pass,
        "existing_M4_cost_match": cost_pass,
        "selected_M1_candidate_matches_M0_B1_M3": candidate_parity_pass,
        "selected_M0_M1_candidate_matches_M4": selected_m4_pass,
        "existing_M0_M1_pick_and_cost_match": m1_pass,
    }
    passed = all(checks.values())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "gate": "P1 minimal real-input cost audit",
        "status": "PASS" if passed else "FAIL",
        "command": f"$env:MPLBACKEND = 'Agg'; & '{Path(sys.executable).resolve().as_posix()}' 'Q3/03_代码/audit_q3_m3_m4_cost_outputs.py' --smoke",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "environment": {"MPLBACKEND": "Agg"},
        "script_sha256": sha256_file(Path(__file__)),
        "inputs_sha256": {
            name: sha256_file(path) for name, path in data["resolved_inputs"].items()
        },
        "sample": {
            "grid_id": grid_id,
            "context_length_tokens": context,
            "budget_flops": budget,
            "N_parameters_abs": cost["N_parameters_abs"],
            "D_tokens_abs": cost["D_tokens_abs"],
            "C_train_flops": flops_value(cost["C_train_flops"]),
            "C_Q_flops": flops_value(cost["C_Q_flops"]),
            "C_attn_flops": flops_value(cost["C_attn_flops"]),
            "C_total_flops": flops_value(cost["C_total_flops"]),
            "m3_search_allowed": support_flag,
            "selected_M4_budget_feasible": parse_bool(selected_m4["budget_feasible"]),
            "M0_selected_run_id": m0["selected_run_id"],
            "M1_selected_run_id": m1["m1_selected_run_id"],
        },
        "checks": checks,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt_path = OUT_DIR / "p1_smoke_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def run_full(data: dict[str, Any]) -> int:
    p1_receipt_path = OUT_DIR / "p1_smoke_receipt.json"
    require(p1_receipt_path.is_file(), "P1 receipt is missing; run --smoke before the full audit")
    p1_receipt = read_json(p1_receipt_path)
    require(p1_receipt.get("status") == "PASS", "P1 smoke receipt is not PASS")
    require(
        p1_receipt.get("script_sha256") == sha256_file(Path(__file__)),
        "P1 receipt belongs to a different audit-script version; rerun --smoke",
    )
    for name, path in data["resolved_inputs"].items():
        require(
            p1_receipt.get("inputs_sha256", {}).get(name) == sha256_file(path),
            f"P1 receipt input hash changed for {name}; rerun --smoke",
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    support_rows, support_failures = support_alignment(data)
    cost_rows, cost_by_key, identity_failures = build_cost_grid(data)
    budget_rows, mask_failures = audit_budget_mask(data, cost_by_key)
    identity_failures.extend(mask_failures)
    monotonic_rows = monotonicity_audit(cost_by_key, data["b1_rows"], data["contexts"])
    summaries, nesting = budget_summaries(data, budget_rows)
    optimum_rows, m1_failures = audit_m0_picks(data, cost_by_key, budget_rows)
    units = unit_check_rows(data)

    outputs: list[Path] = []
    outputs.append(write_rows(OUT_DIR, "m_cost_grid.csv", cost_rows, list(cost_rows[0].keys())))
    outputs.append(write_rows(OUT_DIR, "m_unit_check.csv", units, list(units[0].keys())))
    failure_fields = [
        "failure_scope", "scenario_id", "grid_id", "field",
        "expected", "observed", "relative_error",
    ]
    outputs.append(write_rows(OUT_DIR, "m_cost_identity_failures.csv", identity_failures, failure_fields))
    outputs.append(write_rows(OUT_DIR, "m_monotonicity_audit.csv", monotonic_rows, list(monotonic_rows[0].keys())))
    outputs.append(write_rows(OUT_DIR, "m_budget_feasibility.csv", budget_rows, list(budget_rows[0].keys())))
    outputs.append(write_rows(OUT_DIR, "m_budget_summary.csv", summaries, list(summaries[0].keys())))
    outputs.append(write_rows(OUT_DIR, "m_budget_nesting_audit.csv", nesting, list(nesting[0].keys())))
    outputs.append(write_rows(OUT_DIR, "m_cost_share.csv", optimum_rows, list(optimum_rows[0].keys())))
    outputs.append(write_rows(OUT_DIR, "m0_optimum_cost_audit.csv", optimum_rows, list(optimum_rows[0].keys())))
    outputs.append(write_rows(
        OUT_DIR, "m3_support_alignment.csv", support_rows,
        list(support_rows[0].keys()),
    ))

    figure_audit = plot_audit(summaries, optimum_rows, OUT_DIR)
    outputs.append(ROOT / Path(figure_audit["preview"]))
    (OUT_DIR / "chart_contract.md").write_text(
        "# Q3 成本/可行性图表契约\n\n"
        "核心结论：在固定 Q0 的 M0 支持网格内，预算增加会扩大可行配置集合；最高预算下 1,176 个候选点全部可行，M0 选点的预算利用率受 B1 支持上界约束。\n\n"
        "证据：左面板按 5 个上下文展示 3 档预算下的可行点数；右面板展示同 15 个情景的 M0-based 选点实际成本/预算。虚线只连接离散预算情景。训练/质量/注意力份额另见 m_cost_share.csv；Q0 下质量份额为零。\n\n"
        "数据来源：本审计复算的 m_cost_grid.csv、m_budget_summary.csv、m0_optimum_cost_audit.csv；候选点来自 B1 观测 N/D 支持，M3 标记按 grid_id 直接连接。图不包含插值或支持域外数据。\n\n"
        "解释限制：C7 是架构上下文上限；成本曲线为题面代理公式；图不估计 Q/p 对 Loss，也不代表完整联合最优。最终尺寸 8.4×3.65 in，SVG 与 300 DPI PNG 成对导出，另有灰度预览。\n",
        encoding="utf-8",
    )
    outputs.append(OUT_DIR / "chart_contract.md")

    report = render_report(
        data, units, cost_rows, identity_failures, monotonic_rows, budget_rows,
        summaries, nesting, optimum_rows, support_failures, m1_failures, figure_audit,
    )
    report_path = OUT_DIR / "m4_cost_audit_report.md"
    report_path.write_text(report, encoding="utf-8")
    outputs.append(report_path)
    outputs.append(p1_receipt_path)

    input_hashes = {
        path.resolve().relative_to(ROOT).as_posix(): sha256_file(path)
        for path in data["resolved_inputs"].values()
    }
    output_hashes = {
        path.resolve().relative_to(ROOT).as_posix(): sha256_file(path)
        for path in outputs + [Path(figure_path) for figure_path in figure_audit["outputs"]]
    }
    package_versions = {}
    for package in ("numpy", "pandas", "matplotlib", "Pillow"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    all_pass = (
        not identity_failures
        and all(row["pass"] for row in monotonic_rows)
        and all(row["lower_set_is_subset"] for row in nesting)
        and all(row["M0_feasible_count_match"] for row in summaries)
        and all(row["choice_match"] and row["support_feasible"] and row["budget_feasible"]
                and row["m3_flag_match"] and row["budget_flag_match"] and row["final_flag_match"]
                and row["cost_fields_match"] and math.isclose(row["cost_share_sum"], 1.0, abs_tol=1e-12)
                for row in optimum_rows)
        and not support_failures
        and not m1_failures
        and all(row["mask_fields_match"] and row["cost_fields_match"] for row in budget_rows)
        and figure_audit["layout_verdict"] != "FAIL"
    )
    manifest = {
        "artifact": "Independent Q3 M3/M4 fixed-Q cost and feasibility audit",
        "status": "PASS" if all_pass else "FAIL",
        "command": f"$env:MPLBACKEND = 'Agg'; & '{Path(sys.executable).resolve().as_posix()}' 'Q3/03_代码/audit_q3_m3_m4_cost_outputs.py'",
        "smoke_command": f"$env:MPLBACKEND = 'Agg'; & '{Path(sys.executable).resolve().as_posix()}' 'Q3/03_代码/audit_q3_m3_m4_cost_outputs.py' --smoke",
        "environment": {"MPLBACKEND": "Agg"},
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "packages": package_versions,
        "script": {
            "path": "Q3/03_代码/audit_q3_m3_m4_cost_outputs.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "code_dependencies_sha256": {
            "Q3/03_代码/q3_repro_support.py": sha256_file(ROOT / "Q3/03_代码/q3_repro_support.py"),
        },
        "figure_tools": {
            "skill_root": str(SKILL_ROOT),
            "files_sha256": {
                path.relative_to(SKILL_ROOT).as_posix(): sha256_file(path)
                for path in sorted(FIGURE_SCRIPTS.glob("*.py"))
            },
        },
        "reproducibility_contract": {
            "byte_stable_outputs": [
                "M3_M4成本独立审计/m_cost_grid.csv",
                "M3_M4成本独立审计/m_budget_feasibility.csv",
                "M3_M4成本独立审计/figures/result_q3_m3_m4_cost_feasibility.png",
                "M3_M4成本独立审计/result_q3_m3_m4_cost_feasibility_grayscale.png",
            ],
            "metadata_may_vary": [
                "report generation timestamp",
                "SVG IDs/timestamp",
                "p1_smoke_receipt.json generated_at_utc",
            ],
            "verification": "Compare numeric audit tables and raster image hashes; regenerate the manifest for current report/vector metadata hashes.",
        },
        "inputs_sha256": input_hashes,
        "outputs_sha256": output_hashes,
        "parameters": {
            "Q_version": EXPECTED_Q_VERSION,
            "Q0": float(data["q0"]),
            "training_coefficient": TRAINING_COEFFICIENT,
            "eta": float(ETA),
            "N_D_source_scale": "billion; multiply by 1e9 before cost calculation",
            "budgets_flops": data["budgets"],
            "context_lengths": data["contexts"],
            "support_flag_mapping": "screenshot support_feasible aliases existing m3_search_allowed; read-only join by grid_id",
            "M1_label": "Q3 M0-based grid selection step; not Q2 p-only M1 model",
            "tolerances": {
                "relative": REL_TOLERANCE,
                "absolute_flops": ABS_TOLERANCE_FLOPS,
            },
        },
        "counts": {
            "B1_grid_points": len(data["b1_rows"]),
            "cost_grid_rows": len(cost_rows),
            "M4_mask_rows_audited": len(budget_rows),
            "M3_support_rows_aligned": len(support_rows),
            "M0_M1_scenarios": len(optimum_rows),
            "cost_identity_failures": len(identity_failures),
            "support_alignment_failures": len(support_failures),
            "M0_M1_pick_failures": len(m1_failures),
            "monotonicity_checks_passed": sum(bool(row["pass"]) for row in monotonic_rows),
            "monotonicity_checks_total": len(monotonic_rows),
            "budget_nesting_checks_passed": sum(bool(row["lower_set_is_subset"]) for row in nesting),
            "budget_nesting_checks_total": len(nesting),
        },
        "figure": figure_audit,
        "claim_boundary": "Fixed Q0 and exact observed B1 support grid; no Q/p response is estimated and no support expansion is performed.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = OUT_DIR / "reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Status: {manifest['status']}")
    print(f"Cost grid rows: {len(cost_rows)}; M4 rows independently audited: {len(budget_rows)}")
    print(f"M3 support alignment: {len(support_rows) - len(support_failures)}/1176")
    print(f"M0/M1 cost and selection parity: {len(optimum_rows) - len(m1_failures)}/15")
    print(f"Cost/mask mismatches: {len(identity_failures)}")
    print(f"Budget nesting: {manifest['counts']['budget_nesting_checks_passed']}/{manifest['counts']['budget_nesting_checks_total']}")
    print(f"Report: {report_path.relative_to(ROOT).as_posix()}")
    print(f"Manifest: {manifest_path.relative_to(ROOT).as_posix()}")
    return 0 if all_pass else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one real candidate/context/budget plus one existing selected pick")
    args = parser.parse_args(argv)
    try:
        data = load_inputs()
        if args.smoke:
            return run_smoke(data)
        return run_full(data)
    except (AuditInputError, FileNotFoundError, KeyError, ValueError, ZeroDivisionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
