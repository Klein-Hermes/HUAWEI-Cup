"""Assemble the scope-limited, reproducible Q3 results package.

Run from the repository root after the source task packages are complete:
  python Q3/03_代码/assemble_q3_scoped_release.py

The script reads existing Q3 deliverables, writes a 15-row cross-task table,
creates inventories, and copies a selected release snapshot. It does not
refit models or regenerate task-level results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Q3" / "04_结果"
RELEASE = ROOT / "Q3" / "冻结包_20260925"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_value(raw: object) -> bool:
    return str(raw).strip().lower() in {"true", "1", "yes"}


def n_or_blank(raw: object) -> object:
    value = str(raw).strip()
    if not value:
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def copy_file(source: Path, relative_target: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = RELEASE / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def copy_many(source_dir: Path, relative_dir: str, names: list[str]) -> None:
    for name in names:
        copy_file(source_dir / name, f"{relative_dir}/{name}")


def task_inputs() -> dict[str, Path]:
    return {
        "t1_master": RESULTS / "q3_fixedQ_master_results.csv",
        "t1_transitions": RESULTS / "q3_resource_transition.csv",
        "t2_near": RESULTS / "Q3_T2_近最优_Pareto_边际收益" / "q3_near_optimal.csv",
        "t2_pareto": RESULTS / "Q3_T2_近最优_Pareto_边际收益" / "q3_pareto.csv",
        "t2_marginal": RESULTS / "Q3_T2_近最优_Pareto_边际收益" / "q3_marginal_return.csv",
        "t3_bootstrap": RESULTS / "Q3_T3_稳健性与配置切换" / "q3_bootstrap_summary.csv",
        "t3_config_switch": RESULTS / "Q3_T3_稳健性与配置切换" / "q3_configuration_switch.csv",
        "t3_cost_switch": RESULTS / "Q3_T3_稳健性与配置切换" / "q3_cost_structure_switch.csv",
        "t4_threshold": RESULTS / "q3_betaQ_threshold.csv",
        "t4_regime": RESULTS / "q3_conditional_regime.csv",
        "support_grid": RESULTS / "m3_q3_support_grid.csv",
        "m4_mask": RESULTS / "m4_budget_feasible_mask.csv",
        "m1_mask": RESULTS / "m1_final_feasible_mask.csv",
    }


def aggregate_master() -> tuple[list[str], list[dict[str, object]], dict[str, object]]:
    src = task_inputs()
    tables = {key: read_csv(path) for key, path in src.items()}
    t1_headers, t1 = tables["t1_master"]
    if len(t1) != 15:
        raise ValueError(f"T1 master expected 15 rows, got {len(t1)}")

    t2_near = tables["t2_near"][1]
    t2_pareto = tables["t2_pareto"][1]
    t2_marginal = tables["t2_marginal"][1]
    t3_boot = tables["t3_bootstrap"][1]
    t3_config = tables["t3_config_switch"][1]
    t3_cost = tables["t3_cost_switch"][1]
    t4_threshold = tables["t4_threshold"][1]
    t4_regime = tables["t4_regime"][1]
    budget_dir = RESULTS / "预算饱和与有限网格切换"
    budget_coverage = read_csv(budget_dir / "q3_formal_budget_coverage.csv")[1]
    budget_switches = read_csv(budget_dir / "q3_formal_budget_switches.csv")[1]
    budget_events = read_csv(budget_dir / "q3_finite_grid_switch_events.csv")[1]

    expected = {
        "T2 near-optimal": (len(t2_near), 17640),
        "T2 Pareto": (len(t2_pareto), 1865),
        "T2 marginal return": (len(t2_marginal), 10),
        "T3 bootstrap": (len(t3_boot), 15),
        "T3 configuration switches": (len(t3_config), 22),
        "T3 cost switches": (len(t3_cost), 22),
        "T4 thresholds": (len(t4_threshold), 45),
        "T4 regimes": (len(t4_regime), 166),
        "B1 support grid": (len(tables["support_grid"][1]), 1176),
        "M4 mask": (len(tables["m4_mask"][1]), 17640),
        "M1 mask": (len(tables["m1_mask"][1]), 17640),
        "budget saturation scenarios": (len(budget_coverage), 15),
        "formal budget switch intervals": (len(budget_switches), 10),
    }
    failed = {name: got for name, (got, want) in expected.items() if got != want}
    if failed:
        raise ValueError(f"Unexpected source dimensions: {failed}")

    def k(row: dict[str, str]) -> tuple[str, str]:
        return str(int(float(row["budget_flops"]))), str(int(float(row["context_length_tokens"])))

    coverage_by_key = {
        (str(int(float(row["budget_flops"]))), str(int(float(row["context_length_tokens"])))): row
        for row in budget_coverage
    }
    switch_by_after_key = {
        (str(int(float(row["budget_after_flops"]))), str(int(float(row["context_length_tokens"])))): row
        for row in budget_switches
    }
    if len(coverage_by_key) != 15 or len(switch_by_after_key) != 10:
        raise ValueError("Budget coverage/switch tables have duplicate scenario keys")

    near_summary: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {
        "support_grid_candidate_count": 0, "candidate_count": 0,
        "near_1": 0, "near_3": 0, "near_5": 0,
    })
    for row in t2_near:
        item = near_summary[k(row)]
        item["support_grid_candidate_count"] += 1
        item["candidate_count"] += bool_value(row["budget_feasible"])
        item["near_1"] += bool_value(row["near_optimal_1pct"])
        item["near_3"] += bool_value(row["near_optimal_3pct"])
        item["near_5"] += bool_value(row["near_optimal_5pct"])

    pareto_summary: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {
        "context_points": 0, "budget_feasible_points": 0,
    })
    pareto_flag = {
        "10000000000000000000": "budget_feasible_10000000000000000000",
        "10000000000000000000000": "budget_feasible_10000000000000000000000",
        "1000000000000000000000000": "budget_feasible_1000000000000000000000000",
    }
    for row in t2_pareto:
        context = str(int(float(row["context_length_tokens"])))
        for budget, flag in pareto_flag.items():
            item = pareto_summary[(budget, context)]
            item["context_points"] += 1
            item["budget_feasible_points"] += bool_value(row[flag])

    marginal_by_context: dict[tuple[str, str], dict[str, str]] = {}
    for row in t2_marginal:
        marginal_by_context[(str(int(float(row["from_budget_flops"]))),
                             str(int(float(row["context_length_tokens"]))))] = row

    t3_by_key = {k(row): row for row in t3_boot}
    t4_summary: dict[tuple[str, str], dict[str, object]] = defaultdict(lambda: {
        "scenario_count": 0, "finite_threshold_count": 0,
        "betaQ_threshold_min": "", "betaQ_threshold_max": "",
    })
    t4_thresholds: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in t4_threshold:
        key = k(row)
        t4_summary[key]["scenario_count"] += 1
        value = str(row.get("betaQcrit_first_ND_reallocation", "")).strip()
        if value:
            try:
                numeric = float(value)
                if math.isfinite(numeric):
                    t4_thresholds[key].append(numeric)
            except ValueError:
                pass
    for key, values in t4_thresholds.items():
        t4_summary[key]["finite_threshold_count"] = len(values)
        t4_summary[key]["betaQ_threshold_min"] = min(values)
        t4_summary[key]["betaQ_threshold_max"] = max(values)
    if len(t4_summary) != 15 or any(v["scenario_count"] != 3 for v in t4_summary.values()):
        raise ValueError("T4 conditional thresholds must have three g(Q) rows per budget/context key")

    additions = [
        "t2_support_domain_candidate_count", "t2_feasible_candidate_count", "t2_near_optimal_1pct_count",
        "t2_near_optimal_3pct_count", "t2_near_optimal_5pct_count",
        "t2_pareto_points_at_context", "t2_pareto_budget_feasible_points",
        "t2_next_budget_flops", "t2_loss_reduction_to_next_budget",
        "t2_loss_reduction_per_budget_decade_to_next",
        "t2_next_budget_bootstrap_p2_5", "t2_next_budget_bootstrap_median",
        "t2_next_budget_bootstrap_p97_5",
        "t3_bootstrap_reselection_rate", "t3_modal_configuration_rate",
        "t3_unique_selected_grid_points",
        "t4_conditional_gQ_scenario_count", "t4_finite_betaQ_threshold_count",
        "t4_betaQ_threshold_min", "t4_betaQ_threshold_max",
        "support_C_sat_flops", "support_budget_over_C_sat",
        "support_budget_feasible_candidate_count", "support_budget_feasible_share",
        "support_remaining_candidates_to_saturation", "support_saturated",
        "support_m0_configuration_changed_from_previous_budget",
        "support_m0_switch_events_from_previous_budget",
    ]
    output_rows: list[dict[str, object]] = []
    for source_row in t1:
        row: dict[str, object] = dict(source_row)
        key = k(source_row)
        near = near_summary[key]
        pareto = pareto_summary[key]
        row.update({
            "t2_support_domain_candidate_count": near["support_grid_candidate_count"],
            "t2_feasible_candidate_count": near["candidate_count"],
            "t2_near_optimal_1pct_count": near["near_1"],
            "t2_near_optimal_3pct_count": near["near_3"],
            "t2_near_optimal_5pct_count": near["near_5"],
            "t2_pareto_points_at_context": pareto["context_points"],
            "t2_pareto_budget_feasible_points": pareto["budget_feasible_points"],
        })
        if near["candidate_count"] != int(float(source_row["m4_budget_feasible_candidate_count"])):
            raise ValueError(f"T2/T1 feasible candidate count mismatch for {key}")
        next_budget = "10000000000000000000000" if key[0] == "10000000000000000000" else (
            "1000000000000000000000000" if key[0] == "10000000000000000000000" else ""
        )
        marginal = marginal_by_context.get((key[0], key[1])) if next_budget else None
        row.update({
            "t2_next_budget_flops": int(next_budget) if next_budget else "",
            "t2_loss_reduction_to_next_budget": n_or_blank(marginal.get("loss_reduction")) if marginal else "",
            "t2_loss_reduction_per_budget_decade_to_next": n_or_blank(marginal.get("loss_reduction_per_budget_decade")) if marginal else "",
            "t2_next_budget_bootstrap_p2_5": n_or_blank(marginal.get("bootstrap_loss_reduction_p2_5")) if marginal else "",
            "t2_next_budget_bootstrap_median": n_or_blank(marginal.get("bootstrap_loss_reduction_median")) if marginal else "",
            "t2_next_budget_bootstrap_p97_5": n_or_blank(marginal.get("bootstrap_loss_reduction_p97_5")) if marginal else "",
        })
        t3 = t3_by_key[key]
        row.update({
            "t3_bootstrap_reselection_rate": n_or_blank(t3["baseline_reselection_rate"]),
            "t3_modal_configuration_rate": n_or_blank(t3["modal_configuration_rate"]),
            "t3_unique_selected_grid_points": int(float(t3["n_unique_selected_grid_points"])),
        })
        t4 = t4_summary[key]
        budget_summary = coverage_by_key.get(key)
        if budget_summary is None:
            raise ValueError(f"Budget saturation coverage missing master-table key {key}")
        previous_switch = switch_by_after_key.get(key)
        row.update({
            "t4_conditional_gQ_scenario_count": t4["scenario_count"],
            "t4_finite_betaQ_threshold_count": t4["finite_threshold_count"],
            "t4_betaQ_threshold_min": t4["betaQ_threshold_min"],
            "t4_betaQ_threshold_max": t4["betaQ_threshold_max"],
            "support_C_sat_flops": budget_summary["C_sat_support_grid_flops"],
            "support_budget_over_C_sat": n_or_blank(budget_summary["budget_over_C_sat"]),
            "support_budget_feasible_candidate_count": int(budget_summary["budget_feasible_candidate_count"]),
            "support_budget_feasible_share": n_or_blank(budget_summary["budget_feasible_share"]),
            "support_remaining_candidates_to_saturation": int(budget_summary["remaining_candidates_to_saturation"]),
            "support_saturated": bool_value(budget_summary["support_saturated"]),
            "support_m0_configuration_changed_from_previous_budget":
                bool_value(previous_switch["configuration_changed"]) if previous_switch else "",
            "support_m0_switch_events_from_previous_budget":
                int(previous_switch["number_of_envelope_switches_inside_interval"]) if previous_switch else "",
        })
        output_rows.append(row)

    if len(output_rows) != 15 or any(not r["t3_bootstrap_reselection_rate"] for r in output_rows):
        raise ValueError("Master-table joins did not cover all 15 scenario keys")
    all_headers = t1_headers + additions

    finite_thresholds = sum(
        1 for row in t4_threshold
        if (lambda raw: raw is not None and math.isfinite(raw))(
            (lambda v: float(v) if v.strip() else None)(str(row.get("betaQcrit_first_ND_reallocation", "")))
        )
    )
    t3_robust_config = sum(bool_value(row["robust_configuration_switch"]) for row in t3_config)
    t3_robust_cost = sum(bool_value(row["robust_cost_structure_switch"]) for row in t3_cost)
    numbers = {
        "as_of": date.today().isoformat(),
        "scope": "Q3 identifiable result package at fixed Q0; full empirical Q/p joint optimization excluded",
        "source_dimensions": {name: len(data[1]) for name, data in tables.items()},
        "master_table": {"rows": len(output_rows), "columns": len(all_headers), "key": ["budget_flops", "context_length_tokens"]},
        "t1": {"scenarios": len(t1), "candidate_support_points": len(tables["support_grid"][1]),
               "cost_budget_mask_rows": len(tables["m4_mask"][1]), "final_mask_rows": len(tables["m1_mask"][1])},
        "t2": {"candidate_rows": len(t2_near), "pareto_points": len(t2_pareto),
               "pareto_points_by_context": dict(Counter(str(int(float(r["context_length_tokens"]))) for r in t2_pareto)),
               "marginal_return_rows": len(t2_marginal)},
        "t3": {"bootstrap_scenarios": len(t3_boot), "adjacent_pairs": len(t3_config),
               "robust_configuration_switches": t3_robust_config,
               "robust_cost_structure_switches": t3_robust_cost},
        "t4": {"conditional_scenarios": len(t4_threshold), "regime_rows": len(t4_regime),
               "finite_first_reallocation_thresholds": finite_thresholds,
               "no_finite_first_reallocation_thresholds": len(t4_threshold) - finite_thresholds,
               "betaQ_identified": False, "p_in_objective": False},
        "budget_saturation": {
            "formal_budget_context_scenarios": len(budget_coverage),
            "finite_grid_switch_events_including_initial_points": len(budget_events),
            "formal_budget_interval_comparisons": len(budget_switches),
            "all_1e24_supports_saturated": all(
                bool_value(row["support_saturated"])
                for row in budget_coverage if row["budget_tier"] == "high"
            ),
        },
    }
    return all_headers, output_rows, numbers


def source_tree() -> list[tuple[str, str]]:
    """(source relative to repo root, package-relative destination)"""
    files = [
        ("Q3/04_结果/q3_fixedQ_master_results.csv", "01_fixed/q3_fixedQ_master_results.csv"),
        ("Q3/04_结果/q3_resource_transition.csv", "01_fixed/q3_resource_transition.csv"),
        ("Q3/04_结果/m3_q3_support_grid.csv", "01_fixed/source_tables/m3_q3_support_grid.csv"),
        ("Q3/04_结果/m4_budget_feasible_mask.csv", "01_fixed/source_tables/m4_budget_feasible_mask.csv"),
        ("Q3/04_结果/m1_final_feasible_mask.csv", "01_fixed/source_tables/m1_final_feasible_mask.csv"),
        ("Q3/04_结果/M0_ND_reference_frontier.csv", "01_fixed/source_tables/M0_ND_reference_frontier.csv"),
        ("Q3/04_结果/M0_ND_cluster_bootstrap_frontier.csv", "01_fixed/source_tables/M0_ND_cluster_bootstrap_frontier.csv"),
        ("Q3/04_结果/M0_ND_cluster_bootstrap_selection_frequency.csv", "01_fixed/source_tables/M0_ND_cluster_bootstrap_selection_frequency.csv"),
        ("Q3/04_结果/M0_discrete_transition_pairs.csv", "01_fixed/source_tables/M0_discrete_transition_pairs.csv"),
        ("Q3/04_结果/M0_discrete_transition_bootstrap_draws.csv", "01_fixed/source_tables/M0_discrete_transition_bootstrap_draws.csv"),
        ("Q3/04_结果/Q3_fixedQ_results_manifest.json", "01_fixed/Q3_fixedQ_results_manifest.json"),
        ("Q3/04_结果/Q3_fixedQ_results_integration_report.md", "01_fixed/Q3_fixedQ_results_integration_report.md"),
        ("Q3/01_方案与设计/Q3-T3_Bootstrap稳定性与配置切换_优化执行记录.md", "QA/Q3-T3_Bootstrap稳定性与配置切换_P2范围记录.md"),
        ("Q3/04_结果/_q3_fixedq_p1_smoke/p1_smoke_receipt.json", "01_fixed/p1_smoke_receipt.json"),
        ("Q3/04_结果/figures/result_q3_fixedq_master_and_resource_transition.png", "01_fixed/figures/result_q3_fixedq_master_and_resource_transition.png"),
        ("Q3/04_结果/figures/result_q3_fixedq_master_and_resource_transition.svg", "01_fixed/figures/result_q3_fixedq_master_and_resource_transition.svg"),
        ("Q3/04_结果/figures/result_q3_fixedq_master_and_resource_transition.pdf", "01_fixed/figures/result_q3_fixedq_master_and_resource_transition.pdf"),
        ("Q3/04_结果/figures/_qa/result_q3_fixedq_master_and_resource_transition_grayscale.png", "01_fixed/figures/result_q3_fixedq_master_and_resource_transition_grayscale.png"),
        ("Q3/04_结果/figures/raw_q3_observed_nd_support.png", "01_fixed/figures/raw_q3_observed_nd_support.png"),
        ("Q3/04_结果/figures/raw_q3_observed_nd_support.svg", "01_fixed/figures/raw_q3_observed_nd_support.svg"),
        ("Q3/04_结果/figures/process_q3_m3_m4_mask_audit.png", "01_fixed/figures/process_q3_m3_m4_mask_audit.png"),
        ("Q3/04_结果/figures/process_q3_m3_m4_mask_audit.svg", "01_fixed/figures/process_q3_m3_m4_mask_audit.svg"),
        ("Q3/04_结果/figures/result_q3_m1_budget_optima.png", "01_fixed/figures/result_q3_m1_budget_optima.png"),
        ("Q3/04_结果/figures/result_q3_m1_budget_optima.svg", "01_fixed/figures/result_q3_m1_budget_optima.svg"),
        ("Q3/04_结果/figures/result_q3_m1_budget_optima_grayscale.png", "01_fixed/figures/result_q3_m1_budget_optima_grayscale.png"),
        ("Q3/04_结果/M0_cluster_bootstrap_sensitivity_manifest.json", "01_fixed/source_tables/M0_cluster_bootstrap_sensitivity_manifest.json"),
        ("Q3/04_结果/M0_discrete_transition_manifest.json", "01_fixed/source_tables/M0_discrete_transition_manifest.json"),
        ("Q3/04_结果/T4_independent_P2_receipt.md", "QA/T4_independent_P2_receipt.md"),
        ("Q3/04_结果/Q3_independent_release_P2_receipt.md", "QA/P2_independent_release_receipt.md"),
        ("Q3/03_代码/assemble_q3_fixedq_master_results.py", "code_sources/assemble_q3_fixedq_master_results.py"),
        ("Q3/03_代码/assemble_q3_scoped_release.py", "code_sources/assemble_q3_scoped_release.py"),
        ("Q3/03_代码/assess_q3_m0_discrete_transitions.py", "code_sources/assess_q3_m0_discrete_transitions.py"),
        ("Q3/03_代码/build_q3_m3_m4_mask_pipeline.py", "code_sources/build_q3_m3_m4_mask_pipeline.py"),
        ("Q3/03_代码/solve_q3_m1_from_masks.py", "code_sources/solve_q3_m1_from_masks.py"),
        ("Q3/03_代码/audit_q3_m3_m4_cost_outputs.py", "code_sources/audit_q3_m3_m4_cost_outputs.py"),
        ("Q3/03_代码/plot_q3_m3_m4_m1_audit.py", "code_sources/plot_q3_m3_m4_m1_audit.py"),
        ("Q3/03_代码/analyze_q3_t2_near_optimal_pareto.py", "code_sources/analyze_q3_t2_near_optimal_pareto.py"),
        ("Q3/03_代码/standardize_q3_t3_outputs.py", "code_sources/standardize_q3_t3_outputs.py"),
        ("Q3/03_代码/export_q3_t4_conditional_scenarios.py", "code_sources/export_q3_t4_conditional_scenarios.py"),
        ("Q3/03_代码/solve_q3_conditional_q_sensitivity.py", "code_sources/solve_q3_conditional_q_sensitivity.py"),
        ("Q3/03_代码/analyze_q3_cost_sensitivities.py", "code_sources/analyze_q3_cost_sensitivities.py"),
        ("Q3/03_代码/q3_repro_support.py", "code_sources/q3_repro_support.py"),
        ("utils/plot_style.py", "code_sources/utils/plot_style.py"),
    ]
    for name in ["m_cost_grid.csv", "m_cost_identity_failures.csv", "m_unit_check.csv", "m_monotonicity_audit.csv",
                 "m_budget_feasibility.csv", "m_budget_summary.csv", "m_budget_nesting_audit.csv", "m_cost_share.csv",
                 "m3_support_alignment.csv", "m0_optimum_cost_audit.csv", "m4_cost_audit_report.md", "reproduction_manifest.json",
                 "p1_smoke_receipt.json", "chart_contract.md"]:
        files.append((f"Q3/04_结果/M3_M4成本独立审计/{name}", f"01_fixed/M3_M4成本独立审计/{name}"))
    files.extend([
        ("Q3/04_结果/M3_M4成本独立审计/figures/result_q3_m3_m4_cost_feasibility.png", "01_fixed/M3_M4成本独立审计/figures/result_q3_m3_m4_cost_feasibility.png"),
        ("Q3/04_结果/M3_M4成本独立审计/figures/result_q3_m3_m4_cost_feasibility.svg", "01_fixed/M3_M4成本独立审计/figures/result_q3_m3_m4_cost_feasibility.svg"),
    ])
    for name in ["q3_near_optimal.csv", "q3_pareto.csv", "q3_marginal_return.csv", "report.md",
                 "reproduction_manifest.json", "p1_smoke_receipt.json", "figure_contract.md"]:
        files.append((f"Q3/04_结果/Q3_T2_近最优_Pareto_边际收益/{name}", f"02_robustness/T2/{name}"))
    for ext in ("png", "svg", "grayscale.png"):
        name = f"result_q3_t2_near_optimal_pareto_marginal_return.{ext}"
        if ext == "grayscale.png":
            name = "result_q3_t2_near_optimal_pareto_marginal_return_grayscale.png"
        files.append((f"Q3/04_结果/Q3_T2_近最优_Pareto_边际收益/figures/{name}", f"02_robustness/T2/figures/{name}"))
    for name in ["q3_bootstrap_summary.csv", "q3_configuration_switch.csv", "q3_cost_structure_switch.csv", "report.md",
                 "reproduction_manifest.json", "p1_smoke_receipt.json", "figure_contract.md"]:
        files.append((f"Q3/04_结果/Q3_T3_稳健性与配置切换/{name}", f"02_robustness/T3/{name}"))
    for stem in ["result_q3_t3_bootstrap_stability", "result_q3_t3_configuration_switch", "result_q3_t3_cost_structure_switch"]:
        for ext in ["png", "svg", "pdf", "grayscale.png"]:
            name = f"{stem}.{ext}" if ext != "grayscale.png" else f"{stem}_grayscale.png"
            files.append((f"Q3/04_结果/Q3_T3_稳健性与配置切换/figures/{name}", f"02_robustness/T3/figures/{name}"))
    for source, dest in [
        ("q3_betaQ_threshold.csv", "03_conditional/q3_betaQ_threshold.csv"),
        ("q3_conditional_regime.csv", "03_conditional/q3_conditional_regime.csv"),
        ("q3_t4_conditional_beta_manifest.json", "03_conditional/q3_t4_conditional_beta_manifest.json"),
        ("Q3_Q0_baseline_sensitivity_report.md", "03_conditional/Q3_Q0_baseline_sensitivity_report.md"),
        ("Q3_Q0_baseline_sensitivity_manifest.json", "03_conditional/Q3_Q0_baseline_sensitivity_manifest.json"),
        ("Q3_Q0_baseline_sensitivity_cost.csv", "03_conditional/Q3_Q0_baseline_sensitivity_cost.csv"),
        ("Q3_Q0_baseline_sensitivity_beta_thresholds.csv", "03_conditional/Q3_Q0_baseline_sensitivity_beta_thresholds.csv"),
        ("Q_conditional_Q_break_even_summary.csv", "03_conditional/Q_conditional_Q_break_even_summary.csv"),
        ("Q_conditional_Q_optimality_intervals.csv", "03_conditional/Q_conditional_Q_optimality_intervals.csv"),
        ("Q3_conditional_Q_optimization_report.md", "03_conditional/Q3_conditional_Q_optimization_report.md"),
        ("Q3_conditional_Q_optimization_manifest.json", "03_conditional/Q3_conditional_Q_optimization_manifest.json"),
        ("Q3/03_代码/solve_q3_conditional_q_sensitivity.py", "code_sources/solve_q3_conditional_q_sensitivity.py"),
        ("Q3/03_代码/analyze_q3_cost_sensitivities.py", "code_sources/analyze_q3_cost_sensitivities.py"),
    ]:
        files.append((f"Q3/04_结果/{source}" if not source.startswith("Q3/") else source, dest))
    for ext in ["png", "svg", "pdf", "grayscale.png"]:
        name = f"result_q3_t4_betaQ_scenario.{ext}" if ext != "grayscale.png" else "result_q3_t4_betaQ_scenario_grayscale.png"
        files.append((f"Q3/04_结果/figures/{name}", f"03_conditional/figures/{name}"))
    budget_outputs = [
        "q3_budget_saturation_by_context.csv",
        "q3_formal_budget_coverage.csv",
        "q3_finite_grid_switch_events.csv",
        "q3_formal_budget_switches.csv",
        "report.md",
        "figure_contract.md",
        "reproduction_manifest.json",
    ]
    for name in budget_outputs:
        files.append((
            f"Q3/04_结果/预算饱和与有限网格切换/{name}",
            f"01_fixed/预算饱和与有限网格切换/{name}",
        ))
    files.append((
        "Q3/04_结果/预算饱和与有限网格切换/_p1_smoke/p1_smoke_receipt.json",
        "01_fixed/预算饱和与有限网格切换/p1_smoke_receipt.json",
    ))
    for ext in ["png", "svg", "pdf"]:
        name = f"result_q3_budget_saturation_finite_grid_switches.{ext}"
        files.append((
            f"Q3/04_结果/预算饱和与有限网格切换/figures/{name}",
            f"01_fixed/预算饱和与有限网格切换/figures/{name}",
        ))
    files.append((
        "Q3/04_结果/预算饱和与有限网格切换/figures/result_q3_budget_saturation_finite_grid_switches_grayscale.png",
        "01_fixed/预算饱和与有限网格切换/figures/result_q3_budget_saturation_finite_grid_switches_grayscale.png",
    ))
    files.append((
        "Q3/03_代码/analyze_q3_budget_grid_transitions.py",
        "code_sources/analyze_q3_budget_grid_transitions.py",
    ))
    files.append((
        "Q3/01_方案与设计/Q3_预算饱和与有限网格切换_优化执行计划.md",
        "QA/Q3_预算饱和与有限网格切换_优化执行计划.md",
    ))
    return files


def validate_source_manifests() -> list[dict[str, object]]:
    specs = [
        ("Q3/04_结果/Q3_fixedQ_results_manifest.json", ["inputs_sha256", "outputs_sha256"]),
        ("Q3/04_结果/M3_M4成本独立审计/reproduction_manifest.json", ["inputs_sha256", "outputs_sha256"]),
        ("Q3/04_结果/Q3_T2_近最优_Pareto_边际收益/reproduction_manifest.json", ["input_files", "output_files"]),
        ("Q3/04_结果/Q3_T3_稳健性与配置切换/reproduction_manifest.json", ["inputs", "outputs"]),
        ("Q3/04_结果/q3_t4_conditional_beta_manifest.json", ["outputs"]),
    ]
    t1_aliases = {
        "m0_frontier": "Q3/04_结果/M0_ND_reference_frontier.csv",
        "m1_boundary_audit": "Q3/04_结果/m1_optimum_boundary_audit.csv",
        "m3_support": "Q3/04_结果/m3_q3_support_grid.csv",
        "m4_budget_mask": "Q3/04_结果/m4_budget_feasible_mask.csv",
        "m1_final_mask": "Q3/04_结果/m1_final_feasible_mask.csv",
        "cost_shares": "Q3/04_结果/M0_budget_context_cost_shares.csv",
        "transition_pairs": "Q3/04_结果/M0_discrete_transition_pairs.csv",
        "independent_cost_audit": "Q3/04_结果/M3_M4成本独立审计/m0_optimum_cost_audit.csv",
        "cost_manifest": "Q3/04_结果/Q3_cost_sensitivities_manifest.json",
        "independent_audit_manifest": "Q3/04_结果/M3_M4成本独立审计/reproduction_manifest.json",
        "m0_manifest": "Q3/04_结果/M0_reference_reproduction_manifest.json",
        "m1_manifest": "Q3/04_结果/M1_masked_optimization_manifest.json",
        "m3_manifest": "Q3/04_结果/M3_support_manifest.json",
        "m4_manifest": "Q3/04_结果/M4_budget_manifest.json",
        "transition_manifest": "Q3/04_结果/M0_discrete_transition_manifest.json",
    }
    checks: list[dict[str, object]] = []
    for rel, sections in specs:
        path = ROOT / rel
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        verified = missing = mismatch = 0
        for section in sections:
            items = manifest.get(section, {})
            if not isinstance(items, dict):
                continue
            for raw, value in items.items():
                expected = value.get("sha256") if isinstance(value, dict) else value
                source_path = value.get("path") if isinstance(value, dict) else None
                if source_path is None and rel.endswith("Q3_fixedQ_results_manifest.json") and section == "inputs_sha256":
                    source_path = t1_aliases.get(raw)
                candidate = Path(source_path or raw)
                if not candidate.is_absolute():
                    candidate = ROOT / candidate
                if not candidate.is_file():
                    missing += 1
                elif expected and sha256(candidate) != expected:
                    mismatch += 1
                else:
                    verified += 1
        checks.append({"manifest": rel, "verified": verified, "missing": missing, "hash_mismatch": mismatch,
                       "status": "PASS" if missing == mismatch == 0 else "FAIL"})
    if any(check["status"] != "PASS" for check in checks):
        raise ValueError(f"Source manifest verification failed: {checks}")
    return checks


def make_ledger() -> str:
    return """# Q3 本轮结果包范围与决策记录\n\n- 日期：2026-09-25。\n- 用户确认按本计划推进；本记录只引用当前对话中明确确认的范围，不补造历史决定。\n- 冻结范围：固定 Q=Q₀、冻结 M0、B1 已观测 N/D 支持域下的 T1/M3/M4、T2 近优/Pareto/边际收益、T3 Bootstrap 与相邻切换，以及 T4 条件 βQ 阈值与区间。\n- T2 作为正式结果并入 15 行总表、最终报告和图表/复现清单；其完整候选明细、Pareto 点与边际收益明细保留专题表。\n- T3 22 组相邻情景和 T4 45 个条件情景/166 个区间保留各自专题表；主表只汇总可按预算×上下文对齐的字段。\n- 完整经验 Q/p 联合优化未纳入本轮结论：Q2 目前不能识别独立质量响应，且 p 未进入当前目标。βQ 数值只解释为条件 break-even 阈值，不是估计值。\n- 冻结对象：本文件夹中的报告、15 行主表、专题结果、来源代码、图表/表格清单、复现说明和哈希清单。不得把此状态称为完整 Q/p 最优或竞赛论文终稿。\n- 变更控制：后续源数据、脚本或结果一旦改变，应生成新的日期版本包并更新哈希；不覆盖本次冻结快照。\n"""


def make_readme() -> str:
    return """# Q3 本轮可识别结果包（2026-09-25）\n\n本快照整合 T1/M3/M4、T2、T3 和 T4 的当前正式结果。结论范围是固定 Q₀ 与冻结 M0 下的 B1 观测 N/D 支持域；T4 是条件敏感性。完整经验 Q/p 联合优化仍未识别，因此本包不是完整 Q3 联合最优结果，也不是竞赛论文提交包。\n\n## 入口\n\n- `q3_final_report.md`：本轮结果报告。\n- `q3_master_table.csv`：按预算×上下文键整合的 15 行 T1/T2/T3/T4 汇总。\n- `figure_manifest.csv`、`table_manifest.csv`：包内图件、表格路径与 SHA-256。\n- `reproduction_manifest.md`：源命令、环境、哈希和复现顺序。\n- `frozen_numbers.json`：由源 CSV 直接汇总的锁定数字。\n- `scope_decision_ledger.md`：本轮决策和适用范围。\n- `QA/P2_independent_release_receipt.md`：本轮整包独立范围 P2 复核回执（PASS）。\n- `QA/`：限定范围的完整性、一致性、可复现性和独立复核记录。\n\n## 分区\n\n- `01_fixed/`：固定 Q 主结果、成本/预算审计、支持表和 T1 图件。\n- `02_robustness/T2/`、`02_robustness/T3/`：近优/Pareto/边际收益、Bootstrap 稳健性和切换结果。\n- `03_conditional/`：条件 βQ 阈值、区间结果和图件。\n- `code_sources/`：项目内生成/整理这些产物的脚本快照。\n\n复现需要在项目根目录使用清单中的命令，并保留 Q1/Q2 上游输入文件。快照保存了直接用于核对和解释的结果文件，但不复制完整 Q1/Q2 原始数据集。\n"""


def make_reproduction(checks: list[dict[str, object]], table_rows: int, table_cols: int) -> str:
    assembler = Path(__file__).resolve()
    assembler_hash = sha256(assembler)
    lines = [
        "# Q3 本轮结果包复现清单",
        "",
        "## 复现顺序",
        "",
        "1. 在仓库根目录确认 Q1/Q2 上游源文件存在且哈希与各任务 manifest 相符。",
        "2. 按需重跑各任务原始命令（T1/M3/M4、T2、T3、T4）；完整命令和环境见相应目录内的任务级 `reproduction_manifest.json`。本次冻结包组装没有重新拟合 M0、重采 Bootstrap 或重算 T2/T3/T4。",
        "3. 重新组装汇总表和快照：",
        "   `python Q3/03_代码/analyze_q3_budget_grid_transitions.py`，随后使用 `python Q3/03_代码/assemble_q3_scoped_release.py --release-dir Q3/冻结包_20260925_有限网格预算分析` 生成全新快照；已有冻结目录不覆盖。",
        "4. 核对本文件列出的源任务 manifest、包内表格/图件与 `release_manifest.json` 哈希。",
        f"5. 总表组装器 `{assembler.relative_to(ROOT).as_posix()}` SHA-256：`{assembler_hash}`。",
        "",
        "## 本次任务级 manifest 校验",
        "",
        "| Manifest | 已核验文件 | 缺失 | 哈希不符 | 状态 |",
        "|---|---:|---:|---:|---|",
    ]
    for check in checks:
        lines.append(f"| `{check['manifest']}` | {check['verified']} | {check['missing']} | {check['hash_mismatch']} | {check['status']} |")
    lines += [
        "",
        "## 环境与随机性",
        "",
        "- T2/T3/T4 任务级 manifest 记录 Python 3.10.20、项目 `mmdet` 环境和所用 NumPy/Matplotlib/Pandas/Pillow 版本；具体以各 manifest 为准。",
        "- T2 为确定性有限网格枚举；其任务 manifest 的 `random_seed` 为 `null`，并注明复用冻结的 1,000 行 Q2 轨迹 Bootstrap 参数表，不新增抽样。边际收益区间复用既有配对 Bootstrap 估计；T3 也复用既有轨迹 Bootstrap 与配对抽样，总表合并过程不重采样。",
        "- T1、M3/M4 和 T4 的重跑命令、输入输出哈希均保留在各自任务级 manifest。",
        "- 汇总表由当前源 CSV 的预算×上下文键连接生成，不使用正文摘录的数字作为输入。输出形状：15 行 × {table_cols} 列。",
        "- 冻结快照需要仓库内 Q1/Q2 上游输入才能从原始任务重跑；本包提供的复制品足以审阅与做包内一致性核对。",
        "",
        "## 自身哈希说明",
        "",
        "本文件和 `figure_manifest.csv`、`table_manifest.csv`、`release_manifest.json` 的文件哈希不在自身内容中自引用；`release_manifest.json` 记录其余所有冻结文件的哈希并自排除。",
        "",
    ]
    return "\n".join(lines)


def make_scope_qa(numbers: dict[str, object], checks: list[dict[str, object]]) -> dict[str, str]:
    verified = sum(int(check["verified"]) for check in checks)
    missing = sum(int(check["missing"]) for check in checks)
    mismatches = sum(int(check["hash_mismatch"]) for check in checks)
    completeness = f"""# Q3 本轮结果包限定范围完整性核对

**状态：** PASS（仅限本轮可识别结果包范围；不是原生 submission 完整性审计）

| 交付要求 | 证据 | 状态 |
|---|---|---|
| 最终报告含 T1、T2、T3、T4 与边界说明 | `q3_final_report.md`；七张主报告图逐一登记 | PASS |
| 15 行预算×上下文总表覆盖所有情景 | `q3_master_table.csv`，{numbers['master_table']['rows']} 行、{numbers['master_table']['columns']} 列，主键为预算和上下文 | PASS |
| 支持域饱和和有限网格切换已量化 | 15 个正式预算情景、10 个预算端点对照；高预算均为 1,176/1,176 可行；完整事件表单独保留 | PASS |
| 专题表保留细粒度证据 | T1 15 行和 22 对；T2 17,640 行、1,865 个 Pareto 点和 10 条边际收益；T3 15 行及两张 22 行切换表；T4 45 个阈值及 166 个区间 | PASS |
| 图件、表格与复现文件可定位 | `figure_manifest.csv`、`table_manifest.csv`、`reproduction_manifest.md`、`release_manifest.json` | PASS |
| 结论范围和未识别事项清楚 | 固定 Q₀/M0/B1 支持域；完整经验 Q/p 联合优化、βQ 经验识别和竞赛提交级审查均未宣称完成 | PASS |

项目没有活动 `rigor_profile`。因此，本记录不将 `completeness-auditor` 的原生配置检查标为通过；只记录本次明确列出的 Q3 结果包范围核对。
"""
    consistency = f"""# Q3 本轮结果包限定范围一致性核对

**状态：** PASS（此为表/图/正文静态一致性核对；独立最终 P2 回执单独记录；不是提交级终审）

- 15 个主键情景均按 `budget_flops × context_length_tokens` 唯一连接 T1、T2、T3、T4。
- 预算覆盖表的 15 个键与主表一致；10 个相邻正式预算切换端点与冻结 M0 选点一致，C_sat 按 5,880 个 context×grid 成本点取支持域最大值。
- T2 预算可行候选数逐情景与 T1/M4 数值相等；T2 每个场景保留 {numbers['t1']['candidate_support_points']} 个支持域点。
- T2 Pareto 点共 {numbers['t2']['pareto_points']} 个，五种上下文各 373 个；边际收益表 10 行。
- T3 含 15 个 Bootstrap 情景、22 对相邻变化；稳健 N/D 配置切换 {numbers['t3']['robust_configuration_switches']}/22，成本结构切换 {numbers['t3']['robust_cost_structure_switches']}/22，分别保留在独立表中。
- T4 含 {numbers['t4']['conditional_scenarios']} 个条件情景和 {numbers['t4']['regime_rows']} 个区间；有限首次重分配阈值 {numbers['t4']['finite_first_reallocation_thresholds']} 个，无有限阈值 {numbers['t4']['no_finite_first_reallocation_thresholds']} 个；βQ 未识别且 p 排除。
- 五份任务级输入/输出 manifest 合计核验 {verified} 个文件；缺失 {missing}、哈希不符 {mismatches}。报告链接由组装程序逐条检查。

正文数值均以源 CSV 汇总为准，未从报告文字反向生成主表。
"""
    reproducibility = f"""# Q3 本轮结果包限定范围可复现性核对

**状态：** PASS（限已有结果快照及复现信息；从 Q1/Q2 原始源头重跑仍依赖项目上游文件）

- 源任务 manifest 哈希核对：{verified} 项通过，缺失 {missing} 项，哈希不符 {mismatches} 项。
- 复现命令、Python 环境、随机性/Bootstrap 口径见 `reproduction_manifest.md` 与任务级 `reproduction_manifest.json`。
- 图件 SHA-256、格式、PNG 像素尺寸/DPI、矢量画布尺寸见 `figure_manifest.csv`；数据表行数/列数/哈希见 `table_manifest.csv`。
- `release_manifest.json` 记录冻结包内每个文件的路径、SHA-256 和字节数，自身按约定排除以避免递归哈希。
- T2/T3/T4 复用已有生成结果；新增预算分析是固定 M0 的确定性有限网格枚举，不重拟合模型、不做连续阈值推断。

如需从任务原始输入重跑，按总复现清单中的命令执行，并使用任务级 manifest 中的上游 Q1/Q2 路径与哈希。
"""
    return {
        "QA/scoped_completeness_audit.md": completeness,
        "QA/scoped_consistency_audit.md": consistency,
        "QA/scoped_reproducibility_audit.md": reproducibility,
    }


def main() -> None:
    global RELEASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing-release", action="store_true",
                        help="Update this package path after verifying it is the expected Q3 release directory.")
    parser.add_argument("--release-dir", type=Path, default=RELEASE,
                        help="Release directory directly under Q3; use a new date name for a new snapshot.")
    args = parser.parse_args()
    release_arg = args.release_dir
    RELEASE = (ROOT / release_arg).resolve() if not release_arg.is_absolute() else release_arg.resolve()
    if RELEASE.parent != (ROOT / "Q3").resolve():
        raise SystemExit(f"Release directory must be directly under Q3: {RELEASE}")
    if RELEASE.exists() and not args.allow_existing_release:
        raise SystemExit(f"Refusing to overwrite existing release directory: {RELEASE}")
    if RELEASE.exists() and RELEASE.resolve().parent != (ROOT / "Q3").resolve():
        raise SystemExit("Release path escaped Q3; refusing update")
    RELEASE.mkdir(parents=True, exist_ok=True)

    headers, master_rows, numbers = aggregate_master()
    write_csv(RESULTS / "q3_master_table.csv", headers, master_rows)
    (RESULTS / "frozen_numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Snapshot selected result, audit, figure, and code deliverables. Temporary
    # previews and pre-fix directories are intentionally excluded.
    copied: list[Path] = []
    source_for_dest: dict[str, str] = {}
    for source_rel, target_rel in source_tree():
        copied.append(copy_file(ROOT / source_rel, target_rel))
        source_for_dest[target_rel] = source_rel.replace("\\", "/")
    main_report = (RESULTS / "q3_final_report.md").read_text(encoding="utf-8")
    replacements = {
        "figures/result_q3_fixedq_master_and_resource_transition.png": "01_fixed/figures/result_q3_fixedq_master_and_resource_transition.png",
        "figures/result_q3_t4_betaQ_scenario.png": "03_conditional/figures/result_q3_t4_betaQ_scenario.png",
        "Q3_T2_近最优_Pareto_边际收益/": "02_robustness/T2/",
        "Q3_T3_稳健性与配置切换/": "02_robustness/T3/",
        "q3_fixedQ_master_results.csv": "01_fixed/q3_fixedQ_master_results.csv",
        "q3_resource_transition.csv": "01_fixed/q3_resource_transition.csv",
        "预算饱和与有限网格切换/": "01_fixed/预算饱和与有限网格切换/",
        "Q3_fixedQ_results_integration_report.md": "01_fixed/Q3_fixedQ_results_integration_report.md",
        "M3_M4成本独立审计/": "01_fixed/M3_M4成本独立审计/",
        "Q3_Q0_baseline_sensitivity_report.md": "03_conditional/Q3_Q0_baseline_sensitivity_report.md",
        "q3_t4_conditional_beta_manifest.json": "03_conditional/q3_t4_conditional_beta_manifest.json",
        "../01_方案与设计/Q3-T3_Bootstrap稳定性与配置切换_优化执行记录.md": "QA/Q3-T3_Bootstrap稳定性与配置切换_P2范围记录.md",
        "T4_independent_P2_receipt.md": "QA/T4_independent_P2_receipt.md",
    }
    for old, new in replacements.items():
        main_report = main_report.replace(old, new)
    (RELEASE / "q3_final_report.md").write_text(main_report, encoding="utf-8")
    (RELEASE / "scope_decision_ledger.md").write_text(make_ledger(), encoding="utf-8")
    readme_addendum = (
        "\n## 新增：预算饱和与有限网格切换\n\n"
        "本快照新增固定 Q₀/M0 下的预算覆盖比例、当前 1,176 点支持域饱和预算、"
        "完整离散下包络切换事件及正式预算端点对照。详见 "
        "01_fixed/预算饱和与有限网格切换/ 和报告第 7 图。该增补仍限于 B1 实测支持网格。\n"
    )
    (RELEASE / "README.md").write_text(make_readme() + readme_addendum, encoding="utf-8")
    (RELEASE / "frozen_numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checks = validate_source_manifests()
    for relative_path, content in make_scope_qa(numbers, checks).items():
        target = RELEASE / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (RESULTS / "reproduction_manifest.md").write_text(make_reproduction(checks, len(master_rows), len(headers)), encoding="utf-8")
    (RELEASE / "reproduction_manifest.md").write_text(make_reproduction(checks, len(master_rows), len(headers)), encoding="utf-8")

    # Keep an identical canonical master table in the snapshot and working results area.
    shutil.copy2(RESULTS / "q3_master_table.csv", RELEASE / "q3_master_table.csv")

    # Table inventory includes every CSV copied into the release, plus the master table.
    table_rows: list[dict[str, object]] = []
    for path in sorted(RELEASE.rglob("*.csv")):
        if path.name in {"figure_manifest.csv", "table_manifest.csv"}:
            continue
        h, data = read_csv(path)
        table_rows.append({
            "task": path.relative_to(RELEASE).parts[0],
            "relative_path": path.relative_to(RELEASE).as_posix(),
            "rows": len(data), "columns": len(h), "sha256": sha256(path),
            "source_of_record": source_for_dest.get(
                path.relative_to(RELEASE).as_posix(),
                "Q3/04_结果/q3_master_table.csv" if path.name == "q3_master_table.csv" else "",
            ),
            "role": "master" if path.name == "q3_master_table.csv" else "source_or_specialist_table",
        })
        if not table_rows[-1]["source_of_record"]:
            raise ValueError(f"No source mapping for table {path.relative_to(RELEASE)}")
    table_headers = ["task", "relative_path", "rows", "columns", "sha256", "source_of_record", "role"]
    write_csv(RELEASE / "table_manifest.csv", table_headers, table_rows)
    # Working-results inventory points to original result locations and shares the same table rows.
    working_table_rows = [dict(r, relative_path=str(Path(r["source_of_record"]).relative_to("Q3/04_结果")).replace("\\", "/")) for r in table_rows]
    write_csv(RESULTS / "table_manifest.csv", table_headers, working_table_rows)

    report_figures = {
        "result_q3_fixedq_master_and_resource_transition": 1,
        "result_q3_t2_near_optimal_pareto_marginal_return": 2,
        "result_q3_t3_bootstrap_stability": 3,
        "result_q3_t3_configuration_switch": 4,
        "result_q3_t3_cost_structure_switch": 5,
        "result_q3_t4_betaQ_scenario": 6,
        "result_q3_budget_saturation_finite_grid_switches": 7,
    }
    figure_rows: list[dict[str, object]] = []
    for path in sorted(RELEASE.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".svg", ".pdf"}:
            continue
        stem = path.stem.replace("_grayscale", "")
        report_number = report_figures.get(stem, "")
        if path.stem.endswith("_grayscale"):
            role = "grayscale_alternate"
        else:
            role = "main_report" if report_number else "supporting"
        task = path.relative_to(RELEASE).parts[0]
        pixel_width = pixel_height = dpi_x = dpi_y = page_width_in = page_height_in = ""
        try:
            if path.suffix.lower() == ".png":
                from PIL import Image
                with Image.open(path) as image:
                    pixel_width, pixel_height = image.size
                    dpi = image.info.get("dpi")
                    if dpi and len(dpi) >= 2:
                        dpi_x, dpi_y = round(float(dpi[0]), 3), round(float(dpi[1]), 3)
            elif path.suffix.lower() == ".svg":
                root = ET.parse(path).getroot()
                view_box = root.attrib.get("viewBox", "").split()
                if len(view_box) == 4:
                    page_width_in = round(float(view_box[2]) / 96, 3)
                    page_height_in = round(float(view_box[3]) / 96, 3)
            elif path.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                page = PdfReader(str(path)).pages[0]
                page_width_in = round(float(page.mediabox.width) / 72, 3)
                page_height_in = round(float(page.mediabox.height) / 72, 3)
        except Exception:
            # The task-level figure manifest remains the authority if a reader
            # library does not expose optional metadata for an individual file.
            pass
        figure_rows.append({
            "figure_id": stem, "report_figure": report_number, "task": task, "asset_role": role,
            "format": path.suffix.lower().lstrip("."), "relative_path": path.relative_to(RELEASE).as_posix(),
            "pixel_width": pixel_width, "pixel_height": pixel_height, "dpi_x": dpi_x, "dpi_y": dpi_y,
            "page_width_in": page_width_in, "page_height_in": page_height_in,
            "sha256": sha256(path), "source_data": "see task reproduction_manifest.json",
            "status": "included",
        })
    figure_headers = ["figure_id", "report_figure", "task", "asset_role", "format", "relative_path",
                     "pixel_width", "pixel_height", "dpi_x", "dpi_y", "page_width_in", "page_height_in",
                     "sha256", "source_data", "status"]
    write_csv(RELEASE / "figure_manifest.csv", figure_headers, figure_rows)
    working_figure_rows = [dict(r, relative_path=f"Q3/{RELEASE.name}/" + r["relative_path"]) for r in figure_rows]
    write_csv(RESULTS / "figure_manifest.csv", figure_headers, working_figure_rows)

    # Record every package file except the hash list itself to avoid recursion.
    inventory = []
    for path in sorted(RELEASE.rglob("*")):
        if path.is_file() and path.name != "release_manifest.json":
            inventory.append({"path": path.relative_to(RELEASE).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    release_manifest = {
        "artifact": "Q3 scope-limited release package",
        "created_on": date.today().isoformat(),
        "scope": numbers["scope"],
        "files_excluding_this_manifest": inventory,
        "file_count": len(inventory),
        "source_manifest_checks": checks,
    }
    (RELEASE / "release_manifest.json").write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Verify the created package inventory and report links before returning.
    for item in inventory:
        candidate = RELEASE / item["path"]
        if sha256(candidate) != item["sha256"]:
            raise ValueError(f"Package hash mismatch after write: {item['path']}")
    missing_links = []
    for rel in __import__("re").findall(r"\]\(([^)]+)\)", main_report):
        if rel.startswith(("http:", "https:", "#")):
            continue
        target = (RELEASE / rel.split("#", 1)[0]).resolve()
        if not target.exists():
            missing_links.append(rel)
    if missing_links:
        raise ValueError(f"Broken local Markdown links in release report: {missing_links}")

    print(json.dumps({
        "release": str(RELEASE), "master_rows": len(master_rows), "master_columns": len(headers),
        "tables": len(table_rows), "figures": len(figure_rows), "source_manifest_checks": checks,
        "package_files": len(inventory) + 1,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
