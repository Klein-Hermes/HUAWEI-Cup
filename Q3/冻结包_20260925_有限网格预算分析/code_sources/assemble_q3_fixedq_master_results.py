#!/usr/bin/env python3
"""Integrate the frozen fixed-Q0 Q3 scenario and resource-transition results.

No model refit or bootstrap rerun is performed here. The script joins the
already audited M0 frontier, M1/M3/M4 masks, cost-share audit, and 22-pair
paired-bootstrap transition summary into two task-facing CSVs and one figure.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Q3/04_结果"
OUT_MASTER = "q3_fixedQ_master_results.csv"
OUT_TRANSITIONS = "q3_resource_transition.csv"
OUT_REPORT = "Q3_fixedQ_results_integration_report.md"
OUT_CONTRACT = "Q3_fixedQ_master_resource_transition_图表契约.md"
OUT_MANIFEST = "Q3_fixedQ_results_manifest.json"
FIGURE_STEM = "result_q3_fixedq_master_and_resource_transition"
SMOKE_DIR = RESULTS / "_q3_fixedq_p1_smoke"

getcontext().prec = 36

MASTER_FIELDS = [
    "scenario_order", "scenario_id", "budget_flops", "context_length_tokens",
    "critical_context_length_tokens", "q1_q_version", "q0_reference",
    "quality_setting", "p_setting", "candidate_domain", "candidate_count",
    "m3_support_candidate_count", "m4_budget_feasible_candidate_count",
    "m1_final_feasible_candidate_count", "selected_grid_id", "selected_run_id",
    "selected_N_B", "selected_D_B", "m0_predicted_val_loss",
    "b1_observed_val_loss", "m3_support_policy", "m3_search_allowed",
    "m3_support_class", "m3_inside_convex_hull_diagnostic",
    "m3_interpolation_candidate", "m3_extrapolation_flag",
    "m4_budget_feasible", "m1_final_feasible", "C_train_flops",
    "C_Q_flops", "C_attn_flops", "C_total_flops", "budget_slack_flops",
    "budget_utilization", "train_cost_share", "quality_cost_share",
    "attention_cost_share", "b1_feasible_grid_points", "hit_N_min",
    "hit_N_max", "hit_D_min", "hit_D_max", "all_support_points_feasible",
    "independent_choice_match", "independent_m3_flag_match",
    "independent_m4_flag_match", "independent_final_flag_match",
    "independent_cost_fields_match", "independent_audit_pass",
]

TRANSITION_FIELDS = [
    "pair_id", "pair_type", "transition_axis", "left_scenario_id",
    "right_scenario_id", "left_budget_flops", "right_budget_flops",
    "left_context_length_tokens", "right_context_length_tokens",
    "left_selected_grid_id", "right_selected_grid_id", "left_selected_run_id",
    "right_selected_run_id", "left_selected_N_B", "right_selected_N_B",
    "delta_selected_N_B", "left_selected_D_B", "right_selected_D_B",
    "delta_selected_D_B", "point_delta_log_N", "bootstrap_delta_log_N_p2_5",
    "bootstrap_delta_log_N_median", "bootstrap_delta_log_N_p97_5",
    "point_delta_log_D", "bootstrap_delta_log_D_p2_5",
    "bootstrap_delta_log_D_median", "bootstrap_delta_log_D_p97_5",
    "left_m0_predicted_val_loss", "right_m0_predicted_val_loss",
    "delta_m0_predicted_val_loss", "left_C_total_flops", "right_C_total_flops",
    "delta_C_total_flops", "left_budget_utilization", "right_budget_utilization",
    "delta_budget_utilization", "left_train_cost_share", "right_train_cost_share",
    "delta_train_cost_share", "left_attention_cost_share",
    "right_attention_cost_share", "delta_attention_cost_share",
    "left_feasible_candidate_count", "right_feasible_candidate_count",
    "delta_feasible_candidate_count", "paired_bootstrap_replicates",
    "configuration_change_frequency", "cost_share_change_frequency",
    "cost_share_TV_p05_exact", "cost_share_TV_median",
    "robust_ND_allocation_change", "robust_cost_mix_change", "interpretation",
]


class IntegrationError(ValueError):
    """Raised when authoritative Q3 inputs disagree or cannot be joined."""


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise IntegrationError(f"Empty input CSV: {rel(path)}")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        raise IntegrationError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dec(value: Any, name: str) -> Decimal:
    try:
        out = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as error:
        raise IntegrationError(f"Invalid decimal for {name}: {value!r}") from error
    if not out.is_finite():
        raise IntegrationError(f"Non-finite decimal for {name}: {value!r}")
    return out


def dstr(value: Decimal) -> str:
    if value == 0:
        return "0"
    return str(value.normalize())


def as_bool(value: Any, name: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise IntegrationError(f"Invalid boolean for {name}: {value!r}")


def close(a: Any, b: Any, *, rel_tol: Decimal = Decimal("1e-12")) -> bool:
    left, right = dec(a, "comparison left"), dec(b, "comparison right")
    scale = max(abs(left), abs(right), Decimal(1))
    return abs(left - right) <= rel_tol * scale


def require_close(a: Any, b: Any, name: str) -> None:
    if not close(a, b):
        raise IntegrationError(f"{name} mismatch: {a} vs {b}")


def unique_by(rows: list[dict[str, str]], key_fn, label: str):
    result = {}
    for row in rows:
        key = key_fn(row)
        if key in result:
            raise IntegrationError(f"Duplicate {label} key: {key}")
        result[key] = row
    return result


def scenario_key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["budget_flops"]), int(row["context_length_tokens"])


def find_cost_audit() -> Path:
    matches = list(RESULTS.glob("M3_M4*/m0_optimum_cost_audit.csv"))
    if len(matches) != 1:
        raise IntegrationError(
            f"Expected one audited M0 cost table under M3_M4*, found {len(matches)}"
        )
    return matches[0]


def source_paths() -> dict[str, Path]:
    return {
        "m0_frontier": RESULTS / "M0_ND_reference_frontier.csv",
        "m1_boundary_audit": RESULTS / "m1_optimum_boundary_audit.csv",
        "m3_support": RESULTS / "m3_q3_support_grid.csv",
        "m4_budget_mask": RESULTS / "m4_budget_feasible_mask.csv",
        "m1_final_mask": RESULTS / "m1_final_feasible_mask.csv",
        "cost_shares": RESULTS / "M0_budget_context_cost_shares.csv",
        "transition_pairs": RESULTS / "M0_discrete_transition_pairs.csv",
        "independent_cost_audit": find_cost_audit(),
        "cost_manifest": RESULTS / "Q3_cost_sensitivities_manifest.json",
        "independent_audit_manifest": RESULTS / "M3_M4成本独立审计/reproduction_manifest.json",
        "m0_manifest": RESULTS / "M0_reference_reproduction_manifest.json",
        "m1_manifest": RESULTS / "M1_masked_optimization_manifest.json",
        "m3_manifest": RESULTS / "M3_support_manifest.json",
        "m4_manifest": RESULTS / "M4_budget_manifest.json",
        "transition_manifest": RESULTS / "M0_discrete_transition_manifest.json",
    }


def verify_manifest_output(manifest_path: Path, artifact_path: Path) -> bool:
    """If a manifest declares this output, verify its hash; require a declaration."""
    manifest = read_json(manifest_path)
    expected = {}
    for section in ("outputs_sha256", "outputs"):
        if isinstance(manifest.get(section), dict):
            expected.update(manifest[section])
    artifact_rel = rel(artifact_path)
    declared = expected.get(artifact_rel)
    if declared is None:
        raise IntegrationError(
            f"{rel(manifest_path)} does not declare output {artifact_rel}"
        )
    if sha256(artifact_path) != declared:
        raise IntegrationError(f"Upstream manifest hash mismatch: {artifact_rel}")
    return True


def load_and_integrate(paths: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    # Verify that the authoritative input CSVs still match their producing manifests.
    verify_manifest_output(paths["m0_manifest"], paths["m0_frontier"])
    verify_manifest_output(paths["m1_manifest"], paths["m1_boundary_audit"])
    verify_manifest_output(paths["m1_manifest"], paths["m1_final_mask"])
    verify_manifest_output(paths["m3_manifest"], paths["m3_support"])
    verify_manifest_output(paths["m4_manifest"], paths["m4_budget_mask"])
    verify_manifest_output(paths["transition_manifest"], paths["transition_pairs"])
    verify_manifest_output(paths["cost_manifest"], paths["cost_shares"])
    verify_manifest_output(paths["independent_audit_manifest"], paths["independent_cost_audit"])

    frontier_rows = read_csv(paths["m0_frontier"])
    m1_rows = read_csv(paths["m1_boundary_audit"])
    support_rows = read_csv(paths["m3_support"])
    m4_rows = read_csv(paths["m4_budget_mask"])
    m1_mask_rows = read_csv(paths["m1_final_mask"])
    share_rows = read_csv(paths["cost_shares"])
    transition_rows = read_csv(paths["transition_pairs"])
    independent_rows = read_csv(paths["independent_cost_audit"])

    frontier = unique_by(frontier_rows, scenario_key, "M0 scenario")
    m1 = unique_by(m1_rows, scenario_key, "M1 scenario")
    shares = unique_by(share_rows, scenario_key, "cost-share scenario")
    independent = unique_by(independent_rows, scenario_key, "independent-audit scenario")
    support = unique_by(support_rows, lambda row: row["grid_id"], "M3 grid")
    m4_selected = unique_by(m4_rows, lambda row: (row["scenario_id"], row["grid_id"]), "M4 scenario/grid")
    m1_selected = unique_by(m1_mask_rows, lambda row: (row["scenario_id"], row["grid_id"]), "M1 scenario/grid")

    m3_count: Counter[str] = Counter()
    for row in support_rows:
        if as_bool(row["m3_search_allowed"], "m3_search_allowed"):
            m3_count["all"] += 1
    budget_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    for row in m4_rows:
        sid = row["scenario_id"]
        candidate_counts[sid] += 1
        budget_counts[sid] += int(as_bool(row["budget_feasible"], "budget_feasible"))
    for row in m1_mask_rows:
        final_counts[row["scenario_id"]] += int(as_bool(row["final_feasible"], "final_feasible"))

    ordered_keys = sorted(frontier, key=lambda key: (key[1], key[0]))
    if len(ordered_keys) != 15 or len(m1) != 15 or len(shares) != 15 or len(independent) != 15:
        raise IntegrationError("Expected exactly 15 unique budget-context scenarios across source tables")
    if len(support_rows) != 1176 or len(set(support)) != 1176:
        raise IntegrationError("Expected 1,176 unique M3 support rows")
    if len(m4_rows) != 17640 or len(m1_mask_rows) != 17640:
        raise IntegrationError("Expected 17,640 M4 and M1 candidate-scenario mask rows")
    if len(transition_rows) != 22:
        raise IntegrationError(f"Expected 22 adjacent transition pairs, found {len(transition_rows)}")

    master: list[dict[str, Any]] = []
    for order, key in enumerate(ordered_keys, start=1):
        frow, mrow, srow, arow = frontier[key], m1[key], shares[key], independent[key]
        scenario_id = mrow["scenario_id"]
        grid_id = mrow["m1_selected_grid_id"]
        if (scenario_id, grid_id) not in m4_selected or (scenario_id, grid_id) not in m1_selected:
            raise IntegrationError(f"Selected grid not present in M4/M1 masks: {scenario_id}/{grid_id}")
        support_row = support.get(grid_id)
        if support_row is None:
            raise IntegrationError(f"Selected grid absent from M3 support table: {grid_id}")
        budget_row = m4_selected[(scenario_id, grid_id)]
        final_row = m1_selected[(scenario_id, grid_id)]

        for field in ("selected_run_id", "selected_N_B", "selected_D_B", "m0_predicted_val_loss"):
            m1_field = {
                "selected_run_id": "m1_selected_run_id",
                "selected_N_B": "selected_N_B",
                "selected_D_B": "selected_D_B",
                "m0_predicted_val_loss": "m0_predicted_val_loss",
            }[field]
            require_close(frow[field], mrow[m1_field], f"M0/M1 {field} for {scenario_id}")
        if str(frow["selected_run_id"]) != str(grid_id):
            raise IntegrationError(f"M0 selected run and M1 selected grid differ for {scenario_id}")

        m3_selected = as_bool(support_row["m3_search_allowed"], "selected M3 flag")
        m4_selected_ok = as_bool(budget_row["budget_feasible"], "selected M4 flag")
        m1_final_ok = as_bool(final_row["final_feasible"], "selected M1 final flag")
        if not (m3_selected and m4_selected_ok and m1_final_ok):
            raise IntegrationError(f"Selected configuration fails M3/M4/M1 gate for {scenario_id}")
        if not as_bool(mrow["final_feasible"], "M1 boundary final_feasible"):
            raise IntegrationError(f"M1 boundary audit marks selected point infeasible: {scenario_id}")

        # Check point costs and fractions against the separate cost audit.
        for field in ("C_train_flops", "C_Q_flops", "C_attn_flops", "C_total_flops"):
            require_close(frow[field], arow[field], f"independent cost audit {field} for {scenario_id}")
        require_close(frow["budget_utilization"], arow["budget_utilization_existing"], f"budget utilization for {scenario_id}")
        require_close(
            dec(frow["C_total_flops"], "C_total") / dec(frow["budget_flops"], "budget"),
            frow["budget_utilization"], f"C_total/budget for {scenario_id}",
        )
        shares_sum = sum((dec(srow[name], name) for name in ("train_cost_share", "quality_cost_share", "attention_cost_share")), Decimal(0))
        require_close(shares_sum, 1, f"cost shares sum for {scenario_id}")
        if dec(frow["C_Q_flops"], "C_Q") != 0 or dec(srow["quality_cost_share"], "quality share") != 0:
            raise IntegrationError(f"Fixed-Q0 branch must have zero quality cost for {scenario_id}")

        checks = {
            "independent_choice_match": as_bool(arow["choice_match"], "choice_match"),
            "independent_m3_flag_match": as_bool(arow["m3_flag_match"], "m3_flag_match"),
            "independent_m4_flag_match": as_bool(arow["budget_flag_match"], "budget_flag_match"),
            "independent_final_flag_match": as_bool(arow["final_flag_match"], "final_flag_match"),
            "independent_cost_fields_match": as_bool(arow["cost_fields_match"], "cost_fields_match"),
        }
        audit_pass = all(checks.values()) and as_bool(arow["support_feasible"], "independent support flag")
        if not audit_pass:
            raise IntegrationError(f"Independent P1 cost audit failed for {scenario_id}: {checks}")

        row: dict[str, Any] = {
            "scenario_order": order,
            "scenario_id": scenario_id,
            "budget_flops": frow["budget_flops"],
            "context_length_tokens": frow["context_length_tokens"],
            "critical_context_length_tokens": frow["critical_context_length_tokens"],
            "q1_q_version": frow["q1_q_version"],
            "q0_reference": frow["q0_reference_a1_macro"],
            "quality_setting": frow["q_setting"],
            "p_setting": frow["p_setting"],
            "candidate_domain": "Q2 B1 exact observed N/D grid",
            "candidate_count": frow["b1_total_grid_points"],
            "m3_support_candidate_count": m3_count["all"],
            "m4_budget_feasible_candidate_count": budget_counts[scenario_id],
            "m1_final_feasible_candidate_count": final_counts[scenario_id],
            "selected_grid_id": grid_id,
            "selected_run_id": frow["selected_run_id"],
            "selected_N_B": frow["selected_N_B"],
            "selected_D_B": frow["selected_D_B"],
            "m0_predicted_val_loss": frow["m0_predicted_val_loss"],
            "b1_observed_val_loss": frow["b1_observed_val_loss_at_selected_point"],
            "m3_support_policy": support_row["support_policy"],
            "m3_search_allowed": m3_selected,
            "m3_support_class": support_row["support_class"],
            "m3_inside_convex_hull_diagnostic": support_row["inside_convex_hull"],
            "m3_interpolation_candidate": support_row["interpolation_candidate"],
            "m3_extrapolation_flag": support_row["extrapolation_flag"],
            "m4_budget_feasible": m4_selected_ok,
            "m1_final_feasible": m1_final_ok,
            "C_train_flops": frow["C_train_flops"],
            "C_Q_flops": frow["C_Q_flops"],
            "C_attn_flops": frow["C_attn_flops"],
            "C_total_flops": frow["C_total_flops"],
            "budget_slack_flops": frow["budget_slack_flops"],
            "budget_utilization": frow["budget_utilization"],
            "train_cost_share": srow["train_cost_share"],
            "quality_cost_share": srow["quality_cost_share"],
            "attention_cost_share": srow["attention_cost_share"],
            "b1_feasible_grid_points": frow["b1_feasible_grid_points"],
            "hit_N_min": mrow["hit_N_min"],
            "hit_N_max": mrow["hit_N_max"],
            "hit_D_min": mrow["hit_D_min"],
            "hit_D_max": mrow["hit_D_max"],
            "all_support_points_feasible": frow["all_support_points_feasible"],
            **checks,
            "independent_audit_pass": audit_pass,
        }
        if int(row["candidate_count"]) != candidate_counts[scenario_id]:
            raise IntegrationError(f"Candidate count mismatch for {scenario_id}")
        if int(row["b1_feasible_grid_points"]) != budget_counts[scenario_id] or int(mrow["m1_eligible_candidate_count"]) != final_counts[scenario_id]:
            raise IntegrationError(f"Feasible count mismatch across M0/M1/M4 for {scenario_id}")
        master.append(row)

    scenario_by_key = {scenario_key(row): row for row in master}
    transition_output: list[dict[str, Any]] = []
    pair_ids = set()
    for pair in transition_rows:
        pair_id = pair["pair_id"]
        if pair_id in pair_ids:
            raise IntegrationError(f"Duplicate transition pair id: {pair_id}")
        pair_ids.add(pair_id)
        pair_type = pair["pair_type"]
        left_key = int(pair["left_budget_flops"]), int(pair["left_context_tokens"])
        right_key = int(pair["right_budget_flops"]), int(pair["right_context_tokens"])
        if left_key not in scenario_by_key or right_key not in scenario_by_key:
            raise IntegrationError(f"Transition {pair_id} references an unknown scenario")
        left, right = scenario_by_key[left_key], scenario_by_key[right_key]
        if pair_type == "adjacent_budget":
            if left_key[1] != right_key[1] or right_key[0] <= left_key[0]:
                raise IntegrationError(f"Malformed adjacent-budget pair {pair_id}")
            axis = "budget_increase"
        elif pair_type == "adjacent_context":
            if left_key[0] != right_key[0] or right_key[1] <= left_key[1]:
                raise IntegrationError(f"Malformed adjacent-context pair {pair_id}")
            axis = "context_increase"
        else:
            raise IntegrationError(f"Unknown transition pair type {pair_type!r}")

        left_n, right_n = dec(left["selected_N_B"], "left N"), dec(right["selected_N_B"], "right N")
        left_d, right_d = dec(left["selected_D_B"], "left D"), dec(right["selected_D_B"], "right D")
        point_delta_log_n = math.log(float(right_n / left_n))
        point_delta_log_d = math.log(float(right_d / left_d))
        # Existing paired-bootstrap medians should reproduce the point transition
        # when the selected grid point is stable across all bootstrap replicates.
        if close(pair["configuration_change_frequency"], 1):
            require_close(point_delta_log_n, pair["delta_log_N_median"], f"{pair_id} bootstrap median log-N")
            require_close(point_delta_log_d, pair["delta_log_D_median"], f"{pair_id} bootstrap median log-D")

        transition_output.append({
            "pair_id": pair_id,
            "pair_type": pair_type,
            "transition_axis": axis,
            "left_scenario_id": left["scenario_id"],
            "right_scenario_id": right["scenario_id"],
            "left_budget_flops": left["budget_flops"],
            "right_budget_flops": right["budget_flops"],
            "left_context_length_tokens": left["context_length_tokens"],
            "right_context_length_tokens": right["context_length_tokens"],
            "left_selected_grid_id": left["selected_grid_id"],
            "right_selected_grid_id": right["selected_grid_id"],
            "left_selected_run_id": left["selected_run_id"],
            "right_selected_run_id": right["selected_run_id"],
            "left_selected_N_B": left["selected_N_B"],
            "right_selected_N_B": right["selected_N_B"],
            "delta_selected_N_B": dstr(right_n - left_n),
            "left_selected_D_B": left["selected_D_B"],
            "right_selected_D_B": right["selected_D_B"],
            "delta_selected_D_B": dstr(right_d - left_d),
            "point_delta_log_N": format(point_delta_log_n, ".12g"),
            "bootstrap_delta_log_N_p2_5": pair["delta_log_N_p2_5"],
            "bootstrap_delta_log_N_median": pair["delta_log_N_median"],
            "bootstrap_delta_log_N_p97_5": pair["delta_log_N_p97_5"],
            "point_delta_log_D": format(point_delta_log_d, ".12g"),
            "bootstrap_delta_log_D_p2_5": pair["delta_log_D_p2_5"],
            "bootstrap_delta_log_D_median": pair["delta_log_D_median"],
            "bootstrap_delta_log_D_p97_5": pair["delta_log_D_p97_5"],
            "left_m0_predicted_val_loss": left["m0_predicted_val_loss"],
            "right_m0_predicted_val_loss": right["m0_predicted_val_loss"],
            "delta_m0_predicted_val_loss": dstr(dec(right["m0_predicted_val_loss"], "right Loss") - dec(left["m0_predicted_val_loss"], "left Loss")),
            "left_C_total_flops": left["C_total_flops"],
            "right_C_total_flops": right["C_total_flops"],
            "delta_C_total_flops": dstr(dec(right["C_total_flops"], "right C") - dec(left["C_total_flops"], "left C")),
            "left_budget_utilization": left["budget_utilization"],
            "right_budget_utilization": right["budget_utilization"],
            "delta_budget_utilization": dstr(dec(right["budget_utilization"], "right utilization") - dec(left["budget_utilization"], "left utilization")),
            "left_train_cost_share": left["train_cost_share"],
            "right_train_cost_share": right["train_cost_share"],
            "delta_train_cost_share": dstr(dec(right["train_cost_share"], "right train share") - dec(left["train_cost_share"], "left train share")),
            "left_attention_cost_share": left["attention_cost_share"],
            "right_attention_cost_share": right["attention_cost_share"],
            "delta_attention_cost_share": dstr(dec(right["attention_cost_share"], "right attention share") - dec(left["attention_cost_share"], "left attention share")),
            "left_feasible_candidate_count": left["b1_feasible_grid_points"],
            "right_feasible_candidate_count": right["b1_feasible_grid_points"],
            "delta_feasible_candidate_count": int(right["b1_feasible_grid_points"]) - int(left["b1_feasible_grid_points"]),
            "paired_bootstrap_replicates": pair["paired_bootstrap_replicates"],
            "configuration_change_frequency": pair["configuration_change_frequency"],
            "cost_share_change_frequency": pair["cost_share_change_frequency"],
            "cost_share_TV_p05_exact": pair["cost_share_TV_p05_exact"],
            "cost_share_TV_median": pair["cost_share_TV_median"],
            "robust_ND_allocation_change": pair["robust_ND_allocation_change"],
            "robust_cost_mix_change": pair["robust_cost_mix_change"],
            "interpretation": pair["interpretation"],
        })

    counts = Counter(row["pair_type"] for row in transition_output)
    transitions_summary = {}
    for pair_type in ("adjacent_budget", "adjacent_context"):
        group = [row for row in transition_output if row["pair_type"] == pair_type]
        transitions_summary[pair_type] = {
            "pairs": len(group),
            "robust_ND_allocation_changes": sum(as_bool(row["robust_ND_allocation_change"], "robust ND") for row in group),
            "robust_cost_mix_changes": sum(as_bool(row["robust_cost_mix_change"], "robust cost") for row in group),
        }
    if counts != Counter({"adjacent_budget": 10, "adjacent_context": 12}):
        raise IntegrationError(f"Unexpected transition pair counts: {counts}")

    meta = {
        "scenario_count": len(master),
        "candidate_count": len(support_rows),
        "m4_mask_rows": len(m4_rows),
        "m1_mask_rows": len(m1_mask_rows),
        "transition_pair_count": len(transition_output),
        "m3_allowed_candidates": m3_count["all"],
        "independent_audit_pass_scenarios": sum(as_bool(row["independent_audit_pass"], "audit pass") for row in master),
        "transition_summary": transitions_summary,
        "input_paths": {key: rel(path) for key, path in paths.items()},
    }
    return master, transition_output, meta


def write_smoke(master: list[dict[str, Any]], transitions: list[dict[str, Any]], meta: dict[str, Any], out_dir: Path) -> None:
    write_csv(out_dir / OUT_MASTER, master[:1], MASTER_FIELDS)
    write_csv(out_dir / OUT_TRANSITIONS, transitions[:1], TRANSITION_FIELDS)
    smoke_args = "--smoke" if out_dir.resolve() == SMOKE_DIR.resolve() else f'--smoke --output-dir "{out_dir.resolve().relative_to(ROOT).as_posix()}"'
    receipt = {
        "status": "P1 smoke slice ready for independent review",
        "scope": "one real budget-context scenario and one real adjacent transition pair; full upstream tables validated",
        "command": f"$env:MPLBACKEND = 'Agg'; & '{Path(sys.executable).resolve().as_posix()}' 'Q3/03_代码/assemble_q3_fixedq_master_results.py' {smoke_args}",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "script_sha256": sha256(Path(__file__)),
        "environment": {"MPLBACKEND": "Agg"},
        "scenario_rows_written": 1,
        "transition_rows_written": 1,
        "full_source_scenario_count": meta["scenario_count"],
        "full_source_candidate_count": meta["candidate_count"],
        "full_source_transition_count": meta["transition_pair_count"],
        "independent_audit_pass_scenarios": meta["independent_audit_pass_scenarios"],
        "inputs_sha256": {name: sha256(ROOT / path) for name, path in meta["input_paths"].items() if (ROOT / path).suffix == ".csv"},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p1_smoke_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"P1 smoke: scenario_rows=1, transition_rows=1; validated scenarios={meta['scenario_count']}, transitions={meta['transition_pair_count']}")
    print(f"Wrote {out_dir.relative_to(ROOT).as_posix()}/{OUT_MASTER}")
    print(f"Wrote {out_dir.relative_to(ROOT).as_posix()}/{OUT_TRANSITIONS}")


def render_contract() -> str:
    return """# Q3 固定 Q₀ 主结果与资源转移图表契约

**任务：** Q3-T1 固定 Q₀ 主结果整合  
**核心结论：** 固定 Q₀/M0 下，15 个预算—上下文场景的参考最优配置、预算利用率和相邻情景资源变化可以从已审计结果表中无损对齐；22 对变化中，配置改变与成本构成改变并不总是同步。  
**数据源：** `q3_fixedQ_master_results.csv`、`q3_resource_transition.csv`。  
**图型：** 左侧 5×3 情景矩阵，以 M0 预测 Loss 着色，并在格内标出 N*、D*、Loss* 与预算利用率；右侧以 2×2 摘要矩阵展示相邻预算/上下文的 Bootstrap 稳健变化比例。  
**主面板：** 15 个固定 Q₀ 场景的 Loss 与资源配置矩阵。  
**统计口径：** 22 对相邻场景；每对 1,000 个配对轨迹 Bootstrap 复本；配置变化门槛 95%，成本构成变化门槛 95%。  
**图面限制：** 变化频率代表配置/成本构成发生改变，不代表收益方向或因果效应；高预算最优点受 B1 网格上界限制。  
**尺寸与导出：** 7.2×5.0 英寸；SVG、PDF、300 dpi PNG、灰度预览。
"""


def create_core_figure(master: list[dict[str, Any]], transitions: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from utils.plot_style import apply_publication_style, export_figure

    style_info = apply_publication_style(language="zh", width="double")
    contexts = sorted({int(row["context_length_tokens"]) for row in master})
    budgets = sorted({int(row["budget_flops"]) for row in master})
    lookup = {(int(row["context_length_tokens"]), int(row["budget_flops"])): row for row in master}
    loss = np.array([[float(lookup[(ctx, budget)]["m0_predicted_val_loss"]) for budget in budgets] for ctx in contexts])
    fig = plt.figure(figsize=(7.2, 5.0), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.25)
    ax_main = fig.add_subplot(grid[0, 0])
    image = ax_main.pcolormesh(
        np.arange(len(budgets) + 1) - 0.5,
        np.arange(len(contexts) + 1) - 0.5,
        loss,
        shading="flat",
        cmap="YlGnBu",
        edgecolors="white",
        linewidth=1.2,
    )
    ax_main.set_xlim(-0.5, len(budgets) - 0.5)
    ax_main.set_ylim(len(contexts) - 0.5, -0.5)
    budget_exp = [int(round(math.log10(budget))) for budget in budgets]
    ax_main.set_xticks(range(len(budgets)), [rf"$10^{{{exp}}}$" for exp in budget_exp])
    ax_main.set_yticks(range(len(contexts)), [f"{ctx:,}" for ctx in contexts])
    ax_main.set_xlabel("算力预算（FLOPs）")
    ax_main.set_ylabel("上下文长度（Token）")
    ax_main.set_title("15 个场景：Loss* 与 N*/D*")
    ax_main.tick_params(axis="both", labelsize=7)
    for i, ctx in enumerate(contexts):
        for j, budget in enumerate(budgets):
            row = lookup[(ctx, budget)]
            rgba = image.cmap(image.norm(loss[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.48 else "#202124"
            cell = (
                f"N*={float(row['selected_N_B']):.3g}B\n"
                f"D*={float(row['selected_D_B']):.4g}B\n"
                f"Loss*={float(row['m0_predicted_val_loss']):.4f}\n"
                f"U={float(row['budget_utilization']):.1%}"
            )
            ax_main.text(j, i, cell, ha="center", va="center", color=color, fontsize=6.0, linespacing=1.25)
    cbar = fig.colorbar(image, ax=ax_main, fraction=0.04, pad=0.025)
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
    cbar.set_label("M0 预测验证 Loss（低为优）", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax_transition = fig.add_subplot(grid[0, 1])
    pair_types = ["adjacent_budget", "adjacent_context"]
    type_labels = ["相邻预算", "相邻上下文"]
    metric_defs = [
        ("robust_ND_allocation_change", "N/D 配置改变"),
        ("robust_cost_mix_change", "成本构成改变"),
    ]
    value_grid = np.zeros((len(metric_defs), len(pair_types)))
    label_grid: list[list[str]] = [[""] * len(pair_types) for _ in metric_defs]
    for j, pair_type in enumerate(pair_types):
        rows = [row for row in transitions if row["pair_type"] == pair_type]
        for i, (field, _) in enumerate(metric_defs):
            changed = sum(as_bool(row[field], field) for row in rows)
            total = len(rows)
            value_grid[i, j] = changed / total if total else 0
            label_grid[i][j] = f"{changed}/{total}\n{changed / total:.0%}"
    cmap = plt.get_cmap("Blues")
    ax_transition.pcolormesh(
        np.arange(len(pair_types) + 1) - 0.5,
        np.arange(len(metric_defs) + 1) - 0.5,
        value_grid,
        shading="flat",
        vmin=0,
        vmax=1,
        cmap=cmap,
        edgecolors="white",
        linewidth=1.4,
    )
    ax_transition.set_xlim(-0.5, len(pair_types) - 0.5)
    ax_transition.set_ylim(len(metric_defs) - 0.5, -0.5)
    ax_transition.set_xticks(range(len(pair_types)), type_labels)
    ax_transition.set_yticks(range(len(metric_defs)), [name for _, name in metric_defs])
    ax_transition.set_title("转移稳健变化率")
    ax_transition.tick_params(axis="both", labelsize=7)
    for i in range(len(metric_defs)):
        for j in range(len(pair_types)):
            color = "white" if value_grid[i, j] >= 0.65 else "#202124"
            ax_transition.text(j, i, label_grid[i][j], ha="center", va="center", color=color, fontsize=9)
    ax_transition.text(
        0.0, -0.18,
        "22 对相邻情景；每对 1,000 个配对复本。\n"
        "变化不代表改进或因果效应。\n"
        "M3/M4 最优点标记：15/15 通过。",
        transform=ax_transition.transAxes, ha="left", va="top", fontsize=6.3,
    )
    fig.suptitle("固定 Q0/M0：15 个情景结果与离散资源转移", fontsize=9.5)

    output_stem = out_dir / "figures" / FIGURE_STEM
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = export_figure(fig, output_stem, dpi=300, grayscale_preview=True, strict_layout=True, strict_design=True)
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    generated = [Path(paths["svg"]), Path(paths["png"]), Path(paths["grayscale"]), pdf_path]
    return generated


def render_report(master: list[dict[str, Any]], transitions: list[dict[str, Any]], meta: dict[str, Any], figure_paths: list[Path]) -> str:
    lines = [
        "# Q3-T1 固定 Q₀ 主结果整合与资源转移分析",
        "",
        f"**生成时间（UTC）：** {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
        "**范围：** 固定 Q=Q₀、p 不进入 M0；在 Q2 B1 的精确观测 N/D 支持网格中复用既有 M0/M1/M3/M4 与配对 Bootstrap 结果。",
        "**方法优化：** 本整合只做带哈希核验的数据连接和派生差值，不重新拟合 M0、不重跑 1,000 次 Bootstrap；15 个场景使用预算与上下文双键，22 个转移保留原 pair_id。",
        "",
        "## 交付检查",
        "",
        f"- 主结果：{len(master)} 个预算×上下文场景。",
        f"- 资源转移：{len(transitions)} 对相邻预算/上下文场景。",
        f"- B1/M3 支持点：{meta['candidate_count']}；M4/M1 掩码各 {meta['m4_mask_rows']:,} 行（15 个场景，每场景 {meta['candidate_count']:,} 个候选点）。",
        f"- 独立成本审计一致：{meta['independent_audit_pass_scenarios']}/{len(master)} 个场景。",
        "- 15/15 个最优点均满足 M3 支持、M4 预算与 M1 最终可行标记；固定 Q₀ 下 C_Q=0，成本份额合计为 1。",
        "",
        "## 资源转移汇总",
        "",
        "| 比较类型 | 配对数 | Bootstrap 稳健 N/D 配置改变 | Bootstrap 稳健成本构成改变 |",
        "|---|---:|---:|---:|",
    ]
    for pair_type, label in (("adjacent_budget", "相邻预算"), ("adjacent_context", "相邻上下文")):
        item = meta["transition_summary"][pair_type]
        lines.append(
            f"| {label} | {item['pairs']} | {item['robust_ND_allocation_changes']}/{item['pairs']} | {item['robust_cost_mix_changes']}/{item['pairs']} |"
        )
    lines += [
        "",
        "资源转移表保留左右场景配置、Loss、成本、预算利用率、可行点数，并附原分析的 N/D 对数变化区间、成本份额总变差和稳健改变标记。配置改变与成本构成改变分开判断。",
        "",
        "## 核心图",
        "",
        "图左汇总 15 个情景的 N*、D*、M0 Loss* 和预算利用率；图右汇总 22 对配对 Bootstrap 的稳健变化比例。变化比例只表示离散配置或成本构成改变，不表示变化方向更优。",
        "",
    ]
    lines.extend(f"- `{path.relative_to(RESULTS).as_posix()}`" for path in figure_paths)
    lines += [
        "",
        "## 解释边界",
        "",
        "- 本表为固定 Q₀/M0 的 B1 支持网格参考结果；不含 Q/p 响应效应，不是完整 Q3 联合最优。Loss 为 B1 同域拟合 M0 的预测值。",
        "- 当前 M3 只开放精确观测组合；凸包与距离是诊断字段，不授权非观测插值。",
        "- 高预算所有 B1 网格点可行，最优点可能触及支持上界；预算利用率低不代表域外资源没有收益。",
        "- Bootstrap 使用既有 8 条独立轨迹、1,000 个配对复本；本次未重拟合或扩大不确定性解释。",
        "- T1 的 15 行主表用于汇总，不代替候选级 M3/M4/M1 掩码；近优/Pareto 下游任务仍应读取完整候选表。",
        "",
        f"复现命令：`$env:MPLBACKEND = 'Agg'; & '{Path(sys.executable).resolve().as_posix()}' 'Q3/03_代码/assemble_q3_fixedq_master_results.py'`。",
    ]
    return "\n".join(lines) + "\n"


def run_smoke(paths: dict[str, Path], out_dir: Path) -> int:
    master, transitions, meta = load_and_integrate(paths)
    write_smoke(master, transitions, meta, out_dir)
    return 0


def run_full(paths: dict[str, Path], out_dir: Path) -> int:
    master, transitions, meta = load_and_integrate(paths)
    write_csv(out_dir / OUT_MASTER, master, MASTER_FIELDS)
    write_csv(out_dir / OUT_TRANSITIONS, transitions, TRANSITION_FIELDS)
    contract_path = out_dir / OUT_CONTRACT
    contract_path.write_text(render_contract(), encoding="utf-8")
    figure_paths = create_core_figure(master, transitions, out_dir)
    report_path = out_dir / OUT_REPORT
    report_path.write_text(render_report(master, transitions, meta, figure_paths), encoding="utf-8")

    input_hashes = {key: sha256(path) for key, path in paths.items() if path.suffix.lower() in {".csv", ".json"}}
    outputs = [out_dir / OUT_MASTER, out_dir / OUT_TRANSITIONS, contract_path, report_path, *figure_paths]
    runtime_packages = {
        package: importlib.metadata.version(package)
        for package in ("matplotlib", "numpy", "pandas", "Pillow")
    }
    manifest = {
        "artifact": "Q3-T1 fixed-Q0 master scenario results and resource transitions",
        "command": f"$env:MPLBACKEND = 'Agg'; & '{Path(sys.executable).resolve().as_posix()}' 'Q3/03_代码/assemble_q3_fixedq_master_results.py'",
        "script": {"path": rel(Path(__file__)), "sha256": sha256(Path(__file__))},
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "environment": {"MPLBACKEND": "Agg"},
        "packages": runtime_packages,
        "scope": "fixed Q=Q0, no p response; exact B1/M3 supported grid only",
        "join_keys": {
            "scenario": ["budget_flops", "context_length_tokens"],
            "selected_candidate": ["scenario_id", "grid_id"],
            "transition_pair": "pair_id plus left/right budget-context endpoints",
        },
        "source_rows": {
            "master_scenarios": meta["scenario_count"],
            "observed_support_points": meta["candidate_count"],
            "budget_mask_rows": meta["m4_mask_rows"],
            "final_mask_rows": meta["m1_mask_rows"],
            "transition_pairs": meta["transition_pair_count"],
        },
        "reconciliation": {
            "independent_cost_audit_pass_scenarios": meta["independent_audit_pass_scenarios"],
            "all_selected_m3_m4_m1_feasible": all(as_bool(row["m3_search_allowed"], "M3") and as_bool(row["m4_budget_feasible"], "M4") and as_bool(row["m1_final_feasible"], "M1") for row in master),
            "transition_summary": meta["transition_summary"],
        },
        "inputs_sha256": input_hashes,
        "code_dependencies_sha256": {
            "utils/plot_style.py": sha256(ROOT / "utils/plot_style.py"),
        },
        "reproducibility_contract": {
            "byte_stable_outputs": [OUT_MASTER, OUT_TRANSITIONS, "figures/" + FIGURE_STEM + ".png", "figures/_qa/" + FIGURE_STEM + "_grayscale.png"],
            "metadata_may_vary": [OUT_REPORT + " generation timestamp", "SVG IDs/timestamp", "PDF creation metadata"],
            "verification": "Recompute table values and compare CSV/PNG outputs; regenerate this manifest for the current report/vector metadata hashes.",
        },
        "outputs_sha256": {rel(path): sha256(path) for path in outputs},
        "figure": {
            "stem": rel(out_dir / "figures" / FIGURE_STEM),
            "style": "utils.plot_style.apply_publication_style",
            "dpi": 300,
            "size_inches": [7.2, 5.0],
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = out_dir / OUT_MANIFEST
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Integrated {len(master)} scenarios and {len(transitions)} transitions.")
    print(f"Independent cost audit matches: {meta['independent_audit_pass_scenarios']}/{len(master)}.")
    for path in outputs + [manifest_path]:
        print(f"Wrote {rel(path)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the real-data P1 slice; writes one scenario and one pair")
    parser.add_argument("--output-dir", type=Path, help="output directory (relative paths resolve from project root)")
    args = parser.parse_args()
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = SMOKE_DIR if args.smoke else RESULTS
    elif not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    paths = source_paths()
    if args.smoke:
        return run_smoke(paths, out_dir)
    return run_full(paths, out_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, IntegrationError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
