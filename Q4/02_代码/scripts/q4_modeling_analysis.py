"""Q4 provisional aggregation, scale decomposition, and forecast feasibility.

AI disclosure: OpenAI Codex desktop 26.917.71314 (build 10954; prod), model ID
gpt-6-luna (GPT-6 Luna), developer OpenAI, official release date 2026-09-22.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
RESULTS = Q4 / "03_结果" / "results"
RAW_C = ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"
AI_DISCLOSURE = (
    "OpenAI Codex desktop 26.917.71314 (build 10954; prod); model ID "
    "gpt-6-luna (GPT-6 Luna); developer OpenAI; official release date 2026-09-22."
)

FAMILY_ORDER = ["ARC", "BBH", "GPQA", "IFEval", "MATH-Hard", "MMLU-Pro", "MuSR"]
CORE_FAMILIES = ["BBH", "GPQA", "IFEval", "MATH-Hard", "MMLU-Pro", "MuSR"]
EXPECTED_TASKS = {"ARC": 1, "BBH": 24, "GPQA": 3, "IFEval": 1, "MATH-Hard": 7, "MMLU-Pro": 1, "MuSR": 3}
PRIMARY_VERSIONS = {"ARC": "1.0", "BBH": "0.0", "GPQA": "1.0", "IFEval": "2.0", "MATH-Hard": "2.0", "MMLU-Pro": "0.1", "MuSR": "1.0"}
PRIMARY_METRICS = {"ARC": "acc_norm", "BBH": "acc_norm", "GPQA": "acc_norm", "IFEval": "inst_level_strict_acc", "MATH-Hard": "exact_match", "MMLU-Pro": "acc", "MuSR": "acc_norm"}
IFEVAL_SENSITIVITY_METRICS = ["prompt_level_strict_acc", "inst_level_loose_acc", "prompt_level_loose_acc"]
EXPECTED_CORE_LEAF_TASKS = sum(EXPECTED_TASKS[f] for f in CORE_FAMILIES)
EARLY_MONTH = "2024-11"
LATE_MONTH = "2025-03"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")


def family_for_task(task: str) -> str | None:
    if task == "leaderboard_arc_challenge":
        return "ARC"
    if task.startswith("leaderboard_bbh_"):
        return "BBH"
    if task.startswith("leaderboard_gpqa_"):
        return "GPQA"
    if task == "leaderboard_ifeval":
        return "IFEval"
    if task.startswith("leaderboard_math_") and task.endswith("_hard"):
        return "MATH-Hard"
    if task == "leaderboard_mmlu_pro":
        return "MMLU-Pro"
    if task.startswith("leaderboard_musr_"):
        return "MuSR"
    return None


def make_selected_scores(
    source: pd.DataFrame,
    cohort_names: set[str],
    math_version: str = "2.0",
    ifeval_metric: str = "inst_level_strict_acc",
    exclude_fallback: bool = False,
) -> pd.DataFrame:
    s = source.copy()
    s["family"] = s["task_key"].map(family_for_task)
    s = s[s["family"].notna() & s["model_name_raw"].isin(cohort_names)].copy()
    s["task_version"] = s["task_version"].fillna("").astype(str)
    s["value_numeric"] = pd.to_numeric(s["value_numeric"], errors="coerce")
    direction = s["higher_is_better"].astype(str).str.lower().eq("true")
    s["expected_metric"] = s["family"].map(PRIMARY_METRICS)
    s.loc[s["family"].eq("IFEval"), "expected_metric"] = ifeval_metric
    expected_version = s["family"].map(PRIMARY_VERSIONS)
    expected_version.loc[s["family"].eq("MATH-Hard")] = math_version
    keep = (
        s["task_version"].eq(expected_version.astype(str))
        & s["metric_role"].eq("score")
        & s["metric_filter"].fillna("").eq("none")
        & s["metric_name"].eq(s["expected_metric"])
        & direction
        & s["value_status"].eq("numeric")
        & s["value_numeric"].notna()
    )
    s = s[keep].copy()
    if exclude_fallback:
        s = s[~s["latest_status"].eq("latest_parseable_fallback")].copy()
    s["score_raw"] = s["value_numeric"]
    key = ["model_name_raw", "family", "task_key", "task_version", "metric_key"]
    if s.duplicated(key).any():
        raise ValueError("duplicate selected task-score key found")
    s["task_percentile_0_100"] = (
        s.groupby(["task_key", "task_version", "metric_name"])["score_raw"]
        .rank(method="average", pct=True) * 100.0
    )
    return s


def build_family_table(selected: pd.DataFrame, model_names: list[str], math_version: str = "2.0") -> pd.DataFrame:
    grouped = (
        selected.groupby(["model_name_raw", "family"], as_index=False)
        .agg(
            task_count=("task_key", "nunique"),
            family_score_raw=("score_raw", "mean"),
            family_score_percentile_0_100=("task_percentile_0_100", "mean"),
        )
    )
    full = pd.MultiIndex.from_product([model_names, FAMILY_ORDER], names=["model_name_raw", "family"]).to_frame(index=False)
    full = full.merge(grouped, on=["model_name_raw", "family"], how="left")
    full["expected_task_count"] = full["family"].map(EXPECTED_TASKS).astype(int)
    full["task_count"] = full["task_count"].fillna(0).astype(int)
    full["family_complete"] = full["task_count"].eq(full["expected_task_count"])
    full.loc[~full["family_complete"], ["family_score_raw", "family_score_percentile_0_100"]] = np.nan
    full["version"] = full["family"].map(PRIMARY_VERSIONS)
    full.loc[full["family"].eq("MATH-Hard"), "version"] = math_version
    full["metric"] = full["family"].map(PRIMARY_METRICS)
    return full


def make_index_table(selected: pd.DataFrame, family_table: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    names = base["model_name_raw"].tolist()
    pivot = family_table.pivot(index="model_name_raw", columns="family", values="family_score_percentile_0_100")
    raw_pivot = family_table.pivot(index="model_name_raw", columns="family", values="family_score_raw")
    counts = family_table.pivot(index="model_name_raw", columns="family", values="task_count")
    pivot = pivot.reindex(index=names, columns=FAMILY_ORDER)
    raw_pivot = raw_pivot.reindex(index=names, columns=FAMILY_ORDER)
    counts = counts.reindex(index=names, columns=FAMILY_ORDER).fillna(0)
    result = base.set_index("model_name_raw").copy()
    for family in FAMILY_ORDER:
        result[f"{family}_score_raw"] = raw_pivot[family]
        result[f"{family}_score_percentile_0_100"] = pivot[family]
        result[f"{family}_task_count"] = counts[family].astype(int)
    result["core6_family_count"] = pivot[CORE_FAMILIES].notna().sum(axis=1)
    result["core6_index_0_100"] = pivot[CORE_FAMILIES].mean(axis=1, skipna=False)
    result["full7_index_0_100"] = pivot[FAMILY_ORDER].mean(axis=1, skipna=False)
    result["core6_rawmean_index_0_100"] = raw_pivot[CORE_FAMILIES].mean(axis=1, skipna=False) * 100.0
    core_rows = selected[selected["family"].isin(CORE_FAMILIES)]
    core_n = core_rows.groupby("model_name_raw")["task_key"].nunique().reindex(names).fillna(0).astype(int)
    result["core6_leaf_task_count"] = core_n
    result["core6_all_task_index_0_100"] = np.nan
    all_task_mean = core_rows.groupby("model_name_raw")["task_percentile_0_100"].mean().reindex(names)
    all_task_raw_mean = core_rows.groupby("model_name_raw")["score_raw"].mean().reindex(names)
    complete = core_n.eq(EXPECTED_CORE_LEAF_TASKS)
    result.loc[complete, "core6_all_task_index_0_100"] = all_task_mean[complete]
    result["core6_all_task_rawmean_0_100"] = np.nan
    result.loc[complete, "core6_all_task_rawmean_0_100"] = all_task_raw_mean[complete] * 100.0
    complete_family = family_table.pivot(index="model_name_raw", columns="family", values="family_complete").reindex(index=names, columns=FAMILY_ORDER)
    for family in FAMILY_ORDER:
        result[f"{family}_complete"] = complete_family[family].fillna(False).astype(bool)
    return result.reset_index()


def pair_metrics(primary: pd.Series, alternative: pd.Series, same_scale: bool = True) -> dict[str, float]:
    joined = pd.concat([primary.rename("main"), alternative.rename("alt")], axis=1).dropna()
    if len(joined) < 3:
        return {"rho": np.nan, "mae": np.nan}
    # Rank explicitly and use numpy for the rank correlation.
    a = joined["main"].to_numpy(dtype=float)
    b = joined["alt"].to_numpy(dtype=float)
    ar = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    ar = ar - ar.mean()
    br = br - br.mean()
    denominator = math.sqrt(float(np.sum(ar * ar) * np.sum(br * br)))
    rho = float(np.sum(ar * br) / denominator) if denominator else np.nan
    return {"rho": rho, "mae": float(np.mean(np.abs(a - b))) if same_scale else np.nan}


def summary_row(label: str, frame: pd.DataFrame, column: str, paired: dict[str, float] | None = None) -> dict[str, Any]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return {
        "index_variant": label,
        "n_models": int(values.size),
        "mean_0_100": float(values.mean()) if len(values) else np.nan,
        "median_0_100": float(values.median()) if len(values) else np.nan,
        "sd_0_100": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "p10_0_100": float(values.quantile(0.10)) if len(values) else np.nan,
        "p90_0_100": float(values.quantile(0.90)) if len(values) else np.nan,
        "spearman_vs_primary": paired["rho"] if paired else np.nan,
        "mean_abs_difference_vs_primary": paired["mae"] if paired else np.nan,
    }


def normalize_type(value: Any) -> str:
    s = "" if pd.isna(value) else str(value).lower()
    if "continuously pretrained" in s:
        return "continuously_pretrained"
    if "pretrained" in s:
        return "pretrained"
    if "chat models" in s:
        return "chat_posttrained"
    if "fine-tuned on domain-specific" in s:
        return "domain_finetuned"
    if "base merges" in s or "moerges" in s:
        return "merge"
    if "multimodal" in s:
        return "multimodal"
    if "other" in s:
        return "other"
    return "unknown_or_unmapped"


def map_type_groups(series: pd.Series, min_count: int = 20) -> pd.Series:
    groups = series.map(normalize_type)
    rare = groups.value_counts()[lambda v: v < min_count].index
    return groups.where(~groups.isin(rare), "other_rare").fillna("other_rare")


def invert_small_matrix(matrix: list[list[float]]) -> list[list[float]] | None:
    """Gauss-Jordan inverse for the small fixed-effect design matrices here."""
    n = len(matrix)
    left = [list(map(float, row)) for row in matrix]
    right = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(left[row][col]))
        if abs(left[pivot][col]) < 1e-12:
            return None
        left[col], left[pivot] = left[pivot], left[col]
        right[col], right[pivot] = right[pivot], right[col]
        divisor = left[col][col]
        left[col] = [value / divisor for value in left[col]]
        right[col] = [value / divisor for value in right[col]]
        for row in range(n):
            if row == col:
                continue
            factor = left[row][col]
            if factor:
                left[row] = [a - factor * b for a, b in zip(left[row], left[col])]
                right[row] = [a - factor * b for a, b in zip(right[row], right[col])]
    return right


def multiply_small(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows, inner, cols = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def fit_hc3(frame: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if len(frame) < 30:
        return None
    categories = sorted(frame["type_group"].unique())
    months = sorted(frame["submission_month"].unique())
    ref_type, ref_month = categories[0], months[0]
    columns = [np.ones(len(frame)), frame["log10_params_B"].to_numpy(float)]
    terms = ["intercept", "log10_params_B"]
    for group in categories[1:]:
        columns.append(frame["type_group"].eq(group).to_numpy(float))
        terms.append(f"type[{group}]")
    for month in months[1:]:
        columns.append(frame["submission_month"].eq(month).to_numpy(float))
        terms.append(f"month[{month}]")
    x = np.column_stack(columns)
    y = frame["core6_index_0_100"].to_numpy(float)
    p = x.shape[1]
    xtx = [[float(np.sum(x[:, i] * x[:, j])) for j in range(p)] for i in range(p)]
    xtx_inv_list = invert_small_matrix(xtx)
    if xtx_inv_list is None:
        return None
    xtx_inv = np.asarray(xtx_inv_list, dtype=float)
    xty = np.array([float(np.sum(x[:, i] * y)) for i in range(p)])
    beta = np.array([float(np.sum(xtx_inv[i, :] * xty)) for i in range(p)])
    fitted = np.sum(x * beta[None, :], axis=1)
    residual = y - fitted
    x_inv = np.empty_like(x)
    for j in range(p):
        x_inv[:, j] = sum(x[:, i] * xtx_inv[i, j] for i in range(p))
    hat = np.sum(x_inv * x, axis=1)
    weighted_x = x * ((residual / np.maximum(1 - hat, 1e-8)) ** 2)[:, None]
    meat = [[float(np.sum(weighted_x[:, i] * x[:, j])) for j in range(p)] for i in range(p)]
    covariance = multiply_small(multiply_small(xtx_inv_list, meat), xtx_inv_list)
    se = np.sqrt(np.maximum(np.array([covariance[i][i] for i in range(p)]), 0))
    rows = [{
        "scope": scope, "term": term, "estimate": float(coef), "hc3_se": float(err),
        "ci95_low": float(coef - 1.96 * err), "ci95_high": float(coef + 1.96 * err), "n_models": len(frame),
    } for term, coef, err in zip(terms, beta, se)]
    r2 = 1 - float((residual ** 2).sum() / ((y - y.mean()) ** 2).sum())
    info = {
        "terms": terms, "beta": beta, "references": {"type": ref_type, "month": ref_month},
        "residual": residual, "fitted": fitted, "r_squared": r2, "n_models": len(frame),
    }
    return pd.DataFrame(rows), info


def scale_tech_outputs(aggregate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    d = aggregate[
        aggregate["core6_index_0_100"].notna()
        & aggregate["parameter_count_B"].gt(0)
        & aggregate["submission_date"].notna()
    ].copy()
    d["log10_params_B"] = np.log10(d["parameter_count_B"].astype(float))
    d["submission_month"] = d["submission_date"].dt.to_period("M").astype(str)
    d["type_group"] = map_type_groups(d["C1_Type_raw"])
    d = d.sort_values(["submission_month", "model_name_raw"]).reset_index(drop=True)
    month_counts_all = d["submission_month"].value_counts()
    supported_months = sorted(month_counts_all[month_counts_all.ge(20)].index.tolist())
    if not supported_months or supported_months[0] != EARLY_MONTH or supported_months[-1] != LATE_MONTH:
        raise ValueError(
            f"Contract endpoint months unavailable: expected {EARLY_MONTH}/{LATE_MONTH}, "
            f"supported={supported_months}"
        )
    fit_d = d[d["submission_month"].isin(supported_months)].copy()
    fit = fit_hc3(fit_d, "complete_core6_month_n_ge_20")
    if fit is None:
        raise ValueError("Scale/time/type model is not estimable on months with n>=20")
    coef_main, info = fit
    params = dict(zip(info["terms"], info["beta"]))
    type_effects = {info["references"]["type"]: 0.0}
    month_effects = {info["references"]["month"]: 0.0}
    for group in d["type_group"].unique():
        type_effects[group] = float(params.get(f"type[{group}]", 0.0))
    for month in d["submission_month"].unique():
        month_effects[month] = float(params.get(f"month[{month}]", 0.0))
    global_log = float(fit_d["log10_params_B"].mean())
    global_y = float(fit_d["core6_index_0_100"].mean())
    global_type_share = fit_d["type_group"].value_counts(normalize=True).to_dict()
    month_counts = fit_d["submission_month"].value_counts().to_dict()
    time_mean = sum(month_effects.get(m, 0.0) * n for m, n in month_counts.items()) / len(fit_d)
    monthly_rows = []
    for month, group in d.groupby("submission_month", sort=True):
        decomposition_supported = month in supported_months
        if decomposition_supported:
            shares = group["type_group"].value_counts(normalize=True).to_dict()
            scale = float(params["log10_params_B"]) * (float(group["log10_params_B"].mean()) - global_log)
            type_mix = sum(type_effects.get(t, 0.0) * (shares.get(t, 0.0) - global_type_share.get(t, 0.0)) for t in global_type_share)
            time = month_effects.get(month, 0.0) - time_mean
            observed_delta = float(group["core6_index_0_100"].mean()) - global_y
            unexplained = observed_delta - scale - type_mix - time
        else:
            scale = type_mix = time = observed_delta = unexplained = np.nan
        monthly_rows.append({
            "submission_month": month, "n_models": len(group),
            "mean_core6_index_0_100": float(group["core6_index_0_100"].mean()),
            "median_core6_index_0_100": float(group["core6_index_0_100"].median()),
            "p90_core6_index_0_100": float(group["core6_index_0_100"].quantile(.9)),
            "mean_parameter_count_B": float(group["parameter_count_B"].mean()),
            "median_parameter_count_B": float(group["parameter_count_B"].median()),
            "mean_log10_params_B": float(group["log10_params_B"].mean()),
            "mean_C1_average_0_100": float(group["C1_Average_0_100"].mean()),
            "scale_composition_component": scale,
            "model_type_composition_component": type_mix,
            "month_fixed_effect_component": time,
            "unexplained_component": unexplained,
            "observed_mean_change_vs_global": observed_delta,
            "open_weights_yes_count": int(group["open_weights_status"].eq("Yes").sum()),
            "decomposition_supported_month_n_ge_20": decomposition_supported,
        })
    monthly = pd.DataFrame(monthly_rows)
    eligible = supported_months
    common = {"status": "not_assessed", "early_month": "", "late_month": "", "log10_params_low": np.nan, "log10_params_high": np.nan, "n_early": 0, "n_late": 0}
    coefficient_frames = [coef_main]
    if len(eligible) >= 2:
        early, late = eligible[0], eligible[-1]
        e, l = fit_d[fit_d["submission_month"].eq(early)], fit_d[fit_d["submission_month"].eq(late)]
        low = max(e["log10_params_B"].quantile(.05, interpolation="linear"), l["log10_params_B"].quantile(.05, interpolation="linear"))
        high = min(e["log10_params_B"].quantile(.95, interpolation="linear"), l["log10_params_B"].quantile(.95, interpolation="linear"))
        common.update({
            "status": "overlap" if low < high else "no_overlap", "early_month": early, "late_month": late,
            "log10_params_low": float(low), "log10_params_high": float(high),
            "n_early": int(e["log10_params_B"].between(low, high).sum()) if low < high else 0,
            "n_late": int(l["log10_params_B"].between(low, high).sum()) if low < high else 0,
        })
        if low < high:
            restricted = fit_d[fit_d["log10_params_B"].between(low, high)].copy()
            fit_restricted = fit_hc3(restricted, "common_parameter_support_early_late_p05_p95")
            if fit_restricted is not None:
                coefficient_frames.append(fit_restricted[0])
    for table in coefficient_frames:
        table["r_squared"] = info["r_squared"] if table is coef_main else np.nan
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    monthly["common_parameter_support_status"] = common["status"]
    monthly["common_support_early_month"] = common["early_month"]
    monthly["common_support_late_month"] = common["late_month"]
    info_out = {
        "n_models_in_scale_model": int(len(d)),
        "n_models_in_supported_month_regression": int(len(fit_d)),
        "excluded_sparse_months": {str(month): int(n) for month, n in month_counts_all[month_counts_all.lt(20)].items()},
        "n_complete_core6": int(aggregate["core6_index_0_100"].notna().sum()),
        "beta_log10_params_B": float(params["log10_params_B"]),
        "r_squared": float(info["r_squared"]),
        "reference_type_group": info["references"]["type"],
        "reference_month": info["references"]["month"],
        "month_count": int(fit_d["submission_month"].nunique()),
        "common_support": common,
        "type_group_counts": fit_d["type_group"].value_counts().sort_index().to_dict(),
    }
    return monthly, coefficients, fit_d, info_out


def build_frontier(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = meta[
        meta["open_weights_status"].eq("Yes") & meta["submission_date"].notna()
        & meta["C1_Average_0_100"].notna()
    ].copy()
    d["submission_month"] = d["submission_date"].dt.to_period("M").astype(str)
    rows = []
    for month, group in d.groupby("submission_month", sort=True):
        values = group["C1_Average_0_100"].astype(float)
        top = values.idxmax()
        rows.append({
            "submission_month": month, "n_known_open_weights_models": len(group),
            "monthly_entry_median_C1_average_0_100": float(values.median()),
            "monthly_entry_p90_frontier_C1_average_0_100": float(values.quantile(.9)),
            "monthly_entry_max_C1_average_0_100": float(values.max()),
            "monthly_entry_top_model": str(group.loc[top, "model_name_raw"]),
            "monthly_entry_top_model_parameter_count_B": float(group.loc[top, "parameter_count_B"]) if pd.notna(group.loc[top, "parameter_count_B"]) else np.nan,
            "small_cell_lt10": bool(len(group) < 10),
        })
    frontier = pd.DataFrame(rows)
    bt = []
    reliable = frontier[~frontier["small_cell_lt10"]].copy()
    if not reliable.empty:
        reliable["month_num"] = pd.PeriodIndex(reliable["submission_month"], freq="M").astype(int)
        for i in range(3, len(reliable)):
            train, target = reliable.iloc[:i], reliable.iloc[i]
            x, y = train["month_num"].to_numpy(float), train["monthly_entry_p90_frontier_C1_average_0_100"].to_numpy(float)
            x_center, y_center = float(x.mean()), float(y.mean())
            denominator = float(np.sum((x - x_center) ** 2))
            slope = float(np.sum((x - x_center) * (y - y_center)) / denominator) if denominator else 0.0
            pred = y_center + slope * (float(target["month_num"]) - x_center)
            last = float(train.iloc[-1]["monthly_entry_p90_frontier_C1_average_0_100"])
            observed = float(target["monthly_entry_p90_frontier_C1_average_0_100"])
            for method, predicted in [("linear_time_trend", pred), ("last_value", last)]:
                bt.append({
                    "origin_month": str(train.iloc[-1]["submission_month"]), "target_month": str(target["submission_month"]),
                    "method": method, "observed_p90": observed, "predicted_p90": predicted,
                    "error_pred_minus_observed": predicted - observed, "absolute_error": abs(predicted - observed),
                    "training_reliable_months": len(train),
                })
    return frontier, pd.DataFrame(bt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-only", action="store_true", help="Write base task/frontier tables for the dynamic stage, then stop.")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("[q4] reading inputs", flush=True)
    c8_path = RESULTS / "q4_c8_long_latest.csv.gz"
    candidate_path = RESULTS / "q4_c8_c1_match_candidates.csv"
    c1_path = RAW_C / "leaderboard_cleaned.csv"
    coverage_path = RESULTS / "q4_c8_task_metric_coverage_latest.csv"
    contract_path = Q4 / "01_方案说明" / "q4_model_contract_v0.6.md"
    gate_path = RESULTS / "q4_loss_benchmark_feasibility.csv"
    dynamic_report_path = Q4 / "01_方案说明" / "q4_dynamic_decomposition_report.md"
    dynamic_input_paths = [
        RESULTS / "q4_ar1_dynamic.csv", RESULTS / "q4_ar1_backtest_summary.csv",
        RESULTS / "q4_scale_tech_contribution.csv", RESULTS / "q4_scale_tech_support_audit.csv",
        RESULTS / "q4_scale_tech_contribution_coefficients.csv",
        RESULTS / "q4_c3_year_summary.csv", RESULTS / "q4_c4_year_summary.csv",
        RESULTS / "q4_c4_cutoff_row_audit.csv",
    ]
    core_input_paths = [c8_path, candidate_path, c1_path, coverage_path, contract_path, gate_path]
    input_paths = [*core_input_paths, dynamic_report_path, *dynamic_input_paths]

    long = pd.read_csv(c8_path, low_memory=False)
    candidate = pd.read_csv(candidate_path, low_memory=False)
    candidate = candidate[candidate["match_status"].eq("exact_model_id_unique_record")].copy()
    c1 = pd.read_csv(c1_path, low_memory=False)
    gate_table = pd.read_csv(gate_path, encoding="utf-8-sig")
    gate_row = gate_table.loc[gate_table["scope"].eq("pooled_C5_C6_deduplicated")].iloc[0]
    dynamic_ready = not args.bootstrap_only and dynamic_report_path.is_file() and all(path.is_file() for path in dynamic_input_paths)
    print("[q4] joining cohort", flush=True)
    if candidate["c8_model_name_raw"].duplicated().any():
        raise ValueError("unique-name candidate table contains duplicate C8 model names")
    names = set(candidate["c8_model_name_raw"].astype(str))
    c1_unique = c1[c1["Model"].isin(names)].copy()
    if c1_unique["Model"].duplicated().any() or len(candidate) != 1819 or len(c1_unique) != len(candidate):
        raise ValueError("C1 join does not reproduce the 1,819 unique-name/unique-row candidate cohort")

    base = candidate.merge(c1_unique, left_on="c8_model_name_raw", right_on="Model", how="inner", validate="one_to_one")
    base = base.rename(columns={
        "c8_model_name_raw": "model_name_raw", "#Params (B)": "parameter_count_B",
        "Submission Date": "submission_date_raw", "Hub License": "C1_Hub_License_raw",
        "Type": "C1_Type_raw", "Average ⬆️": "C1_Average_0_100", "IFEval": "C1_IFEval_0_100",
        "BBH": "C1_BBH_0_100", "MATH Lvl 5": "C1_MATH_Lvl5_0_100", "GPQA": "C1_GPQA_0_100",
        "MUSR": "C1_MuSR_0_100", "MMLU-PRO": "C1_MMLU_Pro_0_100",
    })
    base["parameter_count_B"] = pd.to_numeric(base["parameter_count_B"], errors="coerce")
    numeric_c1 = ["C1_Average_0_100", "C1_IFEval_0_100", "C1_BBH_0_100", "C1_MATH_Lvl5_0_100", "C1_GPQA_0_100", "C1_MuSR_0_100", "C1_MMLU_Pro_0_100"]
    for col in numeric_c1:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base["submission_date"] = pd.to_datetime(base["submission_date_raw"], errors="coerce")
    base["open_weights_status"] = base["c2_epoch_open_weights_candidate"].where(
        base["c2_epoch_open_weights_candidate"].isin(["Yes", "No"]), "unknown"
    )
    base["version_match_status"] = "source_version_missing"
    base["version_unresolved"] = True
    run_status = long[["run_id", "latest_status"]].drop_duplicates("run_id")
    name_status = candidate[["c8_model_name_raw", "run_id"]].merge(run_status, on="run_id", how="left")
    fallback = name_status.groupby("c8_model_name_raw")["latest_status"].first().to_dict()
    base["c8_latest_status"] = base["model_name_raw"].map(fallback)

    leaf_scores = long[long["task_role"].eq("task_subtask") & long["metric_role"].eq("score")].copy()
    model_names = base["model_name_raw"].tolist()
    selected = make_selected_scores(leaf_scores, names)
    print(f"[q4] selected {len(selected)} task score rows", flush=True)
    family_table = build_family_table(selected, model_names)
    aggregate = make_index_table(selected, family_table, base)
    print("[q4] built primary aggregate", flush=True)
    aggregate["main_cohort_complete_core6"] = aggregate["core6_index_0_100"].notna()
    aggregate["main_cohort_complete_full7"] = aggregate["full7_index_0_100"].notna()
    aggregate["index_scale"] = "task-wise percentile 0-100 within exact-name candidate cohort; equal leaf-task weights within family; equal family weights"
    aggregate["cohort_status"] = "provisional_exact_name_unique_C1_row; version_unresolved"

    write_csv(selected[[
        "observation_id", "model_name_raw", "model_dir", "family", "task_key", "task_version",
        "metric_key", "metric_name", "score_raw", "task_percentile_0_100", "stderr_value_numeric", "n_effective",
        "higher_is_better", "latest_status",
    ]].assign(version_unresolved=True), RESULTS / "q4_selected_task_scores.csv")
    write_csv(family_table, RESULTS / "q4_task_family_scores.csv")
    print("[q4] wrote primary task tables", flush=True)

    # Index and metric sensitivity summaries; alternate MATH versions remain separate cohorts.
    sensitivity = []
    primary_series = aggregate.set_index("model_name_raw")["core6_index_0_100"]
    sensitivity.append(summary_row("Primary: six families, equal family weights", aggregate, "core6_index_0_100"))
    sensitivity.append(summary_row("All seven families incl. ARC", aggregate, "full7_index_0_100", pair_metrics(primary_series, aggregate.set_index("model_name_raw")["full7_index_0_100"])))
    sensitivity.append(summary_row("Core six families, all 39 leaf tasks equal weight", aggregate, "core6_all_task_index_0_100", pair_metrics(primary_series, aggregate.set_index("model_name_raw")["core6_all_task_index_0_100"])))
    sensitivity.append(summary_row("Core six families, raw accuracy equal-family mean", aggregate, "core6_rawmean_index_0_100", pair_metrics(primary_series, aggregate.set_index("model_name_raw")["core6_rawmean_index_0_100"], same_scale=False)))

    for metric in IFEVAL_SENSITIVITY_METRICS:
        print(f"[q4] IFEval sensitivity {metric}", flush=True)
        alt_scores = make_selected_scores(leaf_scores, names, ifeval_metric=metric)
        alt_families = build_family_table(alt_scores, model_names)
        alt = make_index_table(alt_scores, alt_families, base)
        alt_col = f"core6_if_eval_{metric}_0_100"
        aggregate = aggregate.merge(
            alt[["model_name_raw", "core6_index_0_100"]].rename(columns={"core6_index_0_100": alt_col}),
            on="model_name_raw", how="left", validate="one_to_one"
        )
        sensitivity.append(summary_row(f"IFEval sensitivity: {metric}", alt, "core6_index_0_100", pair_metrics(primary_series, alt.set_index("model_name_raw")["core6_index_0_100"])))

    for version in ["1.0", "3.0"]:
        print(f"[q4] MATH sensitivity {version}", flush=True)
        alt_scores = make_selected_scores(leaf_scores, names, math_version=version)
        alt_families = build_family_table(alt_scores, model_names, math_version=version)
        alt = make_index_table(alt_scores, alt_families, base)
        col = f"core6_math_v{version.replace('.', '')}_0_100"
        aggregate[col] = aggregate["model_name_raw"].map(alt.set_index("model_name_raw")["core6_index_0_100"])
        sensitivity.append(summary_row(f"MATH Hard v{version} replacement; separate cohort", alt, "core6_index_0_100"))

    no_fallback = aggregate[~aggregate["c8_latest_status"].eq("latest_parseable_fallback")]
    sensitivity.append(summary_row("Primary with C8 fallback excluded", no_fallback, "core6_index_0_100", pair_metrics(primary_series, no_fallback.set_index("model_name_raw")["core6_index_0_100"])))
    open_known = aggregate[aggregate["open_weights_status"].eq("Yes")]
    sensitivity.append(summary_row("Primary within C2 known Open Weights=Yes", open_known, "core6_index_0_100", pair_metrics(primary_series, open_known.set_index("model_name_raw")["core6_index_0_100"])))
    sensitivity_table = pd.DataFrame(sensitivity)
    sensitivity_table["index_basis"] = np.where(
        sensitivity_table["index_variant"].str.contains("raw accuracy", case=False),
        "raw C8 accuracy family mean",
        "within-leaf-task percentile rank; MATH-version rows use separate cohorts",
    )
    sensitivity_table["note"] = "Percentile-index variants are on a 0-100 relative-rank scale. Raw-accuracy row is a separate scale sensitivity. MATH-version cohorts are not directly comparable."
    write_csv(sensitivity_table, RESULTS / "q4_index_sensitivity.csv")
    print("[q4] wrote index sensitivity", flush=True)

    family_pct = family_table.pivot(index="model_name_raw", columns="family", values="family_score_percentile_0_100")
    family_pct.columns = ["family_" + re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_") + "_percentile_0_100" for c in family_pct.columns]
    family_raw = family_table.pivot(index="model_name_raw", columns="family", values="family_score_raw") * 100.0
    family_raw.columns = ["family_" + re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_") + "_rawmean_0_100" for c in family_raw.columns]
    family_wide = pd.concat([family_pct, family_raw], axis=1).reset_index()
    aggregate = aggregate.merge(family_wide, on="model_name_raw", how="left", validate="one_to_one")
    write_csv(aggregate, RESULTS / "q4_task_aggregate.csv")

    coverage_rows = [
        {"dimension": "cohort", "level": "exact unique C1 row/name candidates", "n_models": len(aggregate), "expected": 1819,
         "note": "Version unresolved; 39 duplicate-name groups and two unmatched C8 names excluded."}
    ]
    for family in FAMILY_ORDER:
        f = family_table[family_table["family"].eq(family)]
        coverage_rows.append({
            "dimension": "task_family", "level": family,
            "n_models_any_numeric_task": int(f["task_count"].gt(0).sum()),
            "n_models_complete_family": int(f["family_complete"].sum()),
            "expected_task_count": EXPECTED_TASKS[family],
            "version": PRIMARY_VERSIONS[family], "metric": PRIMARY_METRICS[family],
            "note": "Complete requires every declared leaf task, selected version/metric, numeric score, and known higher-is-better direction."
        })
    coverage_rows.extend([
        {"dimension": "index", "level": "core6_equal_family", "n_models": int(aggregate["core6_index_0_100"].notna().sum()), "expected": len(aggregate), "note": "Complete in six families; ARC excluded from main index due low coverage."},
        {"dimension": "index", "level": "full7_equal_family", "n_models": int(aggregate["full7_index_0_100"].notna().sum()), "expected": len(aggregate), "note": "Requires ARC-Challenge as seventh family."},
        {"dimension": "parameter", "level": "positive C1 params among core6", "n_models": int((aggregate["core6_index_0_100"].notna() & aggregate["parameter_count_B"].gt(0)).sum()), "expected": int(aggregate["core6_index_0_100"].notna().sum()), "note": "No model-name parameter imputation."},
        {"dimension": "open_weights", "level": "C2 Yes among candidates", "n_models": int(aggregate["open_weights_status"].eq("Yes").sum()), "expected": len(aggregate), "note": "C2 source candidate retained; No and unknown are separate."},
        {"dimension": "fallback", "level": "C8 latest parseable fallback", "n_models": int(aggregate["c8_latest_status"].eq("latest_parseable_fallback").sum()), "expected": len(aggregate), "note": "Included in primary; exclusion reported as sensitivity."},
    ])
    write_csv(pd.DataFrame(coverage_rows), RESULTS / "q4_coverage_by_dimension.csv")

    monthly, coefficients, scale_sample, scale_info = scale_tech_outputs(aggregate)
    scale_universe = aggregate[
        aggregate["core6_index_0_100"].notna() & aggregate["parameter_count_B"].gt(0)
        & aggregate["submission_date"].notna()
    ].copy()
    scale_universe["type_group"] = map_type_groups(scale_universe["C1_Type_raw"])
    type_rows = []
    for raw_type, _model_group in aggregate.groupby("C1_Type_raw", dropna=False, sort=True):
        type_mask = aggregate["C1_Type_raw"].isna() if pd.isna(raw_type) else aggregate["C1_Type_raw"].eq(raw_type)
        eligible_group = scale_universe[scale_universe["C1_Type_raw"].eq(raw_type)] if pd.notna(raw_type) else scale_universe[scale_universe["C1_Type_raw"].isna()]
        fit_group = scale_sample[scale_sample["C1_Type_raw"].eq(raw_type)] if pd.notna(raw_type) else scale_sample[scale_sample["C1_Type_raw"].isna()]
        type_group = str(eligible_group["type_group"].iloc[0]) if len(eligible_group) else "not_in_scale_universe"
        type_rows.append({
            "C1_Type_raw": "(missing)" if pd.isna(raw_type) else str(raw_type),
            "normalized_regression_group": type_group,
            "candidate_count": int(type_mask.sum()),
            "complete_core6_count": int((type_mask & aggregate["core6_index_0_100"].notna()).sum()),
            "scale_universe_positive_params_and_date_count": int(len(eligible_group)),
            "supported_month_regression_count": int(len(fit_group)),
            "rare_groups_pooled_as_other_rare": type_group == "other_rare",
        })
    write_csv(pd.DataFrame(type_rows), RESULTS / "q4_type_group_crosswalk.csv")
    print("[q4] fitted scale/type/time model", flush=True)
    write_csv(monthly, RESULTS / "q4_scale_tech_monthly.csv")
    write_csv(coefficients, RESULTS / "q4_scale_tech_coefficients.csv")
    frontier, short_backtest = build_frontier(aggregate)
    write_csv(frontier, RESULTS / "q4_frontier_monthly.csv")
    write_csv(short_backtest, RESULTS / "q4_forecast_backtest_1month.csv")
    if short_backtest.empty:
        bt_summary = pd.DataFrame(columns=["method", "origins", "mae", "median_absolute_error"])
    else:
        bt_summary = short_backtest.groupby("method", as_index=False).agg(
            origins=("target_month", "nunique"), mae=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median")
        )
    write_csv(bt_summary, RESULTS / "q4_forecast_backtest_1month_summary.csv")
    bt_mae = dict(zip(bt_summary.get("method", []), bt_summary.get("mae", [])))

    frontier_months = frontier["submission_month"].tolist() if not frontier.empty else []
    span = 0
    if len(frontier_months) > 1:
        span = pd.Period(frontier_months[-1], freq="M").ordinal - pd.Period(frontier_months[0], freq="M").ordinal
    short_targets = int(short_backtest["target_month"].nunique()) if not short_backtest.empty else 0
    forecast_rows = []
    for horizon in [12, 24]:
        target = str(pd.Period(frontier_months[-1], freq="M") + horizon) if frontier_months else ""
        forecast_rows.append({
            "horizon_months": horizon, "forecast_origin": frontier_months[-1] if frontier_months else "",
            "target_month": target, "n_observed_month_bins": len(frontier), "calendar_span_months": span,
            "short_horizon_1m_backtest_targets": short_targets, "annual_horizon_rolling_origin_count": 0,
            "point_forecast": np.nan, "prediction_interval_low": np.nan, "prediction_interval_high": np.nan,
            "status": "not_estimable_from_supplied_history",
            "reason": f"Only {len(frontier)} monthly submission cohorts span {span} months, shorter than the {horizon}-month horizon. No complete annual holdout origin exists. C3 has only 2024-2025 leaderboard-source years plus separate non-comparable historical-paper rows."
        })
    forecast = pd.DataFrame(forecast_rows)
    write_csv(forecast, RESULTS / "q4_forecast_feasibility.csv")

    selected_coverage = []
    for family in FAMILY_ORDER:
        sel = selected[selected["family"].eq(family)]
        f = family_table[family_table["family"].eq(family)]
        selected_coverage.append({
            "family": family, "version": PRIMARY_VERSIONS[family], "metric": PRIMARY_METRICS[family],
            "expected_tasks": EXPECTED_TASKS[family], "selected_score_rows": len(sel),
            "models_with_selected_scores": sel["model_name_raw"].nunique(),
            "complete_family_models": int(f["family_complete"].sum()),
        })
    write_csv(pd.DataFrame(selected_coverage), RESULTS / "q4_selected_metric_coverage.csv")

    if not dynamic_ready:
        bootstrap_outputs = [RESULTS / name for name in [
            "q4_selected_task_scores.csv", "q4_task_family_scores.csv", "q4_task_aggregate.csv",
            "q4_index_sensitivity.csv", "q4_coverage_by_dimension.csv", "q4_scale_tech_monthly.csv",
            "q4_scale_tech_coefficients.csv", "q4_type_group_crosswalk.csv", "q4_frontier_monthly.csv",
            "q4_forecast_backtest_1month.csv", "q4_forecast_backtest_1month_summary.csv",
            "q4_forecast_feasibility.csv", "q4_selected_metric_coverage.csv",
        ]]
        bootstrap_manifest = {
            "status": "bootstrap_base_outputs_for_dynamic_stage",
            "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
            "ai_disclosure": AI_DISCLOSURE,
            "script": rel(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "inputs_sha256": {rel(path): sha256(path) for path in core_input_paths},
            "outputs_sha256": {rel(path): sha256(path) for path in bootstrap_outputs},
            "next_steps": [
                "Run q4_registry_coverage.py after these base outputs exist.",
                "Run q4_dynamic_decomposition.py to create dynamic and C3/C4 year outputs.",
                "Run q4_modeling_analysis.py again to write the integrated report and full modeling manifest.",
            ],
        }
        bootstrap_path = RESULTS / "q4_modeling_bootstrap_manifest.json"
        bootstrap_path.write_text(json.dumps(bootstrap_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[q4] base outputs ready; dynamic inputs are not present yet; run registry coverage and dynamic decomposition, then rerun this script", flush=True)
        return

    ar1_row = pd.read_csv(dynamic_input_paths[0], encoding="utf-8-sig").iloc[0]
    ar1_backtest_row = pd.read_csv(dynamic_input_paths[1], encoding="utf-8-sig").iloc[0]
    contribution_table = pd.read_csv(dynamic_input_paths[2], encoding="utf-8-sig").set_index("component")
    support_row = pd.read_csv(dynamic_input_paths[3], encoding="utf-8-sig").iloc[0]
    decomposition_coefficients = pd.read_csv(dynamic_input_paths[4], encoding="utf-8-sig")
    decomposition_beta = decomposition_coefficients.loc[
        decomposition_coefficients["term"].eq("log10_params_B"), "estimate"
    ].iloc[0]
    c3_year_table = pd.read_csv(dynamic_input_paths[5], encoding="utf-8-sig")
    c4_year_table = pd.read_csv(dynamic_input_paths[6], encoding="utf-8-sig")
    c4_row_audit = pd.read_csv(dynamic_input_paths[7], encoding="utf-8-sig")
    c4_date_counts = c4_row_audit["date_status"].value_counts().to_dict()
    c3_2024 = c3_year_table.loc[c3_year_table["Source"].eq("Open LLM Leaderboard") & c3_year_table["Year"].eq(2024)].iloc[0]
    c3_2025 = c3_year_table.loc[c3_year_table["Source"].eq("Open LLM Leaderboard") & c3_year_table["Year"].eq(2025)].iloc[0]
    c4_2024 = c4_year_table.loc[c4_year_table["publication_year"].eq(2024)].iloc[0]
    c4_2025 = c4_year_table.loc[c4_year_table["publication_year"].eq(2025)].iloc[0]
    arc_core6_overlap = int((aggregate["ARC_complete"] & aggregate["main_cohort_complete_core6"]).sum())
    parameter_row = coefficients.loc[
        coefficients["scope"].eq("complete_core6_month_n_ge_20")
        & coefficients["term"].eq("log10_params_B")
    ].iloc[0]
    fallback_count = int(aggregate["c8_latest_status"].eq("latest_parseable_fallback").sum())
    common_support = scale_info["common_support"]

    lines = [
        "# Q4 冻结版分析报告", "",
        f"> AI 辅助说明：{AI_DISCLOSURE} 本报告供内部复核，正式提交前仍须由队伍核验。", "",
        f"> 合同执行版：`q4_model_contract_v0.6.md`。Q4 内部分析包已获用户授权冻结；其他 AI 使用范围须在竞赛论文提交前核实。", "",
        "## 执行口径", "",
        "本版口径已由用户确认并纳入冻结包；候选仍保持 version_unresolved，不写作严格版本匹配。",
        "主指数先在每个叶任务内，对该任务有有效所选分数的名称候选计算 0–100 百分位秩（并列取平均秩；不同任务分母随覆盖而变，见覆盖表），再在 BBH、GPQA、IFEval、MATH Hard v2.0、MMLU-Pro、MuSR 六个任务族内等权，最后族间等权。指数表达候选队列内的相对位置，不是绝对能力概率。IFEval 主指标为 inst_level_strict_acc；MATH Hard 主版本为 v2.0。原始准确率族均值作为尺度敏感性。", "",
        "## 覆盖与指数", "",
        f"- 精确名称且 C1 行唯一候选：{len(aggregate):,}；严格版本确认：0。",
        f"- 完整六族指数：{int(aggregate['core6_index_0_100'].notna().sum()):,} 个模型。",
        f"- 完整七族指数：{int(aggregate['full7_index_0_100'].notna().sum()):,} 个模型。",
        f"- ARC 完整族：{int(aggregate['ARC_complete'].sum()):,} 个模型；与完整六族样本交集 {arc_core6_overlap:,}，因此七族综合分没有可计算样本，ARC 单独报告覆盖和任务表现。",
        f"- 明确 C2 Open Weights=Yes：{int(aggregate['open_weights_status'].eq('Yes').sum()):,} 个候选；其中 {int((aggregate['main_cohort_complete_core6'] & aggregate['open_weights_status'].eq('Yes')).sum()):,} 个有完整六族指数；No/unknown 不补值。",
        "- 逐模型指数、叶任务覆盖和指标敏感性见 `../03_结果/results/q4_task_aggregate.csv`、`../03_结果/results/q4_task_family_scores.csv`、`../03_结果/results/q4_index_sensitivity.csv`。", "",
        "## 规模与时间分解", "",
        f"- 回归样本 {scale_info['n_models_in_supported_month_regression']:,}；只纳入月样本量不少于 20 的月份。响应为六族百分位指数 0–100，解释量为 log10(C1 参数量 B)、C1 类型组和提交月份固定项，标准误采用 HC3。",
        f"- log10(参数量/B) 系数为 {parameter_row['estimate']:.2f} 指数点，HC3 95% 区间 [{parameter_row['ci95_low']:.2f}, {parameter_row['ci95_high']:.2f}]，R²={parameter_row['r_squared']:.3f}；参照类型组为 `{scale_info['reference_type_group']}`、参照月份为 {scale_info['reference_month']}。该系数是控制已列变量后的样本关联。",
        "- 参数系数是横截面关联；月份项与剩余变化同时混合未观测技术、架构细节、样本选择和评测误差，不能解释为因果技术进步。月样本量不足 20 的月份仍列描述统计，不分解。",
        f"- 参数共同支持状态：{common_support['status']}；早/晚月份为 {common_support['early_month']} / {common_support['late_month']}；两期 log10(B) 共同区间 [{common_support['log10_params_low']:.3f}, {common_support['log10_params_high']:.3f}]，区间内样本数分别为 {common_support['n_early']} / {common_support['n_late']}。",
        f"- 主分析包含 {fallback_count} 个 C8 最新可解析回退模型；另有排除回退模型的指数敏感性结果。",
        "- 月度表中的 `unexplained_component` 是观测均值减去已列回归分量的账面恒等项。含月份固定效应的样本内 OLS 会使每个支持月份的平均残差为零，因此该项接近零是模型构造结果，不代表个体误差或遗漏因素为零。",
        "- C1 原始类型没有独立的 Chat 与 Instruct 子字段；按源标签保留 chat/post-training、pretrained、continuously pretrained、domain fine-tuned、merge 等组，不从模型名推断额外类型。进入尺度样本后不足 20 个模型的类型合并为 other_rare；逐项映射与交集见 `../03_结果/results/q4_type_group_crosswalk.csv`。",
        "- 时间轴用 C1 Submission Date，表示排行榜提交日期，不等于模型发布日期；月度分解见 `../03_结果/results/q4_scale_tech_monthly.csv`。", "",
        "## 动态模型与共同支持精确分解", "",
        f"- 月度 AR(1) 为 `{ar1_row['status']}`：{int(ar1_row['monthly_bins'])} 个连续月、{int(ar1_row['transitions'])} 次转移，a={float(ar1_row['intercept_a']):.3f}，95% HC3 区间 [{float(ar1_row['intercept_ci95_low']):.3f}, {float(ar1_row['intercept_ci95_high']):.3f}]；rho={float(ar1_row['rho']):.3f}，区间 [{float(ar1_row['rho_ci95_low']):.3f}, {float(ar1_row['rho_ci95_high']):.3f}]。区间不校正序列相关，只作描述。",
        f"- 扩展窗口只产生 {int(ar1_backtest_row['ar1_origins'])} 个满足训练门槛的一步起点；AR(1) MAE={float(ar1_backtest_row['ar1_mae']):.3f}，持平基线 MAE={float(ar1_backtest_row['persistence_mae']):.3f}。该小样本结果不支持年度外推；12/24 月预测仍无数值点或区间。",
        f"- 精确分解在 {int(support_row['fit_rows_in_common_support'])} 条 P05–P95 共同支持记录上重新拟合，参数量系数 β={float(decomposition_beta):.2f}；这不同于上一节 {scale_info['n_models_in_supported_month_regression']:,} 条支持月份样本的 β={float(parameter_row['estimate']):.2f}，后者不用于分解贡献项。端点 P05–P95 区间交集为 log10(B)=[{support_row['log10_params_B_support_low_p05_p95']:.4f}, {support_row['log10_params_B_support_high_p05_p95']:.4f}]，端点纳入 {int(support_row['early_rows_in_common_support'])}/{int(support_row['late_rows_in_common_support'])} 条记录；设计矩阵秩 {int(support_row['design_rank'])}/{int(support_row['design_columns'])}。",
        f"- 规模项 {contribution_table.loc['scale_composition','estimate_index_points']:.3f} 指数点（95% HC3 [{contribution_table.loc['scale_composition','ci95_low']:.3f}, {contribution_table.loc['scale_composition','ci95_high']:.3f}]）；非规模合计 {contribution_table.loc['non_scale_total','estimate_index_points']:.3f}（[{contribution_table.loc['non_scale_total','ci95_low']:.3f}, {contribution_table.loc['non_scale_total','ci95_high']:.3f}]）；观测均值变化 {contribution_table.loc['observed_mean_change','estimate_index_points']:.3f}（[{contribution_table.loc['observed_mean_change','ci95_low']:.3f}, {contribution_table.loc['observed_mean_change','ci95_high']:.3f}]）。因为观测变化和非规模合计区间含 0，贡献占比留空。分项和闭合误差详见 `../03_结果/results/q4_scale_tech_contribution.csv` 与 `q4_dynamic_decomposition_report.md`；闭合误差只是因月份固定效应而预期接近 0 的算术检查。", "",
        "## C3/C4 年度背景", "",
        f"- C3 Open LLM Leaderboard 的 2024 年去冲突记录数为 {int(c3_2024['Average_numeric_records']):,}，Average 中位数/P90 为 {c3_2024['Average_median']:.2f}/{c3_2024['Average_p90']:.2f}；2025 年为部分年度覆盖，去冲突记录数 {int(c3_2025['Average_numeric_records']):,}，不并入完整年度变化率。历史论文/报告数据按来源独立列示。",
        f"- C4 截止 2025-03-14：{int(c4_date_counts.get('at_or_before_cutoff', 0)):,} 条记录在截止内，{int(c4_date_counts.get('after_cutoff', 0)):,} 条晚于截止，{int(c4_date_counts.get('date_unknown', 0)):,} 条日期未知。2024 年记录数 {int(c4_2024['record_count']):,}，参数量/算力/数据量标量有效数分别 {int(c4_2024['parameter_count_scalar_n'])}/{int(c4_2024['training_compute_scalar_n'])}/{int(c4_2024['dataset_size_scalar_n'])}；2025 截止日前部分年度记录 {int(c4_2025['record_count'])} 条。C4 保持年度背景，不作 C1 个体匹配或算力增长率。", "",
        "## Loss–Benchmark 预拟合门控", "",
        f"- 状态为 `{gate_row['prefit_gate_status']}`；C5/C6 高可比记录按模型名、来源和数值去重后为 {int(gate_row['unique_high_content_pairs'])} 条，模型名 {int(gate_row['unique_model_names'])} 个，但已核验版本数为 {int(gate_row['verified_model_version_count'])}。Loss 数值范围 [{gate_row['high_loss_min']:.4f}, {gate_row['high_loss_max']:.4f}]。",
        f"- Q2 终点候选中 {int(gate_row['target_candidate_rows_inside_high_loss_range'])}/{int(gate_row['target_candidate_rows'])} 个数值落入该范围；它们不是批准的未来目标网格。Q2 B1 Loss 单位、评测协议及 tokenizer/预处理仍有 UNKNOWN，历史 revision/评测运行标识也缺失，因此没有桥接拟合。复核回执见 `../05_审计交付/q4_m1_review_loss_gate.md`。", "",
        "## 12/24 个月预测可行性", "",
        f"- 开放权重前沿为每月新提交、C2 明确标为 Yes 的模型中 C1 Average 的 P90；共 {len(frontier)} 个有数据月，跨 {span} 个月。",
        f"- 1 个月滚动诊断有 {short_targets} 个目标月；last-value MAE={float(bt_mae.get('last_value', np.nan)):.2f}，线性趋势 MAE={float(bt_mae.get('linear_time_trend', np.nan)):.2f}；样本很少，见 `../03_结果/results/q4_forecast_backtest_1month_summary.csv`，不能验证年度外推。",
        "- 对 12/24 个月没有完整滚动回测起点，故不输出数值预测或预测区间；状态及目标月份见 `../03_结果/results/q4_forecast_feasibility.csv`。",
        "- C3 的 OLLB 记录只到 2024–2025；26 条历史论文/报告记录单独隔离，未用于补足年度预测窗口。", "",
        "## 可复核限制", "",
        "- 名称精确且 C1 行唯一仍不证明 C1/C8 是同一模型版本。",
        "- MATH v1/v2/v3 分开计算；版本敏感性队列不能直接比较。",
        "- 开放权重仅将 C2 明确 Yes 列为已知开放。",
        "- 本版为描述性队列分析，不构成因果推断；用户已确认并冻结本文所列口径。是否有其他 AI 工具参与尚未核实，应在竞赛论文提交前按实际情况补充披露。",
        "- 2026 年华为杯 AI 规定要求数据分析结果注明工具名称、版本/型号、开发机构和版本发布日期，并要求 AI 辅助代码在程序前说明；本次确认 Codex 模型 ID 为 `gpt-6-luna`，官方发布日期为 2026-09-22。", "",
        "图表契约与 QA 记录见 q4_figures_contract.md。", ""
    ]
    report_path = Q4 / "01_方案说明" / "q4_modeling_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    output_names = [
        "q4_selected_task_scores.csv", "q4_task_family_scores.csv", "q4_task_aggregate.csv",
        "q4_index_sensitivity.csv", "q4_coverage_by_dimension.csv", "q4_scale_tech_monthly.csv",
        "q4_scale_tech_coefficients.csv", "q4_type_group_crosswalk.csv", "q4_frontier_monthly.csv", "q4_forecast_backtest_1month.csv",
        "q4_forecast_backtest_1month_summary.csv", "q4_forecast_feasibility.csv", "q4_selected_metric_coverage.csv",
    ]
    output_paths = [RESULTS / n for n in output_names] + [report_path]
    manifest = {
        "status": "analysis_complete; package_freeze_recorded; additional_ai_scope_pending_before_competition_submission",
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ai_disclosure": AI_DISCLOSURE, "script": rel(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs_sha256": {rel(p): sha256(p) for p in input_paths},
        "outputs_sha256": {rel(p): sha256(p) for p in output_paths},
        "cohort": {
            "exact_unique_name_candidates": len(candidate), "version_confirmed": 0,
            "status": "version_unresolved",
            "complete_core6": int(aggregate["core6_index_0_100"].notna().sum()),
            "complete_full7": int(aggregate["full7_index_0_100"].notna().sum()),
        },
        "contract": {
            "main_families": CORE_FAMILIES, "main_family_weighting": "equal",
            "within_family_weighting": "equal leaf tasks after within-task candidate-cohort percentile-rank normalization; incomplete family is missing",
            "task_normalization": "higher-is-better average percentile rank among exact-name unique-C1-row candidates with that leaf score",
            "MATH_Hard_version": "2.0", "IFEval_metric": "inst_level_strict_acc",
            "scale": "C1 #Params (B), log10 transformed for regression",
            "time": "C1 Submission Date, not model release date",
            "open_weights": "C2 candidate metadata; only explicit Yes is known Yes",
        },
        "scale_model": scale_info,
        "forecast": {
            "known_open_month_bins": len(frontier), "calendar_span_months": span,
            "12m_rolling_origin_count": 0, "24m_rolling_origin_count": 0,
            "numeric_forecasts_issued": False,
        },
        "limitations": [
            "Exact name and C1 row uniqueness do not confirm C8/C1 model-version identity.",
            "The decomposition is observational and not causal.",
            "The annual forecast horizon exceeds the supplied comparable time support.",
            "Historical paper/report C3 records were not pooled with leaderboard-source records."
        ],
    }
    manifest_path = RESULTS / "q4_modeling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bootstrap_path = RESULTS / "q4_modeling_bootstrap_manifest.json"
    if bootstrap_path.exists():
        bootstrap_path.unlink()
    print(f"candidates={len(candidate)} core6={manifest['cohort']['complete_core6']} full7={manifest['cohort']['complete_full7']}")
    print(f"open_yes={int(aggregate['open_weights_status'].eq('Yes').sum())} months={len(frontier)} span={span} short_backtest_targets={short_targets}")
    print(f"scale_n={scale_info['n_models_in_scale_model']} beta_log10_params={scale_info['beta_log10_params_B']:.6f} common_support={scale_info['common_support']['status']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
