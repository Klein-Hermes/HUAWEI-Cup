"""Run Q4 descriptive AR(1), supported scale/type/time decomposition, C3/C4 summaries.

AI disclosure: OpenAI Codex desktop 26.917.71314 (build 10954; prod), model ID
gpt-6-luna (GPT-6 Luna), developer OpenAI, official release date 2026-09-22.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from q4_modeling_analysis import map_type_groups

Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
RESULTS = Q4 / "03_结果" / "results"
SMOKE = Q4 / "03_结果" / "smoke"
RAW_C = ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"
AI_DISCLOSURE = (
    "OpenAI Codex desktop 26.917.71314 (build 10954; prod); model ID "
    "gpt-6-luna (GPT-6 Luna); developer OpenAI; official release date 2026-09-22."
)
EARLY_MONTH = "2024-11"
LATE_MONTH = "2025-03"
C4_CUTOFF = pd.Timestamp("2025-03-14")
C3_METRICS = ["Params_B", "Average", "IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO"]
C4_AUDIT_FIELDS = [
    "Model", "Publication date", "Training compute (FLOP)", "Training compute notes",
    "Training dataset size (total)", "Dataset size notes", "Parameters", "Parameters notes",
    "Confidence", "Numerical format", "Training compute estimation method",
    "Open model weights?", "Reference", "Link",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")
    return path


def hc3_fit(x: np.ndarray, y: np.ndarray, labels: list[str]) -> dict[str, Any]:
    n, p = x.shape
    rank = int(np.linalg.matrix_rank(x))
    if rank != p:
        raise ValueError(f"Design matrix is rank deficient: rank={rank}, columns={p}")
    if n <= p:
        raise ValueError(f"No residual degrees of freedom: n={n}, parameters={p}")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    adjusted = residual / np.maximum(1.0 - leverage, 1e-12)
    meat = x.T @ (x * (adjusted ** 2)[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "beta": beta, "covariance": covariance, "se": se, "residual": residual,
        "fitted": x @ beta, "rank": rank, "n": n, "p": p,
        "residual_df": n - p, "labels": labels,
    }


def ar_design(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.column_stack([np.ones(len(values) - 1), values[:-1]]), values[1:]


def run_ar1(frontier: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = frontier.copy()
    d["submission_month"] = d["submission_month"].astype(str)
    d["F_t"] = pd.to_numeric(d["monthly_entry_p90_frontier_C1_average_0_100"], errors="coerce")
    d = d.dropna(subset=["F_t"]).sort_values("submission_month").reset_index(drop=True)
    periods = pd.PeriodIndex(d["submission_month"], freq="M")
    complete = len(periods) > 0 and np.array_equal(
        periods.asi8, np.arange(periods.asi8[0], periods.asi8[0] + len(periods))
    )
    summary = {
        "status": "not_estimable", "monthly_bins": int(len(d)),
        "consecutive_month_bins": bool(complete), "transitions": max(int(len(d) - 1), 0),
        "residual_df": "", "intercept_a": "", "intercept_ci95_low": "",
        "intercept_ci95_high": "", "rho": "", "rho_ci95_low": "", "rho_ci95_high": "",
        "r_squared": "", "durbin_watson": "", "residual_lag1_correlation": "",
        "interval_method": "HC3 covariance; normal 1.96 multiplier; heteroskedasticity robust only",
        "reason": "Requires at least 8 consecutive monthly bins, full rank, and at least 5 residual degrees of freedom.",
    }
    residual_rows: list[dict[str, Any]] = []
    backtest_rows: list[dict[str, Any]] = []
    if len(d) >= 8 and complete:
        vals = d["F_t"].to_numpy(float)
        x, y = ar_design(vals)
        fit = hc3_fit(x, y, ["intercept_a", "rho"]) if np.linalg.matrix_rank(x) == x.shape[1] else None
        if fit is None:
            summary["reason"] = "AR(1) design matrix is rank deficient; status remains not_estimable."
        elif fit["residual_df"] < 5:
            summary["reason"] = "Residual degrees of freedom below the contract minimum of 5."
        else:
            beta, se = fit["beta"], fit["se"]
            resid = fit["residual"]
            tss = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - float((resid ** 2).sum()) / tss if tss > 0 else np.nan
            dw = float(np.diff(resid).dot(np.diff(resid)) / resid.dot(resid)) if resid.dot(resid) > 0 else np.nan
            lag_corr = float(np.corrcoef(resid[1:], resid[:-1])[0, 1]) if len(resid) > 2 and np.std(resid[1:]) and np.std(resid[:-1]) else np.nan
            summary.update({
                "status": "estimated_descriptive", "residual_df": int(fit["residual_df"]),
                "intercept_a": float(beta[0]), "intercept_ci95_low": float(beta[0] - 1.96 * se[0]),
                "intercept_ci95_high": float(beta[0] + 1.96 * se[0]), "rho": float(beta[1]),
                "rho_ci95_low": float(beta[1] - 1.96 * se[1]), "rho_ci95_high": float(beta[1] + 1.96 * se[1]),
                "r_squared": r2, "durbin_watson": dw, "residual_lag1_correlation": lag_corr,
                "reason": "Descriptive AR(1) only; HC3 intervals do not correct serial correlation or establish annual forecast skill.",
            })
            for idx, (observed, predicted, residual) in enumerate(zip(y, fit["fitted"], resid), start=1):
                residual_rows.append({
                    "target_month": d.loc[idx, "submission_month"],
                    "previous_month": d.loc[idx - 1, "submission_month"],
                    "observed_F_t": float(observed), "fitted_F_t": float(predicted), "residual": float(residual),
                })
            # Require each expanding-window training fold to meet the same 8-bin AR eligibility rule.
            for target_idx in range(8, len(d)):
                train_values = vals[:target_idx]
                train_x, train_y = ar_design(train_values)
                if np.linalg.matrix_rank(train_x) != train_x.shape[1]:
                    backtest_rows.append({
                        "target_month": d.loc[target_idx, "submission_month"],
                        "training_month_bins": int(target_idx), "observed_F_t": float(vals[target_idx]),
                        "ar1_prediction": np.nan, "ar1_absolute_error": np.nan,
                        "persistence_prediction": float(vals[target_idx - 1]),
                        "persistence_absolute_error": abs(float(vals[target_idx]) - float(vals[target_idx - 1])),
                        "fit_status": "not_estimable_rank_deficient",
                    })
                    continue
                train_fit = hc3_fit(train_x, train_y, ["intercept_a", "rho"])
                predicted = float(train_fit["beta"][0] + train_fit["beta"][1] * vals[target_idx - 1])
                observed = float(vals[target_idx])
                persistence = float(vals[target_idx - 1])
                backtest_rows.append({
                    "target_month": d.loc[target_idx, "submission_month"],
                    "training_month_bins": int(target_idx), "observed_F_t": observed,
                    "ar1_prediction": predicted, "ar1_absolute_error": abs(observed - predicted),
                    "persistence_prediction": persistence,
                    "persistence_absolute_error": abs(observed - persistence),
                    "fit_status": "estimated",
                })
    residuals = pd.DataFrame(residual_rows, columns=["target_month", "previous_month", "observed_F_t", "fitted_F_t", "residual"])
    backtest = pd.DataFrame(backtest_rows, columns=["target_month", "training_month_bins", "observed_F_t", "ar1_prediction", "ar1_absolute_error", "persistence_prediction", "persistence_absolute_error", "fit_status"])
    valid_backtest = backtest[backtest["fit_status"].eq("estimated")] if not backtest.empty else backtest
    excluded_folds = int(len(backtest) - len(valid_backtest)) if not backtest.empty else 0
    if valid_backtest.empty:
        bt_summary = pd.DataFrame([{
            "ar1_origins": 0, "ar1_mae": np.nan, "persistence_origins": 0,
            "persistence_mae": np.nan, "excluded_rank_deficient_folds": excluded_folds,
            "status": "insufficient_eligible_expanding_origins",
        }])
    else:
        bt_summary = pd.DataFrame([{
            "ar1_origins": int(len(valid_backtest)), "ar1_mae": float(valid_backtest["ar1_absolute_error"].mean()),
            "persistence_origins": int(len(valid_backtest)),
            "persistence_mae": float(valid_backtest["persistence_absolute_error"].mean()),
            "excluded_rank_deficient_folds": excluded_folds,
            "status": "descriptive_small_sample_one_step_check",
        }])
    return pd.DataFrame([summary]), residuals, backtest, bt_summary


def build_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, str]]:
    categories = sorted(frame["type_group"].astype(str).unique())
    months = sorted(frame["submission_month"].astype(str).unique())
    if len(categories) < 2 or len(months) < 2:
        raise ValueError("Decomposition requires at least two type groups and two endpoint months.")
    reference_type, reference_month = categories[0], months[0]
    columns = [np.ones(len(frame)), frame["log10_params_B"].to_numpy(float)]
    labels = ["intercept", "log10_params_B"]
    for category in categories[1:]:
        columns.append(frame["type_group"].astype(str).eq(category).to_numpy(float))
        labels.append(f"type[{category}]")
    for month in months[1:]:
        columns.append(frame["submission_month"].astype(str).eq(month).to_numpy(float))
        labels.append(f"month[{month}]")
    return np.column_stack(columns), labels, {"reference_type": reference_type, "reference_month": reference_month}


def contrast_ci(vector: np.ndarray, fit: dict[str, Any]) -> tuple[float, float, float, float]:
    estimate = float(vector @ fit["beta"])
    variance = float(vector @ fit["covariance"] @ vector)
    se = math.sqrt(max(variance, 0.0))
    return estimate, se, estimate - 1.96 * se, estimate + 1.96 * se


def run_decomposition(aggregate: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = aggregate[
        aggregate["core6_index_0_100"].notna()
        & pd.to_numeric(aggregate["parameter_count_B"], errors="coerce").gt(0)
        & aggregate["submission_date"].notna()
    ].copy()
    d["core6_index_0_100"] = pd.to_numeric(d["core6_index_0_100"], errors="coerce")
    d["parameter_count_B"] = pd.to_numeric(d["parameter_count_B"], errors="coerce")
    d["log10_params_B"] = np.log10(d["parameter_count_B"])
    d["submission_month"] = pd.to_datetime(d["submission_date"], errors="coerce").dt.to_period("M").astype(str)
    d["type_group"] = map_type_groups(d["C1_Type_raw"])
    counts = d["submission_month"].value_counts()
    supported_months = sorted(counts[counts.ge(20)].index.tolist())
    if not supported_months or supported_months[0] != EARLY_MONTH or supported_months[-1] != LATE_MONTH:
        raise ValueError(f"Contract endpoint months unavailable: supported={supported_months}")
    endpoint_rows = d[d["submission_month"].isin([EARLY_MONTH, LATE_MONTH])]
    early_raw = endpoint_rows[endpoint_rows["submission_month"].eq(EARLY_MONTH)]
    late_raw = endpoint_rows[endpoint_rows["submission_month"].eq(LATE_MONTH)]
    low = max(float(early_raw["log10_params_B"].quantile(.05, interpolation="linear")), float(late_raw["log10_params_B"].quantile(.05, interpolation="linear")))
    high = min(float(early_raw["log10_params_B"].quantile(.95, interpolation="linear")), float(late_raw["log10_params_B"].quantile(.95, interpolation="linear")))
    if not low < high:
        raise ValueError("Early/late endpoint parameter distributions have no p05/p95 overlap.")
    supported = d[d["submission_month"].isin(supported_months)].copy()
    fit_rows = supported[supported["log10_params_B"].between(low, high, inclusive="both")].copy()
    if smoke:
        fit_rows = fit_rows[fit_rows["submission_month"].isin([EARLY_MONTH, LATE_MONTH])].copy()
    fit_rows = fit_rows.sort_values(["submission_month", "model_name_raw"]).reset_index(drop=True)
    if len(fit_rows) < 30:
        raise ValueError(f"Common-support model sample too small: n={len(fit_rows)}")
    x, labels, references = build_design(fit_rows)
    y = fit_rows["core6_index_0_100"].to_numpy(float)
    fit = hc3_fit(x, y, labels)
    coef_rows = []
    for i, label in enumerate(labels):
        coef_rows.append({
            "term": label, "estimate": float(fit["beta"][i]), "hc3_se": float(fit["se"][i]),
            "ci95_low": float(fit["beta"][i] - 1.96 * fit["se"][i]),
            "ci95_high": float(fit["beta"][i] + 1.96 * fit["se"][i]),
            "n_models": int(fit["n"]), "residual_df": int(fit["residual_df"]),
            "reference_type": references["reference_type"], "reference_month": references["reference_month"],
        })

    early = fit_rows[fit_rows["submission_month"].eq(EARLY_MONTH)].copy()
    late = fit_rows[fit_rows["submission_month"].eq(LATE_MONTH)].copy()
    if early.empty or late.empty:
        raise ValueError("Common-support endpoint sample is empty for one endpoint.")
    def mean_design(group: pd.DataFrame) -> np.ndarray:
        vector = np.zeros(len(labels), dtype=float)
        vector[labels.index("intercept")] = 1.0
        vector[labels.index("log10_params_B")] = float(group["log10_params_B"].mean())
        shares = group["type_group"].value_counts(normalize=True).to_dict()
        for i, label in enumerate(labels):
            if label.startswith("type["):
                vector[i] = float(shares.get(label[5:-1], 0.0))
            elif label.startswith("month["):
                vector[i] = float(group["submission_month"].eq(label[6:-1]).mean())
        return vector

    delta = mean_design(late) - mean_design(early)
    scale_vector = np.zeros(len(labels)); scale_vector[labels.index("log10_params_B")] = delta[labels.index("log10_params_B")]
    type_vector = np.zeros(len(labels))
    for i, label in enumerate(labels):
        if label.startswith("type["):
            type_vector[i] = delta[i]
    month_vector = np.zeros(len(labels))
    for i, label in enumerate(labels):
        if label.startswith("month["):
            month_vector[i] = delta[i]
    nonscale_vector = type_vector + month_vector
    pred_vector = scale_vector + nonscale_vector
    observed_change = float(late["core6_index_0_100"].mean() - early["core6_index_0_100"].mean())

    components = [
        ("scale_composition", scale_vector), ("type_composition", type_vector),
        ("month_fixed_effect", month_vector), ("non_scale_total", nonscale_vector),
        ("predicted_mean_change", pred_vector),
    ]
    rows: list[dict[str, Any]] = []
    component_stats: dict[str, tuple[float, float, float, float]] = {}
    for name, vector in components:
        estimate, se, lo, hi = contrast_ci(vector, fit)
        component_stats[name] = (estimate, se, lo, hi)
        rows.append({
            "component": name, "estimate_index_points": estimate, "hc3_se": se,
            "ci95_low": lo, "ci95_high": hi, "share_of_observed_change_pct": np.nan,
            "interval_basis": "same HC3 covariance matrix; fixed early/late empirical covariate distributions",
        })
    # HC3 interval for the observed difference uses a two-group means model on the same restricted endpoints.
    endpoint = pd.concat([early, late], ignore_index=True)
    observed_x = np.column_stack([
        np.ones(len(endpoint)), endpoint["submission_month"].eq(LATE_MONTH).to_numpy(float)
    ])
    observed_fit = hc3_fit(observed_x, endpoint["core6_index_0_100"].to_numpy(float), ["intercept", "late_minus_early"])
    obs_vec = np.array([0.0, 1.0])
    obs_est, obs_se, obs_lo, obs_hi = contrast_ci(obs_vec, observed_fit)
    residual = observed_change - component_stats["predicted_mean_change"][0]
    rows.extend([
        {
            "component": "observed_mean_change", "estimate_index_points": observed_change,
            "hc3_se": obs_se, "ci95_low": obs_lo, "ci95_high": obs_hi,
            "share_of_observed_change_pct": np.nan,
            "interval_basis": "HC3 two-group mean contrast on common-support endpoint records",
        },
        {
            "component": "arithmetic_closure_residual", "estimate_index_points": residual,
            "hc3_se": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
            "share_of_observed_change_pct": np.nan,
            "interval_basis": "point arithmetic check only; endpoint month fixed effects make the fitted-sample group mean residual zero by OLS normal equations",
        },
    ])
    scale_ci = component_stats["scale_composition"][2:]
    non_ci = component_stats["non_scale_total"][2:]
    shares_allowed = obs_lo > 0 or obs_hi < 0
    shares_allowed = shares_allowed and (scale_ci[0] > 0 or scale_ci[1] < 0) and (non_ci[0] > 0 or non_ci[1] < 0)
    if shares_allowed and abs(observed_change) > 1e-12:
        for row in rows:
            if row["component"] in {"scale_composition", "non_scale_total", "type_composition", "month_fixed_effect"}:
                row["share_of_observed_change_pct"] = 100.0 * row["estimate_index_points"] / observed_change
    component_sum = sum(component_stats[name][0] for name in ["scale_composition", "type_composition", "month_fixed_effect"])
    component_sum_abs_error = abs(component_sum - component_stats["predicted_mean_change"][0])
    closure_abs_error = abs(component_sum + residual - observed_change)
    if component_sum_abs_error > 1e-10 or closure_abs_error > 1e-8:
        raise ValueError(f"Decomposition closure failed: components={component_sum_abs_error}, observed={closure_abs_error}")
    support = pd.DataFrame([{
        "status": "overlap", "early_month": EARLY_MONTH, "late_month": LATE_MONTH,
        "early_rows_before_support": int(len(early_raw)), "late_rows_before_support": int(len(late_raw)),
        "log10_params_B_support_low_p05_p95": low, "log10_params_B_support_high_p05_p95": high,
        "early_rows_in_common_support": int(len(early)), "late_rows_in_common_support": int(len(late)),
        "supported_months_in_fit": ";".join(sorted(fit_rows["submission_month"].unique())),
        "fit_rows_in_common_support": int(len(fit_rows)), "design_rank": fit["rank"],
        "design_columns": fit["p"], "residual_df": fit["residual_df"],
        "type_reference": references["reference_type"], "month_reference": references["reference_month"],
        "contribution_shares_reported": bool(shares_allowed),
        "share_gate_reason": "Observed-change, scale, and non-scale 95% intervals exclude zero." if shares_allowed else "At least one required 95% interval includes zero; shares suppressed.",
        "component_sum_abs_error": component_sum_abs_error,
        "observed_reconstruction_abs_error": closure_abs_error,
        "closure_residual_interpretation": "arithmetic closure check; expected approximately zero because month fixed effects are fit on the endpoint records",
        "smoke_run": bool(smoke),
    }])
    return pd.DataFrame(rows), pd.DataFrame(coef_rows), support


def run_c3(path: Path, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"Model", "Year", "Average", "Source"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"C3 missing required columns: {missing}")
    raw.insert(0, "source_row_1based", np.arange(2, len(raw) + 2))
    raw["Year_numeric"] = pd.to_numeric(raw["Year"], errors="coerce")
    for column in C3_METRICS:
        if column in raw.columns:
            raw[column + "_numeric"] = pd.to_numeric(raw[column], errors="coerce")
    if smoke:
        raw = raw[raw["Source"].eq("Open LLM Leaderboard") & raw["Year_numeric"].eq(2024)].copy()
    grouping = ["Source", "Year", "Model"]
    raw["duplicate_group_rows"] = raw.groupby(grouping, dropna=False)["source_row_1based"].transform("size")
    conflict = pd.Series(False, index=raw.index)
    for column in C3_METRICS:
        numeric = column + "_numeric"
        if numeric in raw.columns:
            distinct = raw.groupby(grouping, dropna=False)[numeric].transform("nunique")
            conflict |= distinct.gt(1)
    raw["conflicting_duplicate_metrics"] = conflict
    raw["kept_for_summary"] = ~conflict & ~raw.duplicated(grouping, keep="first")
    audit_fields = ["source_row_1based", "Source", "Year", "Model", "Average", "Average_numeric", "duplicate_group_rows", "conflicting_duplicate_metrics", "kept_for_summary"]
    audit = raw[audit_fields].copy()
    good = raw[raw["kept_for_summary"]].copy()
    rows = []
    for (source, year), group in good.groupby(["Source", "Year"], dropna=False, sort=True):
        avg = group["Average_numeric"].dropna()
        original = raw[(raw["Source"].eq(source)) & (raw["Year"].eq(year))]
        year_int = int(float(year)) if str(year).strip() and pd.notna(pd.to_numeric(year, errors="coerce")) else None
        if source == "Open LLM Leaderboard" and year_int == 2025:
            coverage_status = "partial_year_coverage_only"
        elif source == "Open LLM Leaderboard" and year_int is not None and year_int <= 2024:
            coverage_status = "leaderboard_source_full_year_candidate"
        elif source == "Historical (papers/reports)":
            coverage_status = "historical_non_comparable_selective_records"
        else:
            coverage_status = "source_year_requires_review"
        rows.append({
            "Source": source, "Year": year, "source_rows": int(len(original)),
            "unique_model_records_after_dedup": int(group["Model"].nunique()),
            "conflicting_model_year_groups_excluded": int(original.loc[original["conflicting_duplicate_metrics"], "Model"].nunique()),
            "Average_numeric_records": int(len(avg)),
            "Average_median": float(avg.median()) if len(avg) else np.nan,
            "Average_p90": float(avg.quantile(.9)) if len(avg) else np.nan,
            "coverage_status": coverage_status,
            "included_in_complete_year_trend": bool(source == "Open LLM Leaderboard" and year_int is not None and year_int <= 2024),
        })
    return pd.DataFrame(rows), audit


def scalar_numeric(value: Any) -> tuple[float, str]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return np.nan, "missing"
    text = str(value).strip().replace(",", "")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return np.nan, "non_scalar_or_range_text"
    if not math.isfinite(number):
        return np.nan, "non_finite"
    return number, "finite_scalar"


def run_c4(path: Path, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", low_memory=False)
    missing = sorted(set(C4_AUDIT_FIELDS) - set(raw.columns))
    if missing:
        raise ValueError(f"C4 missing required provenance columns: {missing}")
    raw.insert(0, "source_row_1based", np.arange(2, len(raw) + 2))
    raw["publication_date_parsed"] = pd.to_datetime(raw["Publication date"], errors="coerce", utc=True).dt.tz_localize(None)
    raw["publication_year"] = raw["publication_date_parsed"].dt.year
    raw["date_status"] = np.select(
        [raw["publication_date_parsed"].isna(), raw["publication_date_parsed"].le(C4_CUTOFF)],
        ["date_unknown", "at_or_before_cutoff"], default="after_cutoff",
    )
    raw["included_as_of_cutoff"] = raw["date_status"].eq("at_or_before_cutoff")
    for source_col, output_col in [
        ("Training compute (FLOP)", "training_compute"),
        ("Training dataset size (total)", "dataset_size"),
        ("Parameters", "parameter_count"),
    ]:
        parsed = [scalar_numeric(v) for v in raw[source_col]]
        raw[output_col + "_numeric"] = [v[0] for v in parsed]
        raw[output_col + "_parse_status"] = [v[1] for v in parsed]
    scoped = raw[raw["included_as_of_cutoff"]].copy()
    if smoke:
        scoped = scoped[scoped["publication_year"].eq(2024)].copy()
    summaries = []
    for year, group in scoped.groupby("publication_year", dropna=False, sort=True):
        compute = group["training_compute_numeric"].dropna()
        data_size = group["dataset_size_numeric"].dropna()
        parameters = group["parameter_count_numeric"].dropna()
        confidence = group["Confidence"].str.strip().str.lower()
        open_weights = group["Open model weights?"].str.strip().str.lower()
        row: dict[str, Any] = {
            "publication_year": int(year), "record_count": int(len(group)),
            "calendar_coverage_status": "partial_calendar_year_at_cutoff" if int(year) == C4_CUTOFF.year else "calendar_year_ended_before_cutoff",
            "training_compute_scalar_n": int(len(compute)),
            "training_compute_median_FLOP": float(compute.median()) if len(compute) else np.nan,
            "training_compute_p90_FLOP": float(compute.quantile(.9)) if len(compute) else np.nan,
            "training_compute_confident_scalar_n": int(group["training_compute_numeric"].notna().mul(confidence.eq("confident")).sum()),
            "dataset_size_scalar_n": int(len(data_size)),
            "dataset_size_median_tokens": float(data_size.median()) if len(data_size) else np.nan,
            "dataset_size_p90_tokens": float(data_size.quantile(.9)) if len(data_size) else np.nan,
            "dataset_size_confident_scalar_n": int(group["dataset_size_numeric"].notna().mul(confidence.eq("confident")).sum()),
            "parameter_count_scalar_n": int(len(parameters)),
            "parameter_count_median": float(parameters.median()) if len(parameters) else np.nan,
            "parameter_count_p90": float(parameters.quantile(.9)) if len(parameters) else np.nan,
            "parameter_count_confident_scalar_n": int(group["parameter_count_numeric"].notna().mul(confidence.eq("confident")).sum()),
            "confidence_confident_n": int(confidence.eq("confident").sum()),
            "confidence_likely_n": int(confidence.eq("likely").sum()),
            "confidence_unknown_n": int(confidence.eq("unknown").sum()),
            "confidence_speculative_n": int(confidence.eq("speculative").sum()),
            "confidence_blank_n": int(confidence.eq("").sum()),
            "open_weights_yes_n": int(open_weights.isin(["yes", "true"]).sum()),
            "open_weights_no_n": int(open_weights.isin(["no", "false"]).sum()),
            "open_weights_unknown_n": int((~open_weights.isin(["yes", "true", "no", "false"])).sum()),
            "numerical_format_counts": json.dumps(group["Numerical format"].replace("", "<blank>").value_counts().sort_index().to_dict(), ensure_ascii=False),
            "estimation_method_counts": json.dumps(group["Training compute estimation method"].replace("", "<blank>").value_counts().sort_index().to_dict(), ensure_ascii=False),
            "cutoff": C4_CUTOFF.date().isoformat(),
            "summary_scope": "scalar numeric values only; Confidence and estimation-method strata retained; C4 is annual background and not joined to C1 models",
            "smoke_run": bool(smoke),
        }
        summaries.append(row)
    conf_rows = []
    for (year, conf), group in scoped.assign(_confidence=scoped["Confidence"].replace("", "<blank>")).groupby(["publication_year", "_confidence"], sort=True):
        conf_rows.append({
            "publication_year": int(year), "Confidence": conf, "record_count": int(len(group)),
            "training_compute_scalar_n": int(group["training_compute_numeric"].notna().sum()),
            "training_compute_median_FLOP": float(group["training_compute_numeric"].median()) if group["training_compute_numeric"].notna().any() else np.nan,
            "dataset_size_scalar_n": int(group["dataset_size_numeric"].notna().sum()),
            "dataset_size_median_tokens": float(group["dataset_size_numeric"].median()) if group["dataset_size_numeric"].notna().any() else np.nan,
            "parameter_count_scalar_n": int(group["parameter_count_numeric"].notna().sum()),
            "parameter_count_median": float(group["parameter_count_numeric"].median()) if group["parameter_count_numeric"].notna().any() else np.nan,
        })
    audit_cols = ["source_row_1based", *C4_AUDIT_FIELDS, "publication_date_parsed", "publication_year", "date_status", "included_as_of_cutoff", "training_compute_numeric", "training_compute_parse_status", "dataset_size_numeric", "dataset_size_parse_status", "parameter_count_numeric", "parameter_count_parse_status"]
    return pd.DataFrame(summaries), pd.DataFrame(conf_rows), raw[audit_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a reduced-support review package under Q4/03_结果/smoke.")
    args = parser.parse_args()
    output_dir = SMOKE if args.smoke else RESULTS
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = RESULTS / "q4_task_aggregate.csv"
    frontier_path = RESULTS / "q4_frontier_monthly.csv"
    type_mapping_path = Q4 / "02_代码" / "scripts" / "q4_modeling_analysis.py"
    c3_path = RAW_C / "leaderboard_extended_timeseries.csv"
    c4_path = RAW_C / "epoch_all_ai_models.csv"
    contract_path = Q4 / "01_方案说明" / "q4_model_contract_v0.6.md"
    for path in [aggregate_path, frontier_path, c3_path, c4_path, contract_path, type_mapping_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    frontier = pd.read_csv(frontier_path, encoding="utf-8-sig")
    aggregate = pd.read_csv(aggregate_path, low_memory=False, encoding="utf-8-sig")
    ar_summary, ar_residuals, ar_backtest, ar_bt_summary = run_ar1(frontier)
    contributions, coefficients, support = run_decomposition(aggregate, args.smoke)
    decomposition_beta = float(coefficients.loc[coefficients["term"].eq("log10_params_B"), "estimate"].iloc[0])
    c3_summary, c3_audit = run_c3(c3_path, args.smoke)
    c4_summary, c4_confidence, c4_audit = run_c4(c4_path, args.smoke)
    suffix = "_smoke" if args.smoke else ""
    outputs = {
        f"q4_ar1_dynamic{suffix}.csv": ar_summary,
        f"q4_ar1_residuals{suffix}.csv": ar_residuals,
        f"q4_ar1_expanding_backtest{suffix}.csv": ar_backtest,
        f"q4_ar1_backtest_summary{suffix}.csv": ar_bt_summary,
        f"q4_scale_tech_contribution{suffix}.csv": contributions,
        f"q4_scale_tech_contribution_coefficients{suffix}.csv": coefficients,
        f"q4_scale_tech_support_audit{suffix}.csv": support,
        f"q4_c3_year_summary{suffix}.csv": c3_summary,
        f"q4_c3_dedup_audit{suffix}.csv": c3_audit,
        f"q4_c4_year_summary{suffix}.csv": c4_summary,
        f"q4_c4_confidence_summary{suffix}.csv": c4_confidence,
        f"q4_c4_cutoff_row_audit{suffix}.csv": c4_audit,
    }
    output_paths = [write_csv(frame, output_dir / name) for name, frame in outputs.items()]
    if args.smoke:
        report_path = output_dir / "q4_dynamic_decomposition_smoke_report.md"
        status_line = "最小范围运行：C3 限于 2024 年 OLLB，C4 限于截止日前的 2024 年记录，分解只用预设早晚月份。此包用于 P1 代码与数据接口复核，不作为全量结果。"
    else:
        report_path = Q4 / "01_方案说明" / "q4_dynamic_decomposition_report.md"
        status_line = "全量描述性建模已运行；Loss–Benchmark 仍为 BLOCKED_NO_FIT，未执行桥接拟合。"
    report_lines = [
        "# Q4 动态模型、共同支持分解与 C3/C4 年度背景", "",
        f"> {status_line}", "",
        f"> AI 辅助说明：{AI_DISCLOSURE} 其他 AI 对 Q4 交付的参与范围仍待团队核对。", "",
        "## 动态能力项", "",
        f"- AR(1) 状态：`{ar_summary.iloc[0]['status']}`；连续月数 {ar_summary.iloc[0]['monthly_bins']}；转移数 {ar_summary.iloc[0]['transitions']}。",
        f"- 系数为 a={ar_summary.iloc[0]['intercept_a']}、rho={ar_summary.iloc[0]['rho']}；HC3 区间见 `q4_ar1_dynamic{suffix}.csv`。区间不校正序列相关，仅作描述。",
        f"- 扩展窗口 AR(1) 与上月持平基线各有 {int(ar_bt_summary.iloc[0]['ar1_origins'])} 个合格的一步预测起点；误差见 `q4_ar1_backtest_summary{suffix}.csv`。样本不用于年度外推。",
        "",
        "## 规模与非规模组成分解", "",
        f"- 端点为 {support.iloc[0]['early_month']} / {support.iloc[0]['late_month']}；共同支持区间是两端点各自线性插值 P05–P95 参数量区间的交集 log10(B)=[{support.iloc[0]['log10_params_B_support_low_p05_p95']:.4f}, {support.iloc[0]['log10_params_B_support_high_p05_p95']:.4f}]；端点区间内样本数 {support.iloc[0]['early_rows_in_common_support']} / {support.iloc[0]['late_rows_in_common_support']}。该中心区间降低尾部观测影响，结论不外推至区间外记录。",
        f"- 精确分解使用所有 n≥20 月份中落入共同支持区间的 {support.iloc[0]['fit_rows_in_common_support']} 条记录重新拟合，β={decomposition_beta:.4f}，设计矩阵秩 {support.iloc[0]['design_rank']}/{support.iloc[0]['design_columns']}，残差自由度 {support.iloc[0]['residual_df']}。规模、类型、月份与总预测差由同一回归的线性对比精确加总；HC3 区间适用于这些拟合对比，并固定早晚期经验协变量分布。",
        "- 观测变化另以共同支持两组均值差给出 HC3 区间。`arithmetic_closure_residual` 是观测变化减去模型预测变化的算术闭合检查；由于端点月份虚拟变量进入同一 OLS 且端点记录参与拟合，该值按正规方程预期为机器精度下的 0，不代表未解释因素，故只报点值并检查加总误差。贡献占比仅在观测变化、规模项和非规模总项的 95% 区间均不含 0 时显示；占比为点值比率，不另构造区间。",
        f"- 贡献项与系数见 `q4_scale_tech_contribution{suffix}.csv`、`q4_scale_tech_contribution_coefficients{suffix}.csv`；共同支持和占比门记录见 `q4_scale_tech_support_audit{suffix}.csv`。",
        "",
        "## C3 年度排行榜背景", "",
        f"- 年表按 Source 单独列示；Open LLM Leaderboard 按 (Year, Model) 去重，存在不同指标冲突的模型年组排除。2025 标为部分年度覆盖，不纳入完整年度趋势；Historical (papers/reports) 保持独立。详见 `q4_c3_year_summary{suffix}.csv` 与逐行审计表。",
        "",
        "## C4 年度参数量、算力与数据量背景", "",
        f"- 只汇总 Publication date 不晚于 {C4_CUTOFF.date()} 的记录；区间文本和非数值文本不转成数值，数值列只对有限标量做中位数/P90。Confidence、Numerical format、Training compute estimation method、原始值和来源说明均在行审计中保留。",
        "- 年度 C4 只作宏观背景，不与 C1/C8 候选作个体连接，也不直接充当 12/24 月算力增长率。主年表列参数量、训练算力和数据量的标量描述及 Confident 标量计数；置信度分层表保留类别差异。",
        f"- 年表、置信度分层和截止日期行审计见 `q4_c4_year_summary{suffix}.csv`、`q4_c4_confidence_summary{suffix}.csv`、`q4_c4_cutoff_row_audit{suffix}.csv`。", "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "status": "smoke_for_P1_review" if args.smoke else "provisional_full_run",
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ai_disclosure": AI_DISCLOSURE, "script": rel(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()), "contract_sha256": sha256(contract_path),
        "inputs_sha256": {rel(p): sha256(p) for p in [aggregate_path, frontier_path, c3_path, c4_path, contract_path, type_mapping_path]},
        "outputs_sha256": {rel(p): sha256(p) for p in [*output_paths, report_path]},
        "smoke_run": bool(args.smoke), "loss_bridge_fit_executed": False,
        "analysis_scope": {
            "ar1_minimum_consecutive_months": 8, "ar1_annual_forecast_authorized": False,
            "scale_common_support": "intersection of endpoint linearly interpolated p05-p95 log10(parameter count in B); central-overlap inference only",
            "scale_fit_month_scope": "supported months n>=20 within common parameter interval; endpoints fixed to earliest/latest eligible months; smoke uses endpoints only",
            "c3_source_separated": True, "c4_publication_cutoff": C4_CUTOFF.date().isoformat(),
        },
        "row_counts": {
            "ar1_bins": int(ar_summary.iloc[0]["monthly_bins"]),
            "scale_fit_rows": int(support.iloc[0]["fit_rows_in_common_support"]),
            "scale_early_endpoint_rows": int(support.iloc[0]["early_rows_in_common_support"]),
            "scale_late_endpoint_rows": int(support.iloc[0]["late_rows_in_common_support"]),
            "c3_summary_rows": int(len(c3_summary)), "c3_audit_rows": int(len(c3_audit)),
            "c4_year_rows": int(len(c4_summary)), "c4_cutoff_audit_rows": int(len(c4_audit)),
        },
    }
    manifest_path = output_dir / ("q4_dynamic_decomposition_manifest_smoke.json" if args.smoke else "q4_dynamic_decomposition_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={manifest['status']} smoke={args.smoke} bridge_fit_executed=False")
    print(f"ar1={ar_summary.iloc[0]['status']} bins={ar_summary.iloc[0]['monthly_bins']} rho={ar_summary.iloc[0]['rho']}")
    print(f"decomposition_n={support.iloc[0]['fit_rows_in_common_support']} endpoints={support.iloc[0]['early_rows_in_common_support']}/{support.iloc[0]['late_rows_in_common_support']} shares={support.iloc[0]['contribution_shares_reported']}")
    print(f"c3_years={len(c3_summary)} c4_years={len(c4_summary)} report={rel(report_path)} manifest={rel(manifest_path)}")


if __name__ == "__main__":
    main()
