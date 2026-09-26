"""Minimum descriptive scale/non-scale share analysis for Q4.

Uses the original C1 Average on the longest supported month window in the
C1/C8 name-candidate aggregate. It is an observational cohort decomposition,
not a causal estimate of technological progress.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "Q4" / "09_C1全窗占比补充_20260926"
INPUT = ROOT / "Q4" / "03_结果" / "results" / "q4_task_aggregate.csv"
SOURCE_SCRIPT = ROOT / "Q4" / "02_代码" / "scripts" / "q4_dynamic_decomposition.py"
OUTPUT = PACKAGE / "results"
MIN_MONTH_N = 20
Z95 = 1.96

sys.path.insert(0, str(SOURCE_SCRIPT.parent))
import q4_dynamic_decomposition as decomposition  # noqa: E402
from q4_modeling_analysis import normalize_type  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def publisher_key(value: object) -> str:
    name = str(value).strip()
    owner = name.split("/", 1)[0].strip() if "/" in name else name
    return owner.lower()


def fieller(numerator: float, denominator: float, var_n: float,
            var_d: float, cov_nd: float, critical: float) -> dict:
    z2 = critical * critical
    a = denominator * denominator - z2 * var_d
    b = -2.0 * (numerator * denominator - z2 * cov_nd)
    c = numerator * numerator - z2 * var_n
    disc = b * b - 4.0 * a * c
    tol = 1e-12 * max(1.0, abs(a), abs(b), abs(c))
    low = high = math.nan
    if a > tol and disc >= -tol:
        root = math.sqrt(max(disc, 0.0))
        low, high = sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        shape = "bounded_interval" if disc > tol else "singleton"
    elif a > tol:
        shape = "empty"
    elif a < -tol and disc >= -tol:
        shape = "disconnected_unbounded"
        low, high = -math.inf, math.inf
    elif a < -tol:
        shape = "all_real"
        low, high = -math.inf, math.inf
    elif abs(b) <= tol:
        shape = "all_real" if c <= 0 else "empty"
        low, high = (-math.inf, math.inf) if shape == "all_real" else (math.nan, math.nan)
    else:
        root = -c / b
        shape = "one_sided_unbounded"
        low, high = (-math.inf, root) if b > 0 else (root, math.inf)
    den_low = denominator - critical * math.sqrt(max(var_d, 0.0))
    den_high = denominator + critical * math.sqrt(max(var_d, 0.0))
    return {
        "fieller_shape": shape,
        "fieller_low_ratio": low,
        "fieller_high_ratio": high,
        "denominator_ci95_low": den_low,
        "denominator_ci95_high": den_high,
        "denominator_ci_excludes_zero": bool(den_low > 0 or den_high < 0),
        "ratio_bounded": bool(math.isfinite(low) and math.isfinite(high)),
    }


def main() -> None:
    raw = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)
    required = {
        "Model", "model_name_raw", "C1_Average_0_100", "parameter_count_B",
        "submission_date", "C1_Type_raw",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")
    if raw["Model"].isna().any() or raw["Model"].nunique() != len(raw):
        raise ValueError("The input must contain one row per unique C1/C8 model-name candidate.")

    frame = raw.copy()
    frame["score"] = pd.to_numeric(frame["C1_Average_0_100"], errors="coerce")
    frame["parameter_count_B"] = pd.to_numeric(frame["parameter_count_B"], errors="coerce")
    frame["submission_date"] = pd.to_datetime(frame["submission_date"], errors="coerce")
    frame = frame[
        frame["score"].notna()
        & frame["parameter_count_B"].gt(0)
        & frame["submission_date"].notna()
    ].copy()
    frame["log10_params_B"] = np.log10(frame["parameter_count_B"])
    frame["submission_month"] = frame["submission_date"].dt.to_period("M").astype(str)
    normalized_type = frame["C1_Type_raw"].map(normalize_type)
    frame["normalized_type"] = normalized_type
    # Keep only the broad pre-trained / post-trained-or-other distinction. This
    # avoids estimating separate effects for sparse leaderboard labels.
    frame["type_group"] = np.where(
        normalized_type.isin({"pretrained", "continuously_pretrained"}),
        "base_pretrained",
        "chat_finetuned_or_other",
    )

    month_counts = frame["submission_month"].value_counts().sort_index()
    supported_months = sorted(month_counts[month_counts.ge(MIN_MONTH_N)].index.tolist())
    if len(supported_months) < 2:
        raise ValueError("At least two months with 20 or more eligible rows are required.")
    early_month, late_month = supported_months[0], supported_months[-1]
    early_raw = frame[frame["submission_month"].eq(early_month)]
    late_raw = frame[frame["submission_month"].eq(late_month)]
    support_low = max(
        float(early_raw["log10_params_B"].quantile(.05, interpolation="linear")),
        float(late_raw["log10_params_B"].quantile(.05, interpolation="linear")),
    )
    support_high = min(
        float(early_raw["log10_params_B"].quantile(.95, interpolation="linear")),
        float(late_raw["log10_params_B"].quantile(.95, interpolation="linear")),
    )
    if not support_low < support_high:
        raise ValueError("Endpoint P05-P95 parameter-size supports do not overlap.")

    fit_frame = frame[
        frame["submission_month"].isin(supported_months)
        & frame["log10_params_B"].between(support_low, support_high, inclusive="both")
    ].copy()
    fit_frame = fit_frame.sort_values(["submission_month", "model_name_raw"]).reset_index(drop=True)
    if len(fit_frame) < 30:
        raise ValueError("Common-support sample is too small.")
    x, labels, references = decomposition.build_design(fit_frame)
    y = fit_frame["score"].to_numpy(float)
    fit = decomposition.hc3_fit(x, y, labels)
    if fit["residual_df"] <= 0:
        raise ValueError("The common-support design has no residual degrees of freedom.")

    early_mask = fit_frame["submission_month"].eq(early_month).to_numpy()
    late_mask = fit_frame["submission_month"].eq(late_month).to_numpy()
    if not early_mask.any() or not late_mask.any():
        raise ValueError("An endpoint has no rows after common-support filtering.")
    x_early, x_late = x[early_mask], x[late_mask]
    y_early, y_late = y[early_mask], y[late_mask]
    xbar_early, xbar_late = x_early.mean(axis=0), x_late.mean(axis=0)
    delta_x = xbar_late - xbar_early
    denominator = float(y_late.mean() - y_early.mean())

    scale_indices = [labels.index("log10_params_B")]
    type_indices = [i for i, label in enumerate(labels) if label.startswith("type[")]
    month_indices = [i for i, label in enumerate(labels) if label.startswith("month[")]
    component_indices = {
        "scale_composition": scale_indices,
        "type_composition": type_indices,
        "month_fixed_effect": month_indices,
        "non_scale_total": type_indices + month_indices,
    }

    publishers = fit_frame["model_name_raw"].map(publisher_key).to_numpy()
    cluster_ids = sorted(pd.unique(publishers).tolist())
    endpoint_clusters = sorted(pd.unique(publishers[early_mask | late_mask]).tolist())
    n_clusters, n_endpoint_clusters = len(cluster_ids), len(endpoint_clusters)
    if n_endpoint_clusters < 30:
        raise ValueError("Too few publisher clusters contribute endpoint observations.")
    critical = float(student_t.ppf(.975, n_endpoint_clusters - 1))
    # Use the smaller endpoint-cluster count for a common joint finite-sample correction.
    correction = n_endpoint_clusters / (n_endpoint_clusters - 1)
    type_group_cluster_counts = {}
    for group in sorted(fit_frame["type_group"].unique()):
        group_mask = fit_frame["type_group"].eq(group).to_numpy()
        endpoint_group_mask = group_mask & (early_mask | late_mask)
        type_group_cluster_counts[group] = {
            "fit_rows": int(group_mask.sum()),
            "fit_publisher_clusters": int(pd.Series(publishers[group_mask]).nunique()),
            "endpoint_rows": int(endpoint_group_mask.sum()),
            "endpoint_publisher_clusters": int(pd.Series(publishers[endpoint_group_mask]).nunique()),
        }

    xtx_inverse = np.linalg.inv(x.T @ x)
    psi_beta: list[np.ndarray] = []
    psi_delta_x: list[np.ndarray] = []
    psi_denominator: list[float] = []
    n_early, n_late = int(early_mask.sum()), int(late_mask.sum())
    ybar_early, ybar_late = float(y_early.mean()), float(y_late.mean())
    for cluster in cluster_ids:
        in_cluster = publishers == cluster
        score = x[in_cluster].T @ fit["residual"][in_cluster]
        psi_beta.append(xtx_inverse @ score)
        e_cluster = in_cluster & early_mask
        l_cluster = in_cluster & late_mask
        sum_x_e = x[e_cluster].sum(axis=0) if e_cluster.any() else np.zeros(x.shape[1])
        sum_x_l = x[l_cluster].sum(axis=0) if l_cluster.any() else np.zeros(x.shape[1])
        sum_y_e = float(y[e_cluster].sum()) if e_cluster.any() else 0.0
        sum_y_l = float(y[l_cluster].sum()) if l_cluster.any() else 0.0
        psi_delta_x.append(
            (sum_x_l - int(l_cluster.sum()) * xbar_late) / n_late
            - (sum_x_e - int(e_cluster.sum()) * xbar_early) / n_early
        )
        psi_denominator.append(
            (sum_y_l - int(l_cluster.sum()) * ybar_late) / n_late
            - (sum_y_e - int(e_cluster.sum()) * ybar_early) / n_early
        )
    psi_beta_array = np.asarray(psi_beta)
    psi_dx_array = np.asarray(psi_delta_x)
    psi_d_array = np.asarray(psi_denominator)
    var_d = float(correction * (psi_d_array @ psi_d_array))

    rows: list[dict] = []
    for name, indices in component_indices.items():
        if not indices:
            continue
        indices_array = np.asarray(indices, dtype=int)
        numerator = float(fit["beta"][indices_array] @ delta_x[indices_array])
        psi_n = (
            psi_beta_array[:, indices_array] @ delta_x[indices_array]
            + psi_dx_array[:, indices_array] @ fit["beta"][indices_array]
        )
        var_n = float(correction * (psi_n @ psi_n))
        cov_nd = float(correction * (psi_n @ psi_d_array))
        se_n = math.sqrt(max(var_n, 0.0))
        inference = fieller(numerator, denominator, var_n, var_d, cov_nd, critical)
        rows.append({
            "component": name,
            "estimate_C1_Average_points": numerator,
            "publisher_cluster_IF_approx_se": se_n,
            "publisher_cluster_IF_approx_ci95_low": numerator - critical * se_n,
            "publisher_cluster_IF_approx_ci95_high": numerator + critical * se_n,
            "share_point_pct": 100.0 * numerator / denominator if abs(denominator) > 1e-12 else math.nan,
            "share_fieller_ci95_low_pct": 100.0 * inference["fieller_low_ratio"],
            "share_fieller_ci95_high_pct": 100.0 * inference["fieller_high_ratio"],
            "share_fieller_shape": inference["fieller_shape"],
            "ratio_bounded": inference["ratio_bounded"],
            "denominator_ci95_low": inference["denominator_ci95_low"],
            "denominator_ci95_high": inference["denominator_ci95_high"],
            "denominator_ci_excludes_zero": inference["denominator_ci_excludes_zero"],
        })

    decomposition_table = pd.DataFrame(rows)
    point_map = decomposition_table.set_index("component")["estimate_C1_Average_points"].to_dict()
    predicted_change = point_map["scale_composition"] + point_map["type_composition"] + point_map["month_fixed_effect"]
    closure = abs(point_map["type_composition"] + point_map["month_fixed_effect"] - point_map["non_scale_total"])
    observed_error = abs(predicted_change - denominator)
    if closure > 1e-8 or observed_error > 1e-8:
        raise ValueError(f"Decomposition closure failed: component={closure}, observed={observed_error}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT / "q4_c1_full_window_decomposition_and_shares.csv"
    decomposition_table.to_csv(table_path, index=False, encoding="utf-8-sig")
    diagnostics = {
        "status": "descriptive_share_estimable_with_bounded_Fieller_sets",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "input": INPUT.relative_to(ROOT).as_posix(),
        "input_sha256": sha256(INPUT),
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_model_script": SOURCE_SCRIPT.relative_to(ROOT).as_posix(),
        "source_model_script_sha256": sha256(SOURCE_SCRIPT),
        "outcome": "C1_Average_0_100, the single leaderboard Average field used consistently over all supported months",
        "cohort": "one row per C8/C1 exact-name candidate in q4_task_aggregate.csv; exact model revisions and license permissions are not implied",
        "candidate_rows": len(raw),
        "unique_model_names": int(raw["Model"].nunique()),
        "strict_version_confirmed_rows": int(
            raw["strict_version_confirmed"].astype("string").str.strip().str.lower().isin({"true", "1", "yes"}).sum()
        ) if "strict_version_confirmed" in raw else None,
        "time_axis": "C1 submission_date; earliest and latest months with at least 20 eligible records",
        "early_month": early_month,
        "late_month": late_month,
        "eligible_month_counts": {key: int(month_counts[key]) for key in supported_months},
        "support_log10_parameter_count_B": [support_low, support_high],
        "endpoint_rows_in_common_support": {early_month: n_early, late_month: n_late},
        "fit_rows": len(fit_frame),
        "design_rank": int(fit["rank"]),
        "design_columns": int(fit["p"]),
        "residual_df": int(fit["residual_df"]),
        "type_reference": references["reference_type"],
        "type_grouping_rule": "base_pretrained combines pretrained and continuously_pretrained; chat_finetuned_or_other combines chat_posttrained, domain_finetuned, merge, multimodal, other, and unknown_or_unmapped",
        "month_reference": references["reference_month"],
        "type_group_counts_in_fit": {
            str(key): int(value) for key, value in fit_frame["type_group"].value_counts().items()
        },
        "type_group_publisher_cluster_counts": type_group_cluster_counts,
        "normalized_type_counts_in_fit": {
            str(key): int(value) for key, value in fit_frame["normalized_type"].value_counts().items()
        },
        "observed_endpoint_change_C1_Average_points": denominator,
        "publisher_cluster_key": "lowercased namespace before first slash in model_name_raw",
        "publisher_clusters_in_fit": n_clusters,
        "publisher_clusters_in_endpoints": n_endpoint_clusters,
        "joint_cluster_correction": correction,
        "critical_t_975_df": critical,
        "uncertainty": "joint publisher-cluster influence-function sandwich, including uncertainty in fitted coefficients and endpoint covariate distributions; uses analyst-defined endpoint-cluster correction G/(G-1) and t(df=G-1), with Fieller ratio sets from the joint covariance",
        "inference_caveat": "This finite-sample adjustment is a custom approximation, not the canonical OLS CR1 factor. The base_pretrained subgroup contributes only 15 endpoint publisher clusters, so its type-composition interval may understate uncertainty; interpret all intervals as approximate descriptive uncertainty, not guaranteed 95% coverage.",
        "causal_status": "not_causal; time fixed effect is only a non-scale evolution proxy and absorbs unobserved time-varying composition, data, alignment, and evaluation differences",
        "known_limits": [
            "C1 Average is the raw leaderboard summary, not the C8 six-family index used in the shorter G4 decomposition.",
            "C1 and C8 are matched by model name; exact revisions are not confirmed.",
            "C1 Average is treated as a consistent score field across this 10-month source window; hidden protocol changes cannot be ruled out from supplied rows alone.",
            "A bounded share interval establishes a finite descriptive ratio, not that the time term is pure technical progress or a causal effect.",
        ],
        "component_sum_abs_error": closure,
        "predicted_observed_change_abs_error": observed_error,
        "output": table_path.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(table_path),
    }
    diagnostic_path = OUTPUT / "q4_c1_full_window_share_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": diagnostics["status"],
        "window": [early_month, late_month],
        "n": len(fit_frame),
        "support": [support_low, support_high],
        "observed_change": denominator,
        "publisher_clusters": n_clusters,
        "endpoint_publisher_clusters": n_endpoint_clusters,
        "outputs": [str(table_path), str(diagnostic_path)],
    }, ensure_ascii=False, indent=2))
    print(decomposition_table.to_string(index=False))


if __name__ == "__main__":
    main()
