"""Isolated G4 rerun: decomposition, Fieller shares, publisher-cluster sensitivity.

Reads the frozen Q4 aggregate and existing decomposition implementation; writes
only under Q4/07_G4优化执行_20260926. This is an incremental audit artifact.

AI-use note for any contest code submission: assisted by OpenAI Codex (GPT-6
family) for analysis and code drafting; developer OpenAI. Exact model/version
and official release date are unavailable in this task record. The team must
verify and complete those fields and the full tool-participation scope before
submitting this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
G4 = ROOT / "Q4" / "07_G4优化执行_20260926"
Q4 = ROOT / "Q4"
RESULTS = Q4 / "03_结果" / "results"
AGGREGATE_PATH = RESULTS / "q4_task_aggregate.csv"
REFERENCE_COMPONENTS = RESULTS / "q4_scale_tech_contribution.csv"
REFERENCE_SUPPORT = RESULTS / "q4_scale_tech_support_audit.csv"
SOURCE_SCRIPT = Q4 / "02_代码" / "scripts" / "q4_dynamic_decomposition.py"
SOURCE_MODEL = Q4 / "02_代码" / "scripts" / "q4_modeling_analysis.py"
SOURCE_CONTRACT = Q4 / "01_方案说明" / "q4_model_contract_v0.6.md"
EARLY_MONTH = "2024-11"
LATE_MONTH = "2025-03"
Z95 = 1.96
SEED = 20260926

sys.path.insert(0, str(SOURCE_SCRIPT.parent))
import q4_dynamic_decomposition as decomposition  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_support(aggregate: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    d = aggregate[
        aggregate["core6_index_0_100"].notna()
        & pd.to_numeric(aggregate["parameter_count_B"], errors="coerce").gt(0)
        & aggregate["submission_date"].notna()
    ].copy()
    d["core6_index_0_100"] = pd.to_numeric(d["core6_index_0_100"], errors="coerce")
    d["parameter_count_B"] = pd.to_numeric(d["parameter_count_B"], errors="coerce")
    d["log10_params_B"] = np.log10(d["parameter_count_B"])
    d["submission_month"] = pd.to_datetime(d["submission_date"], errors="coerce").dt.to_period("M").astype(str)
    d["type_group"] = decomposition.map_type_groups(d["C1_Type_raw"])
    counts = d["submission_month"].value_counts()
    months = sorted(counts[counts.ge(20)].index.tolist())
    if not months or months[0] != EARLY_MONTH or months[-1] != LATE_MONTH:
        raise ValueError(f"Expected fixed endpoint months unavailable: {months}")
    endpoints = d[d["submission_month"].isin([EARLY_MONTH, LATE_MONTH])]
    early_raw = endpoints[endpoints["submission_month"].eq(EARLY_MONTH)]
    late_raw = endpoints[endpoints["submission_month"].eq(LATE_MONTH)]
    low = max(float(early_raw["log10_params_B"].quantile(.05, interpolation="linear")),
              float(late_raw["log10_params_B"].quantile(.05, interpolation="linear")))
    high = min(float(early_raw["log10_params_B"].quantile(.95, interpolation="linear")),
               float(late_raw["log10_params_B"].quantile(.95, interpolation="linear")))
    fit_rows = d[d["submission_month"].isin(months)
                 & d["log10_params_B"].between(low, high, inclusive="both")]
    fit_rows = fit_rows.sort_values(["submission_month", "model_name_raw"]).reset_index(drop=True)
    if not low < high or len(fit_rows) < 30:
        raise ValueError("Common-support interval or sample size is invalid")
    return fit_rows, low, high


def component_vectors(frame: pd.DataFrame, labels: list[str]) -> tuple[dict[str, np.ndarray], float]:
    early = frame[frame["submission_month"].eq(EARLY_MONTH)]
    late = frame[frame["submission_month"].eq(LATE_MONTH)]
    if early.empty or late.empty:
        raise ValueError("Both endpoint groups must contain rows")

    def mean_design(group: pd.DataFrame) -> np.ndarray:
        v = np.zeros(len(labels), dtype=float)
        v[labels.index("intercept")] = 1.0
        v[labels.index("log10_params_B")] = float(group["log10_params_B"].mean())
        shares = group["type_group"].value_counts(normalize=True).to_dict()
        for i, label in enumerate(labels):
            if label.startswith("type["):
                v[i] = float(shares.get(label[5:-1], 0.0))
            elif label.startswith("month["):
                v[i] = float(group["submission_month"].eq(label[6:-1]).mean())
        return v

    delta = mean_design(late) - mean_design(early)
    scale = np.zeros(len(labels), dtype=float)
    scale[labels.index("log10_params_B")] = delta[labels.index("log10_params_B")]
    type_part = np.zeros(len(labels), dtype=float)
    month_part = np.zeros(len(labels), dtype=float)
    for i, label in enumerate(labels):
        if label.startswith("type["):
            type_part[i] = delta[i]
        elif label.startswith("month["):
            month_part[i] = delta[i]
    components = {
        "scale_composition": scale,
        "type_composition": type_part,
        "month_fixed_effect": month_part,
        "non_scale_total": type_part + month_part,
        "predicted_mean_change": scale + type_part + month_part,
    }
    observed = float(late["core6_index_0_100"].mean() - early["core6_index_0_100"].mean())
    return components, observed


def influence(covariates: np.ndarray, residual: np.ndarray,
              contrast: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(covariates.T @ covariates)
    leverage = np.einsum("ij,jk,ik->i", covariates, inv, covariates)
    adjusted_residual = residual / np.maximum(1.0 - leverage, 1e-12)
    return (covariates @ (inv @ contrast)) * adjusted_residual


def fieller(numer: float, denom: float, var_n: float, var_d: float,
            cov_nd: float) -> dict[str, float | str | bool]:
    z2 = Z95 ** 2
    a = denom * denom - z2 * var_d
    b = -2.0 * (numer * denom - z2 * cov_nd)
    c = numer * numer - z2 * var_n
    disc = b * b - 4.0 * a * c
    tol = 1e-12 * max(1.0, abs(a), abs(b), abs(c))
    lo = math.nan
    hi = math.nan
    root_left = math.nan
    root_right = math.nan
    if abs(a) <= tol:
        if abs(b) <= tol:
            shape = "all_real" if c <= 0 else "empty"
            lo, hi = (-math.inf, math.inf) if shape == "all_real" else (math.nan, math.nan)
        else:
            root = -c / b
            root_left = root
            root_right = root
            shape = "one_sided_unbounded"
            lo, hi = (-math.inf, root) if b > 0 else (root, math.inf)
    elif disc < -tol:
        shape = "empty" if a > 0 else "all_real"
        lo, hi = (math.nan, math.nan) if shape == "empty" else (-math.inf, math.inf)
    else:
        root_delta = math.sqrt(max(0.0, disc))
        r1, r2 = sorted(((-b - root_delta) / (2 * a), (-b + root_delta) / (2 * a)))
        root_left, root_right = r1, r2
        lo, hi = r1, r2
        if a > 0:
            shape = "bounded_interval" if disc > tol else "singleton"
        else:
            shape = "disconnected_unbounded" if disc > tol else "all_real"
            if shape == "disconnected_unbounded":
                lo, hi = -math.inf, math.inf
    finite_bounds = math.isfinite(lo) and math.isfinite(hi)
    return {
        "fieller_shape": shape,
        "fieller_lower_ratio": lo,
        "fieller_upper_ratio": hi,
        "fieller_boundary_1_ratio": root_left,
        "fieller_boundary_2_ratio": root_right,
        "fieller_discriminant": disc,
        "denominator_ci95_low": denom - Z95 * math.sqrt(max(var_d, 0.0)),
        "denominator_ci95_high": denom + Z95 * math.sqrt(max(var_d, 0.0)),
        "denominator_ci_crosses_zero": bool(denom - Z95 * math.sqrt(max(var_d, 0.0)) <= 0 <= denom + Z95 * math.sqrt(max(var_d, 0.0))),
        "ratio_stably_identified": bool(finite_bounds and shape in {"bounded_interval", "singleton"}),
    }


def fixed_design(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    cols = [np.ones(len(frame)), frame["log10_params_B"].to_numpy(float)]
    for label in labels[2:]:
        if label.startswith("type["):
            cols.append(frame["type_group"].astype(str).eq(label[5:-1]).to_numpy(float))
        elif label.startswith("month["):
            cols.append(frame["submission_month"].astype(str).eq(label[6:-1]).to_numpy(float))
        else:
            raise ValueError(f"Unsupported design term: {label}")
    return np.column_stack(cols)


def publisher_key(raw_name: object) -> str:
    name = str(raw_name).strip()
    owner = name.split("/", 1)[0].strip() if "/" in name else name
    return owner.lower()


def publisher_bootstrap(frame: pd.DataFrame, labels: list[str], vectors: dict[str, np.ndarray],
                        original_denom: float, draws: int) -> tuple[pd.DataFrame, dict]:
    d = frame.copy().reset_index(drop=True)
    d["publisher_cluster"] = d["model_name_raw"].map(publisher_key)
    clusters = sorted(d["publisher_cluster"].unique().tolist())
    if len(clusters) < 3:
        raise ValueError(f"Need at least three publisher clusters, got {len(clusters)}")
    groups = {key: d[d["publisher_cluster"].eq(key)] for key in clusters}
    rng = np.random.default_rng(SEED + 11)
    failures: Counter[str] = Counter()
    rows: list[dict] = []
    for replicate in range(draws):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([groups[key] for key in picked], ignore_index=True)
        try:
            early = boot[boot["submission_month"].eq(EARLY_MONTH)]
            late = boot[boot["submission_month"].eq(LATE_MONTH)]
            if early.empty or late.empty:
                raise ValueError("endpoint_missing")
            x = fixed_design(boot, labels)
            y = boot["core6_index_0_100"].to_numpy(float)
            fit = decomposition.hc3_fit(x, y, labels)
            observed = float(late["core6_index_0_100"].mean() - early["core6_index_0_100"].mean())
            boot_vectors, _ = component_vectors(boot, labels)
            for name, vector in boot_vectors.items():
                estimate = float(vector @ fit["beta"])
                share = 100.0 * estimate / observed if abs(observed) > 1e-12 else math.nan
                rows.append({
                    "replicate": replicate + 1,
                    "component": name,
                    "numerator": estimate,
                    "observed_denominator": observed,
                    "share_pct": share,
                    "denominator_sign_opposite_original": bool(np.sign(observed) != np.sign(original_denom)),
                })
        except Exception as exc:  # failures are counted with exact type/message below
            failures[f"{type(exc).__name__}: {str(exc)}"] += 1
    result = pd.DataFrame(rows)
    successful_replicates = len({int(value) for value in result.get("replicate", pd.Series(dtype=int))})
    summary = []
    for component, group in result.groupby("component") if not result.empty else []:
        finite = group[np.isfinite(group["share_pct"])]
        summary.append({
            "component": component,
            "successful_replicates": successful_replicates,
            "requested_replicates": draws,
            "failed_replicates": draws - successful_replicates,
            "ratio_replicates_nonzero_denominator": int(len(finite)),
            "denominator_sign_opposite_original_n": int(group["denominator_sign_opposite_original"].sum()),
            "denominator_sign_opposite_original_pct": float(group["denominator_sign_opposite_original"].mean() * 100),
            "share_q025_pct": float(finite["share_pct"].quantile(.025)) if not finite.empty else math.nan,
            "share_median_pct": float(finite["share_pct"].median()) if not finite.empty else math.nan,
            "share_q975_pct": float(finite["share_pct"].quantile(.975)) if not finite.empty else math.nan,
            "interval_basis": "publisher-cluster bootstrap sensitivity only; not a calibrated CI",
        })
    diagnostics = {
        "cluster_key_source": "q4_task_aggregate.csv:model_name_raw; substring before first slash, stripped and lowercase; no slash uses full model name",
        "publisher_clusters": len(clusters),
        "publisher_cluster_names": clusters,
        "requested_replicates": draws,
        "successful_replicates": successful_replicates,
        "failed_replicates": draws - successful_replicates,
        "failure_reasons": dict(failures),
        "denominator_sign_opposite_original_replicates": int(result.drop_duplicates("replicate")["denominator_sign_opposite_original"].sum()) if not result.empty else 0,
        "denominator_sign_opposite_original_pct": float(result.drop_duplicates("replicate")["denominator_sign_opposite_original"].mean() * 100) if not result.empty else math.nan,
        "support_interval_reselected": False,
        "fixed_support_rows": int(len(frame)),
    }
    return pd.DataFrame(summary), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Use 60 publisher-cluster bootstrap replicates for P1")
    args = parser.parse_args()
    draws = 60 if args.smoke else 2000
    output_dir = G4 / "results" / ("smoke" if args.smoke else "full") / "decomposition"
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = pd.read_csv(AGGREGATE_PATH, low_memory=False, encoding="utf-8-sig")
    contributions, coefficients, support = decomposition.run_decomposition(aggregate, smoke=False)
    frame, low, high = prepare_support(aggregate)
    x, labels, references = decomposition.build_design(frame)
    fit = decomposition.hc3_fit(x, frame["core6_index_0_100"].to_numpy(float), labels)
    vectors, observed = component_vectors(frame, labels)

    endpoint = frame[frame["submission_month"].isin([EARLY_MONTH, LATE_MONTH])].copy()
    observed_x = np.column_stack([np.ones(len(endpoint)), endpoint["submission_month"].eq(LATE_MONTH).to_numpy(float)])
    observed_fit = decomposition.hc3_fit(observed_x, endpoint["core6_index_0_100"].to_numpy(float), ["intercept", "late_minus_early"])
    denominator_contrast = np.array([0.0, 1.0])
    den_psi_local = influence(observed_x, observed_fit["residual"], denominator_contrast)
    den_psi = np.zeros(len(frame), dtype=float)
    den_positions = np.flatnonzero(frame["submission_month"].isin([EARLY_MONTH, LATE_MONTH]).to_numpy())
    den_psi[den_positions] = den_psi_local
    var_d = float(den_psi @ den_psi)
    expected_var_d = float(denominator_contrast @ observed_fit["covariance"] @ denominator_contrast)
    if abs(var_d - expected_var_d) > 1e-8:
        raise ValueError(f"Observed denominator HC3 influence variance mismatch: {var_d} vs {expected_var_d}")

    ratio_rows = []
    for name, vector in vectors.items():
        numerator = float(vector @ fit["beta"])
        num_psi = influence(x, fit["residual"], vector)
        var_n = float(num_psi @ num_psi)
        expected_var_n = float(vector @ fit["covariance"] @ vector)
        if abs(var_n - expected_var_n) > 1e-8:
            raise ValueError(f"Numerator HC3 influence variance mismatch for {name}: {var_n} vs {expected_var_n}")
        cov_nd = float(num_psi @ den_psi)
        interval = fieller(numerator, observed, var_n, var_d, cov_nd)
        share_point = 100.0 * numerator / observed if abs(observed) > 1e-12 else math.nan
        lo, hi = interval["fieller_lower_ratio"], interval["fieller_upper_ratio"]
        ratio_rows.append({
            "component": name,
            "numerator_estimate": numerator,
            "observed_endpoint_mean_difference": observed,
            "descriptive_point_share_pct": share_point,
            "numerator_variance_HC3": var_n,
            "denominator_variance_HC3": var_d,
            "numerator_denominator_covariance_HC3": cov_nd,
            "fieller_lower_pct": lo * 100 if math.isfinite(lo) else lo,
            "fieller_upper_pct": hi * 100 if math.isfinite(hi) else hi,
            **interval,
            "share_interpretation": "stable inferential share" if interval["ratio_stably_identified"] else "ratio not stably identified; point share descriptive only",
        })
    ratio_table = pd.DataFrame(ratio_rows)

    bootstrap_summary, bootstrap_diagnostics = publisher_bootstrap(
        frame, labels, vectors, observed, draws)
    support["decomposition_full_sample_rerun"] = True
    support["publisher_bootstrap_smoke_run"] = bool(args.smoke)
    support["run_scope_note"] = "smoke_run records the upstream decomposition scope (full sample here); publisher_bootstrap_smoke_run records the reduced/full cluster-bootstrap count."
    contribution_out = contributions.copy()
    share_map = ratio_table.set_index("component")["descriptive_point_share_pct"].to_dict()
    contribution_out["descriptive_point_share_pct"] = contribution_out["component"].map(share_map)
    contribution_out["inferential_share_status"] = contribution_out["component"].map(
        ratio_table.set_index("component")["share_interpretation"].to_dict()).fillna("not_applicable")

    frozen_contributions = pd.read_csv(REFERENCE_COMPONENTS, encoding="utf-8-sig")
    merged = contributions.merge(frozen_contributions, on="component", suffixes=("_rerun", "_frozen"), validate="one_to_one")
    compare_cols = ["estimate_index_points", "hc3_se", "ci95_low", "ci95_high"]
    diffs = {col: float((merged[f"{col}_rerun"] - merged[f"{col}_frozen"]).abs().max()) for col in compare_cols}
    frozen_support = pd.read_csv(REFERENCE_SUPPORT, encoding="utf-8-sig").iloc[0]
    support_row = support.iloc[0]
    support_diffs = {
        "common_support_low_abs_diff": abs(float(support_row["log10_params_B_support_low_p05_p95"]) - float(frozen_support["log10_params_B_support_low_p05_p95"])),
        "common_support_high_abs_diff": abs(float(support_row["log10_params_B_support_high_p05_p95"]) - float(frozen_support["log10_params_B_support_high_p05_p95"])),
        "fit_rows_equal": int(support_row["fit_rows_in_common_support"]) == int(frozen_support["fit_rows_in_common_support"]),
        "design_rank_equal": int(support_row["design_rank"]) == int(frozen_support["design_rank"]),
    }
    closure = abs(float(contribution_out.loc[contribution_out["component"].eq("scale_composition"), "estimate_index_points"].iloc[0]
                       + contribution_out.loc[contribution_out["component"].eq("type_composition"), "estimate_index_points"].iloc[0]
                       + contribution_out.loc[contribution_out["component"].eq("month_fixed_effect"), "estimate_index_points"].iloc[0]
                       - contribution_out.loc[contribution_out["component"].eq("predicted_mean_change"), "estimate_index_points"].iloc[0]))
    if max(diffs.values()) > 1e-8 or max(support_diffs[k] for k in ["common_support_low_abs_diff", "common_support_high_abs_diff"]) > 1e-8 or not support_diffs["fit_rows_equal"] or not support_diffs["design_rank_equal"] or closure > 1e-8:
        raise ValueError(f"Frozen-baseline reproduction failed: contribution={diffs}; support={support_diffs}; closure={closure}")

    out_files = {
        "q4_g4_decomposition.csv": contribution_out,
        "q4_g4_coefficients.csv": coefficients,
        "q4_g4_support_audit.csv": support,
        "q4_g4_share_fieller.csv": ratio_table,
        "q4_g4_publisher_bootstrap_summary.csv": bootstrap_summary,
    }
    for filename, table in out_files.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8-sig", na_rep="")
    bootstrap_diagnostics["component_share_summary"] = bootstrap_summary.to_dict(orient="records")
    diagnostics = {
        "status": "smoke_for_P1_review" if args.smoke else "provisional_full_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(AGGREGATE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": sha256(AGGREGATE_PATH),
        "execution_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "execution_script_sha256": sha256(Path(__file__).resolve()),
        "source_decomposition_script_sha256": sha256(SOURCE_SCRIPT),
        "source_modeling_script_sha256": sha256(SOURCE_MODEL),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "frozen_reference_sha256": {
            str(REFERENCE_COMPONENTS.relative_to(ROOT)).replace("\\", "/"): sha256(REFERENCE_COMPONENTS),
            str(REFERENCE_SUPPORT.relative_to(ROOT)).replace("\\", "/"): sha256(REFERENCE_SUPPORT),
        },
        "analysis": {
            "early_month": EARLY_MONTH, "late_month": LATE_MONTH,
            "support_log10_params_B": [low, high],
            "fit_rows": int(len(frame)), "design_rank": int(fit["rank"]),
            "design_columns": int(fit["p"]), "residual_df": int(fit["residual_df"]),
            "observed_endpoint_mean_difference": observed,
            "observed_change_HC3_se": math.sqrt(max(var_d, 0.0)),
            "arithmetic_component_closure_abs_error": closure,
            "frozen_baseline_max_abs_differences": diffs,
            "frozen_support_comparison": support_diffs,
            "hc3_cross_covariance_method": "rowwise product-sum of HC3-adjusted influence contributions on shared common-support endpoint rows; zero denominator influence for non-endpoint months",
            "publisher_cluster_bootstrap": bootstrap_diagnostics,
        },
        "run": {
            "smoke": bool(args.smoke),
            "decomposition_full_sample_rerun": True,
            "publisher_bootstrap_smoke_replicates": draws if args.smoke else None,
            "bootstrap_replicates_requested": draws,
            "seed": SEED,
            "support_audit_smoke_run_semantics": "False means the decomposition itself used all eligible months/full common-support sample; run.smoke only shortens publisher bootstrap.",
        },
        "outputs_sha256": {name: sha256(output_dir / name) for name in out_files},
    }
    diagnostics_path = output_dir / "q4_g4_decomposition_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"support=[{low:.6f},{high:.6f}] n={len(frame)} rank={fit['rank']}/{fit['p']} observed={observed:.6f} closure={closure:.3g}")
    print(f"baseline_max_abs_diffs={diffs}; publisher_bootstrap={bootstrap_diagnostics['successful_replicates']}/{draws} clusters={bootstrap_diagnostics['publisher_clusters']} failures={bootstrap_diagnostics['failure_reasons']}")
    print(ratio_table[["component", "descriptive_point_share_pct", "fieller_shape", "denominator_ci_crosses_zero", "ratio_stably_identified"]].to_string(index=False))
    print(f"outputs={len(out_files)+1} dir={output_dir}")


if __name__ == "__main__":
    main()
