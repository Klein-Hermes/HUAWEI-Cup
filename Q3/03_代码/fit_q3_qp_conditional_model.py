#!/usr/bin/env python3
"""Fit source-specific Q responses and solve Q3 conditional budget scenarios.

The Q-score identity bridge and B1-anchored fits are explicitly conditional
transfer stress tests. B8 extrapolated rows are fitted for diagnosis only and
never used to optimize on the B1 N/D support grid.

Run from the repository root:
    python Q3/03_代码/fit_q3_qp_conditional_model.py --mode p1
    python Q3/03_代码/fit_q3_qp_conditional_model.py --mode full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares, minimize_scalar


ROOT = Path(__file__).resolve().parents[2]
ATT = ROOT / "中文题目" / "F题" / "real_attachments" / "B_scaling_laws"
OUT = ROOT / "Q3" / "04_结果" / "Qp联合模型敏感性_20260926"
P1_OUT = OUT / "p1"

Q2_B1_GRID = ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "b1_fitted_predictions.csv"
Q2_PARAMS = ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "fit_parameters.json"
Q2_BOOT = ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "cluster_bootstrap_estimates.csv"
B6 = ATT / "supplementary_NQ_experiment.csv"
B8 = ATT / "supplementary_NQ_experiment_large.csv"
Q1_P_REPORT = ROOT / "Q1" / "03_结果" / "Q1.3" / "v1" / "q1_3_run_report.md"
Q1_HOLDOUT = ROOT / "Q1" / "03_结果" / "Q1.3" / "v1" / "holdout_metrics.csv"
Q1_FREEZE = ROOT / "Q1" / "03_结果" / "Q1_最终收口" / "q1_final_freeze.json"
Q1_DOMAIN_QUALITY = ROOT / "Q1" / "03_结果" / "Q1_最终收口" / "q1_final_domain_quality.csv"
C7_METADATA = ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution" / "model_architecture_metadata.csv"

Q0 = 0.4984781048
BUDGETS = (10**19, 10**22, 10**24)
CONTEXTS = (2048, 4096, 8192, 32768, 131072)
ETA = 2e-4
KAPPA_BOUNDS = (-4.0, 4.0)
Q_BISECTION_ITERS = 80
KAPPA_STARTS = (-2.0, -0.5, -0.05, 0.05, 0.5, 2.0)
E_FRACTIONS = (0.05, 0.35, 0.70)
EXPONENT_STARTS = (0.08, 0.25, 0.55)
SEED = 20260926
BOOTSTRAPS = 1000
SOURCES = ("B6", "B8_calibrated", "B8_extrapolated")
OPTIMIZATION_SOURCES = ("B6", "B8_calibrated")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_interfaces() -> dict[str, Any]:
    """Validate Q0 and C7 scenario levels against their frozen source inputs."""
    freeze = json.loads(Q1_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_WITH_LIMITATIONS":
        raise ValueError(f"Unexpected Q1 freeze status: {freeze.get('status')!r}")
    if freeze.get("operational_q") != "q_huber" or freeze.get("q_version") != "Q1-q_huber-v1":
        raise ValueError("Q3 requires the frozen Q1 operational interface Q1-q_huber-v1")

    domains = pd.read_csv(Q1_DOMAIN_QUALITY)
    a1 = domains.loc[domains["dataset"].astype(str).str.lower().eq("a1")].copy()
    if len(a1) != 7 or set(a1["q_version"].astype(str)) != {"Q1-q_huber-v1"}:
        raise ValueError("Expected seven A1 domain values at frozen version Q1-q_huber-v1")
    calculated_q0 = float(a1["q_operational"].astype(float).mean())
    if not math.isclose(calculated_q0, Q0, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"Hard-coded Q0 {Q0:.12g} does not match frozen A1 mean {calculated_q0:.12g}")

    c7 = pd.read_csv(C7_METADATA)
    if not {"model_name", "max_position_embeddings"}.issubset(c7.columns):
        raise ValueError("C7 architecture metadata lacks required context columns")
    lengths = pd.to_numeric(c7["max_position_embeddings"], errors="raise")
    if lengths.isna().any() or (lengths <= 0).any() or not np.all(lengths == lengths.astype(int)):
        raise ValueError("C7 context lengths must be positive integers")
    actual_contexts = tuple(int(value) for value in sorted(lengths.astype(int).unique()))
    if actual_contexts != CONTEXTS:
        raise ValueError(f"Frozen C7 contexts {CONTEXTS} do not match input metadata {actual_contexts}")
    if c7["model_name"].isna().any() or c7["model_name"].duplicated().any():
        raise ValueError("C7 model names must be present and unique")

    inputs = [Q1_FREEZE, Q1_DOMAIN_QUALITY, C7_METADATA]
    return {
        "q1_status": freeze["status"], "q1_operational_q": freeze["operational_q"],
        "q1_q_version": freeze["q_version"], "q0_from_a1_domain_mean": calculated_q0,
        "a1_domain_rows": int(len(a1)), "c7_metadata_rows": int(len(c7)),
        "c7_contexts": list(actual_contexts),
        "inputs": {path.resolve().relative_to(ROOT).as_posix(): sha256(path) for path in inputs},
    }


def source_frames() -> dict[str, pd.DataFrame]:
    b6 = pd.read_csv(B6)
    b8 = pd.read_csv(B8)
    frames = {
        "B6": b6.copy(),
        "B8_calibrated": b8.loc[b8["data_type"].eq("calibrated")].copy(),
        "B8_extrapolated": b8.loc[b8["data_type"].eq("extrapolated")].copy(),
    }
    required = ["N_params_B", "D_tokens_B", "Q_score", "val_loss"]
    for name, frame in frames.items():
        if frame.empty or frame[required].isna().any().any():
            raise ValueError(f"{name}: empty source or missing required Q response fields")
        for col in required:
            frame[col] = frame[col].astype(float)
        if (frame[["N_params_B", "D_tokens_B", "Q_score", "val_loss"]] <= 0).any().any():
            raise ValueError(f"{name}: nonpositive model input, Q, or Loss")
    return frames


def load_b1() -> tuple[pd.DataFrame, dict[str, float]]:
    frame = pd.read_csv(Q2_B1_GRID)
    param_file = json.loads(Q2_PARAMS.read_text(encoding="utf-8"))
    if len(frame) != 1176:
        raise ValueError(f"Expected 1,176 B1 points; got {len(frame)}")
    params = {key: float(value) for key, value in param_file["parameters"].items()}
    pred = (
        params["E"] + params["A"] * frame["N_params_B"].to_numpy(float) ** (-params["alpha"])
        + params["B"] * frame["D_tokens_B"].to_numpy(float) ** (-params["beta"])
    )
    if np.max(np.abs(pred - frame["predicted_val_loss"].to_numpy(float))) > 1e-8:
        raise ValueError("B1 frozen M0 coefficients do not reproduce the supplied prediction table")
    return frame, params


def predict_full(theta: np.ndarray, n: np.ndarray, d: np.ndarray, q: np.ndarray,
                 q_ref: float) -> np.ndarray:
    e, log_a, alpha, log_b, beta, kappa = theta
    return e + np.exp(log_a) * n ** (-alpha) + np.exp(log_b) * d ** (-beta) * (q / q_ref) ** (-kappa)


def full_start_vectors(frame: pd.DataFrame) -> list[np.ndarray]:
    y = frame["val_loss"].to_numpy(float)
    n = frame["N_params_B"].to_numpy(float)
    d = frame["D_tokens_B"].to_numpy(float)
    upper_e = float(y.min()) - 1e-7
    if upper_e <= 0:
        raise ValueError("Model requires positive losses above zero for the E constraint")
    starts: list[np.ndarray] = []
    for e_fraction in E_FRACTIONS:
        e0 = min(max(upper_e * e_fraction, 0.0), upper_e - 1e-8)
        residual = max(float(np.median(y)) - e0, 1e-5)
        for exponent in EXPONENT_STARTS:
            a0 = math.log(max(1e-12, 0.5 * residual * float(np.median(n)) ** exponent))
            b0 = math.log(max(1e-12, 0.5 * residual * float(np.median(d)) ** exponent))
            for kappa in KAPPA_STARTS:
                starts.append(np.array([e0, a0, exponent, b0, exponent, kappa], dtype=float))
    if len(starts) != 54:
        raise AssertionError("The frozen source fit contract requires 54 deterministic starts")
    return starts


def fit_full(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    n = frame["N_params_B"].to_numpy(float)
    d = frame["D_tokens_B"].to_numpy(float)
    q = frame["Q_score"].to_numpy(float)
    y = frame["val_loss"].to_numpy(float)
    q_ref = float(np.median(q))
    upper_e = float(y.min()) - 1e-7
    lower = np.array([0.0, -30.0, 1e-4, -30.0, 1e-4, KAPPA_BOUNDS[0]])
    upper = np.array([upper_e, 30.0, 2.0, 30.0, 2.0, KAPPA_BOUNDS[1]])
    results = []
    for start in full_start_vectors(frame):
        fit = least_squares(
            lambda t: predict_full(t, n, d, q, q_ref) - y,
            start,
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=3000,
        )
        if fit.success and np.isfinite(fit.x).all() and np.isfinite(fit.cost):
            results.append(fit)
    if not results:
        raise RuntimeError("No deterministic multistart source model fit converged")
    best = min(results, key=lambda fit: fit.cost)
    near = [fit for fit in results if fit.cost <= best.cost * (1 + 1e-5) + 1e-15]
    theta = best.x
    spread = np.ptp(np.vstack([fit.x for fit in near]), axis=0) / np.maximum(np.abs(theta), 1e-8)
    prediction = predict_full(theta, n, d, q, q_ref)
    rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
    return theta, {
        "Q_ref": q_ref,
        "attempts": 54,
        "successful": len(results),
        "near_optimal": len(near),
        "near_optimal_relative_sse_tolerance": 1e-5,
        "max_relative_parameter_spread": float(np.max(spread)),
        "relative_parameter_spread": spread.tolist(),
        "boundary_kappa": bool(abs(theta[5] - KAPPA_BOUNDS[0]) < 1e-4 or abs(theta[5] - KAPPA_BOUNDS[1]) < 1e-4),
        "sse": float(np.sum((prediction - y) ** 2)),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(prediction - y))),
        "r2": float(1 - np.sum((prediction - y) ** 2) / np.sum((y - y.mean()) ** 2)),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
    }


def predict_b1_anchored(n: np.ndarray, d: np.ndarray, q: np.ndarray,
                        p: dict[str, float], kappa: float, q0: float = Q0) -> np.ndarray:
    return (p["E"] + p["A"] * n ** (-p["alpha"])
            + p["B"] * d ** (-p["beta"]) * (q / q0) ** (-kappa))


def fit_anchored_kappa(frame: pd.DataFrame, p: dict[str, float],
                       weights: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
    n = frame["N_params_B"].to_numpy(float)
    d = frame["D_tokens_B"].to_numpy(float)
    q = frame["Q_score"].to_numpy(float)
    y = frame["val_loss"].to_numpy(float)
    w = np.ones(len(frame), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(frame) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("Invalid sample weights for B1-anchored fit")

    def objective(kappa: float) -> float:
        residual = predict_b1_anchored(n, d, q, p, float(kappa)) - y
        return float(np.sum(w * residual * residual))

    opt = minimize_scalar(objective, bounds=KAPPA_BOUNDS, method="bounded",
                          options={"xatol": 1e-10, "maxiter": 1000})
    candidates = [(float(opt.x), float(opt.fun)), (KAPPA_BOUNDS[0], objective(KAPPA_BOUNDS[0])),
                  (KAPPA_BOUNDS[1], objective(KAPPA_BOUNDS[1]))]
    kappa, sse = min(candidates, key=lambda item: item[1])
    pred = predict_b1_anchored(n, d, q, p, kappa)
    y_mean = float(np.average(y, weights=w))
    sst = float(np.sum(w * (y - y_mean) ** 2))
    return kappa, {
        "sse": sse,
        "rmse": float(np.sqrt(sse / np.sum(w))),
        "mae": float(np.average(np.abs(pred - y), weights=w)),
        "r2": float(1 - sse / sst) if sst > 0 else float("nan"),
        "boundary_kappa": float(kappa) in KAPPA_BOUNDS,
        "n_rows_weighted": float(np.sum(w)),
    }


def evaluate_fit(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - y
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - sse / sst) if sst > 0 else float("nan"),
        "bias_pred_minus_obs": float(np.mean(residual)),
        "sse": sse, "test_sum_y": float(np.sum(y)),
        "test_sum_y2": float(np.sum(y ** 2)),
    }


def leave_one_q_cv(frame: pd.DataFrame, p: dict[str, float], mode: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    q_values = sorted(frame["Q_score"].unique())
    for q_value in q_values:
        train = frame.loc[frame["Q_score"].ne(q_value)]
        test = frame.loc[frame["Q_score"].eq(q_value)]
        if mode == "full":
            theta, diag = fit_full(train)
            pred = predict_full(theta, test["N_params_B"].to_numpy(float),
                                test["D_tokens_B"].to_numpy(float), test["Q_score"].to_numpy(float),
                                float(diag["Q_ref"]))
            record = {"kappa": float(theta[5]), "fit_rmse_train": float(diag["rmse"])}
        elif mode == "anchored":
            kappa, diag = fit_anchored_kappa(train, p)
            pred = predict_b1_anchored(test["N_params_B"].to_numpy(float),
                                       test["D_tokens_B"].to_numpy(float),
                                       test["Q_score"].to_numpy(float), p, kappa)
            record = {"kappa": kappa, "fit_rmse_train": float(diag["rmse"])}
        else:
            raise ValueError(mode)
        y = test["val_loss"].to_numpy(float)
        record.update({"held_out_Q": float(q_value), "n_test": int(len(test)),
                       **evaluate_fit(y, pred)})
        output.append(record)
    return output


def source_support(frame: pd.DataFrame, b1: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    nmin, nmax = float(frame["N_params_B"].min()), float(frame["N_params_B"].max())
    dmin, dmax = float(frame["D_tokens_B"].min()), float(frame["D_tokens_B"].max())
    mask = (b1["N_params_B"].between(nmin, nmax, inclusive="both")
            & b1["D_tokens_B"].between(dmin, dmax, inclusive="both"))
    metadata = {
        "source_N_min_B": nmin, "source_N_max_B": nmax,
        "source_D_min_B": dmin, "source_D_max_B": dmax,
        "source_Q_min": float(frame["Q_score"].min()), "source_Q_max": float(frame["Q_score"].max()),
        "b1_total_grid_points": int(len(b1)), "b1_support_intersection_points": int(mask.sum()),
    }
    return mask.to_numpy(), metadata


def g_exp(q: float) -> float:
    return 1e7 * math.exp(6 * q)


def g_pow(q: float) -> float:
    return 5e9 * q ** 4


def g_log(q: float) -> float:
    return 2e9 * math.log1p(10 * q)


G_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "exponential": g_exp,
    "power": g_pow,
    "logarithmic": g_log,
}


def q_upper_from_budget(budget: float, base_cost: float, d_abs: float,
                        g_name: str, qmax: float = 1.0) -> float:
    if base_cost > budget:
        return Q0 - 1e-12
    if qmax < Q0:
        raise ValueError(f"Source Q upper bound {qmax} is below the frozen baseline Q0={Q0}")
    g = G_FUNCTIONS[g_name]
    if math.fsum((base_cost, d_abs * max(0.0, g(qmax) - g(Q0)))) <= budget:
        return qmax
    lo, hi = Q0, qmax
    for _ in range(Q_BISECTION_ITERS):
        mid = (lo + hi) / 2
        c_q = d_abs * max(0.0, g(mid) - g(Q0))
        if math.fsum((base_cost, c_q)) <= budget:
            lo = mid
        else:
            hi = mid
    return float(lo)


def cost_components(n_b: float, d_b: float, q: float, context: int,
                    g_name: str) -> tuple[float, float, float]:
    n_abs = n_b * 1e9
    d_abs = d_b * 1e9
    c_train = 6.0 * n_abs * d_abs
    c_attn = ETA * n_abs * d_abs * context
    g = G_FUNCTIONS[g_name]
    c_q = d_abs * max(0.0, g(q) - g(Q0))
    return c_train, c_q, c_attn


def solve_scenario(b1: pd.DataFrame, support_mask: np.ndarray, p: dict[str, float],
                   kappa: float, budget: int, context: int, g_name: str,
                   qmax_source: float = 1.0) -> dict[str, Any]:
    n_all = b1["N_params_B"].to_numpy(float)
    d_all = b1["D_tokens_B"].to_numpy(float)
    rows = b1.loc[support_mask]
    base_candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row in rows.itertuples(index=False):
        base_cost = (6 + ETA * context) * (row.N_params_B * 1e9) * (row.D_tokens_B * 1e9)
        if base_cost > budget:
            continue
        q_upper = q_upper_from_budget(budget, base_cost, row.D_tokens_B * 1e9, g_name, qmax_source)
        q_star = q_upper if kappa > 1e-10 else Q0
        q_star = min(q_star, qmax_source)
        c_train, c_q, c_attn = cost_components(row.N_params_B, row.D_tokens_B, q_star, context, g_name)
        total = math.fsum((c_train, c_q, c_attn))
        # Inverse g(Q) can round the affordable boundary upward by one ULP.
        # Step Q toward its feasible lower bound until the reported component
        # sum satisfies the strict budget cap; never emit negative budget slack.
        while total > budget and q_star > Q0:
            q_star = float(np.nextafter(q_star, Q0))
            c_train, c_q, c_attn = cost_components(row.N_params_B, row.D_tokens_B, q_star, context, g_name)
            total = math.fsum((c_train, c_q, c_attn))
        if total > budget:
            continue
        prediction = float(p["E"] + p["A"] * row.N_params_B ** (-p["alpha"])
                           + p["B"] * row.D_tokens_B ** (-p["beta"])
                           * (q_star / Q0) ** (-kappa))
        if total > budget:
            raise ValueError(f"Budget constraint violated by {total / budget:.12g}")
        result = {
            "status": "feasible",
            "budget_flops": int(budget), "context_length_tokens": int(context),
            "g_function": g_name, "kappa_B1_anchored": float(kappa),
            "selected_run_id": row.run_id,
            "selected_N_B": float(row.N_params_B), "selected_D_B": float(row.D_tokens_B),
            "selected_Q": float(q_star), "predicted_loss": prediction,
            "C_train_flops": c_train, "C_Q_flops": c_q, "C_attn_flops": c_attn,
            "C_total_flops": total, "budget_utilization": total / budget,
            "budget_slack_flops": budget - total,
            "Q_upper_affordable": float(q_upper),
            "selected_at_Q_boundary": abs(q_star - q_upper) <= 1e-10 if kappa > 1e-10 else abs(q_star - Q0) <= 1e-10,
            "p_setting": "B1 common mixture policy fixed; numeric 17-domain vector unavailable; p effect absorbed in M0 baseline",
            "Q_mapping_assumption": "Q2 Q_score numerically identical to Q1 q_huber; unvalidated conditional scenario",
        }
        key = (prediction, total, float(row.N_params_B), float(row.D_tokens_B), str(row.run_id), q_star)
        base_candidates.append((key, result))
    if not base_candidates:
        return {
            "status": "infeasible", "budget_flops": int(budget),
            "context_length_tokens": int(context), "g_function": g_name,
            "kappa_B1_anchored": float(kappa), "selected_run_id": "",
            "selected_N_B": math.nan, "selected_D_B": math.nan, "selected_Q": math.nan,
            "predicted_loss": math.nan, "C_train_flops": math.nan, "C_Q_flops": math.nan,
            "C_attn_flops": math.nan, "C_total_flops": math.nan,
            "budget_utilization": math.nan, "budget_slack_flops": math.nan,
            "Q_upper_affordable": math.nan, "selected_at_Q_boundary": False,
            "p_setting": "B1 common mixture policy fixed; numeric 17-domain vector unavailable",
            "Q_mapping_assumption": "Q2 Q_score numerically identical to Q1 q_huber; unvalidated conditional scenario",
        }
    return min(base_candidates, key=lambda item: item[0])[1]


def build_edge_list() -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    # Scenario keys use (budget, context); ordering makes the grid geometry explicit.
    edges: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for context in CONTEXTS:
        for left, right in zip(BUDGETS[:-1], BUDGETS[1:]):
            edges.append(("adjacent_budget", (left, context), (right, context)))
    for budget in BUDGETS:
        for left, right in zip(CONTEXTS[:-1], CONTEXTS[1:]):
            edges.append(("adjacent_context", (budget, left), (budget, right)))
    if len(edges) != 22:
        raise AssertionError("Expected the 3 by 5 scenario grid to have 22 adjacency edges")
    return edges


def shares(row: dict[str, Any]) -> np.ndarray:
    total = float(row["C_total_flops"])
    return np.array([row["C_train_flops"], row["C_Q_flops"], row["C_attn_flops"]], dtype=float) / total


def transition_rows(scenario_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    lookup = {(r["source"], r["g_function"], r["budget_flops"], r["context_length_tokens"]): r
              for r in scenario_rows}
    for source in OPTIMIZATION_SOURCES:
        for g_name in G_FUNCTIONS:
            for ix, (kind, left_key, right_key) in enumerate(build_edge_list(), start=1):
                left = lookup[(source, g_name, left_key[0], left_key[1])]
                right = lookup[(source, g_name, right_key[0], right_key[1])]
                both = left["status"] == "feasible" and right["status"] == "feasible"
                base = {
                    "source": source, "g_function": g_name,
                    "edge_id": f"T{ix:02d}", "edge_kind": kind,
                    "left_budget": left_key[0], "left_context": left_key[1],
                    "right_budget": right_key[0], "right_context": right_key[1],
                    "left_status": left["status"], "right_status": right["status"],
                }
                if not both:
                    output.append({**base, "edge_class": "feasibility_boundary",
                                   "configuration_changed": "", "cost_share_tv": math.nan,
                                   "delta_log_N": math.nan, "delta_log_D": math.nan,
                                   "delta_Q": math.nan,
                                   "interpretation": "feasible-domain entry/exit; no numeric configuration difference"})
                    continue
                lshare, rshare = shares(left), shares(right)
                tv = float(0.5 * np.sum(np.abs(rshare - lshare)))
                dlogn = math.log(right["selected_N_B"] / left["selected_N_B"])
                dlogd = math.log(right["selected_D_B"] / left["selected_D_B"])
                dq = float(right["selected_Q"] - left["selected_Q"])
                changed = (left["selected_run_id"] != right["selected_run_id"]
                           or abs(dq) > 1e-10)
                output.append({**base, "edge_class": "both_feasible",
                               "configuration_changed": bool(changed), "cost_share_tv": tv,
                               "delta_log_N": dlogn, "delta_log_D": dlogd, "delta_Q": dq,
                               "interpretation": "discrete conditional solution comparison"})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def q0_support_rows(b1: pd.DataFrame, supports: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in OPTIMIZATION_SOURCES:
        mask = supports[source]
        for context in CONTEXTS:
            costs = (6 + ETA * context) * (b1.loc[mask, "N_params_B"].to_numpy(float) * 1e9) * (
                b1.loc[mask, "D_tokens_B"].to_numpy(float) * 1e9)
            for budget in BUDGETS:
                rows.append({"source": source, "budget_flops": budget,
                             "context_length_tokens": context,
                             "support_grid_points": int(mask.sum()),
                             "feasible_at_Q0": int(np.sum(costs <= budget)),
                             "status_at_Q0": "feasible" if np.any(costs <= budget) else "infeasible"})
    return rows


def bootstrap_anchored_kappas(frames: dict[str, pd.DataFrame], p: dict[str, float],
                              mode: str) -> dict[str, np.ndarray]:
    if mode != "full":
        return {}
    rng = np.random.default_rng(SEED)
    out: dict[str, np.ndarray] = {}
    for source in OPTIMIZATION_SOURCES:
        frame = frames[source].reset_index(drop=True)
        group_codes, group_values = pd.factorize(list(zip(frame["N_params_B"], frame["D_tokens_B"])))
        n_groups = len(group_values)
        samples = np.empty(BOOTSTRAPS, dtype=float)
        for b in range(BOOTSTRAPS):
            selected = rng.integers(0, n_groups, size=n_groups)
            counts = np.bincount(selected, minlength=n_groups)
            weights = counts[group_codes].astype(float)
            samples[b] = fit_anchored_kappa(frame, p, weights)[0]
        out[source] = samples
    return out


def optimize_all(b1: pd.DataFrame, p: dict[str, float], supports: dict[str, np.ndarray],
                 kappas: dict[str, float], qmax_sources: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in OPTIMIZATION_SOURCES:
        for g_name in G_FUNCTIONS:
            for context in CONTEXTS:
                for budget in BUDGETS:
                    result = solve_scenario(b1, supports[source], p, kappas[source],
                                            budget, context, g_name, qmax_sources[source])
                    rows.append({"source": source, **result})
    return rows


def bootstrap_transition_summary(
    b1: pd.DataFrame,
    p_boot: pd.DataFrame,
    kappas_boot: dict[str, np.ndarray],
    supports: dict[str, np.ndarray],
    qmax_sources: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not kappas_boot:
        return [], []
    rng = np.random.default_rng(SEED + 1)
    b1_draws = p_boot.iloc[rng.integers(0, len(p_boot), size=BOOTSTRAPS)].reset_index(drop=True)
    selection: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    edge_defs = build_edge_list()
    for source in OPTIMIZATION_SOURCES:
        source_kappas = kappas_boot[source]
        by_scenario: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for g_name in G_FUNCTIONS:
            for context in CONTEXTS:
                for budget in BUDGETS:
                    by_scenario[(g_name, budget, context)] = []
        for draw in range(BOOTSTRAPS):
            theta = {key: float(b1_draws.iloc[draw][key])
                     for key in ("E", "A", "alpha", "B", "beta")}
            for g_name in G_FUNCTIONS:
                for context in CONTEXTS:
                    for budget in BUDGETS:
                        result = solve_scenario(b1, supports[source], theta,
                                                float(source_kappas[draw]), budget,
                                                context, g_name, qmax_sources[source])
                        by_scenario[(g_name, budget, context)].append(result)
        # Compact per-scenario uncertainty and paired transition frequencies.
        for (g_name, budget, context), rows in by_scenario.items():
            feasible = [r for r in rows if r["status"] == "feasible"]
            if not feasible:
                selection.append({"source": source, "g_function": g_name,
                                  "budget_flops": budget, "context_length_tokens": context,
                                  "feasible_frequency": 0.0, "top_configuration_frequency": 0.0,
                                  "top_run_id": "", "Q_median": math.nan,
                                  "Q_p025": math.nan, "Q_p975": math.nan,
                                  "Loss_median": math.nan})
            else:
                counts = pd.Series([r["selected_run_id"] for r in feasible]).value_counts()
                top_run = str(counts.index[0])
                selection.append({"source": source, "g_function": g_name,
                                  "budget_flops": budget, "context_length_tokens": context,
                                  "feasible_frequency": len(feasible) / BOOTSTRAPS,
                                  "top_configuration_frequency": float(counts.iloc[0] / BOOTSTRAPS),
                                  "top_run_id": top_run,
                                  "Q_median": float(np.median([r["selected_Q"] for r in feasible])),
                                  "Q_p025": float(np.quantile([r["selected_Q"] for r in feasible], 0.025)),
                                  "Q_p975": float(np.quantile([r["selected_Q"] for r in feasible], 0.975)),
                                  "Loss_median": float(np.median([r["predicted_loss"] for r in feasible]))})
        for g_name in G_FUNCTIONS:
            for ix, (kind, left_key, right_key) in enumerate(edge_defs, start=1):
                left_rows = by_scenario[(g_name, left_key[0], left_key[1])]
                right_rows = by_scenario[(g_name, right_key[0], right_key[1])]
                both = [(a, b) for a, b in zip(left_rows, right_rows)
                        if a["status"] == "feasible" and b["status"] == "feasible"]
                boundary_count = BOOTSTRAPS - len(both)
                if both:
                    changed = []
                    share_tv = []
                    q_delta = []
                    for a, b in both:
                        changed.append(a["selected_run_id"] != b["selected_run_id"]
                                       or abs(a["selected_Q"] - b["selected_Q"]) > 1e-10)
                        share_tv.append(0.5 * float(np.sum(np.abs(shares(a) - shares(b)))))
                        q_delta.append(b["selected_Q"] - a["selected_Q"])
                    summary.append({
                        "source": source, "g_function": g_name, "edge_id": f"T{ix:02d}",
                        "edge_kind": kind, "bootstrap_replicates": BOOTSTRAPS,
                        "both_feasible_replicates": len(both),
                        "boundary_replicates": boundary_count,
                        "configuration_change_frequency": float(np.mean(changed)),
                        "configuration_change_95pct": bool(np.mean(changed) >= 0.95),
                        "cost_share_tv_p05": float(np.quantile(share_tv, 0.05)),
                        "cost_share_change_frequency": float(np.mean(np.asarray(share_tv) > 1e-12)),
                        "delta_Q_median": float(np.median(q_delta)),
                    })
                else:
                    summary.append({
                        "source": source, "g_function": g_name, "edge_id": f"T{ix:02d}",
                        "edge_kind": kind, "bootstrap_replicates": BOOTSTRAPS,
                        "both_feasible_replicates": 0, "boundary_replicates": boundary_count,
                        "configuration_change_frequency": math.nan,
                        "configuration_change_95pct": "", "cost_share_tv_p05": math.nan,
                        "cost_share_change_frequency": math.nan, "delta_Q_median": math.nan,
                    })
    return selection, summary


def source_metrics_rows(frames: dict[str, pd.DataFrame],
                        thetas: dict[str, np.ndarray],
                        full_diags: dict[str, dict[str, Any]],
                        anchored: dict[str, tuple[float, dict[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        frame = frames[source]
        theta, diag = thetas[source], full_diags[source]
        kappa, adiag = anchored[source]
        rows.append({
            "source": source, "n": len(frame),
            "n_ND_cells": int(frame.groupby(["N_params_B", "D_tokens_B"]).ngroups),
            "n_Q_levels": int(frame["Q_score"].nunique()),
            "N_min_B": float(frame["N_params_B"].min()), "N_max_B": float(frame["N_params_B"].max()),
            "D_min_B": float(frame["D_tokens_B"].min()), "D_max_B": float(frame["D_tokens_B"].max()),
            "Q_min": float(frame["Q_score"].min()), "Q_max": float(frame["Q_score"].max()),
            "Q_ref_median": float(diag["Q_ref"]),
            "E": float(theta[0]), "A": float(np.exp(theta[1])), "alpha": float(theta[2]),
            "B": float(np.exp(theta[3])), "beta": float(theta[4]), "kappa_full": float(theta[5]),
            "full_in_sample_rmse": diag["rmse"], "full_in_sample_mae": diag["mae"],
            "full_in_sample_r2": diag["r2"], "full_54_starts_successful": diag["successful"],
            "full_54_starts_near_optimal": diag["near_optimal"],
            "full_max_relative_parameter_spread": diag["max_relative_parameter_spread"],
            "full_kappa_at_bound": diag["boundary_kappa"],
            "B1_anchored_kappa_identity_map": kappa,
            "B1_anchored_rmse_identity_map": adiag["rmse"],
            "B1_anchored_mae_identity_map": adiag["mae"],
            "B1_anchored_r2_identity_map": adiag["r2"],
            "B1_anchored_kappa_at_bound": adiag["boundary_kappa"],
        })
    return rows


def write_report(
    mode: str,
    frames: dict[str, pd.DataFrame],
    full_rows: list[dict[str, Any]],
    anchored_rows: list[dict[str, Any]],
    cv_rows: list[dict[str, Any]],
    support_meta: dict[str, dict[str, Any]],
    feasibility_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    transition_rows_out: list[dict[str, Any]],
    transition_boot: list[dict[str, Any]],
    kappa_boot: dict[str, np.ndarray],
) -> str:
    p_gate_text = (
        "Q1.3 的 A4/A5 `simplex_ridge` p-only 模型已做 A 侧估计：同尺度 1M held-out R²=0.6189，"
        "跨尺度 60M R²=-7.9336、1B R²=-810.1196；因此没有把 A 系数移植到 B1 标量 Loss。"
    )
    lines = [
        "# Q3：Q 响应估计与固定配比预算情景",
        "",
        f"**模式：** {mode}；生成时间 UTC `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`。",
        "**范围：** 按题目要求实际估计 Q 响应并运行固定 p 的预算情景；B6/B8 条件估计不改称真实 B1 Q 效应。B 侧 17 维 p 效应没有可审计联结数据，仍未估计。",
        "",
        "## 数据与参数估计",
        "",
        "分来源拟合完整 `M_Q_NDQ`：`L_s=E_s+A_s N^{-alpha_s}+B_s D^{-beta_s}(Q_s/Q_ref,s)^{-kappa_s}`。各来源 54 组确定多初值、非线性最小二乘拟合；`E≥0`、`A,B>0`、`alpha,beta∈[1e-4,2]`、`kappa∈[-4,4]`。留一 Q 水平验证按来源单独汇总。",
        "",
        "| 来源 | n | N/D 单元 | Q 水平 | κ 全曲面 | LOQ RMSE | LOQ R² | κ B1 锚定 | 锚定 RMSE | 锚定 R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cv_by_source = {name: [r for r in cv_rows if r["source"] == name and r["model"] == "full"]
                    for name in SOURCES}
    anchored_by_source = {r["source"]: r for r in anchored_rows}
    full_by_source = {r["source"]: r for r in full_rows}
    for source in SOURCES:
        diag = full_by_source[source]
        acv = [r for r in cv_rows if r["source"] == source and r["model"] == "anchored"]
        def pooled_cv(rows: list[dict[str, Any]]) -> tuple[float, float]:
            if not rows:
                return math.nan, math.nan
            n = sum(int(row["n_test"]) for row in rows)
            sse = sum(float(row["sse"]) for row in rows)
            sum_y = sum(float(row["test_sum_y"]) for row in rows)
            sum_y2 = sum(float(row["test_sum_y2"]) for row in rows)
            sst = sum_y2 - sum_y * sum_y / n
            return float(np.sqrt(sse / n)), float(1 - sse / sst) if sst > 0 else math.nan
        loq_rmse, loq_r2 = pooled_cv(cv_by_source[source])
        aloq_rmse, aloq_r2 = pooled_cv(acv)
        a = anchored_by_source[source]
        lines.append(
            f"| {source} | {diag['n']} | {diag['n_ND_cells']} | {diag['n_Q_levels']} | "
            f"{diag['kappa']:.6g} | {loq_rmse:.6g} | {loq_r2:.5f} | "
            f"{a['kappa_B1_anchored']:.6g} | {a['rmse']:.6g} | {a['r2']:.5f} |"
        )
    lines.extend([
        "",
        "B1 锚定列固定五个 M0 系数，只估 κ，并假设 B 侧 `Q_score` 与 Q1 `q_huber` 数值相同。该映射未经验证；锚定误差、κ 与预算结果均是迁移压力测试。完整参数、所有留一 Q 水平指标、多初值诊断分别存入 CSV。B8 extrapolated 的 N 范围从 20B 起，与 B1 网格没有交集，只呈现来源拟合，不用于 B1 优化。",
        "",
        "## p 估计与迁移",
        "",
        p_gate_text,
        "Q3 按题面固定配比支路把 B1 的共同训练政策记为 `p0`。其数值 17 维组成未知，B1 M0 只能解释为给定该政策的基线，故结果不报告数值 p 最优。",
        "",
        "## 支持域与预算可行性",
        "",
        "按每个 Q 来源的 N/D 观测范围与 B1 1,176 点网格取交集；质量成本在 Q=Q0 时为零。15 个情景候选计数见 `scenario_support_feasibility.csv`。最低预算、最长上下文没有可行点，故该节点的两条相邻边不计算 N/D 或成本份额差，而另标为可行域边界变化。",
        "",
        "| 来源 | B1 点×来源 N/D 支持交集 | 预算 1e19 各上下文可行数（2048,4096,8192,32768,131072） |",
        "|---|---:|---|",
    ])
    for source in OPTIMIZATION_SOURCES:
        counts = [r["feasible_at_Q0"] for r in feasibility_rows
                  if r["source"] == source and r["budget_flops"] == 10**19]
        lines.append(f"| {source} | {support_meta[source]['b1_support_intersection_points']} | {counts} |")
    lines.extend([
        "",
        "## 条件预算结果与结构转移",
        "",
        "预算使用原题三档 FLOPs，C7 五种上下文与三种 `g(Q)`。N/D 只从来源支持域与 B1 实测网格交集中选；每个可行 `(N,D)` 用单调成本函数二分法求 Q 可负担上界，再取 `[Q0,1]` 内的该上界（κ>0）或 Q0（κ≤0）。二分迭代固定为 80 次，并在成本分项求和后再次执行严格预算边界校正。这求的是显式 Q 尺度假设下的条件最优。",
        "",
        "逐情景结果见 `conditional_scenario_results.csv`；相邻配置与成本份额差见 `conditional_transition_pairs.csv`。每个来源/成本模型下 20 条双端可行边独立判定，2 条边界边只记录可行性进出。Bootstrap 复本采用 B1 轨迹参数复本与 B6/B8 calibrated 固定 (N,D) 单元重抽样的 κ 复本独立配对；它表示条件模型参数稳定性，不是 Q 分数或 B 侧真实训练效果的抽样区间。",
        "",
        "| 来源 | g(Q) | 双端可行边 | N/D/Q 改变频率≥95%的边 | 成本份额改变频率≥95%的边 |",
        "|---|---|---:|---:|---:|",
    ])
    for source in OPTIMIZATION_SOURCES:
        for g_name in G_FUNCTIONS:
            rows = [r for r in transition_boot if r["source"] == source and r["g_function"] == g_name
                    and r["both_feasible_replicates"] > 0]
            both = len(rows)
            cfg = sum(bool(r["configuration_change_95pct"]) for r in rows)
            shares_changed = sum(r["cost_share_change_frequency"] >= 0.95 for r in rows)
            lines.append(f"| {source} | {g_name} | {both} | {cfg} | {shares_changed} |")
    lines.extend([
        "",
        "分支间 κ 符号、最优 Q 和资源转移差异按来源/模型敏感性解释，不合并为单一经验结论。仅当来源 Q 量尺与 Q1 主 Q 的数值身份映射成立时，预算结果才对应该 Q1 成本标尺；目前没有独立证据验证此映射。",
        "",
        "## 解释边界",
        "",
        "1. 题面 Q/p 的参数估计要求已实际尝试：A 侧 p 效应有来源内估计和跨尺度验证；B6/B8 有来源内 Q 曲面估计与留出 Q 水平预测。",
        "2. B6 是半合成校准，B8 calibrated/extrapolated 是分来源校准/扩展数据；这些分支不能取代 B1 上独立可比的 Q/p 训练实验。",
        "3. B 侧 p 效应没有估计，因为 B1–B5 中与 Loss 样本行核验连接的 17 维配比向量为 0；固定政策的向量组成也未恢复。",
        "4. B8 extrapolated 与 B1 参数量没有支持重叠，故不进 B1 网格优化；高预算解仍可能受 B1 网格上界限制。",
        "5. 不能据此声称已识别统一经验 `Loss=f(N,D,p,Q)`，但不能因此跳过已完成的来源内参数拟合和条件预算比较。",
        "",
        "## 复现",
        "",
        "从项目根目录运行 `python Q3/03_代码/fit_q3_qp_conditional_model.py --mode full`。",
        "",
    ])
    return "\n".join(lines)


def run(mode: str, output_dir: Path | None = None) -> None:
    outdir = output_dir if output_dir is not None else (P1_OUT if mode == "p1" else OUT)
    outdir = outdir.resolve()
    try:
        outdir.relative_to(OUT.resolve())
    except ValueError as exc:
        raise ValueError(f"Output directory must stay under {OUT}") from exc
    outdir.mkdir(parents=True, exist_ok=True)
    interface_verification = verify_frozen_interfaces()
    frames = source_frames()
    b1, m0 = load_b1()
    thetas: dict[str, np.ndarray] = {}
    full_diags: dict[str, dict[str, Any]] = {}
    anchored: dict[str, tuple[float, dict[str, float]]] = {}
    full_rows: list[dict[str, Any]] = []
    anchored_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    supports: dict[str, np.ndarray] = {}
    support_meta: dict[str, dict[str, Any]] = {}

    for source in SOURCES:
        theta, diag = fit_full(frames[source])
        thetas[source], full_diags[source] = theta, diag
        n, d, q = (frames[source][col].to_numpy(float) for col in ("N_params_B", "D_tokens_B", "Q_score"))
        anchored_kappa, anchored_diag = fit_anchored_kappa(frames[source], m0)
        anchored[source] = (anchored_kappa, anchored_diag)
        full_rows.append({
            "source": source, "n": len(frames[source]),
            "n_ND_cells": int(frames[source].groupby(["N_params_B", "D_tokens_B"]).ngroups),
            "n_Q_levels": int(frames[source]["Q_score"].nunique()),
            "N_min_B": float(n.min()), "N_max_B": float(n.max()),
            "D_min_B": float(d.min()), "D_max_B": float(d.max()),
            "Q_min": float(q.min()), "Q_max": float(q.max()),
            "Q_ref": diag["Q_ref"], "E": theta[0], "A": np.exp(theta[1]),
            "alpha": theta[2], "B": np.exp(theta[3]), "beta": theta[4], "kappa": theta[5],
            "in_sample_rmse": diag["rmse"], "in_sample_mae": diag["mae"], "in_sample_r2": diag["r2"],
            "starts_attempted": diag["attempts"], "starts_successful": diag["successful"],
            "near_optimal_starts": diag["near_optimal"],
            "max_relative_parameter_spread": diag["max_relative_parameter_spread"],
            "kappa_at_bound": diag["boundary_kappa"],
            "interpretation": "source-specific Q-response fit; not a validated B1 transfer",
        })
        anchored_rows.append({
            "source": source, "kappa_B1_anchored": anchored_kappa,
            **anchored_diag,
            "bridge_assumption": "Q_score numerically identical to Q1 q_huber; unvalidated",
            "interpretation": "conditional transfer stress test only",
        })
        if mode == "full":
            for item in leave_one_q_cv(frames[source], m0, "full"):
                cv_rows.append({"source": source, "model": "full", **item})
            for item in leave_one_q_cv(frames[source], m0, "anchored"):
                cv_rows.append({"source": source, "model": "anchored", **item})
        if source in OPTIMIZATION_SOURCES:
            supports[source], support_meta[source] = source_support(frames[source], b1)

    source_estimate_rows = source_metrics_rows(frames, thetas, full_diags, anchored)
    # source_metrics_rows is not used to avoid mixing aggregate CV calculations into this table;
    # full_rows and anchored_rows are the canonical model outputs.
    write_csv(outdir / "q_response_full_fits.csv", full_rows)
    write_csv(outdir / "q_response_b1_anchored_fits.csv", anchored_rows)

    feasibility_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    transition_out: list[dict[str, Any]] = []
    transition_boot: list[dict[str, Any]] = []
    kappa_boot: dict[str, np.ndarray] = {}
    selection_boot: list[dict[str, Any]] = []
    bootstrap_kappa_rows: list[dict[str, Any]] = []
    if mode == "full":
        for source in OPTIMIZATION_SOURCES:
            qmax = float(frames[source]["Q_score"].max())
            for row in q0_support_rows(b1, supports):
                if row["source"] == source:
                    feasibility_rows.append(row)
            for g_name in G_FUNCTIONS:
                for context in CONTEXTS:
                    for budget in BUDGETS:
                        scenario_rows.append({
                            "source": source,
                            **solve_scenario(b1, supports[source], m0, anchored[source][0],
                                            budget, context, g_name, qmax),
                        })
        transition_out = transition_rows(scenario_rows)
        kappa_boot = bootstrap_anchored_kappas(frames, m0, mode)
        for source, values in kappa_boot.items():
            bootstrap_kappa_rows.extend({"source": source, "replicate": i + 1, "kappa": float(v)}
                                        for i, v in enumerate(values))
        p_boot = pd.read_csv(Q2_BOOT)
        selection_boot, transition_boot = bootstrap_transition_summary(
            b1, p_boot, kappa_boot, supports,
            {source: float(frames[source]["Q_score"].max()) for source in OPTIMIZATION_SOURCES},
        )
        write_csv(outdir / "q_response_leave_one_q_validation.csv", cv_rows)
        write_csv(outdir / "scenario_support_feasibility.csv", feasibility_rows)
        write_csv(outdir / "conditional_scenario_results.csv", scenario_rows)
        write_csv(outdir / "conditional_transition_pairs.csv", transition_out)
        write_csv(outdir / "conditional_transition_bootstrap_summary.csv", transition_boot)
        write_csv(outdir / "conditional_selection_bootstrap_summary.csv", selection_boot)
        write_csv(outdir / "q_response_anchored_kappa_bootstrap.csv", bootstrap_kappa_rows)
    else:
        # P1 vertical slice: all source fits + all candidate feasibility counts + one true budget solution per usable family.
        for source in OPTIMIZATION_SOURCES:
            for row in q0_support_rows(b1, supports):
                if row["source"] == source:
                    feasibility_rows.append(row)
        p1_scenarios = []
        for source in OPTIMIZATION_SOURCES:
            result = solve_scenario(b1, supports[source], m0, anchored[source][0],
                                    10**22, 8192, "logarithmic",
                                    float(frames[source]["Q_score"].max()))
            p1_scenarios.append({"source": source, **result})
            if result["status"] != "feasible" or result["C_total_flops"] > 10**22:
                raise ValueError(f"P1 optimization did not return a feasible solution for {source}")
        write_csv(outdir / "q_response_p1_fits.csv", full_rows)
        write_csv(outdir / "q_response_p1_b1_anchored_fits.csv", anchored_rows)
        write_csv(outdir / "p1_scenario_support_feasibility.csv", feasibility_rows)
        write_csv(outdir / "p1_minimal_conditional_solutions.csv", p1_scenarios)

    if mode == "full":
        report = write_report(mode, frames, full_rows, anchored_rows, cv_rows,
                              support_meta, feasibility_rows, scenario_rows,
                              transition_out, transition_boot, kappa_boot)
        (outdir / "q3_qp_conditional_report.md").write_text(report, encoding="utf-8")
        inputs = [Q2_B1_GRID, Q2_PARAMS, Q2_BOOT, B6, B8, Q1_P_REPORT, Q1_HOLDOUT,
                  Q1_FREEZE, Q1_DOMAIN_QUALITY, C7_METADATA]
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": "python Q3/03_代码/fit_q3_qp_conditional_model.py --mode full",
            "mode": mode, "seed": SEED, "q0": Q0,
            "budgets_flops": list(BUDGETS), "contexts": list(CONTEXTS),
            "eta": ETA, "kappa_bounds": list(KAPPA_BOUNDS),
            "q_upper_bisection_iterations": Q_BISECTION_ITERS,
            "bootstrap_replicates": BOOTSTRAPS,
            "q_score_identity_bridge": "Q2 Q_score = Q1 q_huber; unverified assumption, conditional scenarios only",
            "frozen_interface_verification": interface_verification,
            "sources": SOURCES,
            "optimization_sources": OPTIMIZATION_SOURCES,
            "script": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                       "sha256": sha256(Path(__file__))},
            "python": sys.version, "platform": platform.platform(),
            "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
            "inputs": {path.resolve().relative_to(ROOT).as_posix(): sha256(path) for path in inputs},
            "outputs": {path.name: sha256(path) for path in sorted(outdir.iterdir())
                        if path.is_file() and path.name != "repro_manifest.json"},
        }
        (outdir / "repro_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        status = {
            "status": "P1_MINIMAL_SLICE_COMPLETE",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": "python Q3/03_代码/fit_q3_qp_conditional_model.py --mode p1",
            "full_model_source_fits": 3,
            "source_support_rows": len(feasibility_rows),
            "minimal_solutions": len(p1_scenarios),
            "q_upper_bisection_iterations": Q_BISECTION_ITERS,
            "solutions": p1_scenarios,
            "script": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                       "sha256": sha256(Path(__file__))},
            "output_sha256": {path.name: sha256(path) for path in sorted(outdir.iterdir())
                              if path.is_file() and path.name != "p1_status.json"},
            "frozen_interface_verification": interface_verification,
            "inputs": {path.resolve().relative_to(ROOT).as_posix(): sha256(path)
                       for path in [Q2_B1_GRID, Q2_PARAMS, B6, B8,
                                    Q1_FREEZE, Q1_DOMAIN_QUALITY, C7_METADATA]},
        }
        (outdir / "p1_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"mode={mode}; full_source_fits={len(full_rows)}; B1-anchored fits={len(anchored_rows)}")
    if mode == "p1":
        print(f"support feasibility rows={len(feasibility_rows)}; minimal solutions={len(p1_scenarios)}")
    else:
        print(f"Q3 scenarios={len(scenario_rows)}; transition edges={len(transition_out)}; bootstrap summaries={len(transition_boot)}")
    for row in (p1_scenarios if mode == "p1" else scenario_rows[:3]):
        print(f"{row['source']}: status={row['status']}; C={row['budget_flops']:.0e}; L={row['context_length_tokens']}; "
              f"N={row['selected_N_B']}; D={row['selected_D_B']}; Q={row['selected_Q']}; loss={row['predicted_loss']}; "
              f"cost={row['C_total_flops']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("p1", "full"), default="p1")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional isolated output directory under Q3/04_结果/Qp联合模型敏感性_20260926")
    args = parser.parse_args()
    run(args.mode, args.output_dir)


if __name__ == "__main__":
    main()
