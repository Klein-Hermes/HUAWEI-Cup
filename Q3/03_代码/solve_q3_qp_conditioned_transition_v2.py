#!/usr/bin/env python3
"""Source-separated Q response fits and Q3 budget-transition scenarios.

The analysis follows Q3_Qp条件联合模型与转移判定方案_20260926.md. It fits
the supplied B6/B8 sources independently, anchors a signed Q exponent to the
frozen B1 M0 surface for B6 and B8 calibrated, and optimizes only on each
source's intersection with the observed B1 N/D grid. These are conditional
cross-source scenarios, not real joint p/Q training effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import analyze_q3_cost_sensitivities as cost_module  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("Q3/04_结果/Qp条件转移敏感性_20260926_v2")
Q2_GRID = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q2_PARAMS = Path("Q2/03_结果/经典ScalingLaw基线/v1/fit_parameters.json")
Q2_B6_ROWS = Path("中文题目/F题/real_attachments/B_scaling_laws/supplementary_NQ_experiment.csv")
Q2_B8_ROWS = Path("中文题目/F题/real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv")
Q3_FRONTIER = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
Q3_PLAN = Path("Q3/01_方案与设计/Q3_Qp条件联合模型与转移判定方案_20260926.md")
Q1_FREEZE = Path("Q1/03_结果/Q1_最终收口/q1_final_freeze.json")
Q1_DOMAIN = Path("Q1/03_结果/Q1_最终收口/q1_final_domain_quality.csv")
Q1_MACRO = Path("Q1/03_结果/Q1.1/v1/a1_macro_summary.csv")
SEED = 260926
BOOTSTRAPS = 1000
TRAINING_COEFFICIENT = 6
ETA = Fraction(1, 5000)
PARAM_BOUNDS = ([-np.inf, -15, 1e-4, -15, 1e-4, -4],
                [np.inf, 15, 2.0, 15, 2.0, 4])


def read_csv(path: Path) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], empty_fields: list[str] | None = None) -> None:
    if not rows and empty_fields is None:
        raise ValueError(f"Refusing to write an empty table: {path}")
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(rows[0]) if rows else empty_fields
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def arrays(rows: list[dict[str, str]], qref: float) -> tuple[np.ndarray, ...]:
    n = np.asarray([float(r["N_params_B"]) for r in rows], dtype=float)
    d = np.asarray([float(r["D_tokens_B"]) for r in rows], dtype=float)
    q = np.asarray([float(r["Q_score"]) for r in rows], dtype=float)
    y = np.asarray([float(r["val_loss"]) for r in rows], dtype=float)
    if np.any(n <= 0) or np.any(d <= 0) or np.any(q <= 0) or np.any(y <= 0):
        raise ValueError("N, D, Q, and Loss must all be positive")
    return n, d, q / qref, y


def predict_full(theta: np.ndarray, n: np.ndarray, d: np.ndarray,
                 q_ratio: np.ndarray) -> np.ndarray:
    e, log_a, alpha, log_b, beta, kappa = theta
    return e + np.exp(log_a) * n ** (-alpha) + np.exp(log_b) * d ** (-beta) * q_ratio ** (-kappa)


def fit_full(rows: list[dict[str, str]], qref: float, starts: int = 54,
             seed_theta: np.ndarray | None = None) -> dict[str, Any]:
    n, d, qr, y = arrays(rows, qref)
    y_min = float(np.min(y))
    e_hi = y_min * (1 - 1e-8)
    lower = np.asarray([0.0, -15, 1e-4, -15, 1e-4, -4])
    upper = np.asarray([e_hi, 15, 2.0, 15, 2.0, 4])
    start_points: list[np.ndarray] = []
    if seed_theta is not None:
        k_grid = [-3.5, -2, -0.5, 0.5, 2, 3.5]
        for k in k_grid[:starts]:
            candidate = seed_theta.copy()
            candidate[-1] = k
            start_points.append(candidate)
    else:
        e_grid = [min(e_hi * f, y_min * 0.9) for f in (0.15, 0.4, 0.7)]
        exponent_grid = [0.15, 0.55, 1.1]
        k_grid = [-3.5, -2, -0.5, 0.5, 2, 3.5]
        for e, exponent, k in itertools.product(e_grid, exponent_grid, k_grid):
            amp = max(float(np.median(y - e)) / 2, 1e-6)
            log_a = np.log(amp / np.median(n ** (-exponent)))
            log_b = np.log(amp / np.median(d ** (-exponent)))
            start_points.append(np.asarray([e, log_a, exponent, log_b, exponent, k]))
        start_points = start_points[:starts]
    results = []
    for x0 in start_points:
        x0 = np.minimum(np.maximum(x0, lower + 1e-9), upper - 1e-9)
        try:
            fit = least_squares(lambda x: predict_full(x, n, d, qr) - y,
                                x0=x0, bounds=(lower, upper), x_scale="jac",
                                max_nfev=2500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if np.all(np.isfinite(fit.fun)):
                results.append((float(np.dot(fit.fun, fit.fun)), fit, bool(fit.success)))
        except (ValueError, FloatingPointError):
            continue
    if not results:
        raise RuntimeError("All full-curve fit starts failed")
    results.sort(key=lambda item: item[0])
    sse, best, _ = results[0]
    fitted = predict_full(best.x, n, d, qr)
    near = [fit.x for fit_sse, fit, _ in results
            if fit_sse <= sse * (1 + 1e-5) + 1e-14]
    physical_near = np.asarray([[x[0], np.exp(x[1]), x[2], np.exp(x[3]), x[4], x[5]] for x in near])
    names = ["E", "A", "alpha", "B", "beta", "kappa"]
    near_spans = {}
    for idx, name in enumerate(names):
        values = physical_near[:, idx]
        median = float(np.median(values))
        span = float(np.max(values) - np.min(values))
        near_spans[name] = {"min": float(np.min(values)), "max": float(np.max(values)),
                            "absolute_span": span,
                            "relative_span": span / max(abs(median), 1e-12)}
    raw_lower = np.asarray([0.0, -15, 1e-4, -15, 1e-4, -4])
    raw_upper = np.asarray([e_hi, 15, 2.0, 15, 2.0, 4])
    boundary_names = ["E", "logA", "alpha", "logB", "beta", "kappa"]
    boundary_hits = []
    for name, value, lo, hi in zip(boundary_names, best.x, raw_lower, raw_upper):
        tol = 1e-6 * max(1.0, abs(hi - lo))
        if abs(value - lo) <= tol or abs(value - hi) <= tol:
            boundary_hits.append(name)
    return {
        "theta": best.x,
        "sse": sse,
        "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))),
        "mae": float(np.mean(np.abs(fitted - y))),
        "successful_starts": sum(ok for _, _, ok in results),
        "attempted_starts": len(start_points),
        "relative_sse_spread": float((results[-1][0] - sse) / max(sse, 1e-30)),
        "near_optimal_count": len(near),
        "near_optimal_parameter_spans": near_spans,
        "boundary_hits": boundary_hits,
    }


def full_curve_loo(rows: list[dict[str, str]], smoke: bool) -> tuple[dict[str, float], list[dict[str, Any]]]:
    q_levels = sorted({float(r["Q_score"]) for r in rows})
    fold_rows = []
    all_errors = []
    fold_starts = 6 if smoke else 54
    for q in q_levels:
        train = [r for r in rows if not np.isclose(float(r["Q_score"]), q, atol=1e-12)]
        test = [r for r in rows if np.isclose(float(r["Q_score"]), q, atol=1e-12)]
        if not train or not test:
            continue
        # Rebuild the Q reference and all starts using training levels only;
        # no held-out Loss or full-sample fitted parameter enters this fold.
        qref_train = float(np.median([float(r["Q_score"]) for r in train]))
        fit = fit_full(train, qref_train, starts=fold_starts)
        n, d, qr, y = arrays(test, qref_train)
        pred = predict_full(fit["theta"], n, d, qr)
        errors = pred - y
        all_errors.extend(errors.tolist())
        fold_rows.append({"Q_held_out": q, "n_test": len(test),
                          "rmse": float(np.sqrt(np.mean(errors**2))),
                          "mae": float(np.mean(np.abs(errors))),
                          "kappa_train": float(fit["theta"][-1]),
                          "Q_reference_train_only": qref_train,
                          "multistart_attempts": fit["attempted_starts"]})
    err = np.asarray(all_errors)
    return {"loo_q_rmse": float(np.sqrt(np.mean(err**2))),
            "loo_q_mae": float(np.mean(np.abs(err))),
            "loo_q_levels": len(fold_rows)}, fold_rows


def predict_anchor(kappa: float, rows: list[dict[str, str]], params: dict[str, float], q0: float) -> np.ndarray:
    n = np.asarray([float(r["N_params_B"]) for r in rows])
    d = np.asarray([float(r["D_tokens_B"]) for r in rows])
    q = np.asarray([float(r["Q_score"]) for r in rows])
    return (params["E"] + params["A"] * n ** (-params["alpha"])
            + params["B"] * d ** (-params["beta"]) * (q / q0) ** (-kappa))


def fit_anchor(rows: list[dict[str, str]], params: dict[str, float], q0: float) -> tuple[float, float]:
    y = np.asarray([float(r["val_loss"]) for r in rows])
    def objective(k: float) -> float:
        residual = predict_anchor(float(k), rows, params, q0) - y
        return float(np.dot(residual, residual))
    result = minimize_scalar(objective, bounds=(-4, 4), method="bounded",
                             options={"xatol": 1e-9, "maxiter": 500})
    candidates = [(-4.0, objective(-4.0)), (float(result.x), float(result.fun)),
                  (4.0, objective(4.0))]
    kappa, sse = min(candidates, key=lambda item: item[1])
    if not np.isfinite(sse):
        raise RuntimeError("B1-anchored kappa fit failed")
    return kappa, float(np.sqrt(sse / len(rows)))


def anchor_loo(rows: list[dict[str, str]], params: dict[str, float], q0: float) -> tuple[dict[str, float], list[dict[str, Any]]]:
    folds, errors = [], []
    levels = sorted({float(r["Q_score"]) for r in rows})
    for q in levels:
        train = [r for r in rows if not np.isclose(float(r["Q_score"]), q, atol=1e-12)]
        test = [r for r in rows if np.isclose(float(r["Q_score"]), q, atol=1e-12)]
        kappa, _ = fit_anchor(train, params, q0)
        err = predict_anchor(kappa, test, params, q0) - np.asarray([float(r["val_loss"]) for r in test])
        errors.extend(err.tolist())
        folds.append({"Q_held_out": q, "n_test": len(test), "kappa_train": kappa,
                      "rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))})
    e = np.asarray(errors)
    return {"loo_q_rmse": float(np.sqrt(np.mean(e**2))),
            "loo_q_mae": float(np.mean(np.abs(e))), "loo_q_levels": len(folds)}, folds


def anchor_cluster_bootstrap(rows: list[dict[str, str]], params: dict[str, float], q0: float,
                             draws: int, rng: np.random.Generator) -> tuple[float, float]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["N_params_B"], row["D_tokens_B"]), []).append(row)
    keys = list(groups)
    estimates = []
    for _ in range(draws):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        sample = [row for idx in sampled for row in groups[keys[int(idx)]]]
        estimate, _ = fit_anchor(sample, params, q0)
        estimates.append(estimate)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def cost_parts(n_abs: int, d_abs: int, context: int, q: float, q0: float,
               g_name: str) -> tuple[float, float, float, float]:
    # Preserve the exact integer training-FLOP count in the CSV. The total is
    # still represented as float because the Q-dependent quality cost can be
    # fractional under the continuous quality curve.
    train = TRAINING_COEFFICIENT * n_abs * d_abs
    attention = float(ETA * n_abs * d_abs * context)
    q_cost = d_abs * max(0.0, cost_module.G_FUNCTIONS[g_name](q) - cost_module.G_FUNCTIONS[g_name](q0))
    return train, q_cost, attention, float(train) + q_cost + attention


def max_affordable_q(g_name: str, q0: float, d_abs: int, remaining: float) -> float:
    if remaining <= 0:
        return q0
    g = cost_module.G_FUNCTIONS[g_name]
    if d_abs * max(0.0, g(1) - g(q0)) <= remaining:
        return 1.0
    lo, hi = q0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if d_abs * max(0.0, g(mid) - g(q0)) <= remaining:
            lo = mid
        else:
            hi = mid
    return lo


def m0_loss(point: dict[str, Any], params: dict[str, float]) -> float:
    return (params["E"] + params["A"] * point["n_b"] ** (-params["alpha"])
            + params["B"] * point["d_b"] ** (-params["beta"]))


def optimize_scenarios(source: str, kappa: float, support: dict[str, float],
                       points: list[dict[str, Any]], frontier: list[dict[str, str]],
                       params: dict[str, float], q0: float, g_name: str) -> list[dict[str, Any]]:
    supported = [p for p in points if support["N_min"] <= p["n_b"] <= support["N_max"]
                 and support["D_min"] <= p["d_b"] <= support["D_max"]]
    output = []
    for scenario in frontier:
        budget = int(scenario["budget_flops"])
        context = int(scenario["context_length_tokens"])
        feasible = []
        for point in supported:
            base_cost = float(TRAINING_COEFFICIENT * point["n_abs"] * point["d_abs"]
                              + ETA * point["n_abs"] * point["d_abs"] * context)
            if base_cost <= budget:
                feasible.append((point, base_cost))
        row = {"q_response_source": source, "kappa_Q_B1_anchored": kappa,
               "quality_cost_function": g_name, "budget_flops": budget,
               "context_length_tokens": context, "source_support_grid_points": len(supported),
               "feasible_grid_points": len(feasible), "source_N_min_B": support["N_min"],
               "source_N_max_B": support["N_max"], "source_D_min_B": support["D_min"],
               "source_D_max_B": support["D_max"],
               "global_M0_run_id": scenario["selected_run_id"],
               "status": "FEASIBLE" if feasible else "INFEASIBLE"}
        if not feasible:
            row.update({k: "" for k in ["selected_run_id", "selected_N_B", "selected_D_B", "selected_Q",
                                         "predicted_loss", "restricted_M0_run_id", "restricted_M0_loss",
                                         "delta_N_B_vs_restricted_M0", "delta_D_B_vs_restricted_M0",
                                         "delta_Q_vs_Q0", "delta_loss_vs_restricted_M0", "C_train_flops",
                                         "C_Q_flops", "C_attn_flops", "C_total_flops", "budget_utilization",
                                         "budget_residual_flops", "N_D_changed_vs_restricted_M0", "at_Q0_boundary"]})
            output.append(row)
            continue
        baseline = min(feasible, key=lambda item: (m0_loss(item[0], params), item[1],
                                                   item[0]["n_b"], item[0]["d_b"], item[0]["run_id"]))
        candidates = []
        for point, base_cost in feasible:
            remaining = max(0.0, budget - base_cost)
            q_max = max_affordable_q(g_name, q0, point["d_abs"], remaining)
            q = q_max if kappa > 0 else q0
            train, qcost, attention, total = cost_parts(point["n_abs"], point["d_abs"], context, q, q0, g_name)
            loss = m0_loss(point, params) + params["B"] * point["d_b"] ** (-params["beta"]) * ((q / q0) ** (-kappa) - 1)
            if total > budget + max(1.0, budget) * 1e-12:
                raise ValueError(f"Budget exceeded for {source}/{g_name}/{budget}/{context}")
            candidates.append((loss, total, point["n_b"], point["d_b"], point["run_id"],
                               q, train, qcost, attention, point))
        best = min(candidates, key=lambda x: x[:5])
        base_point = baseline[0]
        row.update({
            "selected_run_id": best[4], "selected_N_B": best[2], "selected_D_B": best[3],
            "selected_Q": best[5], "predicted_loss": best[0],
            "restricted_M0_run_id": base_point["run_id"], "restricted_M0_loss": m0_loss(base_point, params),
            "delta_N_B_vs_restricted_M0": best[2] - base_point["n_b"],
            "delta_D_B_vs_restricted_M0": best[3] - base_point["d_b"],
            "delta_Q_vs_Q0": best[5] - q0,
            "delta_loss_vs_restricted_M0": best[0] - m0_loss(base_point, params),
            "C_train_flops": int(best[6]), "C_Q_flops": best[7], "C_attn_flops": best[8],
            "C_total_flops": best[1], "budget_utilization": best[1] / budget,
            "budget_residual_flops": budget - best[1],
            "N_D_changed_vs_restricted_M0": (best[2] != base_point["n_b"] or best[3] != base_point["d_b"]),
            "at_Q0_boundary": abs(best[5] - q0) < 1e-10,
        })
        output.append(row)
    return output


def make_transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["q_response_source"], r["quality_cost_function"], r["budget_flops"],
               r["context_length_tokens"]): r for r in rows}
    sources = sorted({r["q_response_source"] for r in rows})
    g_names = sorted({r["quality_cost_function"] for r in rows})
    budgets = sorted({int(r["budget_flops"]) for r in rows})
    contexts = sorted({int(r["context_length_tokens"]) for r in rows})
    result = []
    for source, g_name in itertools.product(sources, g_names):
        pairs = []
        for c in contexts:
            pairs.extend(("adjacent_budget", b0, c, b1, c) for b0, b1 in zip(budgets, budgets[1:]))
        for b in budgets:
            pairs.extend(("adjacent_context", b, c0, b, c1) for c0, c1 in zip(contexts, contexts[1:]))
        for idx, (kind, lb, lc, rb, rc) in enumerate(pairs, 1):
            left = by_key[(source, g_name, lb, lc)]
            right = by_key[(source, g_name, rb, rc)]
            both = left["status"] == right["status"] == "FEASIBLE"
            row = {"q_response_source": source, "quality_cost_function": g_name,
                   "pair_id": f"{kind}_{idx:02d}", "transition_type": kind,
                   "left_budget_flops": lb, "left_context_tokens": lc,
                   "right_budget_flops": rb, "right_context_tokens": rc,
                   "left_status": left["status"], "right_status": right["status"],
                   "edge_status": "comparable" if both else "feasibility_boundary"}
            for name, col in (("N_B", "selected_N_B"), ("D_B", "selected_D_B"),
                              ("Q", "selected_Q"), ("loss", "predicted_loss")):
                row[f"left_{name}"] = left[col]
                row[f"right_{name}"] = right[col]
                row[f"delta_{name}"] = right[col] - left[col] if both else ""
            row["N_changed"] = bool(both and left["selected_N_B"] != right["selected_N_B"])
            row["D_changed"] = bool(both and left["selected_D_B"] != right["selected_D_B"])
            row["Q_changed"] = bool(both and abs(left["selected_Q"] - right["selected_Q"]) > 1e-10)
            result.append(row)
    return result


def main() -> None:
    global OUT, BOOTSTRAPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="small-start P1 slice on real supplied data")
    args = parser.parse_args()
    if args.smoke:
        OUT = OUT / "_p1_smoke"
        BOOTSTRAPS = 100

    q0, _, qmeta = cost_module.get_frozen_q0()
    param_file = read_json(Q2_PARAMS)
    if param_file.get("fit_scope") != "B1 only":
        raise ValueError("Expected the frozen B1-only M0 parameter file")
    params = {k: float(v) for k, v in param_file["parameters"].items()}
    b1 = read_csv(Q2_GRID)
    frontier_all = read_csv(Q3_FRONTIER)
    frontier = frontier_all[:2] if args.smoke else frontier_all
    if len(b1) != 1176 or len(frontier_all) != 15:
        raise ValueError("Input contract requires 1,176 B1 points and 15 Q3 frontier scenarios")
    points = []
    for r in b1:
        n_b, d_b = float(r["N_params_B"]), float(r["D_tokens_B"])
        n_abs = int(Fraction(r["N_params_B"]) * 10**9)
        d_abs = int(Fraction(r["D_tokens_B"]) * 10**9)
        point = {"run_id": r["run_id"], "n_b": n_b, "d_b": d_b,
                 "n_abs": n_abs, "d_abs": d_abs}
        calc = m0_loss(point, params)
        if abs(calc - float(r["predicted_val_loss"])) > 2e-8:
            raise ValueError(f"Frozen M0 prediction mismatch at run_id={r['run_id']}")
        points.append(point)
    if len({r["run_id"] for r in points}) != 1176:
        raise ValueError("B1 run_id must be unique")

    b6 = read_csv(Q2_B6_ROWS)
    b8 = read_csv(Q2_B8_ROWS)
    source_rows = {
        "B6_semi_synthetic": b6,
        "B8_calibrated": [r for r in b8 if r["data_type"] == "calibrated"],
        "B8_extrapolated_diagnostic_only": [r for r in b8 if r["data_type"] == "extrapolated"],
    }
    expected = {"B6_semi_synthetic": 360, "B8_calibrated": 984,
                "B8_extrapolated_diagnostic_only": 720}
    if {k: len(v) for k, v in source_rows.items()} != expected:
        raise ValueError("B6/B8 source row counts do not match the supplied data contract")

    rng = np.random.default_rng(SEED)
    fit_rows, loo_rows, anchor_rows = [], [], []
    fit_results = {}
    n_starts = 6 if args.smoke else 54
    for source, rows in source_rows.items():
        qref = float(np.median([float(r["Q_score"]) for r in rows]))
        fit = fit_full(rows, qref, starts=n_starts)
        loo_metrics, folds = full_curve_loo(rows, args.smoke)
        fit_results[source] = (fit, qref, rows)
        theta = fit["theta"]
        fit_rows.append({"source": source, "n": len(rows), "Q_reference_median": qref,
                         "N_min_B": min(float(r["N_params_B"]) for r in rows),
                         "N_max_B": max(float(r["N_params_B"]) for r in rows),
                         "D_min_B": min(float(r["D_tokens_B"]) for r in rows),
                         "D_max_B": max(float(r["D_tokens_B"]) for r in rows),
                         "Q_min": min(float(r["Q_score"]) for r in rows),
                         "Q_max": max(float(r["Q_score"]) for r in rows),
                         "E": theta[0], "A": np.exp(theta[1]), "alpha": theta[2],
                         "B": np.exp(theta[3]), "beta": theta[4], "kappa": theta[5],
                         "in_sample_RMSE": fit["rmse"], "in_sample_MAE": fit["mae"],
                         "LOO_Q_RMSE": loo_metrics["loo_q_rmse"], "LOO_Q_MAE": loo_metrics["loo_q_mae"],
                         "LOO_Q_levels": loo_metrics["loo_q_levels"],
                         "multistart_attempts": fit["attempted_starts"],
                         "multistart_successful": fit["successful_starts"],
                         "near_optimal_count": fit["near_optimal_count"],
                         "near_optimal_parameter_spans_json": json.dumps(fit["near_optimal_parameter_spans"], ensure_ascii=False),
                         "boundary_hits": ";".join(fit["boundary_hits"]) or "none",
                         "boundary_hit_count": len(fit["boundary_hits"]),
                         "relative_SSE_spread": fit["relative_sse_spread"],
                         "interpretation": "semi-synthetic/calibrated source-specific full curve; not real B1 effect"})
        for fold in folds:
            loo_rows.append({"source": source, **fold})

    anchor_sources = ["B6_semi_synthetic", "B8_calibrated"]
    anchor_kappas = {}
    anchor_folds = []
    for source in anchor_sources:
        rows = source_rows[source]
        kappa, rmse = fit_anchor(rows, params, q0)
        base_pred = predict_anchor(0.0, rows, params, q0)
        y = np.asarray([float(r["val_loss"]) for r in rows])
        base_rmse = float(np.sqrt(np.mean((base_pred - y) ** 2)))
        cv_metrics, folds = anchor_loo(rows, params, q0)
        ci_low, ci_high = anchor_cluster_bootstrap(rows, params, q0, BOOTSTRAPS, rng)
        support = {"N_min": min(float(r["N_params_B"]) for r in rows),
                   "N_max": max(float(r["N_params_B"]) for r in rows),
                   "D_min": min(float(r["D_tokens_B"]) for r in rows),
                   "D_max": max(float(r["D_tokens_B"]) for r in rows)}
        support_count = sum(support["N_min"] <= p["n_b"] <= support["N_max"]
                            and support["D_min"] <= p["d_b"] <= support["D_max"] for p in points)
        anchor_kappas[source] = kappa
        anchor_rows.append({"source": source, "n": len(rows), "kappa_B1_anchored": kappa,
                            "Q_reference": q0, "baseline_M0_RMSE": base_rmse,
                            "anchored_in_sample_RMSE": rmse, "anchored_LOO_Q_RMSE": cv_metrics["loo_q_rmse"],
                            "anchored_LOO_Q_MAE": cv_metrics["loo_q_mae"],
                            "anchored_LOO_Q_levels": cv_metrics["loo_q_levels"],
                            "cluster_bootstrap_draws": BOOTSTRAPS,
                            "cluster_bootstrap_95_low": ci_low, "cluster_bootstrap_95_high": ci_high,
                            "bootstrap_unit": "matched (N,D) source cell",
                            "B1_source_support_grid_points": support_count,
                            "B1_total_grid_points": len(points),
                            "support_N_min_B": support["N_min"], "support_N_max_B": support["N_max"],
                            "support_D_min_B": support["D_min"], "support_D_max_B": support["D_max"],
                            "mapping_assumption": "Q_score numerically identified with Q1 q_huber; unvalidated conditional scenario"})
        for fold in folds:
            anchor_folds.append({"source": source, **fold})

    # Only B6 and B8 calibrated enter B1-anchored optimization. B8 extrapolated
    # has no N support overlap with B1 and remains a full-curve diagnostic.
    source_support = {
        name: {"N_min": float(anchor_rows[i]["support_N_min_B"]),
               "N_max": float(anchor_rows[i]["support_N_max_B"]),
               "D_min": float(anchor_rows[i]["support_D_min_B"]),
               "D_max": float(anchor_rows[i]["support_D_max_B"])}
        for i, name in enumerate(anchor_sources)
    }
    grid_coverage = []
    for source, bounds in source_support.items():
        count = sum(bounds["N_min"] <= p["n_b"] <= bounds["N_max"]
                    and bounds["D_min"] <= p["d_b"] <= bounds["D_max"] for p in points)
        grid_coverage.append({"source": source, "B1_points_in_source_support": count,
                              "B1_total_points": len(points), "excluded_points": len(points)-count})
    ext = source_rows["B8_extrapolated_diagnostic_only"]
    ext_bounds = {"N_min": min(float(r["N_params_B"]) for r in ext),
                  "N_max": max(float(r["N_params_B"]) for r in ext),
                  "D_min": min(float(r["D_tokens_B"]) for r in ext),
                  "D_max": max(float(r["D_tokens_B"]) for r in ext)}
    ext_overlap = sum(ext_bounds["N_min"] <= p["n_b"] <= ext_bounds["N_max"]
                      and ext_bounds["D_min"] <= p["d_b"] <= ext_bounds["D_max"] for p in points)
    grid_coverage.append({"source": "B8_extrapolated_diagnostic_only", "B1_points_in_source_support": ext_overlap,
                          "B1_total_points": len(points), "excluded_points": len(points)-ext_overlap})

    scenario_rows = []
    opt_frontier = frontier
    for source in anchor_sources:
        for g_name in cost_module.G_FUNCTIONS:
            scenario_rows.extend(optimize_scenarios(source, anchor_kappas[source], source_support[source],
                                                    points, opt_frontier, params, q0, g_name))
    transition_rows = make_transitions(scenario_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "q2_source_full_curve_fits.csv", fit_rows)
    write_csv(OUT / "q2_leave_one_Q_level_predictions.csv", loo_rows)
    write_csv(OUT / "q3_B1_anchored_Q_response.csv", anchor_rows)
    write_csv(OUT / "q3_anchored_leave_one_Q_predictions.csv", anchor_folds)
    write_csv(OUT / "q3_source_support_intersections.csv", grid_coverage)
    write_csv(OUT / "q3_qp_conditional_scenarios.csv", scenario_rows)
    transition_fields = ["q_response_source", "quality_cost_function", "pair_id", "transition_type",
                         "left_budget_flops", "left_context_tokens", "right_budget_flops", "right_context_tokens",
                         "left_status", "right_status", "edge_status", "left_N_B", "right_N_B", "delta_N_B",
                         "left_D_B", "right_D_B", "delta_D_B", "left_Q", "right_Q", "delta_Q",
                         "left_loss", "right_loss", "delta_loss", "N_changed", "D_changed", "Q_changed"]
    write_csv(OUT / "q3_qp_adjacent_transitions.csv", transition_rows, transition_fields)

    transition_summaries = []
    for source in anchor_sources:
        for g_name in cost_module.G_FUNCTIONS:
            rows = [r for r in scenario_rows if r["q_response_source"] == source and r["quality_cost_function"] == g_name]
            edges = [r for r in transition_rows if r["q_response_source"] == source and r["quality_cost_function"] == g_name]
            feasible_edges = [r for r in edges if r["edge_status"] == "comparable"]
            transition_summaries.append({
                "source": source, "g_Q": g_name, "kappa": anchor_kappas[source],
                "scenarios": len(rows), "feasible_scenarios": sum(r["status"] == "FEASIBLE" for r in rows),
                "infeasible_scenarios": sum(r["status"] == "INFEASIBLE" for r in rows),
                "Q_above_Q0_scenarios": sum(r["status"] == "FEASIBLE" and not r["at_Q0_boundary"] for r in rows),
                "N_D_reselected_vs_restricted_M0": sum(r["status"] == "FEASIBLE" and r["N_D_changed_vs_restricted_M0"] for r in rows),
                "adjacent_edges": len(edges), "comparable_edges": len(feasible_edges),
                "feasibility_boundary_edges": len(edges)-len(feasible_edges),
                "comparable_edges_with_N_change": sum(r["N_changed"] for r in feasible_edges),
                "comparable_edges_with_D_change": sum(r["D_changed"] for r in feasible_edges),
                "comparable_edges_with_Q_change": sum(r["Q_changed"] for r in feasible_edges),
            })

    report_lines = [
        "# Q/p 条件响应下的 Q3 资源转移敏感性（按来源分支）", "",
        "日期：2026-09-26。此处完成的是现有附件的条件模型拟合与迁移压力测试，不是新增真实训练实验。", "",
        "## 计算口径", "",
        f"- 冻结 B1 M0 参数；Q1 q_huber 参考值 Q0={q0:.10f}。假设 Q2 `Q_score` 与 Q1 `q_huber` 数值同尺度，此映射尚未验证。",
        "- 来源曲面逐一拟合 L=E+A N^(-alpha)+B D^(-beta)(Q/Qref)^(-kappa)，不跨来源拼参数；报告留一 Q 水平预测。",
        "- 转移分支冻结 B1 的 E,A,alpha,B,beta，仅在 B6 与 B8 calibrated 上拟合有符号 kappa；按各自 N/D 观测矩形与 B1 网格取交集。",
        "- B8 extrapolated 只参与来源内曲面诊断；N 支持域与 B1 无交集，不进入预算优化。B7 未当作独立重复。",
        "- 分别使用指数、四次幂、对数型 g(Q)；预算不足的情景明确记为 INFEASIBLE。固定 p 为 B1 共同数据政策条件，不填造 17 维配比。", "",
        "## B1 锚定 Q 响应", "",
        "| 来源 | κ | M0 RMSE | 锚定 RMSE | 留一 Q RMSE | cluster bootstrap 95% | 支持域内 B1 点 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in anchor_rows:
        report_lines.append(f"| {r['source']} | {r['kappa_B1_anchored']:.6g} | {r['baseline_M0_RMSE']:.5g} | {r['anchored_in_sample_RMSE']:.5g} | {r['anchored_LOO_Q_RMSE']:.5g} | [{r['cluster_bootstrap_95_low']:.5g}, {r['cluster_bootstrap_95_high']:.5g}] | {r['B1_source_support_grid_points']}/{r['B1_total_grid_points']} |")
    report_lines += ["", "bootstrap 以来源内匹配 (N,D) 单元为重抽样单位；它刻画半合成/校准网格稳定性，不代表真实训练随机性。", "",
                     "## 来源曲面与留一 Q 验证", "",
                     "完整参数表见 `q2_source_full_curve_fits.csv`；逐 Q 留出误差见 `q2_leave_one_Q_level_predictions.csv`。B8 extrapolated 结果仅解释其外推来源内拟合，不迁移到 B1 优化。", "",
                     "## Q3 转移摘要", "",
                     "| 来源 | g(Q) | κ | 可行/15 | 不可行情景 | Q>Q0 | 相对受限 M0 重选 N/D | 可比相邻边/22 | N变边 | D变边 | Q变边 |",
                     "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in transition_summaries:
        report_lines.append(f"| {s['source']} | {s['g_Q']} | {s['kappa']:.6g} | {s['feasible_scenarios']}/{len(rows)} | {s['infeasible_scenarios']} | {s['Q_above_Q0_scenarios']} | {s['N_D_reselected_vs_restricted_M0']} | {s['comparable_edges']}/{len(edges)} | {s['comparable_edges_with_N_change']} | {s['comparable_edges_with_D_change']} | {s['comparable_edges_with_Q_change']} |")
    report_lines += ["", "## 结论与边界", "",
                     "1. B6 与 B8 calibrated 的锚定 κ 分别估计，不合并。若符号相反，Q 导致的结构转移没有来源稳健的唯一方向。",
                     "2. 这些曲线把半合成/校准来源的质量响应迁移到 B1 的固定 M0 结构上，只能回答条件情景；不构成 B1 的实测 Q 效应。",
                     "3. A 侧 p→Loss 的跨尺度检验没有通过，且 B1 缺 checkpoint 级 p；所以本结果按题面允许分支固定 p，不报告 p 的 B 侧边际效应或完整 p/Q 联合最优。",
                     "4. Q1 `q_huber` 与 Q2 `Q_score` 映射未经验证，B8/B6 数据为半合成/校准来源；真实独立 p/Q 识别仍需受控训练或可追溯运行日志。",
                     "5. 逐情景成本和可行性见 `q3_qp_conditional_scenarios.csv`；邻接边见 `q3_qp_adjacent_transitions.csv`；来源域交集见 `q3_source_support_intersections.csv`。", "",
                     "复现命令：`D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/solve_q3_qp_conditioned_transition_v2.py`。", ""]
    report_path = ROOT / OUT / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    inputs = [Q2_GRID, Q2_PARAMS, Q2_B6_ROWS, Q2_B8_ROWS, Q3_FRONTIER, Q3_PLAN,
              Q1_FREEZE, Q1_DOMAIN, Q1_MACRO]
    outputs = [OUT / "q2_source_full_curve_fits.csv", OUT / "q2_leave_one_Q_level_predictions.csv",
               OUT / "q3_B1_anchored_Q_response.csv", OUT / "q3_anchored_leave_one_Q_predictions.csv",
               OUT / "q3_source_support_intersections.csv", OUT / "q3_qp_conditional_scenarios.csv",
               OUT / "q3_qp_adjacent_transitions.csv", OUT / "report.md"]
    manifest = {
        "artifact": "Q3 source-separated Q response transfer sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/solve_q3_qp_conditioned_transition_v2.py",
        "script_sha256": sha256(Path("Q3/03_代码/solve_q3_qp_conditioned_transition_v2.py")),
        "python": sys.version, "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
        "random_seed": SEED, "anchor_cluster_bootstrap_draws": BOOTSTRAPS,
        "quality_reference": {**qmeta, "Q0": q0,
                               "mapping_assumption": "Q_score numerically identified with q_huber; unvalidated"},
        "scope": "B6/B8 calibrated Q-only migration pressure test; fixed p; not empirical joint p/Q identification",
        "full_curve_fits": {r["source"]: {"kappa": r["kappa"], "LOO_Q_RMSE": r["LOO_Q_RMSE"]} for r in fit_rows},
        "B1_anchored_kappa": {r["source"]: r["kappa_B1_anchored"] for r in anchor_rows},
        "input_sha256": {str(p): sha256(p) for p in inputs},
        "output_sha256": {str(p): sha256(p) for p in outputs},
        "row_counts": {"B1_grid": len(b1), "frontier_scenarios": len(frontier),
                       "source_curve_fits": len(fit_rows), "source_curve_LOO_rows": len(loo_rows),
                       "B1_anchored_fits": len(anchor_rows), "conditional_scenarios": len(scenario_rows),
                       "adjacent_edges": len(transition_rows)},
        "checks": {"B1_M0_predictions_reconciled": True, "all_budget_constraints_checked": True,
                   "source_support_intersections_applied": True, "three_g_functions_run": True,
                   "B8_extrapolated_omitted_from_B1_optimization": True,
                   "independent_p_Q_effect_not_claimed": True},
    }
    (ROOT / OUT / "reproduction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "fit_rows": fit_rows, "anchor_rows": anchor_rows,
                      "grid_coverage": grid_coverage, "transition_summaries": transition_summaries,
                      "row_counts": manifest["row_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
