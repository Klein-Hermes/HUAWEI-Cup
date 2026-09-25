#!/usr/bin/env python3
"""Assess trajectory-omission and scaling-form sensitivity of Q3's M0 frontier.

Run from the repository root:
    D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/analyze_q3_trajectory_form_sensitivity.py --p1
    D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/analyze_q3_trajectory_form_sensitivity.py

The analysis keeps Q fixed at Q0 and p outside the objective. It uses the
existing 1,176-point B1 N/D support grid and exact M4 feasibility masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "Q3/04_结果/轨迹留一与函数形式敏感性"
Q2_DIR = ROOT / "Q2/03_结果/经典ScalingLaw基线/v1"
Q3_DIR = ROOT / "Q3/04_结果"

INPUTS = {
    "b1_predictions": Q2_DIR / "b1_fitted_predictions.csv",
    "trajectory_mapping": Q2_DIR / "gate0_b1_trajectory_mapping.csv",
    "baseline_loo_parameters": Q2_DIR / "leave_one_size_out.csv",
    "baseline_parameters": Q2_DIR / "fit_parameters.json",
    "candidate_grid": Q3_DIR / "m3_candidate_grid.csv",
    "budget_mask": Q3_DIR / "m4_budget_feasible_mask.csv",
    "baseline_frontier": Q3_DIR / "M0_ND_reference_frontier.csv",
    "context_audit": ROOT / "Q3/02_数据审计/C7_context_by_model.csv",
}

MODEL_FORMULAS = {
    "m0_current": "L = E + A*N^(-alpha) + B*D^(-beta)",
    "m0_no_floor": "L = A*N^(-alpha) + B*D^(-beta); E=0",
    "m0_shared_exponent": "L = E + A*N^(-gamma) + B*D^(-gamma); alpha=beta=gamma",
}
MODEL_PARAMETER_NAMES = {
    "m0_current": ("E", "A", "alpha", "B", "beta"),
    "m0_no_floor": ("A", "alpha", "B", "beta"),
    "m0_shared_exponent": ("E", "A", "gamma", "B"),
}
MODEL_PARAMETER_COUNTS = {name: len(fields) for name, fields in MODEL_PARAMETER_NAMES.items()}
MODEL_FORMS = tuple(MODEL_FORMULAS)
EXPECTED_TRACKS = 8
EXPECTED_ROWS_PER_TRACK = 147
EXPECTED_SUPPORT_POINTS = 1176
EXPECTED_SCENARIOS = 15
EXPECTED_MASK_ROWS = 17640
TRAINING_COEFFICIENT = 6
ATTENTION_DENOMINATOR = 5000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def n_key(value: float) -> str:
    return f"{float(value):.6f}"


def predict(form: str, params: dict[str, float], n: np.ndarray, d: np.ndarray) -> np.ndarray:
    if form == "m0_current":
        return params["E"] + params["A"] * np.power(n, -params["alpha"]) + params["B"] * np.power(d, -params["beta"])
    if form == "m0_no_floor":
        return params["A"] * np.power(n, -params["alpha"]) + params["B"] * np.power(d, -params["beta"])
    if form == "m0_shared_exponent":
        gamma = params["gamma"]
        return params["E"] + params["A"] * np.power(n, -gamma) + params["B"] * np.power(d, -gamma)
    raise ValueError(f"Unknown model form: {form}")


def parameter_dict(form: str, vector: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(MODEL_PARAMETER_NAMES[form], vector)}


def predict_rows(form: str, params: dict[str, float], frame: pd.DataFrame) -> np.ndarray:
    return predict(
        form,
        params,
        frame["N_params_B"].to_numpy(dtype=float),
        frame["D_tokens_B"].to_numpy(dtype=float),
    )


def current_parameters(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in MODEL_PARAMETER_NAMES["m0_current"]}


def _candidate_starts(
    form: str,
    n: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    anchor: dict[str, float],
    count: int,
) -> list[np.ndarray]:
    n_center = float(np.exp(np.mean(np.log(n))))
    d_center = float(np.exp(np.mean(np.log(d))))
    min_y = float(np.min(y))
    median_y = float(np.median(y))
    split_cycle = (0.2, 0.35, 0.5, 0.65, 0.8)
    if form == "m0_no_floor":
        alpha0 = anchor["alpha"]
        beta0 = anchor["beta"]
        exponent_pairs = (
            (alpha0, beta0), (alpha0 * 0.65, beta0), (alpha0, beta0 * 0.65),
            (alpha0 * 1.35, beta0), (alpha0, beta0 * 1.35),
            (alpha0 * 0.7, beta0 * 1.3), (alpha0 * 1.3, beta0 * 0.7),
            (0.1, 0.1), (0.2, 0.2), (0.35, 0.35), (0.5, 0.5),
            (alpha0 * 0.8, beta0 * 0.8), (alpha0 * 1.2, beta0 * 1.2),
        )
        starts: list[np.ndarray] = []
        for i in range(count):
            alpha, beta = exponent_pairs[i % len(exponent_pairs)]
            split = split_cycle[i % len(split_cycle)]
            a = max(median_y * split * n_center**alpha, 1e-10)
            b = max(median_y * (1 - split) * d_center**beta, 1e-10)
            starts.append(np.array([a, max(alpha, 1e-6), b, max(beta, 1e-6)], dtype=float))
        return starts

    e_cap = max(min_y - max(1e-10, min_y * 1e-10), 1e-9)
    base_e = min(max(anchor["E"], e_cap * 1e-8), e_cap * (1 - 1e-8))
    gamma0 = math.sqrt(max(anchor["alpha"], 1e-8) * max(anchor["beta"], 1e-8))
    gamma_seeds = (gamma0, anchor["alpha"], anchor["beta"], 0.1, 0.2, 0.35, 0.5, 0.7, gamma0 * 0.7, gamma0 * 1.3)
    e_seeds = (base_e, e_cap * 0.1, e_cap * 0.35, e_cap * 0.65, e_cap * 0.9)
    starts = []
    for i in range(count):
        gamma = max(float(gamma_seeds[i % len(gamma_seeds)]), 1e-6)
        e = min(max(float(e_seeds[(i * 2) % len(e_seeds)]), 0.0), e_cap * (1 - 1e-8))
        excess = max(median_y - e, min_y * 0.02)
        split = split_cycle[i % len(split_cycle)]
        a = max(excess * split * n_center**gamma, 1e-10)
        b = max(excess * (1 - split) * d_center**gamma, 1e-10)
        starts.append(np.array([e, a, gamma, b], dtype=float))
    return starts


def fit_restricted_form(
    train: pd.DataFrame,
    form: str,
    anchor: dict[str, float],
    starts: int = 16,
) -> tuple[dict[str, float], dict[str, Any]]:
    n = train["N_params_B"].to_numpy(dtype=float)
    d = train["D_tokens_B"].to_numpy(dtype=float)
    y = train["val_loss"].to_numpy(dtype=float)
    min_y = float(np.min(y))
    if form == "m0_no_floor":
        lower = np.array([1e-12, 1e-8, 1e-12, 1e-8], dtype=float)
        upper = np.full(4, np.inf)
    elif form == "m0_shared_exponent":
        e_upper = max(min_y - max(1e-10, min_y * 1e-10), 1e-9)
        lower = np.array([0.0, 1e-12, 1e-8, 1e-12], dtype=float)
        upper = np.array([e_upper, np.inf, np.inf, np.inf], dtype=float)
    else:
        raise ValueError(f"Restricted fitter does not handle {form}")

    def residual(vector: np.ndarray) -> np.ndarray:
        return predict(form, parameter_dict(form, vector), n, d) - y

    successes = []
    failures = []
    for start_id, x0 in enumerate(_candidate_starts(form, n, d, y, anchor, starts)):
        x0 = np.maximum(x0, lower + np.where(np.isfinite(lower), 1e-12, 0.0))
        finite_upper = np.isfinite(upper)
        x0[finite_upper] = np.minimum(x0[finite_upper], upper[finite_upper] - np.maximum(1e-12, np.abs(upper[finite_upper]) * 1e-10))
        try:
            fit = least_squares(
                residual,
                x0=x0,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
                max_nfev=10000,
            )
            finite = bool(np.isfinite(fit.x).all() and np.isfinite(fit.fun).all())
            if fit.success and finite:
                sse = float(np.dot(fit.fun, fit.fun))
                successes.append((sse, fit, start_id))
            else:
                failures.append({"start_id": start_id, "status": int(fit.status), "message": str(fit.message)})
        except Exception as exc:  # retain per-start evidence while trying remaining starts
            failures.append({"start_id": start_id, "status": -1, "message": f"{type(exc).__name__}: {exc}"})
    if not successes:
        raise RuntimeError(f"No successful optimization for {form}; failures={failures[:3]}")
    sse, best, best_start = min(successes, key=lambda item: item[0])
    params = parameter_dict(form, np.asarray(best.x, dtype=float))
    if form == "m0_shared_exponent":
        require(0 <= params["E"] < min_y, "Shared-exponent model violates E bound")
    require(all(math.isfinite(value) and value > 0 for key, value in params.items() if key != "E"), "Nonpositive or nonfinite coefficient/exponent")
    require("E" not in params or params["E"] >= 0, "Negative loss floor")
    return params, {
        "successful_starts": len(successes),
        "attempted_starts": starts,
        "best_start_id": best_start,
        "training_sse": sse,
        "optimizer_status": int(best.status),
        "optimizer_message": str(best.message),
    }


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    require(np.isfinite(residual).all(), "Nonfinite prediction residual")
    return {
        "n": int(residual.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_pred_minus_obs": float(np.mean(residual)),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def exact_cost(point: dict[str, Any], context: int) -> Fraction:
    nd = int(point["N_parameters"]) * int(point["D_tokens"])
    return Fraction(TRAINING_COEFFICIENT * nd, 1) + Fraction(nd * context, ATTENTION_DENOMINATOR)


def load_inputs() -> dict[str, Any]:
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    require(not missing, f"Missing required inputs: {missing}")
    b1 = read_csv(INPUTS["b1_predictions"])
    mapping = read_csv(INPUTS["trajectory_mapping"])
    loo = read_csv(INPUTS["baseline_loo_parameters"])
    base_meta = json.loads(INPUTS["baseline_parameters"].read_text(encoding="utf-8"))
    candidates = read_csv(INPUTS["candidate_grid"])
    mask = read_csv(INPUTS["budget_mask"])
    frontier = read_csv(INPUTS["baseline_frontier"])
    context_audit = read_csv(INPUTS["context_audit"])

    required_b1 = {"run_id", "N_params_B", "D_tokens_B", "val_loss"}
    require(required_b1.issubset(b1.columns), f"B1 input missing {sorted(required_b1 - set(b1.columns))}")
    required_map = {"N_params_B", "model_repo", "B1_rows"}
    require(required_map.issubset(mapping.columns), f"Trajectory mapping missing {sorted(required_map - set(mapping.columns))}")
    mapping_by_n = {n_key(row.N_params_B): str(row.model_repo) for row in mapping.itertuples(index=False)}
    require(len(mapping_by_n) == EXPECTED_TRACKS, "Expected 8 unique B12-verified trajectories")
    b1["trajectory_id"] = b1["N_params_B"].map(lambda value: mapping_by_n.get(n_key(value)))
    require(b1["trajectory_id"].notna().all(), "Some B1 rows do not map to a verified trajectory")
    track_sizes = b1.groupby("trajectory_id").size()
    require(len(track_sizes) == EXPECTED_TRACKS and (track_sizes == EXPECTED_ROWS_PER_TRACK).all(), f"Expected 8 tracks × 147 rows; got {track_sizes.to_dict()}")
    require(b1["run_id"].nunique() == EXPECTED_SUPPORT_POINTS, "B1 run_id support grid is not 1,176 unique rows")

    candidate_required = {"grid_id", "source_run_id", "N_params_B", "D_tokens_B", "N_parameters", "D_tokens", "observed_val_loss"}
    require(candidate_required.issubset(candidates.columns), f"Candidate grid missing {sorted(candidate_required - set(candidates.columns))}")
    require(len(candidates) == EXPECTED_SUPPORT_POINTS and candidates["grid_id"].nunique() == EXPECTED_SUPPORT_POINTS, "Candidate support grid must have 1,176 unique points")
    candidates["trajectory_id"] = candidates["N_params_B"].map(lambda value: mapping_by_n.get(n_key(value)))
    require(candidates["trajectory_id"].notna().all(), "Candidate rows do not map to verified trajectories")
    candidates["source_run_id"] = candidates["source_run_id"].astype(int)
    candidates["grid_id"] = candidates["grid_id"].astype(int)
    # Keep NumPy's parsed int64 columns: on Windows, astype(int) can downcast
    # absolute token counts and parameter counts to 32-bit and silently wrap.

    mask_required = {"scenario_id", "budget_flops", "context_length_tokens", "grid_id", "budget_feasible"}
    require(mask_required.issubset(mask.columns), f"M4 mask missing {sorted(mask_required - set(mask.columns))}")
    require(len(mask) == EXPECTED_MASK_ROWS, f"Expected 17,640 M4 rows; got {len(mask)}")
    # FLOPs budgets reach 1e24, beyond signed 64-bit integer range.
    mask["budget_flops"] = mask["budget_flops"].map(lambda value: int(str(value)))
    mask["context_length_tokens"] = mask["context_length_tokens"].astype(int)
    mask["grid_id"] = mask["grid_id"].astype(int)
    mask["budget_feasible_bool"] = mask["budget_feasible"].astype(str).str.lower().eq("true")

    c7_contexts = set(context_audit["max_position_embeddings"].astype(int).unique())
    mask_contexts = set(mask["context_length_tokens"].unique())
    require(c7_contexts == mask_contexts and len(c7_contexts) == 5, f"C7/M4 context mismatch: {c7_contexts} vs {mask_contexts}")
    require(mask[["budget_flops", "context_length_tokens"]].drop_duplicates().shape[0] == EXPECTED_SCENARIOS, "Expected 15 budget-context scenarios")
    for (budget, context), group in mask.groupby(["budget_flops", "context_length_tokens"]):
        require(len(group) == EXPECTED_SUPPORT_POINTS, f"Scenario {budget}/{context} has {len(group)} mask rows")
        by_grid = candidates.set_index("grid_id")
        for row in group.itertuples(index=False):
            point = by_grid.loc[int(row.grid_id)].to_dict()
            exact_feasible = exact_cost(point, int(context)) <= int(budget)
            require(
                exact_feasible == bool(row.budget_feasible_bool),
                f"M4 mask disagrees with exact cost at budget={budget}, context={context}, grid_id={row.grid_id}; "
                f"exact_feasible={exact_feasible}, mask={row.budget_feasible_bool}, exact_cost={float(exact_cost(point, int(context)))}, point={point}",
            )

    loo_n_values = set(loo["held_out_N_params_B"].map(n_key))
    require(loo_n_values == set(mapping_by_n), "Existing Q2 leave-one-scale fits do not cover the 8 mapped trajectories")
    require(loo["held_out_rows"].astype(int).eq(EXPECTED_ROWS_PER_TRACK).all(), "Each existing leave-one-scale fit must hold out a full trajectory")
    require(len(frontier) == EXPECTED_SCENARIOS, "Baseline frontier must contain 15 scenarios")

    return {
        "b1": b1,
        "mapping": mapping,
        "loo": loo,
        "base_meta": base_meta,
        "candidates": candidates,
        "mask": mask,
        "frontier": frontier,
        "tracks": sorted(mapping_by_n.items(), key=lambda item: float(item[0])),
        "mapping_by_n": mapping_by_n,
    }


def candidate_scenarios(data: dict[str, Any]) -> dict[tuple[int, int], pd.DataFrame]:
    scenarios: dict[tuple[int, int], pd.DataFrame] = {}
    for (budget, context), mask_group in data["mask"].groupby(["budget_flops", "context_length_tokens"]):
        candidate_fields = data["candidates"][
            ["grid_id", "N_parameters", "D_tokens", "observed_val_loss", "trajectory_id"]
        ]
        joined = mask_group.merge(candidate_fields, on="grid_id", validate="one_to_one")
        feasible = joined[joined["budget_feasible_bool"]].copy()
        require(not feasible.empty, f"No feasible candidate for {budget}/{context}")
        feasible["exact_cost"] = [exact_cost(row._asdict(), int(context)) for row in feasible.itertuples(index=False)]
        feasible["exact_cost_float"] = feasible["exact_cost"].map(float)
        scenarios[(int(budget), int(context))] = feasible
    require(len(scenarios) == EXPECTED_SCENARIOS, "Scenario construction failed")
    return scenarios


def select_scenario(form: str, params: dict[str, float], candidates: pd.DataFrame) -> dict[str, Any]:
    predicted = predict_rows(form, params, candidates)
    require(np.isfinite(predicted).all(), f"Nonfinite Q3 loss predictions for {form}")
    work = candidates.copy()
    work["predicted_loss"] = predicted
    work = work.sort_values(
        ["predicted_loss", "exact_cost_float", "N_params_B", "D_tokens_B", "source_run_id"],
        kind="mergesort",
    )
    row = work.iloc[0]
    return {
        "selected_run_id": int(row["source_run_id"]),
        "selected_grid_id": int(row["grid_id"]),
        "selected_N_B": float(row["N_params_B"]),
        "selected_D_B": float(row["D_tokens_B"]),
        "predicted_loss": float(row["predicted_loss"]),
        "cost_flops": float(row["exact_cost_float"]),
        "feasible_grid_points": int(len(work)),
        "selected_trajectory_id": str(row["trajectory_id"]),
    }


def get_baseline_folds(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    loo = data["loo"]
    fits: dict[str, dict[str, float]] = {}
    for row in loo.to_dict(orient="records"):
        cluster = data["mapping_by_n"][n_key(row["held_out_N_params_B"])]
        fits[cluster] = current_parameters(row)
    return fits


def verify_frozen_m0_and_frontier(data: dict[str, Any], scenarios: dict[tuple[int, int], pd.DataFrame]) -> dict[str, Any]:
    params = {name: float(value) for name, value in data["base_meta"]["parameters"].items()}
    pred = predict_rows("m0_current", params, data["b1"])
    observed_prediction = data["b1"]["predicted_val_loss"].to_numpy(dtype=float)
    max_prediction_difference = float(np.max(np.abs(pred - observed_prediction)))
    require(max_prediction_difference <= 1e-8, f"Frozen M0 prediction mismatch: {max_prediction_difference}")
    selections = []
    frontier_lookup = {
        (int(str(row.budget_flops)), int(row.context_length_tokens)): int(row.selected_run_id)
        for row in data["frontier"].itertuples(index=False)
    }
    for key, frame in sorted(scenarios.items()):
        selected = select_scenario("m0_current", params, frame)
        expected = frontier_lookup[key]
        require(selected["selected_run_id"] == expected, f"Frozen M0 did not reproduce Q3 run_id for {key}: {selected['selected_run_id']} != {expected}")
        selections.append({"budget_flops": key[0], "context_length_tokens": key[1], "selected_run_id": selected["selected_run_id"]})
    return {"max_prediction_abs_difference": max_prediction_difference, "baseline_scenario_matches": len(selections), "baseline_scenario_total": EXPECTED_SCENARIOS}


def parameter_row(form: str, scope: str, omitted_track: str | None, params: dict[str, float], fit_info: dict[str, Any], training_rows: int) -> dict[str, Any]:
    return {
        "model_form": form,
        "formula": MODEL_FORMULAS[form],
        "fit_scope": scope,
        "omitted_trajectory": omitted_track or "",
        "training_rows": int(training_rows),
        "training_trajectories": EXPECTED_TRACKS if scope == "full" else EXPECTED_TRACKS - 1,
        "successful_starts": fit_info.get("successful_starts", "reused_authoritative_Q2_fit"),
        "attempted_starts": fit_info.get("attempted_starts", "reused_authoritative_Q2_fit"),
        "training_sse": fit_info.get("training_sse", ""),
        **{name: params.get(name, "") for name in ("E", "A", "alpha", "B", "beta", "gamma")},
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    rows = []
    for record in frame[columns].to_dict(orient="records"):
        values = []
        for value in record.values():
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}g}")
            else:
                values.append(str(value))
        rows.append(values)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    model_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    holdout: pd.DataFrame,
    frozen_check: dict[str, Any],
    elapsed: float,
) -> str:
    chosen_cols = [
        "model_form", "pooled_oof_rmse", "mean_track_rmse", "pooled_oof_mae",
        "pooled_oof_bias", "q3_full_fit_changed_scenarios_vs_current_m0",
        "loto_exact_match_rate_vs_current_full_m0",
    ]
    key_scenarios = scenario_summary[
        scenario_summary["budget_flops"] == scenario_summary["budget_flops"].max()
    ].copy()
    scenario_cols = [
        "model_form", "context_length_tokens", "unique_loo_selected_points",
        "modal_selection_frequency", "match_current_full_m0_rate",
        "excluded_track_selected_folds", "selected_N_B_min", "selected_N_B_max",
        "selected_D_B_min", "selected_D_B_max",
    ]
    worst_holdout = holdout.sort_values("rmse", ascending=False).groupby("model_form", as_index=False).head(1)
    worst_cols = ["model_form", "omitted_trajectory", "n", "rmse", "mae", "bias_pred_minus_obs"]
    return f"""# Q3 轨迹留一与函数形式敏感性

## 范围

分析固定 $Q=Q_0$，不包含 $p$，只在冻结 M0/B1 的 1,176 个观测 $(N,D)$ 候选点和现有 15 个预算×上下文情景内比较配置。每折整条留出一个 B12 核验的 Pythia 轨迹（147 个检查点）；其余 7 条轨迹用于拟合。留出 Loss 未用于该折参数拟合。决策候选坐标与成本仍固定使用 Q3 支持网格，因此当选点来自被留出轨迹时单独标记。

当前 M0 的逐规模留一参数取自既有 Q2 权威 `leave_one_size_out.csv`，并用其留出误差复核。两个预先限定的结构约束为：令 $E=0$；以及令 $\\alpha=\\beta$。比较的是受限敏感性，不是全面模型搜索。

## 冻结结果复核

- B1 行数/轨迹：1,176 / 8 条，每条 147 行。
- M4 成本掩码：17,640 行；精确复算成本可行性与掩码逐点一致。
- 当前 M0 预测与冻结预测最大绝对差：{frozen_check['max_prediction_abs_difference']:.3g}。
- 原 Q3 预算×上下文选点复现：{frozen_check['baseline_scenario_matches']}/{frozen_check['baseline_scenario_total']}。

## 留出轨迹预测

所有模型使用同样的 8 折。主要比较量是 pooled out-of-trajectory RMSE；同时报告每条留出轨迹的误差，避免把 1,176 个检查点误当成独立训练实验。最差单轨迹如下：

{_markdown_table(worst_holdout, worst_cols)}

## 函数形式与配置敏感性

{_markdown_table(model_summary, chosen_cols)}

最高预算下各上下文的留一选点范围：

{_markdown_table(key_scenarios, scenario_cols)}

逐折参数、留出预测、各情景选点和完整指标分别见本目录 CSV。小样本下不报告显著性检验；选点一致率是 8 个删除情景上的经验稳定性摘要，不是概率置信度。

## 解释限制

- 当前模型的参数留一已在 Q2 完成；本分析新增的是把这些拟合传播到 Q3 的离散预算决策，并与两种受限形式比较。
- Bootstrap 稳定不涵盖函数形式误差；这两种受限模型也不代表完整模型不确定性。交叉验证只覆盖 Pythia 同来源轨迹，不是独立模型族验证。
- M0 的 Q3 前沿仍固定 $Q=Q_0$，排除 $p$；本分析不识别质量或配比效应。
- 最高预算的选点如触及 B1 的 N/D 支持边界，不能外推为更大模型或更多数据量下的最优。
- 既有 B1 残差结构图已复核，本次不重复出图；留一预测误差为本次新增的轨迹外预测诊断。

计算耗时：{elapsed:.2f} s。唯一完整复现命令、输入/输出 SHA-256 与依赖版本见 `reproduction_manifest.json`。
"""


def run_p1(data: dict[str, Any], scenarios: dict[tuple[int, int], pd.DataFrame], output_dir: Path) -> dict[str, Any]:
    loo = data["loo"].copy()
    held_n = float(loo.sort_values("held_out_N_params_B").iloc[0]["held_out_N_params_B"])
    held_track = data["mapping_by_n"][n_key(held_n)]
    train = data["b1"][data["b1"]["trajectory_id"] != held_track].copy()
    test = data["b1"][data["b1"]["trajectory_id"] == held_track].copy()
    base_row = loo[np.isclose(loo["held_out_N_params_B"].astype(float), held_n)].iloc[0]
    params_by_form = {"m0_current": current_parameters(base_row)}
    fit_evidence: dict[str, Any] = {"m0_current": {"fit_source": "authoritative Q2 leave_one_size_out.csv"}}
    anchor = current_parameters(base_row)
    for form in ("m0_no_floor", "m0_shared_exponent"):
        params, info = fit_restricted_form(train, form, anchor, starts=4)
        params_by_form[form] = params
        fit_evidence[form] = info
    holdout_evidence = {}
    for form, params in params_by_form.items():
        holdout_evidence[form] = metrics(test["val_loss"].to_numpy(dtype=float), predict_rows(form, params, test))
    scenario_key = sorted(scenarios)[0]
    selections = {
        form: select_scenario(form, params, scenarios[scenario_key])
        for form, params in params_by_form.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "P1_SMOKE_PASS",
        "scope": "one real held-out trajectory, all three forms, one real Q3 scenario",
        "held_out_trajectory": held_track,
        "training_rows": len(train),
        "held_out_rows": len(test),
        "scenario": {"budget_flops": scenario_key[0], "context_length_tokens": scenario_key[1]},
        "fit_evidence": fit_evidence,
        "holdout_metrics": holdout_evidence,
        "scenario_selections": selections,
        "all_predictions_finite": all(math.isfinite(float(value["rmse"])) for value in holdout_evidence.values()),
        "all_selections_feasible": all(value["feasible_grid_points"] > 0 for value in selections.values()),
    }
    require(receipt["all_predictions_finite"] and receipt["all_selections_feasible"], "P1 smoke result has invalid predictions or selections")
    (output_dir / "p1_smoke_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def write_full_outputs(data: dict[str, Any], scenarios: dict[tuple[int, int], pd.DataFrame], started: float) -> dict[str, Any]:
    output_dir = RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    full_current = {name: float(value) for name, value in data["base_meta"]["parameters"].items()}
    current_folds = get_baseline_folds(data)
    full_fits: dict[str, dict[str, float]] = {"m0_current": full_current}
    full_fit_info: dict[str, dict[str, Any]] = {"m0_current": {"fit_source": "authoritative Q2 fit_parameters.json"}}
    for form in ("m0_no_floor", "m0_shared_exponent"):
        params, info = fit_restricted_form(data["b1"], form, full_current, starts=16)
        full_fits[form] = params
        full_fit_info[form] = info

    full_selection: dict[tuple[str, int, int], dict[str, Any]] = {}
    full_fit_metrics: dict[str, dict[str, Any]] = {}
    for form in MODEL_FORMS:
        pred = predict_rows(form, full_fits[form], data["b1"])
        full_fit_metrics[form] = metrics(data["b1"]["val_loss"].to_numpy(dtype=float), pred)
        for key, frame in scenarios.items():
            full_selection[(form, key[0], key[1])] = select_scenario(form, full_fits[form], frame)

    frontier_lookup = {
        (int(str(row.budget_flops)), int(row.context_length_tokens)): int(row.selected_run_id)
        for row in data["frontier"].itertuples(index=False)
    }
    parameter_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for form in MODEL_FORMS:
        parameter_rows.append(parameter_row(form, "full", None, full_fits[form], full_fit_info[form], len(data["b1"])))
    tracks = [track for _, track in data["tracks"]]
    for omitted_track in tracks:
        train = data["b1"][data["b1"]["trajectory_id"] != omitted_track].copy()
        test = data["b1"][data["b1"]["trajectory_id"] == omitted_track].copy()
        require(len(train) == EXPECTED_ROWS_PER_TRACK * (EXPECTED_TRACKS - 1), f"Unexpected training rows when omitting {omitted_track}")
        require(len(test) == EXPECTED_ROWS_PER_TRACK, f"Unexpected test rows when omitting {omitted_track}")
        base_row = data["loo"][data["loo"]["held_out_N_params_B"].map(n_key) == n_key(float(test["N_params_B"].iloc[0]))]
        require(len(base_row) == 1, f"Missing authoritative Q2 LOO parameters for {omitted_track}")
        base_params = current_parameters(base_row.iloc[0])
        fits_for_fold: dict[str, dict[str, float]] = {"m0_current": base_params}
        fit_info_for_fold: dict[str, dict[str, Any]] = {"m0_current": {"fit_source": "authoritative Q2 leave_one_size_out.csv"}}
        for form in ("m0_no_floor", "m0_shared_exponent"):
            params, info = fit_restricted_form(train, form, base_params, starts=16)
            fits_for_fold[form] = params
            fit_info_for_fold[form] = info

        train_n_min = float(train["N_params_B"].min())
        train_n_max = float(train["N_params_B"].max())
        for form in MODEL_FORMS:
            params = fits_for_fold[form]
            parameter_rows.append(parameter_row(form, "leave_one_trajectory_out", omitted_track, params, fit_info_for_fold[form], len(train)))
            predictions = predict_rows(form, params, test)
            score = metrics(test["val_loss"].to_numpy(dtype=float), predictions)
            if form == "m0_current":
                stored_rmse = float(base_row.iloc[0]["holdout_rmse"])
                require(abs(float(score["rmse"]) - stored_rmse) <= 1e-9, f"Q2 LOO holdout RMSE mismatch for {omitted_track}: {score['rmse']} vs {stored_rmse}")
            holdout_rows.append({"model_form": form, "formula": MODEL_FORMULAS[form], "omitted_trajectory": omitted_track, **score})
            for row_idx, row in enumerate(test.itertuples(index=False)):
                pred = float(predictions[row_idx])
                residual = pred - float(row.val_loss)
                prediction_rows.append({
                    "model_form": form,
                    "omitted_trajectory": omitted_track,
                    "run_id": int(row.run_id),
                    "N_params_B": float(row.N_params_B),
                    "D_tokens_B": float(row.D_tokens_B),
                    "observed_val_loss": float(row.val_loss),
                    "predicted_val_loss": pred,
                    "residual_pred_minus_obs": residual,
                    "outside_training_N_range": bool(float(row.N_params_B) < train_n_min or float(row.N_params_B) > train_n_max),
                })
            for key, feasible in scenarios.items():
                selected = select_scenario(form, params, feasible)
                baseline_full_run = frontier_lookup[key]
                same_form_full = full_selection[(form, key[0], key[1])]["selected_run_id"]
                selection_rows.append({
                    "model_form": form,
                    "omitted_trajectory": omitted_track,
                    "budget_flops": key[0],
                    "context_length_tokens": key[1],
                    "original_full_m0_run_id": baseline_full_run,
                    "same_form_full_fit_run_id": same_form_full,
                    **selected,
                    "matches_original_full_m0": selected["selected_run_id"] == baseline_full_run,
                    "matches_same_form_full_fit": selected["selected_run_id"] == same_form_full,
                    "selected_point_is_omitted_trajectory": selected["selected_trajectory_id"] == omitted_track,
                    "selected_N_outside_training_range": selected["selected_N_B"] < train_n_min or selected["selected_N_B"] > train_n_max,
                    "training_N_min_B": train_n_min,
                    "training_N_max_B": train_n_max,
                })

    parameter_frame = pd.DataFrame(parameter_rows)
    holdout_frame = pd.DataFrame(holdout_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    selection_frame = pd.DataFrame(selection_rows)
    require(len(holdout_frame) == EXPECTED_TRACKS * len(MODEL_FORMS), "Holdout result count mismatch")
    require(len(selection_frame) == EXPECTED_TRACKS * len(MODEL_FORMS) * EXPECTED_SCENARIOS, "LOO selection result count mismatch")
    require(len(prediction_frame) == EXPECTED_TRACKS * len(MODEL_FORMS) * EXPECTED_ROWS_PER_TRACK, "OOF prediction count mismatch")

    summary_rows = []
    for form in MODEL_FORMS:
        form_predictions = prediction_frame[prediction_frame["model_form"] == form]
        pooled = metrics(form_predictions["observed_val_loss"].to_numpy(dtype=float), form_predictions["predicted_val_loss"].to_numpy(dtype=float))
        per_track = holdout_frame[holdout_frame["model_form"] == form]
        form_selections = selection_frame[selection_frame["model_form"] == form]
        changed_full = sum(full_selection[(form, int(row.budget_flops), int(row.context_length_tokens))]["selected_run_id"] != frontier_lookup[(int(row.budget_flops), int(row.context_length_tokens))] for row in data["frontier"].itertuples(index=False))
        summary_rows.append({
            "model_form": form,
            "formula": MODEL_FORMULAS[form],
            "parameter_count": MODEL_PARAMETER_COUNTS[form],
            "full_fit_rmse_in_sample": full_fit_metrics[form]["rmse"],
            "pooled_oof_rmse": pooled["rmse"],
            "mean_track_rmse": float(per_track["rmse"].mean()),
            "median_track_rmse": float(per_track["rmse"].median()),
            "worst_track_rmse": float(per_track["rmse"].max()),
            "pooled_oof_mae": pooled["mae"],
            "pooled_oof_bias": pooled["bias_pred_minus_obs"],
            "pooled_oof_max_abs_error": pooled["max_abs_error"],
            "q3_full_fit_changed_scenarios_vs_current_m0": int(changed_full),
            "q3_full_fit_same_as_current_m0_scenarios": int(EXPECTED_SCENARIOS - changed_full),
            "loto_exact_match_count_vs_current_full_m0": int(form_selections["matches_original_full_m0"].sum()),
            "loto_exact_match_rate_vs_current_full_m0": float(form_selections["matches_original_full_m0"].mean()),
            "loto_exact_match_count_vs_same_form_full": int(form_selections["matches_same_form_full_fit"].sum()),
            "loto_exact_match_rate_vs_same_form_full": float(form_selections["matches_same_form_full_fit"].mean()),
        })
    model_summary = pd.DataFrame(summary_rows)

    scenario_rows = []
    for (form, budget, context), group in selection_frame.groupby(["model_form", "budget_flops", "context_length_tokens"], sort=True):
        counts = Counter(int(value) for value in group["selected_run_id"])
        modal_run, modal_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        scenario_rows.append({
            "model_form": form,
            "budget_flops": int(budget),
            "context_length_tokens": int(context),
            "original_full_m0_run_id": int(group["original_full_m0_run_id"].iloc[0]),
            "same_form_full_fit_run_id": int(group["same_form_full_fit_run_id"].iloc[0]),
            "unique_loo_selected_points": len(counts),
            "modal_selected_run_id": modal_run,
            "modal_selection_frequency": modal_count / EXPECTED_TRACKS,
            "match_current_full_m0_count": int(group["matches_original_full_m0"].sum()),
            "match_current_full_m0_rate": float(group["matches_original_full_m0"].mean()),
            "match_same_form_full_fit_count": int(group["matches_same_form_full_fit"].sum()),
            "match_same_form_full_fit_rate": float(group["matches_same_form_full_fit"].mean()),
            "excluded_track_selected_folds": int(group["selected_point_is_omitted_trajectory"].sum()),
            "selected_N_B_min": float(group["selected_N_B"].min()),
            "selected_N_B_max": float(group["selected_N_B"].max()),
            "selected_D_B_min": float(group["selected_D_B"].min()),
            "selected_D_B_max": float(group["selected_D_B"].max()),
            "selected_loss_min": float(group["predicted_loss"].min()),
            "selected_loss_max": float(group["predicted_loss"].max()),
        })
    scenario_summary = pd.DataFrame(scenario_rows)

    outputs = {
        "q3_loto_parameters.csv": parameter_frame,
        "q3_loto_holdout_metrics.csv": holdout_frame,
        "q3_loto_oof_predictions.csv": prediction_frame,
        "q3_loto_scenario_selections.csv": selection_frame,
        "q3_loto_scenario_summary.csv": scenario_summary,
        "q3_function_form_summary.csv": model_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig", float_format="%.12g")

    frozen_check = verify_frozen_m0_and_frontier(data, scenarios)
    elapsed = time.perf_counter() - started
    report = build_report(model_summary, scenario_summary, holdout_frame, frozen_check, elapsed)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "analysis": "Q3 trajectory leave-one-out decision sensitivity and restricted scaling-form comparison",
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "fixed Q=Q0; p excluded; 8 whole Pythia trajectory folds; 1,176 observed B1 N/D support points; 15 budget-context scenarios",
        "models": MODEL_FORMULAS,
        "random_seed": None,
        "optimizer": {"method": "scipy.optimize.least_squares(method='trf', x_scale='jac')", "alternative_form_starts": 16, "max_nfev": 10000},
        "runtime_seconds": elapsed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "command": "D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/analyze_q3_trajectory_form_sensitivity.py",
        "input_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in INPUTS.values()},
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_m0_reconciliation": frozen_check,
        "row_counts": {name: int(len(frame)) for name, frame in outputs.items()},
        "output_sha256": {
            name: sha256_file(output_dir / name)
            for name in [*outputs, "report.md"]
        },
        "limitations": [
            "Only eight trajectory clusters are available; no inferential test is reported.",
            "LOO decision candidates retain the full fixed Q3 support grid; a selected omitted-trajectory point is flagged.",
            "The two restricted models do not exhaust model-form uncertainty.",
            "All Q3 outputs remain fixed-Q0 M0 reference results; p and Q response are not identified here.",
        ],
    }
    (output_dir / "reproduction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"manifest": manifest, "model_summary": model_summary, "scenario_summary": scenario_summary, "holdout": holdout_frame}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", action="store_true", help="Run a real-data single-fold/one-scenario smoke slice only")
    parser.add_argument("--p1-output", type=Path, help="Optional P1 receipt directory, relative to project root unless absolute")
    args = parser.parse_args()
    started = time.perf_counter()
    data = load_inputs()
    scenarios = candidate_scenarios(data)
    frozen_check = verify_frozen_m0_and_frontier(data, scenarios)
    if args.p1:
        output = args.p1_output or (Path("Q3/04_结果/轨迹留一与函数形式敏感性/_p1_smoke"))
        if not output.is_absolute():
            output = ROOT / output
        receipt = run_p1(data, scenarios, output)
        print(json.dumps({"p1_output": output.relative_to(ROOT).as_posix(), "frozen_check": frozen_check, "receipt": receipt}, ensure_ascii=False, indent=2))
        return
    result = write_full_outputs(data, scenarios, started)
    print("Completed Q3 trajectory/form sensitivity.")
    print(result["model_summary"].to_string(index=False))
    print(f"Outputs: {(RESULT_DIR.relative_to(ROOT)).as_posix()}")


if __name__ == "__main__":
    main()
