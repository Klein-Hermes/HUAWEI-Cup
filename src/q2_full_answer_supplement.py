"""Q2 full-answer supplement: semi-synthetic Q audit and large-scale scenario.

This module does not change the frozen B1 M0 fit, refit on B9/B10, or claim
that semi-synthetic rows are real observations. It audits the permitted uses
of B6-B10 and records a conditional Q response from B6/B7 only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from q2_a_side_complementarity_sensitivity import run_analysis as run_a_side_complementarity
from q2_full_answer_figures import FIGURE_SKILL_ROOT, create_q2_full_answer_figures


ROOT = Path(__file__).resolve().parents[1]
ATT = ROOT / "中文题目" / "F题" / "real_attachments" / "B_scaling_laws"
BASELINE = ROOT / "results" / "q2_scaling_baseline" / "v1"
Q13 = ROOT / "results" / "q1_3" / "v1"
Q13_PQ = ROOT / "results" / "q1_3" / "pq_extension_v1"
OUT = ROOT / "results" / "q2_full_answer_supplement" / "v1"
Q2_OUT = ROOT / "Q2" / "03_结果" / "完整回答补充" / "v1"
DOCS = ROOT / "docs"
Q1_EVIDENCE = {
    "q1_3_holdout_metrics.csv": Q13 / "holdout_metrics.csv",
    "q1_3_substitution_effects_10pp_bootstrap.csv": Q13 / "substitution_effects_10pp_bootstrap.csv",
    "q1_3_pq_model_summary.csv": Q13_PQ / "model_summary.csv",
    "q1_3_pq_paired_comparison.csv": Q13_PQ / "paired_comparison.csv",
    "q1_3_run_report.md": ROOT / "Q1" / "03_结果" / "Q1.3" / "v1" / "q1_3_run_report.md",
    "q1_3_pq_extension_report.md": ROOT / "Q1" / "03_结果" / "Q1.3" / "pq_extension_v1" / "q1_3_pq_extension_report.md",
}
REFERENCE_EVIDENCE = {
    "m0_fit_parameters.json": BASELINE / "fit_parameters.json",
    "m0_validation_metrics_by_source.csv": BASELINE / "validation_metrics_by_source.csv",
    "m0_validation_comparability.json": BASELINE / "validation_comparability.json",
    "m0_marginal_effects_report.md": ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "q2_m0_marginal_effects_report.md",
    "m0_validation_scope_closeout.md": ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "q2_validation_scope_closeout.md",
    "b3_validation_closeout.md": ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "b3_validation_closeout.md",
    "b4_b5_loss_comparability_audit.md": ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "b4_b5_loss_comparability_audit.md",
    "loss_protocol_evidence_sources.md": ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "loss_protocol_evidence_sources.md",
    "loss_protocol_matrix.csv": ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "loss_protocol_matrix.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_identifiability_proof(p_audit: dict[str, object],
                                q_audit: pd.DataFrame) -> str:
    q_huber = q_audit[q_audit["q_scenario"] == "q_huber_candidate"].copy()
    q_deterministic = int(q_audit["qcovered_is_deterministic_function_of_p"].astype(bool).sum())
    q_scenarios = int(len(q_audit))
    q_huber_deterministic = int(q_huber["qcovered_is_deterministic_function_of_p"].astype(bool).sum())
    q_huber_scenarios = int(len(q_huber))
    direct_huber = q_huber[q_huber["mapping_scenario"] == "direct_only"].iloc[0]
    p_vectors = int(p_audit["verified_p_vectors_joined_to_loss"])
    p_rows = int(p_audit["loss_rows_with_p_joined"])
    b1 = p_audit["B1"]
    return "\n".join([
        "# Q2 M1/M2 当前数据不可识别性的代数说明",
        "",
        "状态：当前 B1–B5 不具备正式估计 M1/M2 的联合观测条件。本文件证明在已审计的数据结构下，缺少变化并非只影响精度，而是会造成参数合并或效应分解不唯一。此结论不表示 p 或 Q 没有作用。",
        "",
        "## 审计到的设计事实",
        "",
        f"- B1 有 {int(b1['rows'])} 个检查点、{int(b1['model_trajectories'])} 条模型轨迹；8 个模型规模共享同一训练数据顺序政策（1 个政策组），项目 17 域数值配方向量为 {int(b1['formal_numeric_recipe_vectors'])} 个。",
        f"- B1–B5 经核验并连接到 Loss 的 p 向量为 {p_vectors} 个，含 p 的 Loss 行为 {p_rows} 行；本地 B1 轨迹日志没有按检查点的来源域 token 累计计数，因此 p(D) 未重建。",
        f"- Q1.3 的 `identifiability_audit.csv` 共审计 {q_scenarios} 个映射/Q 情景，其中 {q_deterministic} 个将 `Q_covered` 判定为 p 的确定性函数；正式 `q_huber_candidate` 情景为 {q_huber_deterministic}/{q_huber_scenarios}。这些 A 侧派生特征不能替代 B 侧独立 Q 观测。",
        f"- 限定的 direct-only `q_huber_candidate` 设计矩阵秩从 {int(direct_huber['qcovered_rank_without_q'])} 增至 {int(direct_huber['qcovered_rank_with_q'])}。这允许该特定参数化把 `Q_covered` 当作额外预测基函数，但秩增加本身不构成独立 Q 变化或因果增量证据；见 Q1.3 嵌套 OOF 比较。",
        "",
        "## M1：一个固定配方不能识别 p 增量",
        "",
        "M1 在题目分析报告中的限制形式为：",
        "",
        "`L = E + A N^(-alpha) + B D^(-beta) exp[-beta g_p(p)]`. ",
        "",
        "若所有观测共享同一个固定配方 `p_0`，令 `B_star = B exp[-beta g_p(p_0)]`，则：",
        "",
        "`L = E + A N^(-alpha) + B_star D^(-beta)`,",
        "",
        "它与 M0 完全同形。数据只能估计合并后的 `B_star`，不能把 p 的贡献与基线数据项系数 `B` 分开，也不能从一个配方估计 17 域之间的替代效应。即使将来补齐共同 `p_recipe` 的数值，该静态单配方也只能确认配方身份，不能单独打开 p-only 消融的识别门。",
        "",
        "若考虑累计组成 `p(D)`，它只有在被真实 token 流重建且提供超出 `D` 的、预先规定形式可检验的变化时才可能增加信息。若各轨迹在同一 `D` 共享同一个 `p(D)`，那么观测只沿单一组成路径变化；任意未约束的 `g_p` 与 D 响应不能作一般性分解。当前没有 checkpoint 级源 token 计数，因此不能构造这种路径、计算共线性或估计 M1。",
        "",
        "## M2：若 Q 是 p 的确定函数，p/Q 分解不唯一",
        "",
        "M2 主效应形式为 `h(p,Q)=g_p(p)+g_Q(Q)`。若可用质量量 `Q=q(p)` 完全由配比决定，则在不限制 `g_p` 与 `g_Q` 函数空间时，对任意函数 `a(Q)`，变换：",
        "",
        "`g_p_new(p)=g_p(p)+a(q(p))`,  `g_Q_new(Q)=g_Q(Q)-a(Q)`",
        "",
        "不会改变任何观测到的 `h`。因此在一般函数层面，预测总和不能唯一拆成 `g_p` 与独立 `g_Q`。强加有限维参数形式后，系数有时可凭设计矩阵秩差估计；那只识别所选基函数参数化，不证明 Q 有独立于 p 的变化或因果增量。当前 B1–B5 连行级 p/Q 联合观测都没有，A 侧 `Q_covered(p)` 只用于预测特征敏感性，B6/B7 仅为半合成条件响应。",
        "",
        "## 能改变判定的数据",
        "",
        "1. 提供 B 侧精确模型/训练运行到 17 域的 `p_recipe` 配置或 checkpoint 级来源 token 计数，区分静态配方与累计 `p(D)`，并能映射到 Loss 行。",
        "2. 提供多个相互独立且可比的训练配方，使其在控制 N、D、模型族和来源后仍有变异；若只恢复 B1 的一个共同静态配方，M1 仍与 M0 等价。",
        "3. 提供同一 B 运行/检查点上的版本化 Q，并让 Q 在控制 p、N、D 后仍有独立变化；同时建立 B4/B5 与 B1 的 Loss 评测量尺可比性。",
        "4. 通过上述门禁后，按 M0→M1→M2 的嵌套顺序拟合与验证；只有再有数据支持时才考虑唯一受限交互扩展。",
        "",
        "## 证据来源与范围",
        "",
        "数值事实来自 `q2_p_audit_summary.json`、`q2_p_feasibility_audit.md` 与 Q1.3 `identifiability_audit.csv`；题意和 M0/M1/M2 层级以 `Q2/01_方案说明/题目分析报告.md` 的可见内容为准。Pythia 官方模型卡说明不同规模使用相同数据和顺序（[Pythia-70m model card](https://huggingface.co/EleutherAI/pythia-70m)）；本证明仍不假设未经测量的数值 p(D)，也不把 A 侧或半合成行拼接成 B 侧联合样本。",
        "",
    ])


def predict(theta: np.ndarray, n: np.ndarray, d: np.ndarray, q: np.ndarray,
            with_q: bool) -> np.ndarray:
    if with_q:
        e, log_a, alpha, log_b, beta, kappa = theta
    else:
        e, log_a, alpha, log_b, beta = theta
        kappa = 0.0
    return e + np.exp(log_a) * n ** (-alpha) + np.exp(log_b) * d ** (-beta) * q ** (-kappa)


def fit_response(frame: pd.DataFrame, with_q: bool, return_diagnostics: bool = False):
    n = frame["N_params_B"].to_numpy(float)
    d = frame["D_tokens_B"].to_numpy(float)
    q = frame["Q_score"].to_numpy(float)
    y = frame["val_loss"].to_numpy(float)
    upper_e = float(y.min()) - 1e-7
    results = []
    # Fixed starts make the diagnostic repeatable and expose local instability.
    for e_fraction in (0.05, 0.35, 0.7):
        for exponent in (0.08, 0.25, 0.55):
            e0 = max(0.0, min(upper_e * e_fraction, upper_e - 1e-8))
            rem = max(float(np.median(y)) - e0, 1e-4)
            a0 = np.log(max(1e-8, 0.5 * rem * float(np.median(n)) ** exponent))
            b0 = np.log(max(1e-8, 0.5 * rem * float(np.median(d)) ** exponent))
            x0 = [e0, a0, exponent, b0, exponent]
            lower = [0.0, -30.0, 1e-4, -30.0, 1e-4]
            upper = [upper_e, 30.0, 2.0, 30.0, 2.0]
            kappa_starts = (0.05, 0.35, 1.2) if with_q else (None,)
            for kappa_start in kappa_starts:
                start = list(x0)
                lo = list(lower)
                hi = list(upper)
                if with_q:
                    start.append(kappa_start)
                    lo.append(1e-5)
                    hi.append(4.0)
                result = least_squares(
                    lambda t: predict(t, n, d, q, with_q) - y,
                    np.asarray(start), bounds=(np.asarray(lo), np.asarray(hi)),
                    x_scale="jac", max_nfev=3000,
                )
                results.append(result)
    valid = [r for r in results if r.success and np.isfinite(r.x).all() and np.isfinite(r.cost)]
    if not valid:
        raise RuntimeError("Q response model did not converge from fixed starts")
    best = min(valid, key=lambda r: r.cost)
    if not return_diagnostics:
        return best.x
    near = [r for r in valid if r.cost <= best.cost * (1.0 + 1e-5) + 1e-15]
    spread = np.ptp(np.vstack([r.x for r in near]), axis=0) / np.maximum(np.abs(best.x), 1e-8)
    return best.x, {
        "attempts": len(results), "successful": len(valid), "near_optimal": len(near),
        "near_optimal_cost_relative_tolerance": 1e-5,
        "maximum_relative_parameter_spread": float(np.max(spread)),
        "relative_parameter_spread": [float(v) for v in spread],
    }


def cv_leave_q_level(frame: pd.DataFrame, with_q: bool) -> dict[str, float | int]:
    preds = np.full(len(frame), np.nan, dtype=float)
    q_values = sorted(frame["Q_score"].unique())
    for q_value in q_values:
        test_mask = frame["Q_score"].to_numpy() == q_value
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        theta = fit_response(train, with_q)
        preds[test.index.to_numpy()] = predict(
            theta, test["N_params_B"].to_numpy(float),
            test["D_tokens_B"].to_numpy(float),
            test["Q_score"].to_numpy(float), with_q,
        )
    y = frame["val_loss"].to_numpy(float)
    error = preds - y
    ss_total = float(np.sum((y - y.mean()) ** 2))
    return {
        "held_out_q_levels": len(q_values),
        "n_predictions": int(np.isfinite(preds).sum()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error**2) / ss_total),
    }


def q_direction_audit() -> tuple[pd.DataFrame, dict[str, object]]:
    files = {
        "B6": "supplementary_NQ_experiment.csv",
        "B7": "supplementary_NQ_experiment_expanded.csv",
        "B8": "supplementary_NQ_experiment_large.csv",
    }
    frames = {key: pd.read_csv(ATT / filename) for key, filename in files.items()}
    out_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for label in ("B6", "B7"):
        frame = frames[label]
        if frame[["N_params_B", "D_tokens_B", "Q_score", "val_loss"]].isna().any().any():
            raise ValueError(f"{label} required N,D,Q,Loss field is missing")
        matched_slopes = []
        for (n, d), cell in frame.groupby(["N_params_B", "D_tokens_B"]):
            if cell["Q_score"].nunique() > 1:
                matched_slopes.append(float(np.polyfit(cell["Q_score"], cell["val_loss"], 1)[0]))
        q_mean = frame.groupby("Q_score")["val_loss"].mean()
        out_rows.append({
            "dataset": label, "n": len(frame), "data_type": "semi_synthetic",
            "n_q_levels": int(frame["Q_score"].nunique()),
            "n_matched_ND_cells": len(matched_slopes),
            "matched_slope_negative": int(np.sum(np.asarray(matched_slopes) < 0)),
            "matched_slope_zero": int(np.sum(np.asarray(matched_slopes) == 0)),
            "matched_slope_positive": int(np.sum(np.asarray(matched_slopes) > 0)),
            "median_matched_loss_slope_per_Q": float(np.median(matched_slopes)),
            "pooled_spearman_Q_loss": float(spearmanr(frame["Q_score"], frame["val_loss"]).statistic),
            "mean_loss_lowest_Q": float(q_mean.iloc[0]),
            "mean_loss_highest_Q": float(q_mean.iloc[-1]),
        })
        for with_q, model_name in ((False, "M0_ND"), (True, "M_Q_NDQ")):
            if with_q:
                theta, fit_diag = fit_response(frame, with_q, return_diagnostics=True)
            else:
                theta = fit_response(frame, with_q)
                fit_diag = {}
            in_pred = predict(theta, frame["N_params_B"].to_numpy(float),
                              frame["D_tokens_B"].to_numpy(float),
                              frame["Q_score"].to_numpy(float), with_q)
            cv = cv_leave_q_level(frame, with_q)
            row = {
                "dataset": label, "model": model_name, "n": len(frame),
                "in_sample_rmse": float(np.sqrt(np.mean((in_pred-frame["val_loss"].to_numpy(float))**2))),
                "loo_q_rmse": cv["rmse"], "loo_q_mae": cv["mae"], "loo_q_r2": cv["r2"],
                "loo_q_levels": cv["held_out_q_levels"], "loo_q_predictions": cv["n_predictions"],
                "E": float(theta[0]), "A": float(np.exp(theta[1])), "alpha": float(theta[2]),
                "B": float(np.exp(theta[3])), "beta": float(theta[4]),
                "kappa_Q_exponent": float(theta[5]) if with_q else 0.0,
                "gamma_Q_in_Deff": float(theta[5] / theta[4]) if with_q else 0.0,
                "multistart_attempts": fit_diag.get("attempts"),
                "multistart_successful": fit_diag.get("successful"),
                "multistart_near_optimal": fit_diag.get("near_optimal"),
                "multistart_max_relative_parameter_spread": fit_diag.get("maximum_relative_parameter_spread"),
                "interpretation": "semi-synthetic conditional response; not a real-trial Q effect",
            }
            if with_q:
                log_b = theta[3]
                beta = theta[4]
                kappa = theta[5]
                q_values = frame["Q_score"].to_numpy(float)
                fitted_term = np.exp(log_b) * frame["D_tokens_B"].to_numpy(float) ** (-beta) * q_values ** (-kappa)
                fitted_loss = predict(theta, frame["N_params_B"].to_numpy(float),
                                     frame["D_tokens_B"].to_numpy(float), q_values, True)
                row.update({
                    "dL_dQ_min": float(np.min(-kappa * fitted_term / q_values)),
                    "dL_dQ_median": float(np.median(-kappa * fitted_term / q_values)),
                    "dL_dQ_max": float(np.max(-kappa * fitted_term / q_values)),
                    "epsilon_Q_min": float(np.min(-kappa * fitted_term / fitted_loss)),
                    "epsilon_Q_median": float(np.median(-kappa * fitted_term / fitted_loss)),
                    "epsilon_Q_max": float(np.max(-kappa * fitted_term / fitted_loss)),
                })
            fit_rows.append(row)

    b6 = frames["B6"].merge(
        frames["B7"], on=["experiment_id", "N_params_B", "D_tokens_B", "Q_score"],
        how="left", suffixes=("_b6", "_b7"), indicator=True,
    )
    b6_reuse = {
        "B6_rows": int(len(frames["B6"])),
        "B6_rows_also_in_B7": int((b6["_merge"] == "both").sum()),
        "shared_rows_with_changed_loss": int((
            (b6["_merge"] == "both") &
            (np.abs(b6["val_loss_b6"] - b6["val_loss_b7"]) > 1e-12)
        ).sum()),
        "B7_extra_rows": int(len(frames["B7"]) - (b6["_merge"] == "both").sum()),
    }

    b8 = frames["B8"]
    for data_type, frame in b8.groupby("data_type"):
        matched_slopes = []
        for _, cell in frame.groupby(["N_params_B", "D_tokens_B"]):
            if cell["Q_score"].nunique() > 1:
                matched_slopes.append(float(np.polyfit(cell["Q_score"], cell["val_loss"], 1)[0]))
        q_mean = frame.groupby("Q_score")["val_loss"].mean()
        out_rows.append({
            "dataset": "B8", "n": len(frame), "data_type": str(data_type),
            "n_q_levels": int(frame["Q_score"].nunique()),
            "n_matched_ND_cells": len(matched_slopes),
            "matched_slope_negative": int(np.sum(np.asarray(matched_slopes) < 0)),
            "matched_slope_zero": int(np.sum(np.asarray(matched_slopes) == 0)),
            "matched_slope_positive": int(np.sum(np.asarray(matched_slopes) > 0)),
            "median_matched_loss_slope_per_Q": float(np.median(matched_slopes)),
            "pooled_spearman_Q_loss": float(spearmanr(frame["Q_score"], frame["val_loss"]).statistic),
            "mean_loss_lowest_Q": float(q_mean.iloc[0]),
            "mean_loss_highest_Q": float(q_mean.iloc[-1]),
        })
    return pd.DataFrame(out_rows), {"fit_rows": fit_rows, "B6_B7_overlap": b6_reuse}


def large_scale_projection() -> tuple[pd.DataFrame, dict[str, object]]:
    params = json.loads((BASELINE / "fit_parameters.json").read_text(encoding="utf-8"))["parameters"]
    boot = pd.read_csv(BASELINE / "cluster_bootstrap_estimates.csv")
    large = pd.read_csv(ATT / "supplementary_large_baseline.csv")
    models = pd.read_csv(ATT / "supplementary_large_models.csv")
    b1 = pd.read_csv(ATT / "pythia_training_log_existing.csv")
    n = large["N_params_B"].to_numpy(float)
    d = large["D_tokens_B"].to_numpy(float)
    yhat = params["E"] + params["A"] * n ** (-params["alpha"]) + params["B"] * d ** (-params["beta"])
    bpred = (
        boot["E"].to_numpy()[:, None]
        + boot["A"].to_numpy()[:, None] * n[None, :] ** (-boot["alpha"].to_numpy()[:, None])
        + boot["B"].to_numpy()[:, None] * d[None, :] ** (-boot["beta"].to_numpy()[:, None])
    )
    result = large.copy()
    result["M0_predicted_loss"] = yhat
    result["M0_parameter_bootstrap_p025"] = np.quantile(bpred, 0.025, axis=0)
    result["M0_parameter_bootstrap_p975"] = np.quantile(bpred, 0.975, axis=0)
    result["estimated_minus_M0_loss"] = result["val_loss"] - result["M0_predicted_loss"]
    meta = models[["model_name", "FLOPs", "publication_date", "accessibility"]].copy()
    joined = result.merge(meta, left_on="family", right_on="model_name", how="left", validate="one_to_one")
    summary = {
        "n_B10_rows": int(len(result)),
        "n_converged_rows": int(result["is_converged"].sum()),
        "n_exact_B9_model_matches": int(joined["model_name"].notna().sum()),
        "n_N_above_B1_support": int((n > b1["N_params_B"].max()).sum()),
        "n_D_above_B1_support": int((d > b1["D_tokens_B"].max()).sum()),
        "B1_N_max_B": float(b1["N_params_B"].max()),
        "B1_D_max_B": float(b1["D_tokens_B"].max()),
        "B10_N_min_B": float(n.min()), "B10_N_max_B": float(n.max()),
        "B10_D_min_B": float(d.min()), "B10_D_max_B": float(d.max()),
        "prediction_median": float(np.median(yhat)),
        "prediction_range": [float(np.min(yhat)), float(np.max(yhat))],
        "parameter_bootstrap_median_width": float(np.median(
            result["M0_parameter_bootstrap_p975"] - result["M0_parameter_bootstrap_p025"]
        )),
        "warning": "B10 is an estimated/curated table and all rows exceed B1 N support; parameter bootstrap omits extrapolation/model-form and protocol error.",
    }
    return joined, summary


def q_n_equal_loss_demo(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a finite Q-to-N equal-Loss trade at a shared semi-synthetic grid point."""
    files = {
        "B6": "supplementary_NQ_experiment.csv",
        "B7": "supplementary_NQ_experiment_expanded.csv",
    }
    rows: list[dict[str, object]] = []
    for dataset, filename in files.items():
        frame = pd.read_csv(ATT / filename)
        levels = {
            name: np.sort(frame[name].dropna().unique().astype(float))
            for name in ("N_params_B", "D_tokens_B", "Q_score")
        }
        n0 = float(levels["N_params_B"][len(levels["N_params_B"]) // 2])
        d0 = float(levels["D_tokens_B"][len(levels["D_tokens_B"]) // 2])
        q0 = float(levels["Q_score"][len(levels["Q_score"]) // 2])
        q1 = q0 + 0.1
        if q1 > float(levels["Q_score"].max()):
            raise ValueError(f"{dataset} does not support the fixed +0.1 Q illustration")
        fit = fit_df[(fit_df["dataset"] == dataset) & (fit_df["model"] == "M_Q_NDQ")].iloc[0]
        e, a, alpha, b, beta, kappa = (
            float(fit[key]) for key in ("E", "A", "alpha", "B", "beta", "kappa_Q_exponent")
        )
        def loss(n: float, q: float) -> float:
            return e + a * n ** (-alpha) + b * d0 ** (-beta) * q ** (-kappa)
        loss0 = loss(n0, q0)
        residual_power = loss0 - e - b * d0 ** (-beta) * q1 ** (-kappa)
        if residual_power <= 0:
            raise ValueError(f"{dataset} equal-Loss N solution is outside the fitted model domain")
        n1 = (a / residual_power) ** (1.0 / alpha)
        scale_term = a * n0 ** (-alpha)
        quality_term = b * d0 ** (-beta) * q0 ** (-kappa)
        derivative = -(kappa / alpha) * (quality_term / scale_term) * (n0 / q0)
        rows.append({
            "dataset": dataset,
            "interpretation": "illustrative fitted equal-Loss trade on semi-synthetic grid; not a real-trial resource substitution rate",
            "reference_N_B": n0,
            "reference_D_B": d0,
            "reference_Q_score": q0,
            "reference_fitted_Loss": loss0,
            "Q_after_plus_0p1": q1,
            "N_equal_Loss_after_B": n1,
            "delta_N_B": n1 - n0,
            "local_dN_dQ_B_per_Q": derivative,
            "finite_delta_N_per_plus_0p1_B": n1 - n0,
            "gamma_Q_in_Deff": kappa / beta,
        })
    return pd.DataFrame(rows)


def build_report(qdir: pd.DataFrame, qdetails: dict[str, object],
                 large: pd.DataFrame, large_summary: dict[str, object],
                 qn_demo: pd.DataFrame, a_comp_summary: pd.DataFrame,
                 a_comp_manifest: dict[str, object],
                 figure_manifest: dict[str, object]) -> str:
    m0 = json.loads((BASELINE / "fit_parameters.json").read_text(encoding="utf-8"))
    p_gate = json.loads((ROOT / "Q2" / "03_结果" / "p配比可行性审计" / "v1" / "q2_p_audit_summary.json").read_text(encoding="utf-8"))
    validation = pd.read_csv(BASELINE / "validation_metrics_by_source.csv")
    b1 = pd.read_csv(ATT / "pythia_training_log_existing.csv")
    boot = pd.read_csv(BASELINE / "cluster_bootstrap_estimates.csv")
    n_ref = float(b1["N_params_B"].median())
    d_ref = float(b1["D_tokens_B"].median())
    e0, a0, alpha0, b0, beta0 = (float(m0["parameters"][k]) for k in ("E", "A", "alpha", "B", "beta"))
    loss_ref = e0 + a0 * n_ref ** (-alpha0) + b0 * d_ref ** (-beta0)
    effects_ref = {
        "dL_dN": -alpha0 * a0 * n_ref ** (-alpha0 - 1.0),
        "dL_dD": -beta0 * b0 * d_ref ** (-beta0 - 1.0),
        "epsilon_N": -alpha0 * a0 * n_ref ** (-alpha0) / loss_ref,
        "epsilon_D": -beta0 * b0 * d_ref ** (-beta0) / loss_ref,
    }
    boot_predictions = (boot["E"].to_numpy() + boot["A"].to_numpy() * n_ref ** (-boot["alpha"].to_numpy())
                       + boot["B"].to_numpy() * d_ref ** (-boot["beta"].to_numpy()))
    boot_effects = {
        "dL_dN": -boot["alpha"].to_numpy() * boot["A"].to_numpy() * n_ref ** (-boot["alpha"].to_numpy() - 1.0),
        "dL_dD": -boot["beta"].to_numpy() * boot["B"].to_numpy() * d_ref ** (-boot["beta"].to_numpy() - 1.0),
        "epsilon_N": -boot["alpha"].to_numpy() * boot["A"].to_numpy() * n_ref ** (-boot["alpha"].to_numpy()) / boot_predictions,
        "epsilon_D": -boot["beta"].to_numpy() * boot["B"].to_numpy() * d_ref ** (-boot["beta"].to_numpy()) / boot_predictions,
    }
    effect_intervals = {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in boot_effects.items()}
    q1_metrics = pd.read_csv(Q13 / "holdout_metrics.csv")
    cross = q1_metrics[(q1_metrics["model"] == "simplex_ridge") & q1_metrics["split"].isin(["test_1m", "test_60m", "test_1b"]) & (q1_metrics["variant"] == "closed")]
    cross_r2 = cross.groupby("split")["r2"].mean().to_dict()
    pq_summary = pd.read_csv(Q13_PQ / "model_summary.csv")
    pq_paired = pd.read_csv(Q13_PQ / "paired_comparison.csv")
    q_identifiability = pd.read_csv(Q13_PQ / "identifiability_audit.csv")
    direct_q_ident = q_identifiability[
        (q_identifiability["q_scenario"] == "q_huber_candidate")
        & (q_identifiability["mapping_scenario"] == "direct_only")
    ].iloc[0]
    p_sub = pd.read_csv(Q13 / "substitution_effects_10pp_bootstrap.csv")
    supported = p_sub[p_sub["training_recipes_with_from_share_at_least_10pp"] > 0]
    supported_lo = supported["bootstrap_ci95_lower"]
    supported_hi = supported["bootstrap_ci95_upper"]
    p_sub_summary = {
        "all_domain_pairs": int(p_sub[["from_domain", "to_domain"]].drop_duplicates().shape[0]),
        "supported_domain_pairs": int(supported[["from_domain", "to_domain"]].drop_duplicates().shape[0]),
        "supported_target_pair_intervals": int(len(supported)),
        "ci_fully_positive": int(((supported_lo > 0) & (supported_hi > 0)).sum()),
        "ci_fully_negative": int(((supported_lo < 0) & (supported_hi < 0)).sum()),
        "ci_cross_zero": int(((supported_lo <= 0) & (supported_hi >= 0)).sum()),
    }
    qh = pq_summary[(pq_summary["split"] == "train_1m_nested_cv") &
                    (pq_summary["mapping_scenario"] == "direct_only") &
                    (pq_summary["q_scenario"] == "q_huber_candidate") &
                    (pq_summary["model"] == "p_plus_Qcovered")].iloc[0]
    quadratic = pq_summary[(pq_summary["split"] == "train_1m_nested_cv") &
                           (pq_summary["mapping_scenario"] == "direct_only") &
                           (pq_summary["q_scenario"] == "shared_baseline") &
                           (pq_summary["model"] == "p_only_quadratic")].iloc[0]
    pq_diff_nested = float(qh["nested_macro_normalized_rmse"] - quadratic["nested_macro_normalized_rmse"])
    paired_qh = pq_paired[(pq_paired["split"] == "train_1m_nested_cv") &
                          (pq_paired["mapping_scenario"] == "direct_only") &
                          (pq_paired["q_scenario"] == "q_huber_candidate") &
                          (pq_paired["model"] == "p_plus_Qcovered")].iloc[0]
    p_gate = json.loads((ROOT / "Q2" / "03_结果" / "p配比可行性审计" / "v1" / "q2_p_audit_summary.json").read_text(encoding="utf-8"))
    if p_gate["verified_p_vectors_joined_to_loss"] != 0 or p_gate["loss_rows_with_p_joined"] != 0:
        raise ValueError("Current p gate counts changed; re-audit before generating this supplement")
    fit_df = pd.DataFrame(qdetails["fit_rows"])
    m0_b6 = fit_df[(fit_df.dataset == "B6") & (fit_df.model == "M0_ND")].iloc[0]
    mq_b6 = fit_df[(fit_df.dataset == "B6") & (fit_df.model == "M_Q_NDQ")].iloc[0]
    m0_b7 = fit_df[(fit_df.dataset == "B7") & (fit_df.model == "M0_ND")].iloc[0]
    mq_b7 = fit_df[(fit_df.dataset == "B7") & (fit_df.model == "M_Q_NDQ")].iloc[0]
    trade_b6 = qn_demo[qn_demo.dataset == "B6"].iloc[0]
    trade_b7 = qn_demo[qn_demo.dataset == "B7"].iloc[0]
    b6b7 = qdetails["B6_B7_overlap"]
    b8 = qdir[qdir.dataset == "B8"]
    b6 = qdir[qdir.dataset == "B6"].iloc[0]
    b7 = qdir[qdir.dataset == "B7"].iloc[0]
    a_comp_counts = {
        column: int(a_comp_summary[column].sum())
        for column in (
            "jointly_feasible_direction_pairs",
            "bootstrap_ci_fully_negative_unadjusted",
            "bootstrap_ci_fully_positive_unadjusted",
            "bootstrap_ci_cross_zero_unadjusted",
            "bootstrap_maxT95_fully_negative",
            "bootstrap_maxT95_fully_positive",
            "bootstrap_maxT95_cross_zero",
        )
    }
    a_comp_limits = a_comp_manifest["maxT95_conditional_candidate_set_counts"]

    lines = [
        "# Q2 完整回答补充：质量响应、组成效应与大尺度外推",
        "",
        "版本：v1；用途：补充 Q2 证据链和可计算解析条件。该文件不将半合成补充改称真实观测，也不覆盖冻结的 M0 参数。",
        "",
        "## 题意对应与当前答案",
        "",
        "可见题面把 Q2 定为跨维度融合问题，要求：在 M0 经典 N/D 标度律上纳入 Q 与 p；估计边际效应和无量纲弹性；讨论领域替代/互补；推导 Q 与 N 的等 Loss 替代条件；完成估计、检验与验证。数据说明指定 B1 主拟合、B2/B3 轨迹或族外检查、B4/B5 跨族/文献检查、B6–B8 质量补充、B9/B10 百亿参数以上外推。",
        "",
        "当前可以给出三条分来源结果：B1 的真实 N/D 参考基线；A4/A5 的来源内 p—Loss 配比关系；B6/B7 的半合成 N/D/Q 条件响应。它们不构成同一批真实训练运行上的联合 (N,D,p,Q,Loss) 估计。B 侧可验证 p 向量与 Loss 行连接仍为 0，因此 B 侧 M1 未识别；M2 也不能声称独立于 p 的真实 Q 增量。",
        "",
        "## 1. B1 经典基线的数值锚点与分源验证",
        "",
        f"B1-only 冻结模型为 L=E+A N^(-alpha)+B D^(-beta)：E={e0:.9f}，A={a0:.9f}，alpha={alpha0:.9f}，B={b0:.9f}，beta={beta0:.9f}。1,176 个检查点的训练内 RMSE={m0['metrics']['rmse']:.9g}、R²={m0['metrics']['r2']:.9f}；这不是独立验证。Gate 0 对应 8 条独立 Pythia 轨迹、每条 147 点；24/24 固定多初值成功，参数诊断未提示不稳定。",
        "",
        f"在 B1 的中位参考点 N={n_ref:.7g}B、D={d_ref:.7g}B 处，冻结模型预测 Loss={loss_ref:.7f}，dL/dN={effects_ref['dL_dN']:.7g} Loss/十亿参数，dL/dD={effects_ref['dL_dD']:.7g} Loss/十亿 token，epsilon_N={effects_ref['epsilon_N']:.7g}，epsilon_D={effects_ref['epsilon_D']:.7g}。保存的 1,000 次轨迹 cluster Bootstrap 在该固定点给出的 95% 区间：dL/dN [{effect_intervals['dL_dN'][0]:.7g}, {effect_intervals['dL_dN'][1]:.7g}]；dL/dD [{effect_intervals['dL_dD'][0]:.7g}, {effect_intervals['dL_dD'][1]:.7g}]；epsilon_N [{effect_intervals['epsilon_N'][0]:.7g}, {effect_intervals['epsilon_N'][1]:.7g}]；epsilon_D [{effect_intervals['epsilon_D'][0]:.7g}, {effect_intervals['epsilon_D'][1]:.7g}]。仅 8 个 cluster，区间应视为稳定性提示。",
        "",
        "冻结 M0 对 B2–B5 的分源结果如下；不合并不同来源的误差，也不把同来源插值或不同比较协议说成独立外部验证：",
        "",
        "| 来源 | n | RMSE | MAE | Bias (预测−观测) | R² | Spearman | 解释边界 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    validation_labels = {
        "B2": "半合成压力测试，不是真实外部验证",
        "B3": "Pythia 同来源插值；4,000 点来自 8 条轨迹",
        "B4": "NOT_COMPARABLE；仅描述性",
        "B5": "NOT_COMPARABLE；仅描述性",
    }
    for _, vr in validation.iterrows():
        lines.append(
            f"| {vr['dataset']} | {int(vr['n'])} | {vr['rmse']:.6g} | {vr['mae']:.6g} | {vr['bias_pred_minus_obs']:.6g} | {vr['r2']:.6g} | {vr['spearman']:.6g} | {validation_labels[str(vr['dataset'])]} |"
        )
    lines += [
        "",
        "B5 的原文协议核对补充：Chinchilla 标度分析将平滑训练 Loss 作为测试风险的近似；Gopher 的 Pile benchmark 子集报告 bits-per-byte。B5 本地数据均只标为 `val_loss`，又没有逐点表/图定位，因此无法判断每行对应何种原文评测指标。两者维持 `NOT_COMPARABLE`，不作跨来源换算，也不升级为已验证的来源内趋势。细节见同目录的 `loss_protocol_evidence_sources.md`、`loss_protocol_matrix.csv` 和 `b4_b5_loss_comparability_audit.md`。",
        "",
        "## 2. 解析边际效应、弹性与 Q—N 替代条件",
        "",
        "令 S=A N^{-alpha}、T=B D^{-beta} exp(-beta h(Q,p))，L=E+S+T，且 A,B,alpha,beta>0。若 h 不直接依赖 N,D，则：",
        "",
        "- dL/dN = -alpha S/N；epsilon_N = -alpha S/L。",
        "- dL/dD = -beta T/D；epsilon_D = -beta T/L。",
        "- dL/dQ = -beta T h_Q；epsilon_Q = -beta T Q h_Q/L。",
        "- 在固定 D,p 和目标 L*=L 下，dN/dQ|L = -L_Q/L_N = -beta T h_Q N/(alpha S)。当 h_Q>0 时该斜率为负：更高 Q 可容许更小 N，但斜率大小需要可识别的 Q 系数。",
        "- 若采用 h(Q,p)=gamma log(Q/Q_ref)+g_p(p)，则 D_eff 中的质量倍率为 (Q/Q_ref)^gamma；固定有效数据量时 D_new/D_old=(Q_old/Q_new)^gamma。若同时约束目标 Loss，应用 N(Q)=[A/(L*-E-B D^{-beta} exp(-beta h))]^(1/alpha)，且括号分母必须为正。",
        "",
        "这些是带条件的解析结果，不是已估计的数值替代率；Q1 的 q_huber 操作接口冻结，也不能替代 B 侧独立 Q 变化。该替代率是等 Loss 关系，不是等算力/等成本关系。",
        "",
        "## 3. 配比的可行替代与互补定义",
        "",
        "p 位于 17 维单纯形，故不能把 17 个分量当独立普通变量。域 j 向域 i 转移 delta 时，p(delta)=p+delta(e_i-e_j)，可行范围为 -p_i <= delta <= p_j（若 delta 定义为从 j 转给 i；相反约定需相应换号）。其局部损失变化为 -beta T [h_{p_i}-h_{p_j}]。它回答一个 10 个百分点配比替代对 Loss 的局部影响。",
        "",
        "领域互补性需要先指定两个保持单纯形约束的转移方向 u、v，并定义混合有限差分 Delta_u Delta_v L。局部二阶项为 D_uD_vL=T[beta^2(D_uh)(D_vh)-beta D_uD_vh]。按 Loss 约定，若该二阶项小于 0，第二种转移增强第一种转移的降 Loss 收益，称为该基点与方向下互补；大于 0 表示边际收益减弱，称为替代；接近 0 则无可辨互补/替代。该结论依赖参照配比和方向，不能把无约束 Hessian 元素当成普遍的域关系。",
        "",
        f"A4/A5 当前线性 p-only 主模型可估计来源内方向替代；由于其对 p 为线性，在此主模型设定下任意两个单纯形内转移方向的二阶交叉导数都为 0。这是模型施加的无曲率限制，不是数据证明领域间没有互补。另对 512 个精确训练配方组实施 {a_comp_manifest['bootstrap_resamples']} 次配方组 Bootstrap 的二次敏感性分析：全样本筛出的 {a_comp_counts['jointly_feasible_direction_pairs']:,} 个可行转移对中，未校正逐点区间有 {a_comp_counts['bootstrap_ci_fully_negative_unadjusted']:,} 对全负、{a_comp_counts['bootstrap_ci_fully_positive_unadjusted']:,} 对全正、{a_comp_counts['bootstrap_ci_cross_zero_unadjusted']:,} 对跨零；对候选对作条件 max-|t| 家族校正后，{a_comp_counts['bootstrap_maxT95_fully_negative']:,} 对全负、{a_comp_counts['bootstrap_maxT95_fully_positive']:,} 对全正、{a_comp_counts['bootstrap_maxT95_cross_zero']:,} 对跨零。正向区间表示该基点和两个单独有利转移组合下呈边际收益递减，不等于普遍的域互补/替代结论。候选方向由同一全样本筛选，家族校正条件于该候选集（max-|t| 临界值 {a_comp_limits['critical_value']:.4f}），未调整选择步骤；结果属 A 侧探索性敏感性，不能外推到 B 侧。A 侧外推到 60M/1B 失败。当前 B 侧没有 p 行级连接，所以不能把 A 侧替代系数或探索性曲率迁移为 B 侧标度律参数。逐配对区间、汇总和完整条件限制见 `a_side_complementarity/` 子目录。",
        "",
        "## 4. A 侧配比证据与跨尺度边界",
        "",
        f"A4/A5 的 simplex_ridge 在 1M 同尺度留出集 13 目标宏平均 R2={cross_r2.get('test_1m', float('nan')):.4f}；60M={cross_r2.get('test_60m', float('nan')):.4f}；1B={cross_r2.get('test_1b', float('nan')):.4f}。因此只能写来源内、同尺度关系，不支持跨尺度通用 p 系数。",
        "",
        f"17 域 10pp 替代 Bootstrap 统计沿用 Q1.3 已冻结文件：136 个域对中 {p_sub_summary['supported_domain_pairs']} 个有训练支持；对应 {p_sub_summary['supported_target_pair_intervals']} 个目标×域对区间，其中全正 {p_sub_summary['ci_fully_positive']}、全负 {p_sub_summary['ci_fully_negative']}、跨零 {p_sub_summary['ci_cross_zero']}。这些是模型条件下的逐点区间，没有多重比较调整，不作为因果或 B 侧推断。A 侧 p+Q 模型中 Q_covered(p) 是配比的确定性函数。direct-only q_huber 的嵌套流程标准化 RMSE={qh['nested_macro_normalized_rmse']:.9f}，二次 p-only={quadratic['nested_macro_normalized_rmse']:.9f}，差={pq_diff_nested:+.9f}；配对表的非嵌套 macro_normalized_rmse 差={paired_qh['delta_normalized_rmse_vs_quadratic']:+.9f}，指标定义不同、数值略异，两者都显示加入 Q_covered(p) 未改善结果。路线复审中误把 q_equal 的 0.7533 标成 q_huber，现以 q_huber 的机器结果为准。",
        "",
        "## 5. B6/B7 半合成 Q 响应与 B8 冲突审计",
        "",
        f"B6 有 {int(b6.n)} 行、{int(b6.n_q_levels)} 个 Q 水平；固定 N,D 的 {int(b6.n_matched_ND_cells)} 组中 Loss-Q 线性斜率为负 {int(b6.matched_slope_negative)} 组、正 {int(b6.matched_slope_positive)} 组。B7 有 {int(b7.n)} 行、{int(b7.n_q_levels)} 个 Q 水平；固定 N,D 的 {int(b7.n_matched_ND_cells)} 组斜率为负 {int(b7.matched_slope_negative)} 组、正 {int(b7.matched_slope_positive)} 组。B7 与 B6 的重合 {b6b7['B6_rows_also_in_B7']}/{b6b7['B6_rows']} 行完全复用且 Loss 改动 {b6b7['shared_rows_with_changed_loss']} 行，因此 B6/B7 不是独立复制实验。",
        "",
        f"在 B6 上，带 Q 的 h=gamma log(Q) 形式留一 Q 水平 RMSE={mq_b6.loo_q_rmse:.6g}，M0 N/D RMSE={m0_b6.loo_q_rmse:.6g}；拟合 gamma={mq_b6.gamma_Q_in_Deff:.6g}，合成网格内 Q 弹性中位数={mq_b6.epsilon_Q_median:.6g}，多初值最优解相对参数跨度最大值={mq_b6.multistart_max_relative_parameter_spread:.3g}。B7 对应 RMSE 为 {mq_b7.loo_q_rmse:.6g} 与 {m0_b7.loo_q_rmse:.6g}，gamma={mq_b7.gamma_Q_in_Deff:.6g}，网格内 Q 弹性中位数={mq_b7.epsilon_Q_median:.6g}，多初值最大相对跨度={mq_b7.multistart_max_relative_parameter_spread:.3g}。这些分数只检验半合成网格对预设形式的响应/插值能力，不是对真实 B1–B5 的验证，也不能证明 Q1 的 q_huber 与 CSV 的 Q_score 同尺度。",
        f"等 Loss 数值示例固定共同网格中心 N={trade_b6.reference_N_B:g}B、D={trade_b6.reference_D_B:g}B、Q={trade_b6.reference_Q_score:g}：B6 拟合在 Q 增加 0.1 后对应 N={trade_b6.N_equal_Loss_after_B:.6g}B（变化 {trade_b6.delta_N_B:+.6g}B）；B7 为 N={trade_b7.N_equal_Loss_after_B:.6g}B（变化 {trade_b7.delta_N_B:+.6g}B）。这只是半合成拟合曲面上固定 D 的等 Loss 代数示例，不是实际模型缩放建议或真实资源替代率；完整点值保存在 `b6_b7_q_n_equal_loss_demo.csv`。",
        "",
    ]
    for _, row in b8.iterrows():
        lines.append(
            f"B8 {row['data_type']}：n={int(row['n'])}，固定 N,D 组={int(row['n_matched_ND_cells'])}，Q 斜率负/零/正={int(row['matched_slope_negative'])}/{int(row['matched_slope_zero'])}/{int(row['matched_slope_positive'])}，中位配对斜率={row['median_matched_loss_slope_per_Q']:.6g}。"
        )
    lines += [
        "",
        "B8 与 B6/B7 的方向相反，且在固定 N,D 后依然相反；所以这不是只靠总体 Q 分组造成的表象。B8 的 calibrated/extrapolated 层也分别呈正向。未找到可验证生成规则前，不合并三份数据拟合唯一 Q 弹性；B6/B7 只作为数据说明允许的半合成补充，B8 保留为数据/生成一致性问题。",
        "",
        "## 6. B9/B10 大尺度外推情景",
        "",
        f"对 B10 的 {large_summary['n_B10_rows']} 行估算 Loss，直接代入冻结 B1 M0 参数，不重拟合。B1 支持上限为 N={large_summary['B1_N_max_B']:.6g}B、D={large_summary['B1_D_max_B']:.6g}B；B10 有 {large_summary['n_N_above_B1_support']} 行 N 超出 B1 范围、{large_summary['n_D_above_B1_support']} 行 D 超出 B1 范围。B10 M0 预测 Loss 范围 {large_summary['prediction_range'][0]:.6g}–{large_summary['prediction_range'][1]:.6g}，中位数 {large_summary['prediction_median']:.6g}。逐模型点值和由 8 个独立 B1 轨迹 Bootstrap 参数传播得到的范围在 `b9_b10_m0_extrapolation.csv`。",
        "",
        "B10 是估算/整理 Loss，不是独立真实验证集。区间只传播 M0 参数变异，不含跨协议 Loss 偏差、模型形式错设及远距离外推误差；因此这里只讨论量级与不确定性，不报告合并误差分数或验证通过。B9 模型元数据按精确模型名与 B10 匹配，未匹配项保持缺失。",
        "",
        "## 7. 按题面要求组织的四项回答与证据门禁",
        "",
        "《题目分析报告》§4.2 将 Q2 定义为一个整合任务，没有正式编号为 Q2.1–Q2.4。为逐项核对题面要求，本节把回答整理为四项：广义标度律、边际效应/弹性、领域配比替代/互补、质量—规模替代及验证；这是报告组织方式，不是题面新增的小问。",
        "",
        "| 要求 | 当前可答内容 | 仍缺的实证证据 |",
        "|---|---|---|",
        "| 1. 广义标度律：在 N、D 基线上纳入 p、Q | B1 的 M0 已拟合并冻结；M1/M2 嵌套形式及可识别条件已说明 | B 侧 p 与 Loss 联接为 0，缺少 p 之外独立 Q 变化，因此 M1/M2 均不能作真实数据拟合 |",
        f"| 2. 边际效应与弹性 | B1 范围内 N、D 的检查点级效应和弹性已计算；A4/A5 提供 1M 来源内 p 关联；B6/B7 仅有半合成 Q 条件响应 | B 侧 p 效应、真实独立 Q 效应及相应弹性未识别；B8 的方向与 B6/B7 冲突。B 侧 p-Loss 连接行数为 {p_gate['loss_rows_with_p_joined']} |",
        f"| 3. 领域配比替代与互补 | A4/A5 的 10pp 替代结果和二次敏感性配方组 Bootstrap/max-T 区间可作来源内探索 | max-T 未发现稳健互补方向；{a_comp_counts['bootstrap_maxT95_fully_positive']:,} 对递减收益信号仅条件于同样本筛选的候选集。没有 B 侧配比证据，不能推广成通用效应 |",
        "| 4. 质量—规模替代条件及验证 | 已给等 Loss 解析条件；B6/B7 有明确标注的半合成数值示例；B2/B3 与 B4/B5 的验证边界已分源整理 | 缺可识别的真实 Q 系数，不能估计经验替代率；B4/B5 对 B1 均不可比，联合 p/Q 模型尚无真实联合验证 |",
        "",
        "结论：四项要求均已逐项给出当前可答内容和证据缺口；其中经典基线、部分来源内配比结果和条件推导有实证/计算支持，完整的真实联合经验标度律仍未识别。若竞赛稿要求完整论证，应把跨源可加性作为未验证假设，并将 p/Q 联合识别失败列为结果；不得将半合成拟合包装成真实统一规律。",
        "",
        "## 8. M1/M2 当前数据不可识别性的代数说明",
        "",
        f"B1 只有一个共同配方政策组。若 p 为固定 `p_0`，M1 中 `B exp[-beta g_p(p_0)]` 可并入单一数据项系数，模型与 M0 完全同形；不能从中估计 p 增量或域替代关系。若改用公共累计组成 `p(D)`，则必须从 token 流真实重建，并依赖预先规定的低维 `g_p` 及可区分于 D 的支持；当前没有 checkpoint 级源 token 计数。对 M2，若 `Q=Q_covered(p)`，一般函数层面的 p/Q 分解不唯一；受限参数化可以增加预测基函数设计秩（本例 direct-only q_huber 为 {int(direct_q_ident['qcovered_rank_without_q'])}→{int(direct_q_ident['qcovered_rank_with_q'])}），但这不证明独立 Q 变化，且嵌套 OOF p+Qcovered 标准化 RMSE={qh['nested_macro_normalized_rmse']:.6f}，高于二次 p-only 的 {quadratic['nested_macro_normalized_rmse']:.6f}。详细证明、条件和重开门禁见 [`q2_m1_m2_identifiability_proof.md`](q2_m1_m2_identifiability_proof.md)。",
        "",
        "## 数据投毒与复现边界",
        "",
        "本分析遵循《题目分析报告》中的投毒处理：原始 PDF 低可见度文本不作为题意、公式、先验、变量或结论。本轮数据说明只从 `数据说明(无隐藏字段版本）.pdf` 的可见文本层确认 B6–B10 数据角色；未读取原始带隐藏字段版本的文本层。对 B8 冲突只报告可复核的数值模式，不推断恶意来源。输入原件只读。",
        "",
        "完整回答数值分析与图表复现命令：`D:\\Anaconda\\python.exe src/q2_full_answer_supplement.py`。",
        "",
        "先刷新 M0 弹性剖面、p Gate 审计和 A 侧 p-only 跨尺度图的只读派生文件：`D:\\Anaconda\\python.exe Q2\\归档_20260925\\plot_q2_supplementary_figures.py`；随后运行完整回答入口。入口会校验补充图的 SHA-256，将三图接入正式图组，并同步到 `Q2/04_图表/q2_full_answer/` 与 `docs/q2_full_answer_figures/`。输入和图表口径见 `Q2/归档_20260925/Q2补充图表/复现清单.json` 与 `图表契约.md`。",
        "",
        "## 图表索引",
        "",
        "以下图按原始数据、分析过程和结果分开呈现；图表合同、导出清单和布局 QA 记录位于 `q2_full_answer_figures_contract.csv` 与 `q2_full_answer_figures_qa.json`。A、B、半合成 Q 仍按来源分开展示。",
        "图表生成调用 math-modeling figure skill 的 `setup_style.py`、`export_figure.py`、`visual_qa.py`；如技能未安装在默认目录，运行前设置 `CODEX_MATH_MODELING_SKILL_ROOT`。",
        "",
        "| 类别 | 图 |",
        "|---|---|",
        "| 原始数据 | [A 侧 17 域配方支持范围](q2_full_answer_figures/raw_q2_p_recipe_support.png) |",
        "| 过程 | [p 互补候选区间校正](q2_full_answer_figures/process_q2_p_interval_adjustment.png) |",
        "| 结果 | [A 侧 p-only 同尺度与跨尺度评估（RMSE、R²）](q2_full_answer_figures/result_q2_p_cross_scale_rmse.png) |",
        "| 原始数据 | [半合成 Q–Loss 来源分布](q2_full_answer_figures/raw_q2_q_source_patterns.png) |",
        "| 过程 | [B6/B7 留一 Q 水平比较](q2_full_answer_figures/process_q2_q_leave_q_comparison.png) |",
        "| 结果 | [固定 N,D 的 Q 斜率冲突](q2_full_answer_figures/result_q2_q_matched_slope_conflict.png) |",
        "| 识别门禁 | [B1–B5 p Gate 审计矩阵](q2_full_answer_figures/result_q2_p_gate_audit.png) |",
        "| M0 结果 | [N、D 弹性随训练量变化](q2_full_answer_figures/result_q2_m0_elasticity_profile.png) |",
        "| A 侧补充 | [A 侧 p-only 同尺度与跨尺度评估](q2_full_answer_figures/result_q2_A_p_only_cross_scale.png)（与既有跨尺度图内容重复） |",
        "| 流程图 | [Q2 证据分支与识别门禁](q2_full_answer_figures/flow_q2_full_answer.png) |",
        "",
        f"本次入口收录 {figure_manifest['n_data_figures']} 张数据图和 {figure_manifest['n_workflow_figures']} 张流程图；各图导出 PNG、SVG、PDF 与灰度 PNG。A 侧 p-only 跨尺度图是既有同内容图的来源标记副本，不增加评估证据；弹性区间仅作稳定性提示；p Gate 图中的 0 表示当前没有已核验并连接到 Loss 的配比向量。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference_dir = OUT / "reference_evidence"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for name, source in REFERENCE_EVIDENCE.items():
        shutil.copy2(source, reference_dir / name)
    (reference_dir / "README.md").write_text(
        "# Q2 M0 来源证据副本\n\n"
        "以下为冻结 M0 拟合、边际效应、分源验证和 Loss 可比性审计的只读副本，用于本补充报告的数值锚点；不重拟合 M0，不将 NOT_COMPARABLE 来源升级为验证通过。SHA-256 登记在上级 repro_manifest.json。\n",
        encoding="utf-8",
    )
    q1_evidence_dir = OUT / "q1_source_evidence"
    q1_evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, source in Q1_EVIDENCE.items():
        shutil.copy2(source, q1_evidence_dir / name)
    (q1_evidence_dir / "README.md").write_text(
        "# Q1.3 来源证据副本\n\n"
        "以下文件是 Q1.3 已冻结分析结果的逐字节副本，仅用于支持 Q2 补充报告中的 A 侧来源内结果；本副本不重跑 Q1，也不把 A 侧证据改称 B 侧识别。原始来源仍位于 Q1/03_结果/Q1.3。SHA-256 同时登记在上级 repro_manifest.json 的 inputs 与 outputs。\n",
        encoding="utf-8",
    )
    qdir, qdetails = q_direction_audit()
    fits = pd.DataFrame(qdetails["fit_rows"])
    qn_demo = q_n_equal_loss_demo(fits)
    large, large_summary = large_scale_projection()
    a_comp_manifest = run_a_side_complementarity(
        OUT / "a_side_complementarity", seed=20260924, n_boot=500
    )
    a_comp_summary = pd.read_csv(
        OUT / "a_side_complementarity" / "a4_a5_transfer_pair_complementarity_summary.csv"
    )
    p_audit_path = ROOT / "Q2" / "03_结果" / "p配比可行性审计" / "v1" / "q2_p_audit_summary.json"
    p_audit = json.loads(p_audit_path.read_text(encoding="utf-8"))
    q_ident_path = Q13_PQ / "identifiability_audit.csv"
    q_ident_audit = pd.read_csv(q_ident_path)
    proof_path = OUT / "q2_m1_m2_identifiability_proof.md"
    proof_path.write_text(build_identifiability_proof(p_audit, q_ident_audit), encoding="utf-8")
    qdir.to_csv(OUT / "b6_b8_q_direction_audit.csv", index=False)
    fits.to_csv(OUT / "b6_b7_q_model_supplement.csv", index=False)
    qn_demo.to_csv(OUT / "b6_b7_q_n_equal_loss_demo.csv", index=False)
    large.to_csv(OUT / "b9_b10_m0_extrapolation.csv", index=False)
    (OUT / "b9_b10_extrapolation_summary.json").write_text(
        json.dumps(large_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    q1_holdout_metrics = pd.read_csv(Q13 / "holdout_metrics.csv")
    figure_manifest = create_q2_full_answer_figures(
        ROOT, OUT,
        p_complementarity_summary=a_comp_summary,
        q_model_summary=fits,
        q_direction_summary=qdir,
        q1_holdout_metrics=q1_holdout_metrics,
    )

    report = build_report(qdir, qdetails, large, large_summary, qn_demo,
                          a_comp_summary, a_comp_manifest, figure_manifest)
    report_path = OUT / "q2_full_answer_evidence_supplement.md"
    report_path.write_text(report, encoding="utf-8")

    inputs = [
        ATT / name for name in (
            "supplementary_NQ_experiment.csv", "supplementary_NQ_experiment_expanded.csv",
            "supplementary_NQ_experiment_large.csv", "supplementary_large_baseline.csv",
            "supplementary_large_models.csv", "pythia_training_log_existing.csv",
        )
    ] + [
        BASELINE / "fit_parameters.json", BASELINE / "cluster_bootstrap_estimates.csv",
        BASELINE / "validation_metrics_by_source.csv", BASELINE / "validation_comparability.json",
        Q13 / "holdout_metrics.csv", Q13 / "substitution_effects_10pp_bootstrap.csv",
        Q13_PQ / "model_summary.csv", Q13_PQ / "paired_comparison.csv",
        ROOT / "Q1" / "03_结果" / "Q1.3" / "v1" / "q1_3_run_report.md",
        ROOT / "Q1" / "03_结果" / "Q1.3" / "pq_extension_v1" / "q1_3_pq_extension_report.md",
        REFERENCE_EVIDENCE["m0_marginal_effects_report.md"],
        REFERENCE_EVIDENCE["m0_validation_scope_closeout.md"],
        REFERENCE_EVIDENCE["b3_validation_closeout.md"],
        REFERENCE_EVIDENCE["b4_b5_loss_comparability_audit.md"],
        REFERENCE_EVIDENCE["loss_protocol_evidence_sources.md"],
        REFERENCE_EVIDENCE["loss_protocol_matrix.csv"],
        ROOT / "Q2" / "01_方案说明" / "题目分析报告.md",
        ROOT / "题目分析报告.md",
        ROOT / "中文题目" / "F题" / "算力约束下提升大语言模型能力的资源配置建模.docx",
        ROOT / "中文题目" / "F题" / "数据说明(无隐藏字段版本）.pdf",
        ROOT / "Q2" / "03_结果" / "p配比可行性审计" / "v1" / "q2_p_audit_summary.json",
        ROOT / "Q2" / "03_结果" / "p配比可行性审计" / "v1" / "q2_p_feasibility_audit.md",
        ROOT / "Q2" / "03_结果" / "p配比可行性审计" / "v1" / "q2_p_trajectory_mapping.csv",
        ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "q2_m0_marginal_effects_by_checkpoint.csv",
        ROOT / "Q2" / "归档_20260925" / "plot_q2_supplementary_figures.py",
        ROOT / "Q2" / "归档_20260925" / "Q2补充图表" / "复现清单.json",
        *[
            ROOT / "Q2" / "归档_20260925" / "Q2补充图表" / (
                f"{stem}_{suffix}" if suffix == "grayscale.png" else f"{stem}.{suffix}"
            )
            for stem in (
                "result_q2_p_gate_audit",
                "result_q2_m0_elasticity_profile",
                "result_q2_A_p_only_cross_scale",
            )
            for suffix in ("pdf", "svg", "png", "grayscale.png")
        ],
        q_ident_path,
        ROOT / "中文题目" / "F题" / "real_attachments" / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv",
        ROOT / "中文题目" / "F题" / "real_attachments" / "A_data_value" / "regmix_tables" / "train_pile_loss_1m.csv",
        Q13 / "lambda_selection_quadratic_sensitivity.csv",
        Path(__file__).resolve().with_name("q2_a_side_complementarity_sensitivity.py"),
        Path(__file__).resolve().with_name("q2_full_answer_figures.py"),
        ROOT / "utils" / "plot_style.py",
        FIGURE_SKILL_ROOT / "tools" / "figure" / "scripts" / "setup_style.py",
        FIGURE_SKILL_ROOT / "tools" / "figure" / "scripts" / "export_figure.py",
        FIGURE_SKILL_ROOT / "tools" / "figure" / "scripts" / "visual_qa.py",
    ]
    manifest = {
        "task": "Q2 full-answer evidence supplement",
        "status": "SUPPLEMENT_COMPLETE_JOINT_P_Q_IDENTIFICATION_UNRESOLVED",
        "command": "D:\\Anaconda\\python.exe src/q2_full_answer_supplement.py",
        "code": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "python": "3.10+",
        "resampling": {
            "A4_A5_complementarity": {"seed": 20260924, "recipe_group_bootstrap_replicates": 500},
            "B6_B7_fits": "deterministic fixed multi-start; no bootstrap",
        },
        "A4_A5_complementarity": a_comp_manifest,
        "B1_m0_refit": False,
        "B6_B7_role": "semi-synthetic conditional response only",
        "B8_role": "direction-conflict audit only; not pooled with B6/B7",
        "B9_B10_role": "out-of-support scenario projection; B10 not an independent validation set",
        "figure_runtime": {
            "skill_root": str(FIGURE_SKILL_ROOT),
            "scripts": ["setup_style.py", "export_figure.py", "visual_qa.py"],
        },
        "inputs": [{
            "path": str(p.relative_to(ROOT).as_posix()) if p.is_relative_to(ROOT) else str(p.resolve()),
            "sha256": sha256(p),
        } for p in inputs],
        "outputs": {},
        "B6_B7_overlap": qdetails["B6_B7_overlap"],
        "B9_B10_summary": large_summary,
    }
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "repro_manifest.json":
            manifest["outputs"][p.relative_to(OUT).as_posix()] = sha256(p)
    (OUT / "repro_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    Q2_OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copytree(OUT, Q2_OUT, dirs_exist_ok=True)
    shutil.copy2(report_path, DOCS / report_path.name)
    shutil.copy2(proof_path, DOCS / proof_path.name)
    shutil.copytree(OUT / "q2_full_answer_figures", DOCS / "q2_full_answer_figures", dirs_exist_ok=True)
    q2_figures_dir = ROOT / "Q2" / "04_图表" / "q2_full_answer"
    shutil.copytree(OUT / "q2_full_answer_figures", q2_figures_dir, dirs_exist_ok=True)
    shutil.copy2(Path(__file__), ROOT / "Q2" / "02_代码" / Path(__file__).name)
    shutil.copy2(Path(__file__).resolve().with_name("q2_a_side_complementarity_sensitivity.py"),
                 ROOT / "Q2" / "02_代码" / "q2_a_side_complementarity_sensitivity.py")
    shutil.copy2(Path(__file__).resolve().with_name("q2_full_answer_figures.py"),
                 ROOT / "Q2" / "02_代码" / "q2_full_answer_figures.py")
    print(json.dumps({
        "status": manifest["status"],
        "B6_B7_overlap": qdetails["B6_B7_overlap"],
        "B6_q_lOO_rmse": fits[(fits.dataset == "B6") & (fits.model == "M_Q_NDQ")].iloc[0]["loo_q_rmse"],
        "B8_matched_direction": qdir[qdir.dataset == "B8"][
            ["data_type", "matched_slope_negative", "matched_slope_zero", "matched_slope_positive"]
        ].to_dict(orient="records"),
        "B9_B10": large_summary,
        "outputs": str(Q2_OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
