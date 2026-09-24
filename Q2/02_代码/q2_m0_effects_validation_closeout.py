#!/usr/bin/env python3
"""Derive M0 marginal effects and close out B2-B5 validation scope.

This postprocessor consumes frozen Q2 baseline outputs. It does not refit M0,
resample clusters, modify source data, or estimate p/Q effects.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "q2_scaling_baseline" / "v1"
Q2_BASELINE_COPY = ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1"
Q2_CODE = ROOT / "Q2" / "02_代码"
Q2_STRESS = ROOT / "Q2" / "03_结果" / "B2半合成压力测试" / "v1"
Q2_P_AUDIT = ROOT / "Q2" / "03_结果" / "p配比可行性审计" / "v1"
Q1_FINAL = ROOT / "Q1" / "03_结果" / "Q1_最终收口"
DOCS = ROOT / "docs"

BOOTSTRAP_COLUMNS = ["E", "A", "alpha", "B", "beta"]
EFFECT_COLUMNS = [
    "dLoss_dN_per_B_params",
    "dLoss_dD_per_B_tokens",
    "elasticity_N",
    "elasticity_D",
]
EXPECTED_VALIDATION_N = {"B2": 1029, "B3": 4000, "B4": 57, "B5": 44}
PERCENTILES = (2.5, 97.5)
ARTIFACT_NAMES = [
    "q2_m0_marginal_effects_by_checkpoint.csv",
    "q2_m0_marginal_effects_by_size.csv",
    "q2_m0_marginal_effects_report.md",
    "q2_validation_scope_closeout.md",
]
MANIFEST_NAME = "q2_m0_effects_validation_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen input is missing: {relative(path)}")


def verify_frozen_output_hashes(baseline_manifest: dict) -> dict[str, str]:
    """Verify the frozen baseline outputs against their original manifest."""
    expected_outputs = baseline_manifest.get("outputs_sha256", {})
    names = [
        "fit_parameters.json",
        "b1_fitted_predictions.csv",
        "cluster_bootstrap_estimates.csv",
        "validation_metrics_by_source.csv",
        "validation_metrics_by_stratum.csv",
        "validation_comparability.json",
    ]
    verified = {}
    for name in names:
        path = BASELINE / name
        require_file(path)
        manifest_key = f"results/q2_scaling_baseline/v1/{name}"
        expected = expected_outputs.get(manifest_key)
        actual = sha256(path)
        if not expected:
            raise ValueError(f"Frozen baseline manifest has no hash for {manifest_key}")
        if actual != expected:
            raise ValueError(f"Frozen baseline hash mismatch for {manifest_key}")
        verified[relative(path)] = actual
    return verified


def effect_arrays(parameters: np.ndarray, n_params_b: np.ndarray, d_tokens_b: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate M0 predictions, partial derivatives, and elasticities."""
    parameters = np.atleast_2d(np.asarray(parameters, dtype=float))
    n = np.asarray(n_params_b, dtype=float)[None, :]
    d = np.asarray(d_tokens_b, dtype=float)[None, :]
    e = parameters[:, 0, None]
    a = parameters[:, 1, None]
    alpha = parameters[:, 2, None]
    b = parameters[:, 3, None]
    beta = parameters[:, 4, None]
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        loss = e + a * np.power(n, -alpha) + b * np.power(d, -beta)
        d_n = -alpha * a * np.power(n, -alpha - 1.0)
        d_d = -beta * b * np.power(d, -beta - 1.0)
        elasticity_n = d_n * n / loss
        elasticity_d = d_d * d / loss
    return {
        "predicted_val_loss": loss,
        "dLoss_dN_per_B_params": d_n,
        "dLoss_dD_per_B_tokens": d_d,
        "elasticity_N": elasticity_n,
        "elasticity_D": elasticity_d,
    }


def load_and_validate_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict[str, str]]:
    baseline_manifest_path = BASELINE / "复现清单.json"
    fit_path = BASELINE / "fit_parameters.json"
    predictions_path = BASELINE / "b1_fitted_predictions.csv"
    bootstrap_path = BASELINE / "cluster_bootstrap_estimates.csv"
    mapping_path = Q2_BASELINE_COPY / "gate0_b1_trajectory_mapping.csv"

    for path in [baseline_manifest_path, fit_path, predictions_path, bootstrap_path, mapping_path]:
        require_file(path)

    baseline_manifest = read_json(baseline_manifest_path)
    verified_hashes = verify_frozen_output_hashes(baseline_manifest)
    fit = read_json(fit_path)
    parameters = fit.get("parameters", {})
    if any(name not in parameters for name in BOOTSTRAP_COLUMNS):
        raise ValueError("fit_parameters.json is missing one or more M0 parameters")

    estimate = np.array([parameters[name] for name in BOOTSTRAP_COLUMNS], dtype=float)
    if not np.isfinite(estimate).all() or estimate[0] < 0 or np.any(estimate[1:] <= 0):
        raise ValueError("Frozen M0 parameters do not satisfy the declared constraints")

    predictions = pd.read_csv(predictions_path)
    required_prediction_columns = {
        "run_id", "N_params_B", "D_tokens_B", "observed_val_loss", "predicted_val_loss"
    }
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError("B1 frozen predictions do not have the required columns")
    if len(predictions) != 1176 or predictions["run_id"].nunique() != 1176:
        raise ValueError("B1 is expected to contain 1,176 unique checkpoint rows")
    if (predictions[["N_params_B", "D_tokens_B"]].to_numpy(dtype=float) <= 0).any():
        raise ValueError("B1 contains non-positive N or D values")

    mapping = pd.read_csv(mapping_path)
    required_mapping_columns = {
        "N_params_B", "B12_model_size", "model_repo", "B1_rows", "cluster_key"
    }
    if not required_mapping_columns.issubset(mapping.columns):
        raise ValueError("Gate 0 trajectory mapping does not have the required columns")
    if len(mapping) != 8 or mapping["model_repo"].nunique() != 8:
        raise ValueError("Gate 0 mapping must identify exactly eight model trajectories")
    if not (mapping["cluster_key"].astype(str) == mapping["model_repo"].astype(str)).all():
        raise ValueError("Frozen Gate 0 cluster_key differs from model_repo")
    if not (mapping["B1_rows"].astype(int) == 147).all():
        raise ValueError("Each frozen B1 trajectory is expected to have 147 checkpoints")

    predictions = predictions.merge(
        mapping[["N_params_B", "B12_model_size", "model_repo"]],
        on="N_params_B",
        how="left",
        validate="many_to_one",
    )
    if predictions["model_repo"].isna().any():
        raise ValueError("Some B1 checkpoints could not be joined to the frozen trajectory map")
    actual_track_rows = predictions.groupby("model_repo", sort=False).size()
    if len(actual_track_rows) != 8 or not (actual_track_rows == 147).all():
        raise ValueError("B1 checkpoints do not match the frozen eight-by-147 track structure")

    point_effects = effect_arrays(estimate, predictions["N_params_B"], predictions["D_tokens_B"])
    recomputed = point_effects["predicted_val_loss"][0]
    frozen = predictions["predicted_val_loss"].to_numpy(dtype=float)
    max_prediction_delta = float(np.max(np.abs(recomputed - frozen)))
    if max_prediction_delta > 1e-10:
        raise ValueError(f"M0 prediction reproduction error too large: {max_prediction_delta:.3g}")

    bootstrap_path_data = pd.read_csv(bootstrap_path)
    if not set(BOOTSTRAP_COLUMNS).issubset(bootstrap_path_data.columns):
        raise ValueError("Cluster bootstrap file does not contain all M0 parameters")
    expected_bootstrap = int(baseline_manifest.get("bootstrap_replicates", 0))
    actual_bootstrap = len(bootstrap_path_data)
    if actual_bootstrap != expected_bootstrap:
        raise ValueError(
            "Incomplete cluster-bootstrap parameter file: "
            f"expected {expected_bootstrap}, found {actual_bootstrap}; refusing to generate intervals"
        )
    if "replicate" in bootstrap_path_data and bootstrap_path_data["replicate"].duplicated().any():
        raise ValueError("Cluster bootstrap file contains duplicate replicate identifiers")
    bootstrap_parameters = bootstrap_path_data[BOOTSTRAP_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(bootstrap_parameters).all():
        raise ValueError("Cluster bootstrap parameter file contains non-finite values")
    if (bootstrap_parameters[:, 0] < 0).any() or (bootstrap_parameters[:, 1:] <= 0).any():
        raise ValueError("Cluster bootstrap parameters violate the declared M0 constraints")
    if (bootstrap_parameters[:, 0] >= predictions["observed_val_loss"].min()).any():
        raise ValueError("A cluster bootstrap E estimate violates the M0 upper-bound constraint")

    inputs_to_hash = [
        baseline_manifest_path,
        fit_path,
        predictions_path,
        bootstrap_path,
        mapping_path,
        BASELINE / "validation_metrics_by_source.csv",
        BASELINE / "validation_metrics_by_stratum.csv",
        BASELINE / "validation_comparability.json",
        Q2_BASELINE_COPY / "b3_validation_closeout.md",
        Q2_BASELINE_COPY / "b4_b5_loss_comparability_audit.md",
        Q2_STRESS / "b2_cerebras_stress_report.md",
        Q2_P_AUDIT / "q2_p_feasibility_audit.md",
        Q1_FINAL / "q1_final_decision.md",
        Path(__file__).resolve(),
    ]
    for path in inputs_to_hash:
        require_file(path)
    input_hashes = {relative(path): sha256(path) for path in inputs_to_hash}
    return fit, predictions, mapping, bootstrap_path_data, baseline_manifest, {
        **verified_hashes,
        **input_hashes,
        "_max_prediction_reproduction_delta": f"{max_prediction_delta:.17g}",
    }


def compute_effect_tables(
    fit: dict, predictions: pd.DataFrame, mapping: pd.DataFrame, bootstrap: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    estimate = np.array([fit["parameters"][name] for name in BOOTSTRAP_COLUMNS], dtype=float)
    n = predictions["N_params_B"].to_numpy(dtype=float)
    d = predictions["D_tokens_B"].to_numpy(dtype=float)
    point = effect_arrays(estimate, n, d)
    bootstrap_params = bootstrap[BOOTSTRAP_COLUMNS].to_numpy(dtype=float)
    sampled = effect_arrays(bootstrap_params, n, d)

    result = predictions[
        ["run_id", "model_repo", "B12_model_size", "N_params_B", "D_tokens_B", "observed_val_loss"]
    ].copy()
    result["predicted_val_loss"] = point["predicted_val_loss"][0]
    for name in EFFECT_COLUMNS:
        values = point[name][0]
        result[name] = values
        result[f"{name}_lower_95"] = np.percentile(sampled[name], PERCENTILES[0], axis=0)
        result[f"{name}_upper_95"] = np.percentile(sampled[name], PERCENTILES[1], axis=0)

    numeric = result.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Derived checkpoint effect table contains non-finite numeric values")

    summary_rows = []
    for track in mapping.sort_values("N_params_B").to_dict(orient="records"):
        mask = result["model_repo"].eq(track["model_repo"]).to_numpy()
        if int(mask.sum()) != 147:
            raise ValueError(f"Unexpected checkpoint count for {track['model_repo']}")
        row = {
            "model_repo": track["model_repo"],
            "B12_model_size": track["B12_model_size"],
            "N_params_B": float(track["N_params_B"]),
            "n_checkpoints": int(mask.sum()),
            "D_tokens_B_min": float(np.min(d[mask])),
            "D_tokens_B_median": float(np.median(d[mask])),
            "D_tokens_B_max": float(np.max(d[mask])),
        }
        for name in EFFECT_COLUMNS:
            checkpoint_values = result.loc[mask, name].to_numpy(dtype=float)
            replicate_medians = np.median(sampled[name][:, mask], axis=1)
            row[f"{name}_median"] = float(np.median(checkpoint_values))
            row[f"{name}_checkpoint_p10"] = float(np.percentile(checkpoint_values, 10))
            row[f"{name}_checkpoint_p90"] = float(np.percentile(checkpoint_values, 90))
            row[f"{name}_median_lower_95"] = float(np.percentile(replicate_medians, PERCENTILES[0]))
            row[f"{name}_median_upper_95"] = float(np.percentile(replicate_medians, PERCENTILES[1]))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
        raise ValueError("Derived model-size summary contains non-finite numeric values")

    checks = {
        "checkpoint_rows": int(len(result)),
        "trajectory_clusters": int(result["model_repo"].nunique()),
        "bootstrap_replicates": int(len(bootstrap)),
        "all_derived_numeric_finite": True,
    }
    return result, summary, checks


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}g}"


def build_effect_report(
    fit: dict, summary: pd.DataFrame, bootstrap_count: int, max_prediction_delta: float
) -> str:
    p = fit["parameters"]
    table = [
        "| 模型轨迹 | N (十亿参数) | 检查点数 | D 范围 (十亿 Token) | ∂L/∂N 中位数 [Bootstrap 95% CI] | ∂L/∂D 中位数 [Bootstrap 95% CI] | εN 中位数 [Bootstrap 95% CI] | εD 中位数 [Bootstrap 95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        d_range = f"{fmt(row['D_tokens_B_min'])}–{fmt(row['D_tokens_B_max'])}"
        cells = [
            str(row["B12_model_size"]),
            fmt(row["N_params_B"]),
            str(row["n_checkpoints"]),
            d_range,
            f"{fmt(row['dLoss_dN_per_B_params_median'])} [{fmt(row['dLoss_dN_per_B_params_median_lower_95'])}, {fmt(row['dLoss_dN_per_B_params_median_upper_95'])}]",
            f"{fmt(row['dLoss_dD_per_B_tokens_median'])} [{fmt(row['dLoss_dD_per_B_tokens_median_lower_95'])}, {fmt(row['dLoss_dD_per_B_tokens_median_upper_95'])}]",
            f"{fmt(row['elasticity_N_median'])} [{fmt(row['elasticity_N_median_lower_95'])}, {fmt(row['elasticity_N_median_upper_95'])}]",
            f"{fmt(row['elasticity_D_median'])} [{fmt(row['elasticity_D_median_lower_95'])}, {fmt(row['elasticity_D_median_upper_95'])}]",
        ]
        table.append("| " + " | ".join(cells) + " |")

    return rf"""# Q2 M0：边际效应与弹性

## 范围与数据

本报告从冻结的 B1 经典基线直接计算模型蕴含的 N、D 边际效应和无量纲弹性；不重新拟合 M0，也不使用 Q 或 p。B1 共 {len(summary) * 147:,} 个检查点，Gate 0 将其映射为 {len(summary)} 条独立模型轨迹、每条 147 个检查点。`run_id` 仅用于定位检查点，不作为独立轨迹数。

冻结模型为

\[L=E+A N^{{-\alpha}}+B D^{{-\beta}}.\]

本次冻结参数：E={fmt(p['E'], 10)}，A={fmt(p['A'], 10)}，α={fmt(p['alpha'], 10)}，B={fmt(p['B'], 10)}，β={fmt(p['beta'], 10)}。N 与 D 按十亿参数、十亿 Token 输入。弹性以模型预测的 L 为分母，在每个观测到的 B1 检查点计算。

## 定义

\[\frac{{\partial L}}{{\partial N}}=-\alpha A N^{{-\alpha-1}},\qquad
\frac{{\partial L}}{{\partial D}}=-\beta B D^{{-\beta-1}}.\]

\[\varepsilon_N=\frac{{\partial L}}{{\partial N}}\frac{{N}}{{L}}
=-\frac{{\alpha A N^{{-\alpha}}}}{{L}},\qquad
\varepsilon_D=\frac{{\partial L}}{{\partial D}}\frac{{D}}{{L}}
=-\frac{{\beta B D^{{-\beta}}}}{{L}}.\]

在其他输入固定时，负的边际效应表示增加 N 或 D 对应模型预测 Loss 下降。弹性是局部百分比变化率：输入增加约 1% 时，模型预测 Loss 的局部百分比变化约为对应弹性的 1%。这些是 M0 的模型蕴含量，不是 Q 或 p 的效应，也不单独构成因果结论。

## 按模型轨迹的摘要

每行对该轨迹 147 个实际检查点的点估计取中位数；区间通过对每个已保存 Bootstrap 参数样本计算该轨迹检查点效应的中位数，再取 2.5 与 97.5 百分位。D 范围和检查点间差异描述训练轨迹覆盖，不是置信区间。

@@TABLE@@

逐检查点结果见 `q2_m0_marginal_effects_by_checkpoint.csv`，完整轨迹摘要见 `q2_m0_marginal_effects_by_size.csv`。逐点区间是对固定 B1 支持点条件化的参数 Bootstrap 区间。

## 不确定性与复核

- 区间复用 {bootstrap_count:,} 个已保存的完整轨迹 cluster Bootstrap 拟合，没有重新抽样或重新拟合。
- 独立 Bootstrap cluster 只有 8 条 B12 确认的 Pythia 模型轨迹；因此百分位区间对小 cluster 数敏感，不应描述为高精度总体区间。
- 从冻结参数复算 B1 的最大预测差为 {max_prediction_delta:.3g}；导数和弹性由解析公式逐检查点计算，派生数值均通过有限性检查。
- 原始数据与冻结拟合结果保持不变。可复现入口：`D:/Anaconda/python.exe src/q2_m0_effects_validation_closeout.py`。
""".replace("@@TABLE@@", "\n".join(table))


def build_validation_report(metrics: pd.DataFrame, comparability: dict, input_paths: dict[str, Path]) -> str:
    rows = {str(row["dataset"]): row for row in metrics.to_dict(orient="records")}
    required = set(EXPECTED_VALIDATION_N)
    if set(rows) != required:
        raise ValueError(f"Validation source table has datasets {sorted(rows)}, expected {sorted(required)}")
    for dataset, expected_n in EXPECTED_VALIDATION_N.items():
        if int(rows[dataset]["n"]) != expected_n:
            raise ValueError(f"{dataset} validation row count is not {expected_n}")

    comparability_map = comparability.get("source_comparability", {})
    b4b5_audit = input_paths["b4b5_audit"].read_text(encoding="utf-8")
    b3_closeout = input_paths["b3_closeout"].read_text(encoding="utf-8")
    b2_report = input_paths["b2_report"].read_text(encoding="utf-8")
    p_audit = input_paths["p_audit"].read_text(encoding="utf-8")
    q1_decision = input_paths["q1_decision"].read_text(encoding="utf-8")
    if "NOT_COMPARABLE" not in b4b5_audit or "lacks" not in comparability_map.get("B4", "") or "lacks" not in comparability_map.get("B5", ""):
        raise ValueError("B4/B5 comparability evidence does not preserve the frozen NOT_COMPARABLE gate")
    if "interpolat" not in b3_closeout.lower() or "独立真实验证" not in b3_closeout:
        raise ValueError("B3 closeout no longer documents its interpolation/validation limitation")
    if "半合成" not in b2_report:
        raise ValueError("B2 stress report no longer identifies the data as semi-synthetic")
    if "FAIL" not in p_audit:
        raise ValueError("The current p-only data gate is not the expected FAIL state")
    if "q_huber" not in q1_decision or "q_equal" not in q1_decision:
        raise ValueError("Q1 final decision does not confirm the frozen operational/sensitivity Q pair")

    roles = {
        "B2": "半合成压力测试；不是 Cerebras 原始训练日志，也不是独立真实外部验证。",
        "B3": "Pythia 同来源插值轨迹；4,000 点来自 8 条轨迹，不是 4,000 个独立实验。",
        "B4": "跨族快照；对 B1 为 NOT_COMPARABLE，误差只作来源分层描述。",
        "B5": "多来源文献汇编；对 B1 为 NOT_COMPARABLE，误差只作来源分层描述。",
    }
    table = [
        "| 来源 | n（记录数） | RMSE | MAE | 偏差（预测−观测） | R² | Spearman | 可支持的解释 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in ["B2", "B3", "B4", "B5"]:
        row = rows[dataset]
        table.append(
            "| " + " | ".join([
                dataset,
                str(int(row["n"])),
                fmt(row["rmse"], 5),
                fmt(row["mae"], 5),
                fmt(row["bias_pred_minus_obs"], 5),
                fmt(row["r2"], 5),
                fmt(row["spearman"], 5),
                roles[dataset],
            ]) + " |"
        )

    pooled = comparability.get("pooled_validation_metric")
    if pooled is not None:
        raise ValueError("Frozen validation contract unexpectedly contains a pooled B2-B5 metric")
    return rf"""# Q2 B2–B5：验证证据与结论边界

## 结论

B1 的 M0 参数只由 B1 拟合。B2–B5 按来源分别报告；来源性质和评测协议不同或未记录，因此不合并成一个跨源验证分数。B2 与 B3 提供受限的压力/轨迹形状检查；B4 与 B5 当前均未通过相对 B1 的绝对 Loss 可比性门禁。

## 分源指标

指标来自冻结的 `validation_metrics_by_source.csv`，没有在验证集调参或重校准。

@@TABLE@@

## 各来源能支持什么

- **B2**：半合成 Cerebras 情景压力测试。当前 RMSE、MAE 与偏差描述该构造情景上的 M0 表现；不能解释为真实 Cerebras 训练的外部验证。B2 分层指标保留在 `validation_metrics_by_stratum.csv`。
- **B3**：8 条 Pythia 轨迹插值生成的 4,000 个检查点。整体误差小，说明冻结 M0 能跟随该同来源插值轨迹；不能据此声称跨来源、独立训练运行或 p/Q 效应已验证。逐轨迹复核见 `b3_validation_closeout.md`。
- **B4**：57 个跨族快照。缺少逐行来源和完整评测协议；现有误差只能作为描述性诊断。审计记录的 Cerebras-GPT 训练量来源不能由所核对论文表格支持，故不据此改写原始数值。
- **B5**：44 个多来源文献记录。缺少统一 tokenizer、评测集/切分和逐点原文定位；不得将其误差当作 B1 同量纲外部验证。可能的来源内趋势仍是待核候选。

## 门禁与范围

- B4/B5 对 B1 的当前分类均为 `NOT_COMPARABLE`，没有来源达到 `ABSOLUTE_COMPARABLE`；不做线性换算、偏移校准或跨源汇总。
- B1–B5 的 p-only 数据 Gate 当前为 FAIL；本报告不估计 M1/M2，也不报告 p/Q 边际效应或数值替代关系。
- Q1 的正式操作接口现已冻结为 `q_huber`，`q_equal` 保留为敏感性分；该接口冻结不解除 B 侧 p 数据缺失或独立 Q 变化不足的识别门禁。M0 本身不使用 Q。
- 数据污染处置遵循题目分析报告和既有审计：本收口仅复用已整理的结构化指标与可见审计结论，不把低可见度/隐藏文本当作题意、数据或证据。原始附件保持只读。

## 可复核文件

- 来源汇总：`validation_metrics_by_source.csv`
- 分层明细：`validation_metrics_by_stratum.csv`
- 比较合同：`validation_comparability.json`
- B3 逐轨迹收口：`b3_validation_closeout.md`
- B4/B5 协议审计：`b4_b5_loss_comparability_audit.md`
- p 可行性审计：`q2_p_feasibility_audit.md`
""".replace("@@TABLE@@", "\n".join(table))


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def render_artifacts() -> tuple[dict[str, bytes], dict]:
    fit, predictions, mapping, bootstrap, baseline_manifest, input_hashes = load_and_validate_inputs()
    checkpoint, size_summary, checks = compute_effect_tables(fit, predictions, mapping, bootstrap)
    metrics = pd.read_csv(BASELINE / "validation_metrics_by_source.csv")
    comparability = read_json(BASELINE / "validation_comparability.json")
    max_delta = float(input_hashes.pop("_max_prediction_reproduction_delta"))
    effect_report = build_effect_report(fit, size_summary, len(bootstrap), max_delta)
    validation_inputs = {
        "b2_report": Q2_STRESS / "b2_cerebras_stress_report.md",
        "b3_closeout": Q2_BASELINE_COPY / "b3_validation_closeout.md",
        "b4b5_audit": Q2_BASELINE_COPY / "b4_b5_loss_comparability_audit.md",
        "p_audit": Q2_P_AUDIT / "q2_p_feasibility_audit.md",
        "q1_decision": Q1_FINAL / "q1_final_decision.md",
    }
    validation_report = build_validation_report(metrics, comparability, validation_inputs)

    checkpoint_csv = checkpoint.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8-sig")
    size_csv = size_summary.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8-sig")
    artifacts = {
        ARTIFACT_NAMES[0]: checkpoint_csv,
        ARTIFACT_NAMES[1]: size_csv,
        ARTIFACT_NAMES[2]: effect_report.encode("utf-8"),
        ARTIFACT_NAMES[3]: validation_report.encode("utf-8"),
    }
    checks.update({
        "baseline_fit_scope": fit.get("fit_scope"),
        "bootstrap_expected": int(baseline_manifest.get("bootstrap_replicates", 0)),
        "bootstrap_observed": int(len(bootstrap)),
        "max_prediction_reproduction_delta": max_delta,
        "validation_rows_by_source": {str(row["dataset"]): int(row["n"]) for row in metrics.to_dict(orient="records")},
        "validation_pooled_metric": comparability.get("pooled_validation_metric"),
    })
    return artifacts, {"checks": checks, "input_sha256": input_hashes}


def main() -> int:
    artifacts, metadata = render_artifacts()
    primary_dirs = [BASELINE, Q2_BASELINE_COPY, DOCS]
    output_hashes: dict[str, str] = {}
    for directory in primary_dirs:
        for name, content in artifacts.items():
            destination = directory / name
            write_bytes(destination, content)
            output_hashes[relative(destination)] = sha256(destination)
    q2_script_copy = Q2_CODE / Path(__file__).name
    write_bytes(q2_script_copy, Path(__file__).read_bytes())
    output_hashes[relative(q2_script_copy)] = sha256(q2_script_copy)

    manifest = {
        "task": "Q2 M0 marginal effects and B2-B5 validation scope closeout",
        "status": "completed_with_validation_limits",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "D:\\Anaconda\\python.exe src/q2_m0_effects_validation_closeout.py",
        "script": relative(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "executable": sys.executable,
        },
        "method": {
            "m0_formula": "L = E + A*N_params_B**(-alpha) + B*D_tokens_B**(-beta)",
            "elasticity_denominator": "M0 fitted prediction at each fixed B1 checkpoint",
            "bootstrap": "Reuse all saved successful complete-trajectory cluster bootstrap parameter fits; pointwise percentile interval 2.5-97.5; no refit or resampling.",
            "independent_clusters": 8,
            "validation_pooling": "No pooled B2-B5 metric; report by source only.",
        },
        "input_sha256": metadata["input_sha256"],
        "checks": metadata["checks"],
        "output_sha256_excluding_this_manifest": output_hashes,
        "manifest_note": "This manifest is mirrored to all three delivery locations; its own hash is intentionally excluded to avoid recursion.",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for directory in primary_dirs:
        write_bytes(directory / MANIFEST_NAME, manifest_bytes)

    print(f"Generated {len(artifacts)} artifacts and one manifest in results/, Q2/, and docs/.")
    print(f"B1 checkpoints: {metadata['checks']['checkpoint_rows']}; clusters: {metadata['checks']['trajectory_clusters']}.")
    print(f"Bootstrap fits: {metadata['checks']['bootstrap_observed']}.")
    print(f"Max frozen-prediction reproduction delta: {metadata['checks']['max_prediction_reproduction_delta']:.3g}.")
    print("Validation counts: " + json.dumps(metadata["checks"]["validation_rows_by_source"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
