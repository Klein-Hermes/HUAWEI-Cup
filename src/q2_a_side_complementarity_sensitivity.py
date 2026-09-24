"""Exploratory A4/A5 domain-transfer complementarity sensitivity for Q2.

This reuses the frozen Q1.3 quadratic sensitivity form and lambda. It does not
alter Q1 outputs or identify B-side p effects. Recipe-group bootstrap intervals
are pointwise and conditional on full-sample selection of improving transfers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import f_q1_3_mixture_loss as BASE


ROOT = Path(__file__).resolve().parents[1]
Q1_OUT = ROOT / "Q1" / "03_结果" / "Q1.3" / "v1"
OUT = ROOT / "results" / "q2_full_answer_supplement" / "v1" / "a_side_complementarity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interaction_hessian(model: dict[str, Any], n_domains: int) -> np.ndarray:
    """Return target × domain × domain Hessians in the raw closed-share basis."""
    coef = model["coef"]
    n_targets = coef.shape[1]
    hessian = np.zeros((n_targets, n_domains, n_domains), dtype=float)
    term = n_domains
    for j in range(n_domains):
        for k in range(j + 1, n_domains):
            hessian[:, j, k] = coef[term, :]
            hessian[:, k, j] = coef[term, :]
            term += 1
    if term != coef.shape[0]:
        raise ValueError("Quadratic coefficient count does not match the 17-share feature contract")
    return hessian


def build_candidates(p_ref: np.ndarray, model: dict[str, Any], pcols: list[str],
                     ycols: list[str], step: float = 0.1) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    n_domains = len(pcols)
    n_targets = len(ycols)
    baseline = BASE.predict_quadratic(model, p_ref[None, :])[0]
    hessian = interaction_hessian(model, n_domains)
    eligible_donors = np.flatnonzero(p_ref >= step - 1e-12)
    candidates: list[list[dict[str, Any]]] = []
    summary: dict[str, Any] = {"reference_share_at_least_10pp_donors": [pcols[i] for i in eligible_donors]}

    for target_idx in range(n_targets):
        directions: list[dict[str, Any]] = []
        for donor in eligible_donors:
            for recipient in range(n_domains):
                if recipient == donor:
                    continue
                u = np.zeros(n_domains, dtype=float)
                u[recipient] = 1.0
                u[donor] = -1.0
                shifted = p_ref + step * u
                if shifted.min() < -1e-12 or shifted.max() > 1.0 + 1e-12:
                    continue
                change = float(
                    BASE.predict_quadratic(model, shifted[None, :])[0, target_idx]
                    - baseline[target_idx]
                )
                if change < -1e-12:
                    directions.append({
                        "from_idx": int(donor), "to_idx": int(recipient),
                        "direction": u, "delta_loss": change,
                    })

        target_pairs: list[dict[str, Any]] = []
        for first, second in combinations(directions, 2):
            u = first["direction"]
            v = second["direction"]
            joint = p_ref + step * (u + v)
            if joint.min() < -1e-12 or joint.max() > 1.0 + 1e-12:
                continue
            a, b = first["to_idx"], first["from_idx"]
            c, d = second["to_idx"], second["from_idx"]
            point_interaction = float(
                step ** 2 * (hessian[target_idx, a, c] - hessian[target_idx, a, d]
                             - hessian[target_idx, b, c] + hessian[target_idx, b, d])
            )
            target_pairs.append({
                "target_idx": target_idx,
                "first_from_idx": b, "first_to_idx": a,
                "second_from_idx": d, "second_to_idx": c,
                "first_delta_loss": first["delta_loss"],
                "second_delta_loss": second["delta_loss"],
                "point_interaction": point_interaction,
            })
        candidates.append(target_pairs)
        summary[ycols[target_idx]] = {
            "full_model_loss_reducing_10pp_directions": len(directions),
            "jointly_feasible_direction_pairs": len(target_pairs),
            "point_complementary_pairs": sum(r["point_interaction"] < -1e-12 for r in target_pairs),
            "point_diminishing_pairs": sum(r["point_interaction"] > 1e-12 for r in target_pairs),
            "point_near_zero_pairs": sum(abs(r["point_interaction"]) <= 1e-12 for r in target_pairs),
        }
    return candidates, summary


def run_analysis(out_dir: Path = OUT, seed: int = 20260924, n_boot: int = 500) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = BASE.load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
    p = train["p"]
    y = train["y"]
    pcols = list(train["pcols"])
    ycols = list(train["ycols"])
    recipe_groups = BASE.recipe_groups(p)
    p_ref = p.mean(axis=0)

    lambda_table_path = Q1_OUT / "lambda_selection_quadratic_sensitivity.csv"
    lambda_table = pd.read_csv(lambda_table_path)
    selected_lambda = float(lambda_table.loc[lambda_table["standardized_mse"].idxmin(), "lambda"])
    model = BASE.fit_quadratic_ridge(p, y, selected_lambda)
    candidates, candidate_summary = build_candidates(p_ref, model, pcols, ycols)
    total_pairs = sum(map(len, candidates))
    if total_pairs == 0:
        raise ValueError("No jointly feasible pairs of loss-reducing 10pp transfers at the reference recipe")

    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    total_groups = len(recipe_groups)
    for _ in range(n_boot):
        drawn_groups = rng.integers(0, total_groups, size=total_groups)
        row_idx = np.concatenate([recipe_groups[int(g)] for g in drawn_groups])
        boot_model = BASE.fit_quadratic_ridge(p[row_idx], y[row_idx], selected_lambda)
        boot_h = interaction_hessian(boot_model, len(pcols))
        target_values = []
        for target_idx, rows in enumerate(candidates):
            if not rows:
                target_values.append(np.empty(0, dtype=float))
                continue
            idx = np.asarray([
                [r["first_to_idx"], r["first_from_idx"], r["second_to_idx"], r["second_from_idx"]]
                for r in rows
            ], dtype=int)
            first_to, first_from, second_to, second_from = idx.T
            delta = 0.01 * (
                boot_h[target_idx, first_to, second_to]
                - boot_h[target_idx, first_to, second_from]
                - boot_h[target_idx, first_from, second_to]
                + boot_h[target_idx, first_from, second_from]
            )
            target_values.append(delta)
        samples.append(np.concatenate(target_values))
    bootstrap = np.vstack(samples)
    if bootstrap.shape != (n_boot, total_pairs) or not np.isfinite(bootstrap).all():
        raise ArithmeticError("Bootstrap interaction matrix has unexpected shape or non-finite values")
    point_vector = np.concatenate([
        np.asarray([r["point_interaction"] for r in rows], dtype=float) for rows in candidates
    ])
    bootstrap_se = bootstrap.std(axis=0, ddof=1)
    valid_se = bootstrap_se > np.finfo(float).eps
    centered_t = np.zeros_like(bootstrap)
    centered_t[:, valid_se] = np.abs(
        (bootstrap[:, valid_se] - point_vector[None, valid_se]) / bootstrap_se[None, valid_se]
    )
    if np.any(~valid_se):
        exact_zero = np.abs(bootstrap[:, ~valid_se] - point_vector[None, ~valid_se]) > 1e-12
        centered_t[:, ~valid_se] = np.where(exact_zero, np.inf, 0.0)
    max_t_critical = float(np.quantile(np.max(centered_t, axis=1), 0.95))
    simultaneous_lower = point_vector - max_t_critical * bootstrap_se
    simultaneous_upper = point_vector + max_t_critical * bootstrap_se

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    flat_offset = 0
    for target_idx, target_pairs in enumerate(candidates):
        start = flat_offset
        stop = start + len(target_pairs)
        target_boot = bootstrap[:, start:stop]
        full_ci = np.quantile(target_boot, [0.025, 0.975], axis=0) if target_pairs else (np.empty((2, 0)))
        target_values = []
        for j, pair in enumerate(target_pairs):
            lo, hi = (float(full_ci[0, j]), float(full_ci[1, j]))
            global_j = start + j
            sim_lo = float(simultaneous_lower[global_j])
            sim_hi = float(simultaneous_upper[global_j])
            prob_negative = float(np.mean(target_boot[:, j] < -1e-12))
            if hi < -1e-12:
                label = "pointwise_complementarity"
            elif lo > 1e-12:
                label = "pointwise_diminishing_returns"
            else:
                label = "pointwise_interval_crosses_zero"
            record = {
                "target": ycols[target_idx],
                "reference_recipe": "A4/A5 arithmetic mean closed composition",
                "step_per_transfer": 0.1,
                "first_from_domain": pcols[pair["first_from_idx"]],
                "first_to_domain": pcols[pair["first_to_idx"]],
                "second_from_domain": pcols[pair["second_from_idx"]],
                "second_to_domain": pcols[pair["second_to_idx"]],
                "first_individual_delta_loss": pair["first_delta_loss"],
                "second_individual_delta_loss": pair["second_delta_loss"],
                "interaction_delta_loss": pair["point_interaction"],
                "bootstrap_ci95_lower": lo,
                "bootstrap_ci95_upper": hi,
                "bootstrap_maxT95_simultaneous_lower": sim_lo,
                "bootstrap_maxT95_simultaneous_upper": sim_hi,
                "bootstrap_probability_interaction_negative": prob_negative,
                "pointwise_label_unadjusted": label,
                "maxT_label_conditional_candidate_set": (
                    "familywise_complementarity" if sim_hi < -1e-12
                    else "familywise_diminishing_returns" if sim_lo > 1e-12
                    else "familywise_interval_crosses_zero"
                ),
            }
            pair_rows.append(record)
            target_values.append(record)
        if target_pairs:
            cls = [r["pointwise_label_unadjusted"] for r in target_values]
            ints = np.asarray([r["interaction_delta_loss"] for r in target_values])
            summary_rows.append({
                "target": ycols[target_idx],
                "full_model_loss_reducing_10pp_directions": candidate_summary[ycols[target_idx]]["full_model_loss_reducing_10pp_directions"],
                "jointly_feasible_direction_pairs": len(target_pairs),
                "point_complementary_pairs": int(np.sum(ints < -1e-12)),
                "point_diminishing_pairs": int(np.sum(ints > 1e-12)),
                "bootstrap_ci_fully_negative_unadjusted": cls.count("pointwise_complementarity"),
                "bootstrap_ci_fully_positive_unadjusted": cls.count("pointwise_diminishing_returns"),
                "bootstrap_ci_cross_zero_unadjusted": cls.count("pointwise_interval_crosses_zero"),
                "bootstrap_maxT95_fully_negative": int(sum(r["bootstrap_maxT95_simultaneous_upper"] < -1e-12 for r in target_values)),
                "bootstrap_maxT95_fully_positive": int(sum(r["bootstrap_maxT95_simultaneous_lower"] > 1e-12 for r in target_values)),
                "bootstrap_maxT95_cross_zero": int(sum(
                    r["bootstrap_maxT95_simultaneous_lower"] <= 1e-12
                    and r["bootstrap_maxT95_simultaneous_upper"] >= -1e-12
                    for r in target_values
                )),
                "median_interaction_delta_loss": float(np.median(ints)),
                "median_bootstrap_ci_width": float(np.median([r["bootstrap_ci95_upper"] - r["bootstrap_ci95_lower"] for r in target_values])),
            })
        flat_offset = stop

    pair_df = pd.DataFrame(pair_rows)
    summary_df = pd.DataFrame(summary_rows)
    pair_path = out_dir / "a4_a5_transfer_pair_complementarity_bootstrap.csv"
    summary_path = out_dir / "a4_a5_transfer_pair_complementarity_summary.csv"
    pair_df.to_csv(pair_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    total_negative = int(summary_df["bootstrap_ci_fully_negative_unadjusted"].sum())
    total_positive = int(summary_df["bootstrap_ci_fully_positive_unadjusted"].sum())
    total_cross = int(summary_df["bootstrap_ci_cross_zero_unadjusted"].sum())
    total_sim_negative = int(summary_df["bootstrap_maxT95_fully_negative"].sum())
    total_sim_positive = int(summary_df["bootstrap_maxT95_fully_positive"].sum())
    total_sim_cross = int(summary_df["bootstrap_maxT95_cross_zero"].sum())
    total_candidates = int(summary_df["jointly_feasible_direction_pairs"].sum())
    donors = candidate_summary["reference_share_at_least_10pp_donors"]
    report = [
        "# A4/A5 配比转移互补性敏感性分析",
        "",
        "## 范围与定义",
        "",
        f"本分析只复用 A4/A5 的 {len(p)} 个闭合配比和 13 个 Loss 目标，按 Q1.3 已有的零友好二次 p-only 敏感性模型及其冻结正则强度 lambda={selected_lambda:.9g} 估计转移交叉效应；不改写 Q1 主模型、不接入 B 侧、不解释为独立 Q 效应。512 个样本对应 {total_groups} 个精确配方组。",
        "",
        "参照配方取 512 个 A4/A5 配方的算术平均。候选方向限定为：在该参照配方上从份额至少 10pp 的供体域转出 10pp 到任一接收域，且二次敏感性模型预测该单一转移降低目标 Loss。对两个单一有益转移 u、v，计算：",
        "",
        "`I_uv = L(p + 0.1(u+v)) - L(p + 0.1u) - L(p + 0.1v) + L(p)`。",
        "",
        "I_uv < 0 表示两项转移同时实施时的 Loss 降幅超过各自降幅之和，称为该参照配方与方向下的互补；I_uv > 0 表示边际降幅减弱，称为递减/替代；区间跨零则不作方向判断。只保留两个转移共同实施后仍落在单纯形内的组合。",
        "",
        f"参照配方的合格供体域为：{', '.join(donors)}。候选筛选基于全样本二次敏感性点估计；随后按精确配方组有放回 Bootstrap {n_boot} 次，固定 lambda 与候选方向，对每个组合给逐点 percentile 95% 区间。区间未作多重比较调整，且未包含模型族、lambda、参照配方和候选筛选的不确定性。",
        "",
        "## 汇总",
        "",
        f"13 个 Loss 目标共形成 {total_candidates} 个满足条件的转移对；全样本二次模型中互补方向 {int((pair_df['interaction_delta_loss'] < -1e-12).sum())} 个、递减/替代方向 {int((pair_df['interaction_delta_loss'] > 1e-12).sum())} 个。未校正的逐点 Bootstrap 95% 区间全负 {total_negative} 个、全正 {total_positive} 个、跨零 {total_cross} 个。对全部 6,551 个候选转移对采用居中 recipe-bootstrap max-|t| 近似单步家族校正后，全负 {total_sim_negative} 个、全正 {total_sim_positive} 个、跨零 {total_sim_cross} 个；校正仅覆盖全样本模型筛出的候选集合，不覆盖候选筛选本身。",
        "",
        "| Loss 目标 | 可行且单项降 Loss 的转移数 | 转移对数 | 点估计互补 | 点估计递减/替代 | 点区间全负* | 点区间全正* | maxT 全负† | maxT 全正† |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        report.append(
            f"| {row['target']} | {int(row['full_model_loss_reducing_10pp_directions'])} | {int(row['jointly_feasible_direction_pairs'])} | {int(row['point_complementary_pairs'])} | {int(row['point_diminishing_pairs'])} | {int(row['bootstrap_ci_fully_negative_unadjusted'])} | {int(row['bootstrap_ci_fully_positive_unadjusted'])} | {int(row['bootstrap_maxT95_fully_negative'])} | {int(row['bootstrap_maxT95_fully_positive'])} |"
        )
    report += [
        "",
        "*点区间逐项 95% percentile 区间，未作多重比较调整。†maxT 区间在已筛选的 6,551 个候选交叉效应上作近似单步家族校正；候选转移由同一全样本模型选出，所以即使 maxT 也不校正前置筛选，也不是独立确认性检验。",
        "",
        "## 解释边界",
        "",
        "主 p-only 线性模型的转移交叉效应为 0 是其线性形式的结构结果。此处二次面给出的是 A4/A5 内、固定正则化形式下的探索性曲率诊断；不同目标可能呈现不同转移对关系，不能概括成普遍的域对排序。特别是该结果不能迁移成 B1–B5 的 p 边际效应、B 侧 M1 系数或联合 p/Q 标度律。跨尺度 A 侧预测限制、B 侧 p-Loss 连接为 0、B4/B5 不可比的门禁仍有效。",
        "",
        "## 复现",
        "",
        f"`D:\\Anaconda\\python.exe src/q2_a_side_complementarity_sensitivity.py --seed {seed} --bootstrap {n_boot}`",
    ]
    report_path = out_dir / "a4_a5_transfer_pair_complementarity_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    inputs = [
        BASE.DEFAULT_DATA_ROOT / "train_mixture_1m.csv",
        BASE.DEFAULT_DATA_ROOT / "train_pile_loss_1m.csv",
        Q1_OUT / "lambda_selection_quadratic_sensitivity.csv",
        ROOT / "src" / "f_q1_3_mixture_loss.py",
    ]
    manifest = {
        "status": "CONDITIONAL_A_SIDE_EXPLORATORY",
        "seed": seed,
        "bootstrap_resamples": n_boot,
        "bootstrap_unit": "exact closed-composition recipe group; 512 groups in the current source table",
        "lambda": selected_lambda,
        "reference_recipe": "arithmetic mean of the 512 closed A4/A5 training recipes",
        "eligible_donor_domains": donors,
        "n_targets": len(ycols),
        "n_transfer_pairs": total_candidates,
        "pointwise_interval_counts_unadjusted": {
            "fully_negative": total_negative,
            "fully_positive": total_positive,
            "cross_zero": total_cross,
        },
        "maxT95_conditional_candidate_set_counts": {
            "critical_value": max_t_critical,
            "fully_negative": total_sim_negative,
            "fully_positive": total_sim_positive,
            "cross_zero": total_sim_cross,
            "candidate_pair_count": total_candidates,
            "selection_adjusted": False,
        },
        "limitations": [
            "quadratic model is a pre-existing sensitivity form, not Q1.3 primary model",
            "candidate transfer directions selected by the full-sample fitted model",
            "percentile intervals are pointwise without multiplicity adjustment",
            "A4/A5 source-local only; not transferable to B1-B5 p effects",
        ],
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in inputs],
        "outputs": {
            pair_path.name: sha256(pair_path),
            summary_path.name: sha256(summary_path),
            report_path.name: sha256(report_path),
        },
        "code": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    q2_code_dir = ROOT / "Q2" / "02_代码"
    q2_code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), q2_code_dir / Path(__file__).name)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run_analysis(args.output, args.seed, args.bootstrap)
    print(json.dumps(result, ensure_ascii=False, indent=2))
