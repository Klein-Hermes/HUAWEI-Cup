#!/usr/bin/env python
"""Audit how much each BBH leaf task influences the six-family Q4 index."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
HISTORIC_RESULTS = PROJECT_ROOT / "Q4" / "03_结果" / "results"
REVISION_ROOT = SCRIPT_PATH.parents[1]
FAMILIES = ["BBH", "GPQA", "IFEval", "MATH-Hard", "MMLU-Pro", "MuSR"]
TOP_N = 100
NUMERIC_TOLERANCE = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_descending_rank(scores: pd.Series) -> pd.Series:
    """Integer ranks, breaking exact score ties by model name ascending."""
    frame = pd.DataFrame({"model_name_raw": scores.index.astype(str), "score": scores.to_numpy()})
    frame = frame.sort_values(
        ["score", "model_name_raw"], ascending=[False, True], kind="mergesort"
    )
    frame["rank"] = np.arange(1, len(frame) + 1, dtype=int)
    return frame.set_index("model_name_raw")["rank"]


def spearman_r(left: pd.Series, right: pd.Series) -> float:
    """Spearman correlation without scipy (Pearson correlation of average ranks)."""
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    return float(left_rank.corr(right_rank, method="pearson"))


def percentile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q, interpolation="linear"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Run a deterministic 40-model vertical slice."
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override output directory (defaults under this revision)."
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        REVISION_ROOT / "results" / ("smoke" if args.smoke else "full")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = HISTORIC_RESULTS / "q4_task_aggregate.csv"
    family_path = HISTORIC_RESULTS / "q4_task_family_scores.csv"
    leaf_path = HISTORIC_RESULTS / "q4_selected_task_scores.csv"
    for path in (aggregate_path, family_path, leaf_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required historical Q4 result is missing: {path}")

    aggregate = pd.read_csv(aggregate_path, low_memory=False)
    family_long = pd.read_csv(family_path, low_memory=False)
    leaf = pd.read_csv(leaf_path, low_memory=False)

    required_aggregate = {"model_name_raw", "core6_index_0_100"}
    required_family = {"model_name_raw", "family", "family_score_percentile_0_100"}
    required_leaf = {
        "model_name_raw", "family", "task_key", "metric_name", "score_raw",
        "task_percentile_0_100",
    }
    for label, frame, required in (
        ("aggregate", aggregate, required_aggregate),
        ("family", family_long, required_family),
        ("leaf", leaf, required_leaf),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} input missing required columns: {missing}")

    if aggregate["model_name_raw"].astype(str).duplicated().any():
        raise ValueError("Historical aggregate contains duplicate model identifiers.")
    complete = aggregate.loc[aggregate["core6_index_0_100"].notna()].copy()
    complete["model_name_raw"] = complete["model_name_raw"].astype(str)
    complete = complete.set_index("model_name_raw")
    if not complete.index.is_unique:
        raise ValueError("Complete six-family cohort contains duplicate model identifiers.")
    model_names = sorted(complete.index.tolist())
    if args.smoke:
        model_names = model_names[:40]
    if len(model_names) < 20:
        raise ValueError(f"Need at least 20 complete six-family models; got {len(model_names)}.")
    rank_cutoff_n = min(10 if args.smoke else TOP_N, len(model_names))

    family_long = family_long.copy()
    family_long["model_name_raw"] = family_long["model_name_raw"].astype(str)
    family_subset = family_long.loc[
        family_long["model_name_raw"].isin(model_names)
        & family_long["family"].isin(FAMILIES)
    ]
    if family_subset.duplicated(["model_name_raw", "family"]).any():
        raise ValueError("Family score table has duplicate model-family records.")
    family_wide = family_subset.pivot(
        index="model_name_raw", columns="family", values="family_score_percentile_0_100"
    ).reindex(index=model_names, columns=FAMILIES)
    if family_wide.isna().any().any():
        raise ValueError("A complete six-family model is missing at least one family score.")

    reported_index = complete.loc[model_names, "core6_index_0_100"].astype(float)
    reconstructed_index = family_wide.mean(axis=1)
    max_reconstruction_error = float((reported_index - reconstructed_index).abs().max())
    if max_reconstruction_error > NUMERIC_TOLERANCE:
        raise ValueError(
            "Six-family index does not match the equal-family mean of family percentiles; "
            f"maximum absolute difference={max_reconstruction_error:.12g}."
        )

    bbh = leaf.loc[
        leaf["family"].eq("BBH") & leaf["metric_name"].eq("acc_norm")
    ].copy()
    bbh["model_name_raw"] = bbh["model_name_raw"].astype(str)
    if bbh.duplicated(["model_name_raw", "task_key"]).any():
        raise ValueError("BBH leaf table contains duplicate model-task records.")
    tasks = sorted(bbh["task_key"].astype(str).unique().tolist())
    if len(tasks) < 2:
        raise ValueError(f"Expected multiple BBH leaf tasks, found {len(tasks)}.")
    bbh["task_key"] = bbh["task_key"].astype(str)
    bbh_pct = bbh.pivot(
        index="model_name_raw", columns="task_key", values="task_percentile_0_100"
    ).reindex(index=model_names, columns=tasks)
    bbh_raw = bbh.pivot(
        index="model_name_raw", columns="task_key", values="score_raw"
    ).reindex(index=model_names, columns=tasks)
    if bbh_pct.isna().any().any() or bbh_raw.isna().any().any():
        missing_cells = int(bbh_pct.isna().sum().sum() + bbh_raw.isna().sum().sum())
        raise ValueError(
            "Selected complete six-family cohort lacks BBH task-level scores; "
            f"missing cells across raw and percentile matrices={missing_cells}."
        )

    base_rank = stable_descending_rank(reported_index.loc[model_names])
    non_bbh_families = [family for family in FAMILIES if family != "BBH"]
    other_family_total = family_wide[non_bbh_families].sum(axis=1)
    loo_ranks: dict[str, pd.Series] = {}
    loo_rows: list[dict[str, object]] = []
    base_top = set(base_rank.nsmallest(rank_cutoff_n).index)

    for task in tasks:
        reduced_bbh = bbh_pct.drop(columns=[task]).mean(axis=1)
        index_without_task = (other_family_total + reduced_bbh) / len(FAMILIES)
        rank_without_task = stable_descending_rank(index_without_task)
        loo_ranks[task] = rank_without_task
        loo_top = set(rank_without_task.nsmallest(rank_cutoff_n).index)
        abs_score_shift = (index_without_task - reported_index.loc[model_names]).abs()
        abs_rank_shift = (rank_without_task - base_rank).abs()
        max_rank_model = str(abs_rank_shift.idxmax())
        loo_rows.append(
            {
                "omitted_bbh_task": task,
                "model_n": len(model_names),
                "bbh_task_n_in_reaggregation": len(tasks) - 1,
                "task_metric": "acc_norm",
                "spearman_rank_r_vs_baseline": spearman_r(
                    reported_index.loc[model_names], index_without_task
                ),
                "top_n": rank_cutoff_n,
                "top_n_retained_n": len(base_top.intersection(loo_top)),
                "top_n_retained_pct": 100.0 * len(base_top.intersection(loo_top)) / len(base_top),
                "mean_abs_core6_index_change_points": float(abs_score_shift.mean()),
                "max_abs_core6_index_change_points": float(abs_score_shift.max()),
                "median_abs_rank_shift": float(abs_rank_shift.median()),
                "p95_abs_rank_shift": percentile(abs_rank_shift, 0.95),
                "max_abs_rank_shift": int(abs_rank_shift.max()),
                "max_shift_model": max_rank_model,
            }
        )

    loo_table = pd.DataFrame(loo_rows).sort_values("omitted_bbh_task")
    rank_matrix = pd.DataFrame(loo_ranks, index=model_names)
    rank_shift_matrix = rank_matrix.subtract(base_rank, axis="index").abs()
    largest_model_shift = rank_shift_matrix.max(axis=1)
    worst_task_by_model = rank_shift_matrix.idxmax(axis=1)
    model_stability = pd.DataFrame(
        {
            "model_name_raw": model_names,
            "baseline_core6_index_0_100": reported_index.loc[model_names].to_numpy(),
            "baseline_rank": base_rank.loc[model_names].to_numpy(dtype=int),
            "best_rank_across_bbh_loo": rank_matrix.min(axis=1).loc[model_names].to_numpy(dtype=int),
            "worst_rank_across_bbh_loo": rank_matrix.max(axis=1).loc[model_names].to_numpy(dtype=int),
            "median_rank_across_bbh_loo": rank_matrix.median(axis=1).loc[model_names].to_numpy(),
            "max_abs_rank_shift": largest_model_shift.loc[model_names].to_numpy(dtype=int),
            "task_at_max_abs_rank_shift": worst_task_by_model.loc[model_names].to_numpy(),
            "top_n_cutoff": rank_cutoff_n,
            "top_n_count_across_bbh_loo": rank_matrix.le(rank_cutoff_n).sum(axis=1).loc[model_names].to_numpy(dtype=int),
            "bbh_leaf_task_n": len(tasks),
        }
    ).sort_values(["max_abs_rank_shift", "baseline_rank"], ascending=[False, True])

    all_bbh = bbh.loc[bbh["task_key"].isin(tasks)]
    raw_by_task = all_bbh.groupby("task_key", sort=True)["score_raw"]
    core6_raw_by_task = bbh.loc[bbh["model_name_raw"].isin(model_names)].groupby(
        "task_key", sort=True
    )["score_raw"]
    difficulty_rows = []
    for task in tasks:
        values_all = raw_by_task.get_group(task).astype(float)
        values_core = core6_raw_by_task.get_group(task).astype(float)
        difficulty_rows.append(
            {
                "task_key": task,
                "family": "BBH",
                "metric": "acc_norm",
                "valid_model_n_all_candidates": int(values_all.notna().sum()),
                "valid_model_n_core6_complete": int(values_core.notna().sum()),
                "core6_raw_score_mean": float(values_core.mean()),
                "core6_raw_score_median": float(values_core.median()),
                "core6_raw_score_p25": percentile(values_core, 0.25),
                "core6_raw_score_p75": percentile(values_core, 0.75),
            }
        )
    difficulty = pd.DataFrame(difficulty_rows).sort_values(
        "core6_raw_score_median", ascending=True
    )

    output_paths = {
        "leave_one_out": output_dir / "q4_bbh_task_leave_one_out.csv",
        "model_rank_stability": output_dir / "q4_bbh_model_rank_stability.csv",
        "task_difficulty": output_dir / "q4_bbh_task_difficulty_summary.csv",
        "report": output_dir / "q4_bbh_task_influence_report.md",
    }
    loo_table.to_csv(output_paths["leave_one_out"], index=False, encoding="utf-8-sig")
    model_stability.to_csv(output_paths["model_rank_stability"], index=False, encoding="utf-8-sig")
    difficulty.to_csv(output_paths["task_difficulty"], index=False, encoding="utf-8-sig")

    corr_min = loo_table.loc[loo_table["spearman_rank_r_vs_baseline"].idxmin()]
    corr_median = float(loo_table["spearman_rank_r_vs_baseline"].median())
    top_min = loo_table.loc[loo_table["top_n_retained_pct"].idxmin()]
    score_max = loo_table.loc[loo_table["max_abs_core6_index_change_points"].idxmax()]
    rank_max = loo_table.loc[loo_table["max_abs_rank_shift"].idxmax()]
    shift_model = str(rank_max["max_shift_model"])
    difficulty_order = difficulty.sort_values("core6_raw_score_median", ascending=True)
    hardest = difficulty_order.iloc[0]
    easiest = difficulty_order.iloc[-1]
    report = f"""# BBH 逐任务聚合影响分析

**修订状态：** 属于 Q4 继续推进工作包，修订版尚未封包或经队伍复审。

## 分析范围与方法

- 读取历史 Q4 的逐模型逐叶任务得分、逐族聚合分数和六族主指数；输入表没有被改写。
- 分析固定六族完整样本中的 **{len(model_names):,}** 个模型及 BBH **{len(tasks)}** 个 `acc_norm` 叶任务。完整六族既定指数与六族得分等权平均的最大差为 **{max_reconstruction_error:.3g}**。
- 对每个 BBH 叶任务依次执行一次 leave-one-task-out：保留其他 {len(tasks)-1} 个任务的逐任务百分位平均，等权替换 BBH 族分，再与其他五族重新计算六族指数。百分位沿用历史 C8 候选池计算值，不在子样本中重排。
- 排名完全并列时，按模型名称升序稳定打破并列。任务难度表使用同一完整六族样本中的原始 `acc_norm` 分数中位数与四分位数。

## 结果

- 逐个剔除任务后，Spearman 名次相关的最小值为 **{float(corr_min['spearman_rank_r_vs_baseline']):.4f}**（剔除 `{corr_min['omitted_bbh_task']}`），24 项诊断的中位数为 **{corr_median:.4f}**。
- 基线 Top-{rank_cutoff_n} 在逐任务剔除后的最小保留率为 **{float(top_min['top_n_retained_pct']):.1f}%**（剔除 `{top_min['omitted_bbh_task']}`）。
- 六族指数点数变化的最大绝对值为 **{float(score_max['max_abs_core6_index_change_points']):.4f}**，对应剔除 `{score_max['omitted_bbh_task']}`。
- 最大单模型名次位移为 **{int(rank_max['max_abs_rank_shift'])}** 位，发生在剔除 `{rank_max['omitted_bbh_task']}`；模型 `{shift_model}`。
- 在完整六族样本中，BBH 任务原始分数中位数最低/最高的叶任务分别是 `{hardest['task_key']}`（{float(hardest['core6_raw_score_median']):.4f}）和 `{easiest['task_key']}`（{float(easiest['core6_raw_score_median']):.4f}）。此处“难/易”只表示本队列的描述性中位数，不表示题目难度的普遍属性。

## 可复核文件

- `q4_bbh_task_leave_one_out.csv`：24 次任务剔除对应的排序与指数变化诊断。
- `q4_bbh_model_rank_stability.csv`：每个纳入模型跨 24 次剔除的名次范围与最大位移。
- `q4_bbh_task_difficulty_summary.csv`：逐任务有效数及队列原始分数分布。
- `q4_bbh_task_influence_manifest.json`：输入哈希、运行环境、参数和输出哈希。

## 限制

该分析是确定性敏感性检验，不是置信区间或总体抽样不确定性估计；一次只剔除一个 BBH 叶任务，不能代替更广泛的基准有效性验证。模型名称连接仍不确认版本身份。该分析不解除 Loss–Benchmark 拟合或 12/24 月预测的数据门控，也不把描述性差异解释为因果作用。
"""
    output_paths["report"].write_text(report, encoding="utf-8")

    input_paths = [aggregate_path, family_path, leaf_path]
    manifest = {
        "schema_version": 1,
        "analysis_id": "q4_bbh_task_influence_v1",
        "package_status": "revision_in_progress",
        "run_mode": "smoke" if args.smoke else "full",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "inputs": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
        },
        "sample": {
            "complete_core6_models": len(model_names),
            "bbh_leaf_tasks": len(tasks),
            "metric": "acc_norm",
            "top_n_rank_check": rank_cutoff_n,
            "max_reconstructed_index_difference": max_reconstruction_error,
        },
        "method": {
            "aggregation": "equal leaf-task weights within BBH; equal family weights within Core6",
            "sensitivity": "deterministic leave-one-BBH-leaf-task-out",
            "task_percentiles": "reuse historical exact-name candidate-cohort percentiles; do not rerank the complete Core6 subset",
            "tie_break": "descending score, then model_name_raw ascending",
            "random_seed": None,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "reproduction_command": (
            "python Q4/06_修订推进_20260926/scripts/q4_bbh_task_influence.py"
            if not args.smoke
            else "python Q4/06_修订推进_20260926/scripts/q4_bbh_task_influence.py --smoke"
        ),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_paths.values()
        },
    }
    manifest_path = output_dir / "q4_bbh_task_influence_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "run_mode": manifest["run_mode"],
                "model_n": len(model_names),
                "bbh_task_n": len(tasks),
                "max_reconstruction_error": max_reconstruction_error,
                "min_loo_spearman": float(loo_table["spearman_rank_r_vs_baseline"].min()),
                "max_rank_shift": int(loo_table["max_abs_rank_shift"].max()),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - print concise reproducible CLI diagnostic.
        print(f"ERROR: {error}", file=sys.stderr)
        raise
