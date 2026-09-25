#!/usr/bin/env python3
"""Standardize existing Q3-T3 bootstrap and paired-switch evidence.

This is a derivative export only: it reads the frozen 1,000-replicate source
tables, verifies their manifests, and does not run or extend any bootstrap.

AI-assisted programming note (2026-09-25): OpenAI Codex assisted with export
organization and visualization. The participating team must review and
disclose AI assistance under the current competition rules.

Run from the repository root:
    python Q3/03_代码/standardize_q3_t3_outputs.py
Use --validate-only for the read-only P1 handoff; it writes no files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
from q3_repro_support import resolve_math_modeling_skill_root

SKILL_ROOT = resolve_math_modeling_skill_root()
FIGURE_SCRIPTS = SKILL_ROOT / "tools" / "figure" / "scripts"
if str(FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BOOTSTRAP_SUMMARY = Path("Q3/04_结果/M0_ND_cluster_bootstrap_frontier.csv")
BOOTSTRAP_FREQUENCY = Path("Q3/04_结果/M0_ND_cluster_bootstrap_selection_frequency.csv")
BOOTSTRAP_MANIFEST = Path("Q3/04_结果/M0_cluster_bootstrap_sensitivity_manifest.json")
TRANSITION_SUMMARY = Path("Q3/04_结果/M0_discrete_transition_pairs.csv")
TRANSITION_DRAWS = Path("Q3/04_结果/M0_discrete_transition_bootstrap_draws.csv")
TRANSITION_MANIFEST = Path("Q3/04_结果/M0_discrete_transition_manifest.json")

OUTPUT_DIR = Path("Q3/04_结果/Q3_T3_稳健性与配置切换")
FIGURE_DIR = OUTPUT_DIR / "figures"
PREVIEW_DIR = OUTPUT_DIR / "_figure_previews"
BOOTSTRAP_OUT = OUTPUT_DIR / "q3_bootstrap_summary.csv"
CONFIG_SWITCH_OUT = OUTPUT_DIR / "q3_configuration_switch.csv"
COST_SWITCH_OUT = OUTPUT_DIR / "q3_cost_structure_switch.csv"
REPORT_OUT = OUTPUT_DIR / "report.md"
CONTRACT_OUT = OUTPUT_DIR / "figure_contract.md"
MANIFEST_OUT = OUTPUT_DIR / "reproduction_manifest.json"
P1_RECEIPT = OUTPUT_DIR / "p1_smoke_receipt.json"

BOOTSTRAP_REPLICATES = 1000
PAIRED_REPLICATES = 1000
ROBUST_SWITCH_THRESHOLD = 0.95
FIGURE_DPI = 300
FIGURE_SPECS = {
    "result_q3_t3_bootstrap_stability": (7.2, 5.6),
    "result_q3_t3_configuration_switch": (7.2, 8.1),
    "result_q3_t3_cost_structure_switch": (7.2, 8.1),
}

BOOTSTRAP_RENAME = {
    "baseline_selection_frequency": "baseline_reselection_rate",
    "modal_selection_frequency": "modal_configuration_rate",
    "unique_selected_grid_points": "n_unique_selected_grid_points",
}

PAIR_COMMON = [
    "pair_id",
    "pair_type",
    "left_scenario_id",
    "right_scenario_id",
    "left_budget_flops",
    "right_budget_flops",
    "left_context_tokens",
    "right_context_tokens",
    "paired_bootstrap_replicates",
]

CONFIG_FIELDS = PAIR_COMMON + [
    "configuration_switch_rate",
    "delta_log_N_p2_5",
    "delta_log_N_median",
    "delta_log_N_p97_5",
    "delta_log_D_p2_5",
    "delta_log_D_median",
    "delta_log_D_p97_5",
    "robust_configuration_switch",
    "robust_switch_threshold",
    "switch_definition",
]

COST_FIELDS = PAIR_COMMON + [
    "cost_structure_switch_rate",
    "cost_share_TV_p05_exact",
    "cost_share_TV_median",
    "delta_training_share_p2_5",
    "delta_training_share_median",
    "delta_training_share_p97_5",
    "robust_cost_structure_switch",
    "robust_switch_threshold",
    "switch_definition",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with (ROOT / path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    output = ROOT / path
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _budget_label(value: str | int) -> str:
    number = int(value)
    digits = str(number)
    exponent = len(digits) - 1
    significand = number // (10**exponent)
    return f"{significand}e{exponent}"


def scenario_id(budget: str | int, context: str | int) -> str:
    return f"B{_budget_label(budget)}_CTX{int(context)}"


def _assert_quantile_order(row: dict[str, str], fields: tuple[str, str, str], label: str) -> None:
    low, median, high = (float(row[field]) for field in fields)
    if not low <= median <= high:
        raise ValueError(f"Quantiles are not ordered for {label}: {low}, {median}, {high}")


def _assert_hash(path: str, expected: str, label: str) -> None:
    candidate = ROOT / path
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256(candidate)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch for {path}: {actual} != {expected}")


def validate_source_manifest(path: Path, script_rel: str | None = None) -> dict[str, Any]:
    manifest = json.loads((ROOT / path).read_text(encoding="utf-8"))
    for rel, expected in manifest.get("inputs", {}).items():
        _assert_hash(rel, expected, "source input")
    for rel, expected in manifest.get("outputs", {}).items():
        _assert_hash(rel, expected, "source output")

    embedded_script = manifest.get("script")
    if isinstance(embedded_script, dict):
        _assert_hash(embedded_script["path"], embedded_script["sha256"], "source script")
    elif manifest.get("script_sha256"):
        command = manifest.get("reproduction_command", "")
        match = re.search(r"(?:python(?:\.exe)?\s+)?([^\s]+\.py)", command)
        rel = script_rel or (match.group(1) if match else None)
        if not rel:
            raise ValueError(f"Cannot resolve source script path from {path}")
        _assert_hash(rel, manifest["script_sha256"], "source script")
    return manifest


def build_standard_tables(
    bootstrap_rows: list[dict[str, str]],
    frequency_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    draw_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(bootstrap_rows) != 15:
        raise ValueError(f"Expected 15 bootstrap scenarios, received {len(bootstrap_rows)}")
    if len(pair_rows) != 22:
        raise ValueError(f"Expected 22 transition pairs, received {len(pair_rows)}")
    if len(draw_rows) != 22 * PAIRED_REPLICATES:
        raise ValueError(f"Expected 22,000 paired draws, received {len(draw_rows)}")

    bootstrap_keys: set[tuple[str, str]] = set()
    bootstrap_out: list[dict[str, Any]] = []
    for row in bootstrap_rows:
        key = (row["budget_flops"], row["context_length_tokens"])
        if key in bootstrap_keys:
            raise ValueError(f"Duplicate bootstrap scenario: {key}")
        bootstrap_keys.add(key)
        if int(row["bootstrap_replicates"]) != BOOTSTRAP_REPLICATES:
            raise ValueError(f"Unexpected bootstrap replicate count in {key}")
        if float(row["baseline_selection_frequency"]) != 1.0:
            raise ValueError(f"Baseline selection frequency is not 100% in {key}")
        if int(row["unique_selected_grid_points"]) != 1:
            raise ValueError(f"Expected one selected grid point in {key}")
        for prefix in ("selected_N_B", "selected_D_B", "predicted_loss"):
            _assert_quantile_order(
                row,
                (f"{prefix}_p2_5", f"{prefix}_median", f"{prefix}_p97_5"),
                f"Bootstrap {key} {prefix}",
            )
        normalized = {BOOTSTRAP_RENAME.get(k, k): v for k, v in row.items()}
        bootstrap_out.append({"scenario_id": scenario_id(*key), **normalized})

    if len(bootstrap_keys) != 15:
        raise ValueError("Bootstrap scenario coverage is incomplete")
    if len(frequency_rows) == 0:
        raise ValueError("Bootstrap selection-frequency detail is empty")
    frequency_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in frequency_rows:
        frequency_by_key[(row["budget_flops"], row["context_length_tokens"])].append(row)
    if set(frequency_by_key) != bootstrap_keys:
        raise ValueError("Bootstrap frequency table scenario keys differ from summary")
    for key, rows in frequency_by_key.items():
        total = sum(int(row["selection_count"]) for row in rows)
        selected = [row for row in rows if int(row["selection_count"]) > 0]
        if total != BOOTSTRAP_REPLICATES or len(selected) != 1:
            raise ValueError(f"Bootstrap frequency detail is inconsistent for {key}")
        summary = next(r for r in bootstrap_rows if (r["budget_flops"], r["context_length_tokens"]) == key)
        if str(selected[0]["run_id"]) != str(summary["baseline_selected_run_id"]):
            raise ValueError(f"Selected run ID differs from baseline for {key}")

    draws_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in draw_rows:
        draws_by_pair[row["pair_id"]].append(row)
    if set(draws_by_pair) != {r["pair_id"] for r in pair_rows}:
        raise ValueError("Paired-draw IDs differ from transition summary")
    for pair_id, rows in draws_by_pair.items():
        if len(rows) != PAIRED_REPLICATES:
            raise ValueError(f"Expected {PAIRED_REPLICATES} paired draws for {pair_id}")
        if len({r["bootstrap_replicate"] for r in rows}) != PAIRED_REPLICATES:
            raise ValueError(f"Duplicate/missing replicate IDs in {pair_id}")
        for draw in rows:
            changed = _as_bool(draw["configuration_changed"])
            run_id_changed = draw["left_run_id"] != draw["right_run_id"]
            nd_changed = (
                float(draw["left_N_B"]) != float(draw["right_N_B"])
                or float(draw["left_D_B"]) != float(draw["right_D_B"])
            )
            if changed != run_id_changed or changed != nd_changed:
                raise ValueError(f"N/D changed flag disagrees with selected point in {pair_id}")
            expected_tv = abs(
                float(draw["left_training_cost_share"])
                - float(draw["right_training_cost_share"])
            )
            actual_tv = float(draw["total_variation_cost_share"])
            if not math.isclose(actual_tv, expected_tv, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Cost-share total variation mismatch in {pair_id}")

    config_out: list[dict[str, Any]] = []
    cost_out: list[dict[str, Any]] = []
    config_robust = 0
    cost_robust = 0
    pair_type_counts: dict[str, int] = defaultdict(int)
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"configuration": 0, "cost": 0})

    for row in pair_rows:
        pair_id = row["pair_id"]
        draws = draws_by_pair[pair_id]
        for prefix in ("delta_log_N", "delta_log_D", "delta_training_share"):
            _assert_quantile_order(
                row,
                (f"{prefix}_p2_5", f"{prefix}_median", f"{prefix}_p97_5"),
                f"Transition {pair_id} {prefix}",
            )
        if not 0 <= float(row["cost_share_TV_p05_exact"]) <= float(row["cost_share_TV_median"]) <= 1:
            raise ValueError(f"Cost-share total-variation quantiles are invalid in {pair_id}")
        config_rate = sum(_as_bool(d["configuration_changed"]) for d in draws) / len(draws)
        cost_rate = sum(float(d["total_variation_cost_share"]) > 0 for d in draws) / len(draws)
        source_config_rate = float(row["configuration_change_frequency"])
        source_cost_rate = float(row["cost_share_change_frequency"])
        if not math.isclose(config_rate, source_config_rate, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"N/D switch frequency mismatch in {pair_id}: {config_rate} != {source_config_rate}")
        if not math.isclose(cost_rate, source_cost_rate, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Cost-structure switch frequency mismatch in {pair_id}: {cost_rate} != {source_cost_rate}")

        config_flag = config_rate >= ROBUST_SWITCH_THRESHOLD
        cost_flag = cost_rate >= ROBUST_SWITCH_THRESHOLD
        if config_flag != _as_bool(row["robust_ND_allocation_change"]):
            raise ValueError(f"N/D robust-switch classification mismatch in {pair_id}")
        if cost_flag != _as_bool(row["robust_cost_mix_change"]):
            raise ValueError(f"Cost robust-switch classification mismatch in {pair_id}")

        common = {
            **{field: row[field] for field in PAIR_COMMON if field not in {"left_scenario_id", "right_scenario_id"}},
            "left_scenario_id": scenario_id(row["left_budget_flops"], row["left_context_tokens"]),
            "right_scenario_id": scenario_id(row["right_budget_flops"], row["right_context_tokens"]),
        }
        config_out.append({
            **common,
            "configuration_switch_rate": f"{config_rate:.6f}",
            **{field: row[field] for field in [
                "delta_log_N_p2_5", "delta_log_N_median", "delta_log_N_p97_5",
                "delta_log_D_p2_5", "delta_log_D_median", "delta_log_D_p97_5",
            ]},
            "robust_configuration_switch": str(config_flag).lower(),
            "robust_switch_threshold": f"{ROBUST_SWITCH_THRESHOLD:.2f}",
            "switch_definition": "paired-bootstrap frequency of a changed selected N/D grid point >= 0.95",
        })
        cost_out.append({
            **common,
            "cost_structure_switch_rate": f"{cost_rate:.6f}",
            **{field: row[field] for field in [
                "cost_share_TV_p05_exact", "cost_share_TV_median",
                "delta_training_share_p2_5", "delta_training_share_median", "delta_training_share_p97_5",
            ]},
            "robust_cost_structure_switch": str(cost_flag).lower(),
            "robust_switch_threshold": f"{ROBUST_SWITCH_THRESHOLD:.2f}",
            "switch_definition": "paired-bootstrap frequency of nonzero exact cost-share total variation >= 0.95",
        })
        pair_type_counts[row["pair_type"]] += 1
        type_counts[row["pair_type"]]["configuration"] += int(config_flag)
        type_counts[row["pair_type"]]["cost"] += int(cost_flag)
        config_robust += int(config_flag)
        cost_robust += int(cost_flag)

    expected_pair_types = {"adjacent_budget": 10, "adjacent_context": 12}
    if dict(pair_type_counts) != expected_pair_types:
        raise ValueError(f"Unexpected pair type coverage: {dict(pair_type_counts)}")
    if config_robust != 17 or cost_robust != 12:
        raise ValueError(f"Unexpected robust switch counts: N/D={config_robust}, cost={cost_robust}")
    if type_counts["adjacent_budget"] != {"configuration": 10, "cost": 0}:
        raise ValueError(f"Unexpected budget-pair switch pattern: {type_counts['adjacent_budget']}")
    if type_counts["adjacent_context"] != {"configuration": 7, "cost": 12}:
        raise ValueError(f"Unexpected context-pair switch pattern: {type_counts['adjacent_context']}")

    summary = {
        "bootstrap_scenarios": len(bootstrap_out),
        "bootstrap_replicates_per_scenario": BOOTSTRAP_REPLICATES,
        "paired_scenario_pairs": len(pair_rows),
        "paired_replicates_per_pair": PAIRED_REPLICATES,
        "paired_draw_rows": len(draw_rows),
        "robust_ND_configuration_switches": config_robust,
        "robust_cost_structure_switches": cost_robust,
        "pair_type_counts": dict(pair_type_counts),
        "robust_switch_counts_by_pair_type": dict(type_counts),
    }
    return bootstrap_out, config_out, cost_out, summary


def _write_markdown(summary: dict[str, Any]) -> None:
    budget_rows = summary["robust_switch_counts_by_pair_type"]
    report = f"""# Q3-T3 Bootstrap 稳定性与情景切换标准化结果

## 标准化范围

本包从现有权威 CSV 派生三张标准表和三张对应图，不重新抽样、不重拟合、不改变固定 Q0/M0/B1 模型合同。源结果仍是数值真值；本目录中的 CSV 是列名统一、切换概念拆分后的可追溯视图。

## 结果

- **情景内稳定性：**{summary['bootstrap_scenarios']} 个预算—上下文情景各使用 {summary['bootstrap_replicates_per_scenario']:,} 次既有轨迹级 Bootstrap；所有情景均 100% 重选基线最优点，且每个情景仅有 1 个被选中的 N/D 网格点。
- **跨情景 N/D 配置切换：**{summary['robust_ND_configuration_switches']}/{summary['paired_scenario_pairs']} 对达到 ≥95% 稳健切换门槛。其中预算相邻比较 {budget_rows['adjacent_budget']['configuration']}/10 对，上下文相邻比较 {budget_rows['adjacent_context']['configuration']}/12 对。
- **跨情景成本结构切换：**{summary['robust_cost_structure_switches']}/{summary['paired_scenario_pairs']} 对达到 ≥95% 稳健切换门槛。其中预算相邻比较 {budget_rows['adjacent_budget']['cost']}/10 对，上下文相邻比较 {budget_rows['adjacent_context']['cost']}/12 对。

两种跨情景切换不是同一指标：全部 10 组预算相邻比较都改变了 N/D 选点，但成本份额保持不变；12 组上下文相邻比较都改变了成本份额，其中 7 组也改变了 N/D 选点。`q3_configuration_switch.csv` 与 `q3_cost_structure_switch.csv` 分开保存，不能把其中一种切换率写成另一种。

## 方法与解释边界

- Bootstrap 重抽样单位为 8 条训练轨迹；1,000 次复本是既有输入，复本数未增加。重选率描述给定 M0 函数形式和 B1 候选网格内的参数传播稳定性，不是独立验证，也不覆盖模型形式偏差或网格外迁移。
- 配置切换使用同一复本编号配对相邻预算/上下文；N/D 切换定义为所选离散网格点改变，成本结构切换定义为精确成本份额总变差非零。两者均以配对频率 ≥95% 标记为稳健切换。
- 本结果固定 Q=Q0，不估计 Q/p 响应，不代表完整 Q3 联合最优。Q3 仍处于未冻结状态。

## 文件与复现

- `q3_bootstrap_summary.csv`：{summary['bootstrap_scenarios']} 行情景内 Bootstrap 摘要。
- `q3_configuration_switch.csv`：{summary['paired_scenario_pairs']} 行 N/D 配置切换摘要。
- `q3_cost_structure_switch.csv`：{summary['paired_scenario_pairs']} 行成本结构切换摘要。
- 三张独立图件位于 `figures/`；坐标、频率口径和判据见 `figure_contract.md`。
- 输入/输出 SHA-256、环境版本和唯一命令见 `reproduction_manifest.json`。

从项目根目录复现：

```powershell
D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/standardize_q3_t3_outputs.py
```
"""
    (ROOT / REPORT_OUT).write_text(report, encoding="utf-8")


def _write_contract() -> None:
    content = """# Q3-T3 标准化图表契约

| 文件 | 核心结论 | 图型与映射 | 统计口径 | 版面 |
|---|---|---|---|---|
| `result_q3_t3_bootstrap_stability` | 15 个情景中，基线最优 N/D 点在每个 1,000 次重抽样中均被重选；每情景只有一个网格点被选中。 | 单变量横向点图；y 刻度标示预算—上下文情景，x 为重选频率。展示每个真实情景值，不画均值柱、不添加未计算的置信区间。 | 每情景 n=1,000 个既有轨迹级参数复本；独立轨迹簇为 8。 | 7.2 × 5.6 in；300 DPI；PDF/SVG/PNG/灰度 PNG。 |
| `result_q3_t3_configuration_switch` | 22 对相邻情景中，17 对的 N/D 最优离散点改变频率达到 95%。 | 分类点图；y 刻度标示 pair_id，颜色和点形区分预算相邻/上下文相邻；x 为配置改变频率，垂直虚线标 95% 判据。 | 每对 n=1,000 个配对复本；每个 pair 单独显示，不连接离散类别。 | 7.2 × 8.1 in；300 DPI；PDF/SVG/PNG/灰度 PNG。 |
| `result_q3_t3_cost_structure_switch` | 22 对中，12 对成本份额总变差非零频率达到 95%；这一判据与 N/D 切换分表、分图。 | 分类点图；与配置切换图使用同样 pair 顺序、颜色和坐标范围，便于配对比较；仅显示成本结构切换率与 95% 判据。 | 每对 n=1,000 个配对复本；变化定义为训练/注意力成本份额的精确总变差非零。 | 7.2 × 8.1 in；300 DPI；PDF/SVG/PNG/灰度 PNG。 |

预算相邻对为 T01–T10，上下文相邻对为 T11–T22；完整左右端点与数值均在相应 CSV。百分比均是既有 1,000 个配对复本的经验频率，不解释为独立样本置信区间。
"""
    (ROOT / CONTRACT_OUT).write_text(content, encoding="utf-8")


def _plot_bootstrap(rows: list[dict[str, Any]]) -> tuple[list[Path], list[tuple[str, str]]]:
    import matplotlib.pyplot as plt
    from visual_qa import audit_layout, render_preview
    from export_figure import export_figure

    ordered = sorted(rows, key=lambda r: (int(r["budget_flops"]), int(r["context_length_tokens"])))
    y = list(range(len(ordered)))
    labels = [f"{_budget_label(r['budget_flops'])} FLOPs · {int(r['context_length_tokens']):,} tokens" for r in ordered]
    rates = [float(r["baseline_reselection_rate"]) * 100 for r in ordered]

    fig, ax = plt.subplots(figsize=FIGURE_SPECS["result_q3_t3_bootstrap_stability"], constrained_layout=True)
    ax.scatter(rates, y, s=26, color="#0072B2", edgecolor="white", linewidth=0.5, zorder=3)
    for rate, row_y in zip(rates, y):
        ax.annotate(f"{rate:.0f}%", (rate, row_y), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("基线最优 N/D 配置重选频率（每情景 n=1,000）")
    ax.set_ylabel("")
    ax.set_title("情景内 Bootstrap 选点稳定性", loc="left", pad=8)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.6, color="#D5DCE3")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stem = ROOT / FIGURE_DIR / "result_q3_t3_bootstrap_stability"
    preview = ROOT / PREVIEW_DIR / "result_q3_t3_bootstrap_stability_preview.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    render_preview(fig, str(preview), dpi=150)
    issues = audit_layout(fig)
    if any(level in {"WARN", "FAIL"} for level, _ in issues):
        raise RuntimeError(f"Bootstrap figure layout audit failed: {issues}")
    outputs = [Path(p) for p in export_figure(
        fig, basename=str(stem), formats=["pdf", "svg", "png"],
        size_inches=FIGURE_SPECS["result_q3_t3_bootstrap_stability"],
        dpi=FIGURE_DPI, grayscale_preview=False, tight=False,
    )]
    outputs.append(_write_grayscale_preview(stem))
    plt.close(fig)
    return outputs + [preview], issues


def _pair_label_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: r["pair_id"])


def _plot_switch(
    rows: list[dict[str, Any]],
    rate_field: str,
    title: str,
    xlabel: str,
    stem_name: str,
) -> tuple[list[Path], list[tuple[str, str]]]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from visual_qa import audit_layout, render_preview
    from export_figure import export_figure

    ordered = _pair_label_rows(rows)
    y = list(range(len(ordered)))
    pair_labels = [row["pair_id"] for row in ordered]
    rates = [float(row[rate_field]) * 100 for row in ordered]
    marker_color = {"adjacent_budget": "#0072B2", "adjacent_context": "#D55E00"}
    marker_style = {"adjacent_budget": "o", "adjacent_context": "s"}

    fig, ax = plt.subplots(figsize=FIGURE_SPECS[stem_name], constrained_layout=True)
    for pair_type in ("adjacent_budget", "adjacent_context"):
        indices = [i for i, row in enumerate(ordered) if row["pair_type"] == pair_type]
        if not indices:
            continue
        ax.scatter(
            [rates[i] for i in indices], indices,
            s=28, marker=marker_style[pair_type], color=marker_color[pair_type],
            edgecolor="white", linewidth=0.5, zorder=3,
        )
    for rate, row_y in zip(rates, y):
        ax.annotate(
            f"{rate:.0f}%", (rate, row_y),
            xytext=(-6, 0) if rate >= 50 else (6, 0), textcoords="offset points",
            ha="right" if rate >= 50 else "left", va="center", fontsize=7,
        )
    ax.axvline(ROBUST_SWITCH_THRESHOLD * 100, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(y, pair_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 95, 100], ["0%", "25%", "50%", "75%", "95%", "100%"])
    ax.set_xlabel(f"{xlabel}（每对 n=1,000）")
    ax.set_ylabel("")
    ax.set_title(title, loc="left", pad=8)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.6, color="#D5DCE3")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=marker_color["adjacent_budget"],
               markeredgecolor="white", label="预算相邻", markersize=5),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=marker_color["adjacent_context"],
               markeredgecolor="white", label="上下文相邻", markersize=5),
        Line2D([0], [0], color="#555555", linestyle="--", label="稳健切换门槛 95%", linewidth=0.9),
    ]
    ax.legend(handles=legend, loc="center", frameon=True, framealpha=0.92, ncol=1, fontsize=7)

    stem = ROOT / FIGURE_DIR / stem_name
    preview = ROOT / PREVIEW_DIR / f"{stem_name}_preview.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    render_preview(fig, str(preview), dpi=150)
    issues = audit_layout(fig)
    if any(level in {"WARN", "FAIL"} for level, _ in issues):
        raise RuntimeError(f"Switch figure layout audit failed for {stem_name}: {issues}")
    outputs = [Path(p) for p in export_figure(
        fig, basename=str(stem), formats=["pdf", "svg", "png"],
        size_inches=FIGURE_SPECS[stem_name], dpi=FIGURE_DPI,
        grayscale_preview=False, tight=False,
    )]
    outputs.append(_write_grayscale_preview(stem))
    plt.close(fig)
    return outputs + [preview], issues


def _prepare_plot_environment() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    from setup_style import setup_style

    style = setup_style(journal="general", lang="zh", use_sciplots=False)
    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.grid": False,
    })
    return style


def _write_grayscale_preview(stem: Path) -> Path:
    from PIL import Image

    color_path = Path(f"{stem}.png")
    grayscale_path = Path(f"{stem}_grayscale.png")
    with Image.open(color_path) as image:
        image.convert("L").save(grayscale_path, dpi=(FIGURE_DPI, FIGURE_DPI))
    return grayscale_path


def write_outputs(
    bootstrap_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    source_meta: dict[str, Any],
) -> None:
    OUTPUT_DIR_ABS = ROOT / OUTPUT_DIR
    OUTPUT_DIR_ABS.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR_ABS = ROOT / FIGURE_DIR
    FIGURE_DIR_ABS.mkdir(parents=True, exist_ok=True)

    bootstrap_fields = ["scenario_id"] + [BOOTSTRAP_RENAME.get(k, k) for k in read_csv(BOOTSTRAP_SUMMARY)[0]]
    write_csv(BOOTSTRAP_OUT, bootstrap_rows, bootstrap_fields)
    write_csv(CONFIG_SWITCH_OUT, config_rows, CONFIG_FIELDS)
    write_csv(COST_SWITCH_OUT, cost_rows, COST_FIELDS)
    _write_contract()
    _write_markdown(summary)

    style = _prepare_plot_environment()
    figures: list[Path] = []
    figure_qa: dict[str, list[dict[str, str]]] = {}
    for outputs, issues, key in [
        (*_plot_bootstrap(bootstrap_rows), "result_q3_t3_bootstrap_stability"),
        (*_plot_switch(config_rows, "configuration_switch_rate", "跨情景 N/D 配置切换", "改变所选 N/D 网格点的配对复本频率", "result_q3_t3_configuration_switch"), "result_q3_t3_configuration_switch"),
        (*_plot_switch(cost_rows, "cost_structure_switch_rate", "跨情景成本结构切换", "成本份额总变差非零的配对复本频率", "result_q3_t3_cost_structure_switch"), "result_q3_t3_cost_structure_switch"),
    ]:
        figures.extend(outputs)
        figure_qa[key] = [{"severity": level, "message": message} for level, message in issues]

    import matplotlib
    import numpy

    source_files = [
        BOOTSTRAP_SUMMARY, BOOTSTRAP_FREQUENCY, BOOTSTRAP_MANIFEST,
        TRANSITION_SUMMARY, TRANSITION_DRAWS, TRANSITION_MANIFEST,
    ]
    output_files = [
        p.relative_to(ROOT) for p in OUTPUT_DIR_ABS.rglob("*")
        if p.is_file() and p != ROOT / MANIFEST_OUT
    ]
    manifest = {
        "artifact": "Q3-T3 standardized Bootstrap stability and distinct transition outputs",
        "status": "T3 standardized; Q3 remains unfrozen",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bootstrap_recomputed": False,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_replicates_per_pair": PAIRED_REPLICATES,
        "independent_trajectory_clusters": 8,
        "robust_switch_threshold": ROBUST_SWITCH_THRESHOLD,
        "source_manifests": {
            BOOTSTRAP_MANIFEST.as_posix(): sha256(ROOT / BOOTSTRAP_MANIFEST),
            TRANSITION_MANIFEST.as_posix(): sha256(ROOT / TRANSITION_MANIFEST),
        },
        "script": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "source_hashes": {p.as_posix(): sha256(ROOT / p) for p in source_files},
        "source_checks": source_meta,
        "counts": summary,
        "switch_definitions": {
            "configuration": "selected N/D grid point differs between adjacent scenarios; robust if paired-bootstrap frequency >= 0.95",
            "cost_structure": "exact total variation of training/attention cost shares is nonzero; robust if paired-bootstrap frequency >= 0.95",
        },
        "claim_boundary": "fixed Q=Q0 and frozen M0 over the observed B1 support grid; trajectory-bootstrap parameter sensitivity, not independent validation, causality, or full Q3 joint optimization",
        "figure_specs_inches": {k: list(v) for k, v in FIGURE_SPECS.items()},
        "figure_dpi": FIGURE_DPI,
        "figure_qa": figure_qa,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": {"matplotlib": matplotlib.__version__, "numpy": numpy.__version__},
        "reproduction_command": f'& "{Path(sys.executable).resolve().as_posix()}" "Q3/03_代码/standardize_q3_t3_outputs.py"',
        "validate_only_command": f'& "{Path(sys.executable).resolve().as_posix()}" "Q3/03_代码/standardize_q3_t3_outputs.py" --validate-only',
        "code_dependencies_sha256": {
            "Q3/03_代码/q3_repro_support.py": sha256(ROOT / "Q3/03_代码/q3_repro_support.py"),
            "utils/plot_style.py": sha256(ROOT / "utils/plot_style.py"),
        },
        "figure_tools": {
            "skill_root": str(SKILL_ROOT),
            "files_sha256": {
                path.relative_to(SKILL_ROOT).as_posix(): sha256(path)
                for path in sorted(FIGURE_SCRIPTS.glob("*.py"))
            },
        },
        "inputs": {p.as_posix(): sha256(ROOT / p) for p in source_files},
        "outputs": {p.as_posix(): sha256(ROOT / p) for p in sorted(output_files)},
    }
    (ROOT / MANIFEST_OUT).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(bootstrap_rows)} bootstrap scenarios")
    print(f"Wrote {len(config_rows)} N/D configuration transitions")
    print(f"Wrote {len(cost_rows)} cost-structure transitions")
    print(f"Robust switches: N/D {summary['robust_ND_configuration_switches']}/22; cost structure {summary['robust_cost_structure_switches']}/22")
    print(f"Figure style: {style}")
    print(f"Manifest: {MANIFEST_OUT.as_posix()}")
    print("No Bootstrap was rerun or extended.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Read and validate real source evidence without writing files.")
    args = parser.parse_args()

    bootstrap_manifest = validate_source_manifest(BOOTSTRAP_MANIFEST)
    transition_manifest = validate_source_manifest(TRANSITION_MANIFEST)
    bootstrap_fields, bootstrap_rows = read_csv(BOOTSTRAP_SUMMARY)
    frequency_fields, frequency_rows = read_csv(BOOTSTRAP_FREQUENCY)
    pair_fields, pair_rows = read_csv(TRANSITION_SUMMARY)
    draw_fields, draw_rows = read_csv(TRANSITION_DRAWS)
    del bootstrap_fields, frequency_fields, pair_fields, draw_fields

    bootstrap_out, config_out, cost_out, summary = build_standard_tables(
        bootstrap_rows, frequency_rows, pair_rows, draw_rows,
    )
    source_meta = {
        "bootstrap_manifest_phase": bootstrap_manifest.get("artifact"),
        "transition_manifest_scope": transition_manifest.get("scope"),
        "transition_bootstrap_cluster_count": transition_manifest.get("bootstrap_cluster_count"),
        "source_manifest_hashes_verified": True,
    }
    if args.validate_only:
        print(json.dumps({"status": "PASS", "write_mode": "read-only", **summary}, ensure_ascii=False, indent=2))
        return 0

    write_outputs(bootstrap_out, config_out, cost_out, summary, source_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
