#!/usr/bin/env python3
"""Audit support-grid budget saturation and exact discrete M0 allocation changes.

The analysis is limited to the frozen 1,176-point B1 candidate grid, fixed
Q=Q0, and the existing M0 predicted-loss surface. It does not estimate a
continuous KKT threshold or a full Q/p optimum.

Run from the repository root:
    python Q3/03_代码/analyze_q3_budget_grid_transitions.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Q3" / "04_结果"
OUT_DIR = RESULTS / "预算饱和与有限网格切换"
FIG_DIR = OUT_DIR / "figures"
SMOKE_DIR = OUT_DIR / "_p1_smoke"
CANDIDATE_PATH = RESULTS / "m3_candidate_grid.csv"
COST_PATH = RESULTS / "M3_M4成本独立审计" / "m_cost_grid.csv"
FRONTIER_PATH = RESULTS / "M0_ND_reference_frontier.csv"
CONTEXTS = [2048, 4096, 8192, 32768, 131072]
BUDGETS = [10**19, 10**22, 10**24]
EXPECTED_CANDIDATES = 1176
ETA = Fraction(1, 5000)
TRAINING_COEFFICIENT = 6
UNIT_SCALE = 10**9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fraction_text(value: Fraction) -> str:
    with localcontext() as ctx:
        ctx.prec = 32
        out = Decimal(value.numerator) / Decimal(value.denominator)
    return format(out.normalize(), "f")


def ratio_text(numerator: Fraction, denominator: Fraction) -> str:
    with localcontext() as ctx:
        ctx.prec = 16
        out = (Decimal(numerator.numerator) / Decimal(numerator.denominator)) / (
            Decimal(denominator.numerator) / Decimal(denominator.denominator)
        )
    return format(out, ".12g")


def exact_cost(n_b: str, d_b: str, context: int) -> Fraction:
    n_abs = int(Fraction(n_b) * UNIT_SCALE)
    d_abs = int(Fraction(d_b) * UNIT_SCALE)
    nd = n_abs * d_abs
    return Fraction(TRAINING_COEFFICIENT * nd) + ETA * nd * context


def read_inputs() -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, str]]]:
    candidate_rows = read_csv(CANDIDATE_PATH)
    cost_rows = read_csv(COST_PATH)
    frontier_rows = read_csv(FRONTIER_PATH)
    if len(candidate_rows) != EXPECTED_CANDIDATES:
        raise ValueError(f"Expected {EXPECTED_CANDIDATES} B1 points, got {len(candidate_rows)}")
    if len(frontier_rows) != len(CONTEXTS) * len(BUDGETS):
        raise ValueError("The frozen M0 frontier must contain 15 formal budget/context rows")

    candidates: list[dict[str, Any]] = []
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for raw in candidate_rows:
        grid_id = str(raw["grid_id"])
        if grid_id in candidate_by_id:
            raise ValueError(f"Duplicate grid_id in candidate table: {grid_id}")
        if raw["observed_exact"].strip().lower() != "true":
            raise ValueError(f"Candidate {grid_id} is outside the exact observed B1 grid")
        loss = float(raw["m0_predicted_val_loss"])
        n = float(raw["N_params_B"])
        d = float(raw["D_tokens_B"])
        if not (math.isfinite(loss) and math.isfinite(n) and math.isfinite(d) and n > 0 and d > 0):
            raise ValueError(f"Invalid loss or N/D values for grid_id={grid_id}")
        item: dict[str, Any] = {
            "grid_id": grid_id,
            "run_id": str(raw["source_run_id"]),
            "n_b": n,
            "d_b": d,
            "n_raw": raw["N_params_B"],
            "d_raw": raw["D_tokens_B"],
            "loss": loss,
        }
        candidates.append(item)
        candidate_by_id[grid_id] = item

    if len({(c["n_raw"], c["d_raw"]) for c in candidates}) != EXPECTED_CANDIDATES:
        raise ValueError("Expected 1,176 distinct exact observed N/D candidates")

    costs_by_context: dict[int, list[dict[str, Any]]] = defaultdict(list)
    cost_ids: set[tuple[int, str]] = set()
    if len(cost_rows) != EXPECTED_CANDIDATES * len(CONTEXTS):
        raise ValueError("Independent cost table must contain 5,880 rows")
    for raw in cost_rows:
        context = int(raw["context_length_tokens"])
        grid_id = str(raw["grid_id"])
        key = (context, grid_id)
        if key in cost_ids:
            raise ValueError(f"Duplicate context/grid_id cost row: {key}")
        cost_ids.add(key)
        if context not in CONTEXTS or grid_id not in candidate_by_id:
            raise ValueError(f"Unexpected cost row: {key}")
        candidate = candidate_by_id[grid_id]
        if Fraction(raw["N_params_B"]) != Fraction(candidate["n_raw"]):
            raise ValueError(f"N coordinate mismatch between cost and candidate tables: {key}")
        if Fraction(raw["D_tokens_B"]) != Fraction(candidate["d_raw"]):
            raise ValueError(f"D coordinate mismatch between cost and candidate tables: {key}")
        if raw["Q_setting"] != "Q=Q0" or float(raw["C_Q_flops"]) != 0:
            raise ValueError(f"Expected fixed Q=Q0 and zero quality cost: {key}")
        total = exact_cost(candidate["n_raw"], candidate["d_raw"], context)
        if not math.isclose(float(raw["C_total_flops"]), float(total), rel_tol=1e-12, abs_tol=1.0):
            raise ValueError(f"Recomputed cost differs from audited cost table: {key}")
        costs_by_context[context].append({**candidate, "cost": total})

    for context in CONTEXTS:
        rows = costs_by_context[context]
        if len(rows) != EXPECTED_CANDIDATES:
            raise ValueError(f"Context {context} has {len(rows)} cost rows")

    frontier_keys = {(int(r["budget_flops"]), int(r["context_length_tokens"])) for r in frontier_rows}
    expected_keys = {(budget, context) for budget in BUDGETS for context in CONTEXTS}
    if frontier_keys != expected_keys:
        raise ValueError("Frozen M0 frontier keys differ from the expected 15 formal scenarios")
    return candidates, dict(costs_by_context), frontier_rows


def selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["loss"],
        item["cost"],
        item["n_b"],
        item["d_b"],
        item["run_id"],
    )


def select_feasible(rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    feasible = [row for row in rows if row["cost"] <= budget]
    if not feasible:
        raise ValueError(f"No candidate feasible at budget={budget}")
    return min(feasible, key=selection_key)


def support_boundary_label(candidate: dict[str, Any], all_candidates: list[dict[str, Any]]) -> str:
    n_values = [row["n_b"] for row in all_candidates]
    d_values = [row["d_b"] for row in all_candidates]
    hits = []
    for name, value, boundary in (
        ("N_min", candidate["n_b"], min(n_values)),
        ("N_max", candidate["n_b"], max(n_values)),
        ("D_min", candidate["d_b"], min(d_values)),
        ("D_max", candidate["d_b"], max(d_values)),
    ):
        if math.isclose(value, boundary, rel_tol=0, abs_tol=1e-12):
            hits.append(name)
    return ";".join(hits)


def compute_saturation(
    costs_by_context: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, Fraction]]:
    context_rows: list[dict[str, Any]] = []
    formal_rows: list[dict[str, Any]] = []
    csat_by_context: dict[int, Fraction] = {}
    for context in CONTEXTS:
        rows = costs_by_context[context]
        csat = max(row["cost"] for row in rows)
        csat_by_context[context] = csat
        summary: dict[str, Any] = {
            "context_length_tokens": context,
            "candidate_count": EXPECTED_CANDIDATES,
            "C_sat_support_grid_flops": fraction_text(csat),
            "definition": "max candidate cost over the 1,176 exact observed B1 points",
        }
        for budget, tier in zip(BUDGETS, ("low", "mid", "high")):
            feasible_count = sum(row["cost"] <= budget for row in rows)
            saturated = feasible_count == EXPECTED_CANDIDATES
            share = feasible_count / EXPECTED_CANDIDATES
            summary[f"{tier}_budget_flops"] = budget
            summary[f"{tier}_feasible_count"] = feasible_count
            summary[f"{tier}_feasible_share"] = format(share, ".12g")
            summary[f"{tier}_remaining_candidates"] = EXPECTED_CANDIDATES - feasible_count
            summary[f"{tier}_budget_over_C_sat"] = ratio_text(Fraction(budget), csat)
            summary[f"{tier}_status"] = "support_saturated" if saturated else "budget_screening"
            formal_rows.append({
                "context_length_tokens": context,
                "budget_tier": tier,
                "budget_flops": budget,
                "C_sat_support_grid_flops": fraction_text(csat),
                "budget_over_C_sat": ratio_text(Fraction(budget), csat),
                "candidate_count": EXPECTED_CANDIDATES,
                "budget_feasible_candidate_count": feasible_count,
                "budget_feasible_share": format(share, ".12g"),
                "remaining_candidates_to_saturation": EXPECTED_CANDIDATES - feasible_count,
                "support_saturated": saturated,
                "status": "support_saturated" if saturated else "budget_screening",
            })
        context_rows.append(summary)
    if not all(row["support_saturated"] for row in formal_rows if row["budget_tier"] == "high"):
        raise ValueError("At 1e24 FLOPs, not all 1,176 candidates are feasible in every context")
    return context_rows, formal_rows, csat_by_context


def compute_envelopes(
    candidates: list[dict[str, Any]],
    costs_by_context: dict[int, list[dict[str, Any]]],
    csat_by_context: dict[int, Fraction],
    contexts: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    events: list[dict[str, Any]] = []
    events_by_context: dict[int, list[dict[str, Any]]] = {}
    for context in contexts or CONTEXTS:
        rows = sorted(costs_by_context[context], key=lambda row: (row["cost"], selection_key(row)))
        groups: dict[Fraction, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["cost"]].append(row)
        best: dict[str, Any] | None = None
        feasible_count = 0
        context_events: list[dict[str, Any]] = []
        for threshold in sorted(groups):
            group = groups[threshold]
            feasible_count += len(group)
            entering_best = min(group, key=selection_key)
            if best is not None and selection_key(entering_best) >= selection_key(best):
                continue
            previous = best
            best = entering_best
            event = {
                "context_length_tokens": context,
                "transition_index": len(context_events) + 1,
                "budget_threshold_flops": fraction_text(threshold),
                "budget_threshold_over_C_sat": ratio_text(threshold, csat_by_context[context]),
                "feasible_candidate_count_at_threshold": feasible_count,
                "feasible_candidate_share_at_threshold": format(feasible_count / EXPECTED_CANDIDATES, ".12g"),
                "previous_grid_id": previous["grid_id"] if previous else "",
                "previous_run_id": previous["run_id"] if previous else "",
                "previous_N_params_B": format(previous["n_b"], ".12g") if previous else "",
                "previous_D_tokens_B": format(previous["d_b"], ".12g") if previous else "",
                "previous_m0_predicted_val_loss": format(previous["loss"], ".12g") if previous else "",
                "selected_grid_id": best["grid_id"],
                "selected_run_id": best["run_id"],
                "selected_N_params_B": format(best["n_b"], ".12g"),
                "selected_D_tokens_B": format(best["d_b"], ".12g"),
                "selected_m0_predicted_val_loss": format(best["loss"], ".12g"),
                "loss_improvement": format(previous["loss"] - best["loss"], ".12g") if previous else "",
                "configuration_changed": previous is None or previous["grid_id"] != best["grid_id"],
                "support_boundary_hit": support_boundary_label(best, candidates),
                "transition_type": "initial_feasible_optimum" if previous is None else "new_lower_loss_candidate",
            }
            context_events.append(event)
        if not context_events:
            raise ValueError(f"No discrete lower-envelope event found for context={context}")
        events.extend(context_events)
        events_by_context[context] = context_events
    return events, events_by_context


def compute_formal_switches(
    costs_by_context: dict[int, list[dict[str, Any]]],
    events_by_context: dict[int, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    frontier_rows: list[dict[str, str]],
    contexts: list[int] | None = None,
) -> list[dict[str, Any]]:
    frontier = {
        (int(row["budget_flops"]), int(row["context_length_tokens"])): row
        for row in frontier_rows
    }
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for context in contexts or CONTEXTS:
        for budget in BUDGETS:
            item = select_feasible(costs_by_context[context], budget)
            reference = frontier[(budget, context)]
            if item["run_id"] != str(reference["selected_run_id"]):
                raise ValueError(
                    f"Selected run differs from frozen M0 frontier at C={budget}, L={context}: "
                    f"{item['run_id']} != {reference['selected_run_id']}"
                )
            count = sum(row["cost"] <= budget for row in costs_by_context[context])
            if count != int(reference["b1_feasible_grid_points"]):
                raise ValueError(f"Feasible-count mismatch against frozen frontier at C={budget}, L={context}")
            selected[(budget, context)] = item

    results: list[dict[str, Any]] = []
    for context in contexts or CONTEXTS:
        for idx, (before_budget, after_budget) in enumerate(zip(BUDGETS, BUDGETS[1:]), start=1):
            before = selected[(before_budget, context)]
            after = selected[(after_budget, context)]
            switches_inside = [
                event for event in events_by_context[context]
                if before_budget < Fraction(event["budget_threshold_flops"]) <= after_budget
            ]
            changed = before["grid_id"] != after["grid_id"]
            results.append({
                "context_length_tokens": context,
                "budget_interval": "low_to_mid" if idx == 1 else "mid_to_high",
                "budget_before_flops": before_budget,
                "budget_after_flops": after_budget,
                "selected_before_grid_id": before["grid_id"],
                "selected_before_run_id": before["run_id"],
                "selected_before_N_params_B": format(before["n_b"], ".12g"),
                "selected_before_D_tokens_B": format(before["d_b"], ".12g"),
                "selected_before_m0_predicted_val_loss": format(before["loss"], ".12g"),
                "selected_after_grid_id": after["grid_id"],
                "selected_after_run_id": after["run_id"],
                "selected_after_N_params_B": format(after["n_b"], ".12g"),
                "selected_after_D_tokens_B": format(after["d_b"], ".12g"),
                "selected_after_m0_predicted_val_loss": format(after["loss"], ".12g"),
                "configuration_changed": changed,
                "number_of_envelope_switches_inside_interval": len(switches_inside),
                "loss_improvement": format(before["loss"] - after["loss"], ".12g"),
                "support_boundary_hit_after": support_boundary_label(after, candidates),
                "all_candidates_feasible_after": sum(
                    row["cost"] <= after_budget for row in costs_by_context[context]
                ) == EXPECTED_CANDIDATES,
                "formal_frontier_parity": True,
            })
    return results


def make_chart(
    costs_by_context: dict[int, list[dict[str, Any]]],
    csat_by_context: dict[int, Fraction],
    output_dir: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    skill_scripts = Path(r"C:\Users\86147\.codex\skills\math-modeling\tools\figure\scripts")
    if str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    from export_figure import export_figure
    from setup_style import setup_style
    from visual_qa import audit_layout, print_report

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    setup_style(journal="general", lang="zh", use_sciplots=False, constrained_layout=False)
    matplotlib.rcParams["axes.unicode_minus"] = False
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    for budget in BUDGETS:
        ax.axvline(float(budget), color="#AAB0B6", linestyle=":", linewidth=0.8, zorder=0)
    for context, color in zip(CONTEXTS, colors):
        csat = csat_by_context[context]
        ordered = sorted(costs_by_context[context], key=lambda row: row["cost"])
        unique_costs = sorted({row["cost"] for row in ordered})
        shares = [
            sum(row["cost"] <= threshold for row in ordered) / EXPECTED_CANDIDATES
            for threshold in unique_costs
        ]
        ax.step(
            [float(value) for value in unique_costs],
            shares,
            where="post",
            color=color,
            linewidth=1.7,
            label=f"L={context:,}",
        )
        ax.axvline(float(csat), color=color, linestyle="--", linewidth=0.8, alpha=0.65, zorder=1)
        ax.scatter([float(csat)], [1.0], s=16, color=color, zorder=3)
    ax.axhline(1.0, color="#6B7280", linestyle=":", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlim(1e16, 1.1e24)
    ax.set_ylim(0, 1.035)
    ax.set_xticks([1e16, *(float(budget) for budget in BUDGETS)])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"1e{round(math.log10(value))}" if value > 0 else "")
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("预算 C (FLOPs)", fontsize=8, labelpad=5)
    ax.set_ylabel("可行候选比例", fontsize=8, labelpad=4)
    ax.set_title("固定 Q0 下的 B1 预算覆盖；虚线标记 C_sat", fontsize=9)
    ax.grid(True, axis="both", color="#D7DCE1", linewidth=0.55, alpha=0.8)
    ax.legend(ncol=3, loc="upper left", fontsize=7.5)
    fig.tight_layout(pad=1.2)
    issue_lines = audit_layout(fig)
    qa_report = print_report(issue_lines)
    if any(severity == "FAIL" for severity, _ in issue_lines):
        raise RuntimeError(f"Chart layout check failed: {issue_lines}")
    output_stem = output_dir / "result_q3_budget_saturation_finite_grid_switches"
    outputs = export_figure(
        fig,
        output_stem,
        formats=["pdf", "svg", "png"],
        dpi=300,
        size_inches=fig.get_size_inches(),
        grayscale_preview=True,
        tight=True,
    )
    plt.close(fig)
    return {"paths": outputs, "layout_verdict": qa_report, "layout_issues": issue_lines}


def render_report(
    saturation_rows: list[dict[str, Any]],
    formal_switches: list[dict[str, Any]],
    events_by_context: dict[int, list[dict[str, Any]]],
) -> str:
    lines = [
        "# Q3 预算饱和与有限网格切换分析",
        "",
        "**范围：** 固定 Q=Q₀、冻结 M0 预测 Loss、B1 的 1,176 个精确观测 N/D 候选点。成本按 C=6ND+(NDL)/5000 重算，并与独立 M3/M4 成本网格逐点核对。",
        "",
        "## 判定口径",
        "",
        "- 当前离散支持域的饱和预算定义为 C_sat(L)=max_j C_j(L)。预算不低于该值时，这 1,176 个候选点全部预算可行；它不是支持域外配置的成本上界，也不表示全局物理饱和。",
        "- 状态只用 budget_screening（当前网格仍有候选不可行）和 support_saturated（当前网格 1,176/1,176 可行）。同时给出预算/C_sat、可行点数、可行占比及剩余点数，不设置人为“接近饱和”阈值。",
        "- 离散配置转折是有限支持网格下的精确枚举：候选在其实际成本处进入可行集合，若按既有 M1 顺序规则（最低 M0 Loss，其次成本、N、D、run_id）成为新最优点，则记录该成本为切换阈值。没有连续 KKT 临界值或网格间插值。",
        "- 下表的正式预算情景均与既有 M0 前沿逐点核对；所有结论不扩展到 Q/p 联合经验最优。",
        "",
        "## 支持域饱和预算",
        "",
        "| 上下文 L | C_sat (FLOPs) | 1e19 可行 | 1e22 可行 | 1e24 可行 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in saturation_rows:
        lines.append(
            f"| {row['context_length_tokens']:,} | {float(row['C_sat_support_grid_flops']):.6g} | "
            f"{row['low_feasible_count']:,}/1,176 ({float(row['low_feasible_share']):.1%}) | "
            f"{row['mid_feasible_count']:,}/1,176 ({float(row['mid_feasible_share']):.1%}) | "
            f"{row['high_feasible_count']:,}/1,176 ({float(row['high_feasible_share']):.1%}) |"
        )
    lines += [
        "",
        "1e24 FLOPs 在 5 个上下文、15 个预算×上下文情景中均覆盖 1,176/1,176 个候选点。各情景距支持域饱和点的比值、候选可行份额和剩余候选数见 q3_formal_budget_coverage.csv。",
        "",
        "![候选预算覆盖随绝对预算变化](figures/result_q3_budget_saturation_finite_grid_switches.png)",
        "",
        "## 正式预算间配置切换",
        "",
        "| 上下文 L | 预算区间 | 配置是否改变 | 区间内离散切换数 | Loss 下降 | 切换后支持边界 |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in formal_switches:
        lines.append(
            f"| {row['context_length_tokens']:,} | {row['budget_interval']} | "
            f"{'是' if row['configuration_changed'] else '否'} | "
            f"{row['number_of_envelope_switches_inside_interval']} | "
            f"{float(row['loss_improvement']):.6g} | {row['support_boundary_hit_after'] or '无'} |"
        )
    total_events = sum(len(rows) for rows in events_by_context.values())
    lines += [
        "",
        f"完整预算轴上识别到 {total_events} 个有限网格的最优点事件（含每个上下文的首个可行最优点）。所有事件阈值和切换前后配置见 q3_finite_grid_switch_events.csv；10 个正式预算区间的端点对照见 q3_formal_budget_switches.csv。",
        "",
        "## 输出与复现",
        "",
        "- q3_budget_saturation_by_context.csv：每个上下文的 C_sat 与三档预算覆盖摘要。",
        "- q3_formal_budget_coverage.csv：15 个正式预算情景的可行候选数、比例、剩余数及饱和状态。",
        "- q3_finite_grid_switch_events.csv：完整预算轴上的离散下包络切换阈值。",
        "- q3_formal_budget_switches.csv：低→中、中→高两段正式预算端点比较。",
        "- reproduction_manifest.json：脚本、输入、输出哈希及范围声明。",
        "",
        "复现命令：python Q3/03_代码/analyze_q3_budget_grid_transitions.py",
        "",
    ]
    return "\n".join(lines)


def run_smoke(
    candidates: list[dict[str, Any]],
    costs_by_context: dict[int, list[dict[str, Any]]],
    frontier_rows: list[dict[str, str]],
) -> dict[str, Any]:
    context = CONTEXTS[0]
    rows = costs_by_context[context]
    csat = max(row["cost"] for row in rows)
    low_selected = select_feasible(rows, BUDGETS[0])
    mid_selected = select_feasible(rows, BUDGETS[1])
    high_selected = select_feasible(rows, BUDGETS[2])
    refs = {
        int(row["budget_flops"]): row
        for row in frontier_rows if int(row["context_length_tokens"]) == context
    }
    for budget, selected in ((BUDGETS[0], low_selected), (BUDGETS[1], mid_selected), (BUDGETS[2], high_selected)):
        if selected["run_id"] != refs[budget]["selected_run_id"]:
            raise ValueError(f"P1 parity failed for context={context}, budget={budget}")
    if sum(row["cost"] <= BUDGETS[2] for row in rows) != EXPECTED_CANDIDATES:
        raise ValueError("P1 high-budget saturation assertion failed")
    _, events_by_context = compute_envelopes(
        candidates,
        costs_by_context,
        {context: csat},
        contexts=[context],
    )
    formal_switches = compute_formal_switches(
        costs_by_context,
        events_by_context,
        candidates,
        frontier_rows,
        contexts=[context],
    )
    if len(formal_switches) != 2 or len(events_by_context[context]) < 2:
        raise ValueError("P1 discrete-envelope or formal interval check did not run")
    return {
        "artifact": "Q3 budget-grid transition analysis P1 smoke receipt",
        "status": "PASS",
        "scope": "one real context (2048 tokens), all 1,176 exact observed B1 candidates, complete discrete envelope and the three formal budgets",
        "checks": {
            "candidate_count": len(candidates),
            "cost_rows_for_context": len(rows),
            "cost_recomputed_against_independent_grid": True,
            "formal_frontier_selection_parity": True,
            "finite_grid_envelope_event_count": len(events_by_context[context]),
            "formal_budget_interval_comparison_count": len(formal_switches),
            "high_budget_support_saturation": True,
            "C_sat_support_grid_flops": fraction_text(csat),
        },
        "formal_selected_run_ids": {
            "1e19": low_selected["run_id"],
            "1e22": mid_selected["run_id"],
            "1e24": high_selected["run_id"],
        },
    }


def run_full(
    candidates: list[dict[str, Any]],
    costs_by_context: dict[int, list[dict[str, Any]]],
    frontier_rows: list[dict[str, str]],
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    saturation_rows, formal_rows, csat_by_context = compute_saturation(costs_by_context)
    events, events_by_context = compute_envelopes(candidates, costs_by_context, csat_by_context)
    formal_switches = compute_formal_switches(costs_by_context, events_by_context, candidates, frontier_rows)

    saturation_path = OUT_DIR / "q3_budget_saturation_by_context.csv"
    coverage_path = OUT_DIR / "q3_formal_budget_coverage.csv"
    events_path = OUT_DIR / "q3_finite_grid_switch_events.csv"
    switches_path = OUT_DIR / "q3_formal_budget_switches.csv"
    report_path = OUT_DIR / "report.md"
    write_csv(saturation_path, saturation_rows)
    write_csv(coverage_path, formal_rows)
    write_csv(events_path, events)
    write_csv(switches_path, formal_switches)
    report_path.write_text(
        render_report(saturation_rows, formal_switches, events_by_context),
        encoding="utf-8",
    )
    chart_qa = make_chart(costs_by_context, csat_by_context, FIG_DIR)

    chart_contract = OUT_DIR / "figure_contract.md"
    chart_contract.write_text(
        "# 图表契约：固定 Q₀ 下的预算饱和与有限网格切换\n\n"
        "横轴为绝对预算 C（FLOPs），纵轴为 1,176 个精确观测 B1 候选的预算可行比例；"
        "每条阶梯线对应一个上下文，彩色竖虚线和圆点表示该上下文的 C_sat，灰色竖点线表示 1e19、1e22、1e24 三档正式预算，"
        "横虚线表示 100% 候选可行。由于成本可写成 ND×(6+L/5000)，相对 C_sat 的五条曲线会完全重合，因此图中使用绝对预算轴。"
        "数据来自独立审计成本网格，并按题面代理成本公式逐点复算。"
        "图中不插值、不外推，也不估计连续 KKT 转折。正式 1e19/1e22/1e24 情景的精确覆盖数字见 CSV 表。\n\n"
        "输出为 SVG/PDF/300 DPI PNG 和灰度 PNG；使用数学建模 figure helper 做导出及版面预检。"
        "该图是当前 Q3 报告的第 7 图候选，不代表全 Q3 全局图审计通过。\n",
        encoding="utf-8",
    )
    input_paths = [CANDIDATE_PATH, COST_PATH, FRONTIER_PATH]
    output_paths = [
        saturation_path, coverage_path, events_path, switches_path, report_path, chart_contract,
        FIG_DIR / "result_q3_budget_saturation_finite_grid_switches.png",
        FIG_DIR / "result_q3_budget_saturation_finite_grid_switches.svg",
        FIG_DIR / "result_q3_budget_saturation_finite_grid_switches.pdf",
        FIG_DIR / "result_q3_budget_saturation_finite_grid_switches_grayscale.png",
    ]
    manifest_path = OUT_DIR / "reproduction_manifest.json"
    manifest = {
        "artifact": "Q3 support-grid budget saturation and finite-grid transition analysis",
        "status": "PASS",
        "command": "python Q3/03_代码/analyze_q3_budget_grid_transitions.py",
        "generated_on": date.today().isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "script": {
            "path": "Q3/03_代码/analyze_q3_budget_grid_transitions.py",
            "sha256": sha256(Path(__file__)),
        },
        "inputs_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in input_paths
        },
        "outputs_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in output_paths
        },
        "summary": {
            "contexts": len(CONTEXTS),
            "formal_budget_context_scenarios": len(formal_rows),
            "support_grid_candidates_per_context": EXPECTED_CANDIDATES,
            "finite_grid_switch_events_including_initial_points": len(events),
            "formal_budget_interval_comparisons": len(formal_switches),
            "formal_frontier_selection_parity": True,
            "all_1e24_scenarios_support_saturated": True,
        },
        "chart_qa": chart_qa,
        "claim_boundary": (
            "fixed Q=Q0, frozen M0, exact observed 1,176-point B1 support grid; "
            "no continuous KKT threshold, support-external optimum, or full Q/p empirical optimum"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a real-input P1 slice for one context")
    args = parser.parse_args()
    candidates, costs_by_context, frontier_rows = read_inputs()
    if args.smoke:
        receipt = run_smoke(candidates, costs_by_context, frontier_rows)
        SMOKE_DIR.mkdir(parents=True, exist_ok=True)
        receipt["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        receipt["script_sha256"] = sha256(Path(__file__))
        receipt["input_hashes"] = {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (CANDIDATE_PATH, COST_PATH, FRONTIER_PATH)
        }
        path = SMOKE_DIR / "p1_smoke_receipt.json"
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": receipt["status"], "receipt": path.as_posix()}, ensure_ascii=False))
        return 0
    manifest = run_full(candidates, costs_by_context, frontier_rows)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_DIR.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, ZeroDivisionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
