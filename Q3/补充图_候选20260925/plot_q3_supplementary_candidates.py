#!/usr/bin/env python3
"""Render three supplemental Q3 candidate figures from existing result CSVs.

Run from the repository root with the project environment:
    D:\\Anaconda\\python.exe Q3\\补充图_候选20260925\\plot_q3_supplementary_candidates.py

This script reads existing results only. It does not refit models, rerun a
bootstrap, alter source data, or update formal manifests/reports.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
INPUTS = {
    "bootstrap": ROOT / "Q3/04_结果/Q3_T3_稳健性与配置切换/q3_bootstrap_summary.csv",
    "cost": ROOT / "Q3/04_结果/M0_budget_context_cost_shares.csv",
    "near": ROOT / "Q3/04_结果/Q3_T2_近最优_Pareto_边际收益/q3_near_optimal.csv",
}
STEMS = {
    "bootstrap": "result_q3_supp_bootstrap_loss_sensitivity",
    "cost": "result_q3_supp_fixedq_cost_shares",
    "near": "result_q3_supp_near_optimal_count_response",
}
FIGURE_DPI = 301
# Existing outputs are skipped to prevent replacement.
REFRESH_OWN_DRAFTS = False

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from utils.plot_style import (
    apply_publication_style,
    audit_design,
    audit_layout,
    export_figure,
)


CONTEXT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
CONTEXT_MARKERS = ["o", "s", "^", "D", "P"]
CONTEXT_LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1))]
REQUIRED_SCENARIO_COUNT = 15


class DataContractError(ValueError):
    """Raised when an input does not meet its already documented contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataContractError(message)


def read_csv(name: str) -> pd.DataFrame:
    path = INPUTS[name]
    require(path.is_file(), f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    require(not missing, f"{source}: missing required fields {missing}")


def require_values(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = {
        column: int(frame[column].isna().sum() + frame[column].fillna("").astype(str).str.strip().eq("").sum())
        for column in columns
    }
    missing = {column: count for column, count in missing.items() if count}
    require(not missing, f"{source}: missing values in required fields {missing}")


def numeric(frame: pd.DataFrame, column: str, source: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    require(values.notna().all(), f"{source}: non-numeric values in {column}")
    require(np.isfinite(values.to_numpy(dtype=float)).all(), f"{source}: non-finite values in {column}")
    return values.astype(float)


def parse_bool(series: pd.Series, field: str, source: str) -> pd.Series:
    mapping = {"true": True, "1": True, "false": False, "0": False}
    parsed = series.fillna("").astype(str).str.strip().str.lower().map(mapping)
    require(parsed.notna().all(), f"{source}: unrecognized boolean values in {field}")
    return parsed.astype(bool)


def scenario_keys(frame: pd.DataFrame, source: str) -> list[tuple[int, int]]:
    require_columns(frame, ["budget_flops", "context_length_tokens"], source)
    require_values(frame, ["budget_flops", "context_length_tokens"], source)
    try:
        budgets = [int(str(value).strip()) for value in frame["budget_flops"]]
        contexts = [int(str(value).strip()) for value in frame["context_length_tokens"]]
    except ValueError as error:
        raise DataContractError(f"{source}: invalid integer budget/context value") from error
    frame["_budget"] = budgets
    frame["_context"] = contexts
    pairs = sorted(set(zip(budgets, contexts)))
    budget_values = sorted(set(budgets))
    context_values = sorted(set(contexts))
    expected = {(budget, context) for budget in budget_values for context in context_values}
    require(
        len(pairs) == REQUIRED_SCENARIO_COUNT and set(pairs) == expected,
        f"{source}: expected a complete 3-budget × 5-context 15-scenario grid; found {len(pairs)} pairs",
    )
    return pairs


def budget_power(budget: int) -> int:
    text = str(budget)
    require(text.startswith("1") and set(text[1:]) <= {"0"}, f"Unexpected budget support value: {budget}")
    return len(text) - 1


def validate_bootstrap() -> pd.DataFrame:
    source = "bootstrap summary"
    frame = read_csv("bootstrap")
    required = [
        "scenario_id", "budget_flops", "context_length_tokens", "bootstrap_replicates",
        "n_unique_selected_grid_points", "selected_N_B_p2_5", "selected_N_B_median",
        "selected_N_B_p97_5", "selected_D_B_p2_5", "selected_D_B_median",
        "selected_D_B_p97_5", "predicted_loss_p2_5", "predicted_loss_median",
        "predicted_loss_p97_5", "interpretation",
    ]
    require_columns(frame, required, source)
    require_values(frame, required, source)
    require(len(frame) == REQUIRED_SCENARIO_COUNT, f"{source}: expected 15 rows, found {len(frame)}")
    scenario_keys(frame, source)
    reps = numeric(frame, "bootstrap_replicates", source)
    require((reps == 1000).all(), f"{source}: expected 1,000 existing resamples per scenario")
    unique_points = numeric(frame, "n_unique_selected_grid_points", source)
    require((unique_points == 1).all(), f"{source}: selected-grid uniqueness is not one point per scenario")

    for prefix in ("selected_N_B", "selected_D_B"):
        low = numeric(frame, f"{prefix}_p2_5", source)
        median = numeric(frame, f"{prefix}_median", source)
        high = numeric(frame, f"{prefix}_p97_5", source)
        require(((low == median) & (median == high)).all(), f"{source}: {prefix} percentile range is not degenerate")

    low = numeric(frame, "predicted_loss_p2_5", source)
    median = numeric(frame, "predicted_loss_median", source)
    high = numeric(frame, "predicted_loss_p97_5", source)
    require(((low <= median) & (median <= high)).all(), f"{source}: Loss percentile ordering is contradictory")
    require(frame["interpretation"].str.contains("fixed-Q B1/M0 support grid only", regex=False).all(),
            f"{source}: interpretation does not identify the fixed-Q B1/M0 support domain")
    frame["_loss_low"] = low
    frame["_loss_median"] = median
    frame["_loss_high"] = high
    return frame


def validate_cost_shares() -> pd.DataFrame:
    source = "fixed-Q cost shares"
    frame = read_csv("cost")
    required = [
        "budget_flops", "context_length_tokens", "C_train_flops", "C_Q_flops",
        "C_attn_flops", "C_total_flops", "train_cost_share", "quality_cost_share",
        "attention_cost_share", "interpretation",
    ]
    require_columns(frame, required, source)
    require_values(frame, required, source)
    require(len(frame) == REQUIRED_SCENARIO_COUNT, f"{source}: expected 15 rows, found {len(frame)}")
    scenario_keys(frame, source)
    columns = ["C_train_flops", "C_Q_flops", "C_attn_flops", "C_total_flops",
               "train_cost_share", "quality_cost_share", "attention_cost_share"]
    values = {column: numeric(frame, column, source) for column in columns}
    require((values["C_total_flops"] > 0).all(), f"{source}: non-positive total compute cost")
    for column in ("train_cost_share", "quality_cost_share", "attention_cost_share"):
        require(((values[column] >= 0) & (values[column] <= 1)).all(), f"{source}: {column} outside [0, 1]")
    total_share = values["train_cost_share"] + values["quality_cost_share"] + values["attention_cost_share"]
    require(np.allclose(total_share, 1.0, atol=1e-12, rtol=0), f"{source}: shares do not sum to one")
    require((values["quality_cost_share"] == 0).all(), f"{source}: quality share is not zero under fixed Q")
    require((values["C_Q_flops"] == 0).all(), f"{source}: C_Q_flops is not zero under fixed Q")
    require(np.allclose(values["C_train_flops"] / values["C_total_flops"], values["train_cost_share"], atol=1e-12, rtol=1e-12),
            f"{source}: training shares do not match actual FLOPs columns")
    require(np.allclose(values["C_attn_flops"] / values["C_total_flops"], values["attention_cost_share"], atol=1e-12, rtol=1e-12),
            f"{source}: attention shares do not match actual FLOPs columns")
    require(frame["interpretation"].str.contains("descriptive fixed-Q", regex=False).all(),
            f"{source}: interpretation does not identify descriptive fixed-Q scenarios")
    for column, series in values.items():
        frame[f"_{column}"] = series
    return frame


def validate_near_optimal() -> pd.DataFrame:
    source = "near-optimal candidate table"
    frame = read_csv("near")
    required = [
        "scenario_id", "budget_flops", "context_length_tokens", "grid_id", "final_feasible",
        "loss_gap_relative", "near_optimal_1pct", "near_optimal_3pct", "near_optimal_5pct",
    ]
    require_columns(frame, required, source)
    require_values(frame, [column for column in required if column != "loss_gap_relative"], source)
    scenario_keys(frame, source)
    frame["_final_feasible"] = parse_bool(frame["final_feasible"], "final_feasible", source)
    eligible = frame.loc[frame["_final_feasible"]].copy()
    require(not eligible.empty, f"{source}: no final_feasible rows")
    require_values(eligible, ["scenario_id", "budget_flops", "context_length_tokens", "grid_id", "loss_gap_relative"], source)
    eligible["_gap"] = numeric(eligible, "loss_gap_relative", source)
    require((eligible["_gap"] >= 0).all(), f"{source}: negative relative Loss gaps among feasible candidates")
    require(not eligible.duplicated(["scenario_id", "grid_id"]).any(), f"{source}: duplicate feasible candidate keys")
    eligible["_near_1"] = parse_bool(eligible["near_optimal_1pct"], "near_optimal_1pct", source)
    eligible["_near_3"] = parse_bool(eligible["near_optimal_3pct"], "near_optimal_3pct", source)
    eligible["_near_5"] = parse_bool(eligible["near_optimal_5pct"], "near_optimal_5pct", source)
    require(eligible["scenario_id"].nunique() == REQUIRED_SCENARIO_COUNT,
            f"{source}: expected feasible candidates in all 15 scenarios")
    for scenario_id, group in eligible.groupby("scenario_id", sort=False):
        require((group["_gap"] == 0).sum() == 1, f"{source}: expected one enumerated exact optimum in {scenario_id}")
        for threshold, flag in ((0.01, "_near_1"), (0.03, "_near_3"), (0.05, "_near_5")):
            count = int((group["_gap"] <= threshold).sum())
            flag_count = int(group[flag].sum())
            require(count == flag_count,
                    f"{source}: enumerated {threshold:.0%} count ({count}) conflicts with existing flags ({flag_count}) in {scenario_id}")
    return eligible


def check_stem_available(key: str) -> Path | None:
    stem = OUT_DIR / STEMS[key]
    existing = [
        stem.with_suffix(".svg"),
        stem.with_suffix(".png"),
        OUT_DIR / "_qa" / f"{stem.name}_grayscale.png",
    ]
    found = [path for path in existing if path.exists()]
    if len(found) == len(existing):
        if not REFRESH_OWN_DRAFTS:
            print(f"Skipping {key}: all figure files already exist; nothing will be overwritten.")
            return None
        print(f"Refreshing this task's own first-pass draft for {key}.")
    else:
        require(not found, f"Refusing to overwrite an incomplete set of existing figure files: {found}")
    return stem


def export_checked(fig, stem: Path) -> list[str]:
    fig.canvas.draw()
    layout_issues = audit_layout(fig)
    design_issues = audit_design(fig)
    require(not layout_issues, f"Figure layout audit failed: {layout_issues}")
    require(not design_issues, f"Figure design audit failed: {design_issues}")
    paths = export_figure(
        fig,
        stem,
        dpi=FIGURE_DPI,
        grayscale_preview=True,
        strict_layout=True,
        strict_design=True,
    )
    print(f"Exported: {paths}")
    return list(paths.values())


def plot_bootstrap_loss(frame: pd.DataFrame) -> list[str]:
    stem = check_stem_available("bootstrap")
    if stem is None:
        return [
            str(OUT_DIR / f"{STEMS['bootstrap']}.svg"),
            str(OUT_DIR / f"{STEMS['bootstrap']}.png"),
            str(OUT_DIR / "_qa" / f"{STEMS['bootstrap']}_grayscale.png"),
        ]
    budgets = sorted(frame["_budget"].unique())
    contexts = sorted(frame["_context"].unique())
    require(len(budgets) == 3 and len(contexts) == 5, "Bootstrap chart requires 3 budgets and 5 contexts")
    ordered = frame.sort_values(["_budget", "_context"]).reset_index(drop=True)
    positions = np.arange(len(ordered))
    budget_colors = ["#0072B2", "#E69F00", "#009E73"]
    budget_to_color = dict(zip(budgets, budget_colors))
    context_to_marker = dict(zip(contexts, CONTEXT_MARKERS))
    median_values = ordered["_loss_median"].to_numpy(dtype=float)
    low_deviation = (ordered["_loss_low"].to_numpy(dtype=float) / median_values - 1.0) * 100.0
    high_deviation = (ordered["_loss_high"].to_numpy(dtype=float) / median_values - 1.0) * 100.0
    max_deviation = max(float(np.abs(low_deviation).max()), float(np.abs(high_deviation).max()))
    require(max_deviation > 0, "Bootstrap Loss intervals are all collapsed to a point")

    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=False)
    fig.set_layout_engine(None)
    grid = fig.add_gridspec(1, 2, width_ratios=(6.1, 1.0), left=0.205, right=0.98, top=0.80, bottom=0.245, wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    ax_median = fig.add_subplot(grid[0, 1], sharey=ax)
    for row_index, row in ordered.iterrows():
        budget = int(row["_budget"])
        context = int(row["_context"])
        ax.errorbar(
            0.0,
            row_index,
            xerr=np.array([[-low_deviation[row_index]], [high_deviation[row_index]]]),
            fmt=context_to_marker[context],
            color=budget_to_color[budget],
            markersize=3.8,
            capsize=2.0,
            linewidth=1.0,
            elinewidth=1.05,
            zorder=3,
        )
        ax_median.text(0.0, row_index, f"{median_values[row_index]:.6f}", ha="left", va="center", fontsize=6.7)
    for separator in (4.5, 9.5):
        ax.axhline(separator, color="#9299A1", linewidth=0.55, linestyle=(0, (2, 2)), zorder=0)
    scenario_labels = [
        f"10^{budget_power(int(row['_budget']))}  |  {int(row['_context']):,}"
        for _, row in ordered.iterrows()
    ]
    ax.set_yticks(positions, scenario_labels)
    ax.invert_yaxis()
    x_limit = max_deviation * 1.18
    ax.set_xlim(-x_limit, x_limit)
    ax.axvline(0, color="#24292F", linewidth=0.75, zorder=1)
    ax.set_xlabel("Deviation from scenario median Loss (%)")
    ax.set_ylabel("Budget / context scenario")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}%"))
    ax.grid(axis="x", color="#C8CDD3", linewidth=0.45, alpha=0.75)
    ax.tick_params(axis="y", length=0, labelsize=6.7)
    ax_median.set_xlim(0, 1)
    ax_median.set_xticks([])
    ax_median.tick_params(axis="y", left=False, labelleft=False)
    for spine in ax_median.spines.values():
        spine.set_visible(False)
    ax_median.set_title("Median Loss", fontsize=7.0, pad=7)

    legend_handles = [
        plt.Line2D([0], [0], color=budget_to_color[budget], marker="o", linestyle="none", markersize=4,
                   label=fr"$10^{{{budget_power(budget)}}}$ FLOPs")
        for budget in budgets
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], frameon=False,
               ncol=3, loc="upper center", bbox_to_anchor=(0.59, 0.895), handlelength=1.0, columnspacing=1.0)
    fig.suptitle("Bootstrap parameter sensitivity of Loss", x=0.075, y=0.975, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.075, 0.92, "Horizontal ranges are p2.5–p97.5 deviations from the median; 8 B1 trajectories, 1,000 existing resamples.", fontsize=7.2)
    fig.text(
        0.075,
        0.085,
        "Right column reports the absolute predicted_loss_median. Selected N/D percentiles collapse to one grid point per scenario; these fixed-form M0 parameter-sensitivity ranges are not model-form error or out-of-domain uncertainty.",
        fontsize=6.6,
        wrap=True,
    )
    try:
        return export_checked(fig, stem)
    finally:
        plt.close(fig)


def plot_fixedq_cost(frame: pd.DataFrame) -> list[str]:
    stem = check_stem_available("cost")
    if stem is None:
        return [str(OUT_DIR / f"{STEMS['cost']}{suffix}") for suffix in (".svg", ".png", "_grayscale.png")]
    budgets = sorted(frame["_budget"].unique())
    contexts = sorted(frame["_context"].unique())
    ordered = frame.sort_values(["_budget", "_context"]).reset_index(drop=True)
    labels = [
        f"10^{budget_power(int(budget))}  |  {int(context):,}"
        for budget, context in zip(ordered["_budget"], ordered["_context"])
    ]
    y = np.arange(len(ordered))
    train = ordered["_train_cost_share"].to_numpy(dtype=float) * 100
    attention = ordered["_attention_cost_share"].to_numpy(dtype=float) * 100

    fig, ax = plt.subplots(figsize=(7.2, 5.25), constrained_layout=False)
    fig.set_layout_engine(None)
    ax.barh(y, train, height=0.68, color="#0072B2", edgecolor="white", linewidth=0.35, label="Training")
    ax.barh(
        y,
        attention,
        left=train,
        height=0.68,
        color="#E69F00",
        edgecolor="#30343B",
        linewidth=0.45,
        hatch="//",
        label="Attention",
    )
    for separator in (4.5, 9.5):
        ax.axhline(separator, color="#9299A1", linewidth=0.55, linestyle=(0, (2, 2)), zorder=0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of total compute cost (FLOPs)")
    ax.set_ylabel("Budget / context scenario")
    ax.grid(axis="x", color="#C8CDD3", linewidth=0.45, alpha=0.75, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.19), handlelength=1.5)
    fig.suptitle("Fixed-Q scenario cost composition", x=0.205, y=0.975, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.205, 0.925, "Descriptive shares for 15 existing M0 selected scenarios; Q = Q0 and quality-compute share = 0 throughout.", fontsize=7.2)
    fig.text(
        0.205,
        0.075,
        "Training and attention segments are actual cost shares. This is descriptive scenario accounting; it is not a new optimization result, causal effect, or structural transition.",
        fontsize=6.8,
        wrap=True,
    )
    fig.subplots_adjust(left=0.205, right=0.99, top=0.845, bottom=0.19)
    try:
        return export_checked(fig, stem)
    finally:
        plt.close(fig)


def plot_near_optimal_response(frame: pd.DataFrame) -> list[str]:
    stem = check_stem_available("near")
    if stem is None:
        return [
            str(OUT_DIR / f"{STEMS['near']}.svg"),
            str(OUT_DIR / f"{STEMS['near']}.png"),
            str(OUT_DIR / "_qa" / f"{STEMS['near']}_grayscale.png"),
        ]
    budgets = sorted(frame["_budget"].unique())
    contexts = sorted(frame["_context"].unique())
    require(len(budgets) == 3 and len(contexts) == 5, "Near-optimal chart requires 3 budgets and 5 contexts")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.8), sharex=True, constrained_layout=False)
    fig.set_layout_engine(None)

    for ax, budget in zip(axes, budgets):
        budget_rows = frame.loc[frame["_budget"] == budget]
        panel_max = 1
        for index, context in enumerate(contexts):
            subset = budget_rows.loc[budget_rows["_context"] == context]
            gaps = np.sort(subset["_gap"].to_numpy(dtype=float))
            require(len(gaps) > 0, f"No final_feasible candidates for budget={budget}, context={context}")
            events = np.unique(gaps[gaps <= 0.10])
            tolerance = np.unique(np.concatenate(([0.0], events, [0.10])))
            counts = np.searchsorted(gaps, tolerance, side="right")
            panel_max = max(panel_max, int(counts.max()))
            ax.step(
                tolerance * 100,
                counts,
                where="post",
                color=CONTEXT_COLORS[index],
                linestyle=CONTEXT_LINESTYLES[index],
                linewidth=1.15,
                label=f"{context:,} tokens",
                zorder=3,
            )
        for threshold, style in ((1, "--"), (3, "-."), (5, ":")):
            ax.axvline(threshold, color="#555B62", linewidth=0.65, linestyle=style, alpha=0.75, zorder=1)
        ax.set_title(fr"Budget $10^{{{budget_power(budget)}}}$ FLOPs", pad=6)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, max(2, panel_max * 1.08))
        ax.set_xticks([0, 1, 3, 5, 10], ["0%", "1%", "3%", "5%", "10%"])
        ax.set_xlabel("Allowed relative Loss gap")
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.grid(axis="y", color="#C8CDD3", linewidth=0.45, alpha=0.75)

    axes[0].set_ylabel("Feasible candidate count")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.55, 0.84), handlelength=1.8, columnspacing=1.0)
    fig.suptitle("Near-optimal feasible candidate count", x=0.075, y=0.975, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.075, 0.905, "Exact finite-grid enumeration from final_feasible rows; vertical guides mark the existing 1%, 3%, and 5% cutoffs.", fontsize=7.2)
    fig.text(
        0.075,
        0.075,
        "Counts include candidates with loss_gap_relative ≤ tolerance over 0–10%. Budget panels use independent count axes. Counts describe enumerated options; they are not probabilities or elasticities.",
        fontsize=6.7,
        wrap=True,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.745, bottom=0.255, wspace=0.28)
    try:
        return export_checked(fig, stem)
    finally:
        plt.close(fig)


def run_one(label: str, validate, plot, outputs: list[str], errors: list[str]) -> None:
    try:
        data = validate()
        print(f"Validated {label}: {len(data)} rows")
        paths = plot(data)
        outputs.extend(paths)
    except Exception as error:  # keep independent figure tasks moving
        message = f"{label}: {type(error).__name__}: {error}"
        errors.append(message)
        print(f"STOP {label}: {message}", file=sys.stderr)


def main() -> int:
    apply_publication_style(language="en", width="double")
    plt.rcParams["figure.constrained_layout.use"] = False
    outputs: list[str] = []
    errors: list[str] = []
    run_one("Bootstrap Loss interval", validate_bootstrap, plot_bootstrap_loss, outputs, errors)
    run_one("fixed-Q cost shares", validate_cost_shares, plot_fixedq_cost, outputs, errors)
    run_one("near-optimal candidate response", validate_near_optimal, plot_near_optimal_response, outputs, errors)
    print("Generated files:")
    for path in outputs:
        print(f"  {Path(path).resolve()}")
    if errors:
        print(f"Stopped figure task(s): {len(errors)}", file=sys.stderr)
        return 1
    print("All three supplemental candidate figures passed data and layout checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
