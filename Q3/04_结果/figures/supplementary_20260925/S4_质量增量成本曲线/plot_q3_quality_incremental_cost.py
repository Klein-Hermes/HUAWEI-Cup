#!/usr/bin/env python3
"""Plot the already computed prompt-proxy Q cost envelope.

Run from the repository root:
    D:/Anaconda/python.exe Q3/04_结果/figures/supplementary_20260925/S4_质量增量成本曲线/plot_q3_quality_incremental_cost.py

Only the existing Q_cost_increment_envelope.csv is plotted. This script does
not fit a Q-to-Loss response, resample data, or choose an optimal Q.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from PIL import Image


def find_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "Q3/04_结果/Q_cost_increment_envelope.csv").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate project root from this script.")


ROOT = find_project_root()
SKILL_SCRIPTS = Path(r"C:\Users\86147\.codex\skills\math-modeling\tools\figure\scripts")
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report  # noqa: E402


SOURCE = ROOT / "Q3/04_结果/Q_cost_increment_envelope.csv"
DEFAULT_OUT = Path(__file__).resolve().parent
Q0_EXPECTED = 0.49847810484764643
FUNCTIONS = ("exponential", "power4", "logarithmic")
COLORS = {
    "exponential": "#0072B2",
    "power4": "#D55E00",
    "logarithmic": "#009E73",
}
LINES = {
    "exponential": ("-", "o"),
    "power4": ("--", "s"),
    "logarithmic": ("-.", "D"),
}
REFERENCE_LEVELS = ("min", "median", "max")
EXPECTED_D = {
    "min": 134_000_000,
    "median": 146_801_000_000,
    "max": 299_893_000_000,
}
REQUIRED = {
    "g_function",
    "Q0_q_huber",
    "Q_target",
    "incremental_flops_per_token",
    "D_B1_min_tokens",
    "C_Q_at_D_B1_min_flops",
    "D_B1_median_tokens",
    "C_Q_at_D_B1_median_flops",
    "D_B1_max_tokens",
    "C_Q_at_D_B1_max_flops",
}


def load_and_validate() -> pd.DataFrame:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    data = pd.read_csv(SOURCE, encoding="utf-8-sig")
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Required source fields are missing: {sorted(missing)}")
    if len(data) != 36 or data.isna().any().any():
        raise ValueError(f"Expected 36 complete rows; got {len(data)} rows.")
    if set(data["g_function"].unique()) != set(FUNCTIONS):
        raise ValueError("The source does not contain exactly the three contracted g(Q) forms.")
    q0_values = data["Q0_q_huber"].astype(float).unique()
    if len(q0_values) != 1 or not np.isclose(q0_values[0], Q0_EXPECTED, atol=1e-12, rtol=0):
        raise ValueError("Q0 does not match the report's frozen q_huber value.")
    expected_q = np.array([Q0_EXPECTED, *np.arange(0.50, 1.001, 0.05)])
    for function in FUNCTIONS:
        part = data.loc[data["g_function"] == function].sort_values("Q_target")
        if len(part) != 12 or not np.allclose(part["Q_target"].to_numpy(), expected_q, atol=1e-10, rtol=0):
            raise ValueError(f"Unexpected Q support for {function}.")
        baseline = part.loc[np.isclose(part["Q_target"], Q0_EXPECTED, atol=1e-12)]
        if len(baseline) != 1:
            raise ValueError(f"Expected one exact Q0 row for {function}.")
        for level in REFERENCE_LEVELS:
            d_col = f"D_B1_{level}_tokens"
            c_col = f"C_Q_at_D_B1_{level}_flops"
            d_values = part[d_col].astype(float).unique()
            if len(d_values) != 1 or not np.isclose(d_values[0], EXPECTED_D[level], rtol=0, atol=0.5):
                raise ValueError(f"Unexpected B1 {level} D reference.")
            if float(baseline.iloc[0][c_col]) != 0.0:
                raise ValueError(f"Q0 cost must be exactly zero for {function} / {level}.")
            expected_cost = part[d_col].to_numpy(dtype=float) * part["incremental_flops_per_token"].to_numpy(dtype=float)
            if not np.allclose(expected_cost, part[c_col].to_numpy(dtype=float), rtol=1e-12, atol=1e-6):
                raise ValueError(f"Stored cost product does not reconcile for {function} / {level}.")
            if (part.loc[part["Q_target"] > Q0_EXPECTED, c_col] <= 0).any():
                raise ValueError(f"Positive targets must have positive cost for {function} / {level}.")
    return data


def make_figure(data: pd.DataFrame):
    style = setup_style(journal="general", lang="zh", use_sciplots=False)
    plt.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(7.2, 7.8))
    fig.set_layout_engine("none")
    grid = fig.add_gridspec(
        4, 1, height_ratios=(0.54, 1, 1, 1),
        left=0.14, right=0.98, top=0.97, bottom=0.16, hspace=0.28,
    )
    header = fig.add_subplot(grid[0])
    header.axis("off")
    axes = [fig.add_subplot(grid[1])]
    axes.extend(fig.add_subplot(grid[row], sharex=axes[0]) for row in (2, 3))

    for ax, level in zip(axes, REFERENCE_LEVELS):
        d_value = EXPECTED_D[level]
        level_positive = data.loc[
            (data["Q_target"] > Q0_EXPECTED), f"C_Q_at_D_B1_{level}_flops"
        ].astype(float).to_numpy()
        ax.set_yscale("log")
        ax.set_ylim(float(level_positive.min()) / 1.35, float(level_positive.max()) * 1.3)
        ax.set_xlim(0.5, 1.015)
        for function in FUNCTIONS:
            part = data.loc[
                (data["g_function"] == function) & (data["Q_target"] > Q0_EXPECTED)
            ].sort_values("Q_target")
            line_style, marker = LINES[function]
            ax.plot(
                part["Q_target"].to_numpy(dtype=float),
                part[f"C_Q_at_D_B1_{level}_flops"].to_numpy(dtype=float),
                color=COLORS[function],
                linestyle=line_style,
                marker=marker,
                markersize=3.4,
                linewidth=1.3,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=function,
                zorder=3,
            )
        ax.set_title(
            f"B1 D {level} = {d_value:,.0f} tokens",
            loc="left",
            fontsize=8.2,
            pad=4,
            fontweight="semibold",
        )
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.grid(axis="y", which="major", color="#D9DEE3", linewidth=0.55)
        ax.grid(axis="y", which="minor", color="#EBEEF1", linewidth=0.25, alpha=0.45)
        ax.tick_params(axis="both", labelsize=7.2, length=2.8, width=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(np.arange(0.5, 1.01, 0.1))
    axes[-1].set_xlabel("目标质量 Q", labelpad=-1)
    axes[0].set_ylabel("C_Q（FLOPs）", fontsize=7.6, labelpad=6)
    legend_handles = [
        Line2D(
            [0], [0], color=COLORS[name], linestyle=LINES[name][0],
            marker=LINES[name][1], markerfacecolor="white", markersize=4,
            linewidth=1.3, label=name,
        )
        for name in FUNCTIONS
    ]
    header.text(0.5, 0.88, "质量提升的题面代理成本",
        ha="center", va="center", fontsize=11, fontweight="semibold", color="#20252A")
    header.text(
        0.5, 0.51,
        "Q0=0.4984781048 处 C_Q=0 FLOPs（精确零值不绘于对数轴）；"
        "仅显示 Q≥0.50 的正成本，各面板独立对数范围",
        ha="center", va="center", fontsize=7.4, color="#41464B",
    )
    header.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=3, frameon=False, fontsize=7.3, handlelength=2.0, columnspacing=1.8,
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the Q3 prompt-proxy quality cost envelope.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="Output directory; relative paths resolve from project root.")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    stem = output_dir / "result_q3_supp_quality_incremental_cost"
    expected = [stem.with_suffix(".svg"), stem.with_suffix(".png"), output_dir / f"{stem.name}_grayscale.png"]
    if any(path.exists() for path in expected):
        raise FileExistsError("Refusing to overwrite existing figure artifacts: " +
                              ", ".join(str(path) for path in expected if path.exists()))
    data = load_and_validate()
    fig = make_figure(data)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    if any(severity == "FAIL" for severity, _ in issues):
        raise RuntimeError("Programmatic layout audit failed; no final files were exported.")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = export_figure(
        fig,
        basename=str(stem),
        formats=("svg", "png"),
        dpi=301,
        size_inches=(7.2, 7.8),
        grayscale_preview=False,
        tight=True,
        pad_inches=0.08,
    )
    gray_path = output_dir / f"{stem.name}_grayscale.png"
    with Image.open(stem.with_suffix(".png")) as source_image:
        source_image.convert("L").save(gray_path, dpi=(301, 301))
    files.append(str(gray_path))
    plt.close(fig)
    print(f"Rows validated: {len(data)}; functions: {len(FUNCTIONS)}; Q0 zero baselines: 3.")
    print(f"Layout audit: {verdict}; exported: {len(files)} files.")
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
