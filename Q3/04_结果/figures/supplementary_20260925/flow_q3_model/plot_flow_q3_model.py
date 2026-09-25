#!/usr/bin/env python3
"""Generate a Q3-specific model/solver branch diagram from confirmed sources.

Run from the repository root:
    D:/Anaconda/python.exe Q3/04_结果/figures/supplementary_20260925/flow_q3_model/plot_flow_q3_model.py

The diagram distinguishes computed fixed-Q0 analyses, conditional Q-only
sensitivity, and the upstream evidence gate for joint Q/p identification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image


def find_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "Q3/04_结果/M0_ND_reference_frontier.csv").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate project root from this script.")


ROOT = find_project_root()
SKILL_SCRIPTS = Path(r"C:\Users\86147\.codex\skills\math-modeling\tools\figure\scripts")
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent
SOURCE_REQUIREMENTS = {
    "Q3/03_代码/solve_q3_m0_reference_grid.py": (
        "Q2_GRID_REL",
        "C7_REL",
        "EXPECTED_Q_VERSION",
    ),
    "Q3/03_代码/solve_q3_conditional_q_sensitivity.py": (
        "L_cond",
        "beta_Q",
        "p excluded",
    ),
    "Q3/04_结果/M0_ND_reference_frontier_report.md": (
        "不是完整 Q3 联合最优",
        "C7 上下文情景",
    ),
    "Q3/04_结果/Q3_conditional_Q_optimization_report.md": (
        "未知的 Loss 改善系数",
        "p 不进入模型",
    ),
}

COLORS = {
    "fixed": {"band": "#EEF3F6", "edge": "#36677D", "node": "#FFFFFF"},
    "conditional": {"band": "#F4F0E8", "edge": "#9A6B20", "node": "#FFFFFF"},
    "gate": {"band": "#F0F0F0", "edge": "#555B61", "node": "#FFFFFF"},
    "text": "#20252A",
    "muted": "#50575E",
    "arrow": "#3D454C",
}


def verify_confirmed_sources() -> None:
    missing = []
    mismatches = []
    for relative, terms in SOURCE_REQUIREMENTS.items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for term in terms:
            if term not in text:
                mismatches.append(f"{relative}: expected source wording {term!r}")
    if missing or mismatches:
        details = "; ".join([*(f"missing {item}" for item in missing), *mismatches])
        raise ValueError("Diagram source contract failed: " + details)


def parallelogram(ax, x, y, w, h, text, edge, fill="white", fontsize=8.2):
    skew = min(0.22, w * 0.07)
    points = [(x + skew, y), (x + w, y), (x + w - skew, y + h), (x, y + h)]
    ax.add_patch(Polygon(points, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.25, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=COLORS["text"], linespacing=1.32, zorder=4)


def rectangle(ax, x, y, w, h, text, edge, fill="white", fontsize=8.2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=1.25, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=COLORS["text"], linespacing=1.32, zorder=4)


def rounded(ax, x, y, w, h, text, edge, fill="white", fontsize=8.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.12",
        facecolor=fill, edgecolor=edge, linewidth=1.25, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=COLORS["text"], linespacing=1.32, zorder=4)


def diamond(ax, x, y, w, h, text, edge, fill="white", fontsize=8.0):
    points = [(x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y), (x, y + h / 2)]
    ax.add_patch(Polygon(points, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.25, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=COLORS["text"], linespacing=1.25, zorder=4)


def arrow(ax, start, end, label=None, label_xy=None):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.15,
        color=COLORS["arrow"], connectionstyle="arc3,rad=0", zorder=2,
    ))
    if label and label_xy:
        ax.text(*label_xy, label, ha="center", va="bottom", fontsize=7.0,
                color=COLORS["muted"], zorder=5)


def make_figure():
    setup_style(journal="general", lang="zh", use_sciplots=False)
    plt.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(14.0, 8.7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(7, 8.98, "Q3 建模与求解分支", ha="center", va="center",
            fontsize=14, fontweight="semibold", color=COLORS["text"])
    ax.text(7, 8.66, "固定质量参考分析、条件 Q-only 敏感性与联合 Q/p 的识别边界",
            ha="center", va="center", fontsize=8.8, color=COLORS["muted"])

    lanes = [
        (5.92, 2.42, "已计算｜固定 Q0 主支路", COLORS["fixed"]),
        (3.18, 2.40, "条件假设｜Q-only 支路（βQ 未知）", COLORS["conditional"]),
        (0.48, 2.35, "上游门槛｜完整 Q/p 联合优化尚未识别", COLORS["gate"]),
    ]
    for y, h, label, color in lanes:
        ax.add_patch(Rectangle((0.25, y), 13.5, h, facecolor=color["band"],
                               edgecolor="#D5DADF", linewidth=0.7, zorder=0))
        ax.text(0.48, y + h - 0.18, label, ha="left", va="top", fontsize=8.2,
                fontweight="bold", color=color["edge"])

    fixed = COLORS["fixed"]
    parallelogram(
        ax, 0.55, 6.24, 3.95, 1.34,
        "冻结输入\nQ1 q_huber：Q0 = 0.4984781048\nQ2 B1/M0：1,176 个实测 N/D 点\nC7 context + 预算 / 题面成本函数",
        fixed["edge"], fixed["node"], fontsize=8.1,
    )
    rectangle(
        ax, 5.05, 6.24, 4.0, 1.34,
        "固定 Q = Q0，C_Q(Q0) = 0\n训练 / 注意力成本形成可行约束\n仅枚举 B1 支持点，按 M0 预测 Loss 选点\n3 档预算 × 5 档 context",
        fixed["edge"], fixed["node"], fontsize=8.0,
    )
    rounded(
        ax, 9.95, 6.24, 3.42, 1.34,
        "已计算结果\n15 情景 M0 参考前沿\n复用既有轨迹 Bootstrap\n预算 × context 描述分析\n不含 p；不等于完整联合最优",
        fixed["edge"], fixed["node"], fontsize=7.8,
    )
    arrow(ax, (4.51, 6.91), (5.00, 6.91))
    arrow(ax, (9.06, 6.91), (9.90, 6.91))

    conditional = COLORS["conditional"]
    parallelogram(
        ax, 0.55, 3.48, 3.65, 1.28,
        "复用固定 Q0 的 M0 Loss\nB1 实测 N/D 支持网格\n题面 g(Q) 质量成本",
        conditional["edge"], conditional["node"], fontsize=8.0,
    )
    rectangle(
        ax, 4.67, 3.40, 4.10, 1.44,
        "条件目标：L_cond = L_M0 - βQ(Q-Q0)，βQ >= 0\nβQ 未由 Q2 识别；p 不进入模型\n对每个 N/D 点求预算可负担的 Q",
        conditional["edge"], conditional["node"], fontsize=7.9,
    )
    rounded(
        ax, 9.23, 3.48, 4.14, 1.28,
        "条件 Q-only 敏感性\nβQ 分段区间与 N/D 重分配阈值\n只在该显式假设下成立\n不代表实测 Q 收益或联合最优",
        conditional["edge"], conditional["node"], fontsize=7.8,
    )
    arrow(ax, (4.21, 4.12), (4.62, 4.12))
    arrow(ax, (8.78, 4.12), (9.18, 4.12))

    gate = COLORS["gate"]
    parallelogram(
        ax, 0.55, 0.86, 4.12, 1.30,
        "所需上游证据\n可追溯的配比数据 p\n独立变化的 Q\n可连接的 Q / p → Loss 证据",
        gate["edge"], gate["node"], fontsize=8.0,
    )
    diamond(ax, 5.45, 0.78, 2.35, 1.43, "识别门槛\n齐备？", gate["edge"], gate["node"], fontsize=8.0)
    rounded(
        ax, 8.88, 0.86, 4.49, 1.30,
        "当前状态：完整 Q/p 联合优化未识别\n缺少可追溯配比与可连接的响应证据\n本轮没有求得完整联合最优",
        gate["edge"], gate["node"], fontsize=8.0,
    )
    arrow(ax, (4.69, 1.51), (5.40, 1.51))
    arrow(ax, (7.86, 1.51), (8.83, 1.51), "当前证据门未满足", (8.34, 1.63))

    ax.text(
        0.45, 0.20,
        "依据：Q3 fixed-Q M0 reference、T3 既有 Bootstrap 与 T4 conditional Q-only 代码 / 报告；"
        "流程图不表示新增拟合、重抽样或联合优化。",
        ha="left", va="center", fontsize=6.8, color=COLORS["muted"],
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Q3 model/solver branch diagram.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="Output directory; relative paths resolve from project root.")
    args = parser.parse_args()
    verify_confirmed_sources()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    stem = output_dir / "flow_q3_model"
    expected = [stem.with_suffix(".svg"), stem.with_suffix(".png"), output_dir / "flow_q3_model_grayscale.png"]
    if any(path.exists() for path in expected):
        raise FileExistsError("Refusing to overwrite existing diagram artifacts: " +
                              ", ".join(str(path) for path in expected if path.exists()))
    fig = make_figure()
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
        size_inches=(14.0, 8.7),
        grayscale_preview=False,
        tight=True,
        pad_inches=0.08,
    )
    gray_path = output_dir / "flow_q3_model_grayscale.png"
    with Image.open(stem.with_suffix(".png")) as source_image:
        source_image.convert("L").save(gray_path, dpi=(301, 301))
    files.append(str(gray_path))
    plt.close(fig)
    print(f"Source assertions: {len(SOURCE_REQUIREMENTS)} code/report files verified.")
    print(f"Layout audit: {verdict}; exported: {len(files)} files.")
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
