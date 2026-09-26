# AI assistance: OpenAI Codex desktop 26.917.71314 (build 10954), developer OpenAI; used 2026-09-26.
# AI contribution: chart code/layout based on frozen project evidence; the team remains responsible for the claims and final verification.
# Model selector exact display name:
# Model version/publication date:

"""Generate the current F-problem Q1–Q4 evidence and dependency flowchart.

Run with the project Anaconda Python:
    D:\\Anaconda\\python.exe src\\plot_f_overall_model.py

The script uses the already available Matplotlib and Pillow installations. It
writes synchronized copies to the current Q1 figure locations because the
project's only paper-level overview had historically been stored there.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "Q1" / "04_图表" / "q1_2_model_final",
    ROOT / "Q1" / "04_图表" / "q1_2_model",
    ROOT / "figures" / "q1_2_model_final",
    ROOT / "figures" / "q1_2_model",
]


def add_card(ax, x, y, w, h, title, body, color, status=None, body_size=9.2):
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.10",
        facecolor="#FFFFFF", edgecolor="#48515A", linewidth=1.15,
        zorder=2,
    )
    ax.add_patch(card)
    header_h = 0.48
    ax.add_patch(Rectangle(
        (x, y + h - header_h), w, header_h,
        facecolor=color, edgecolor="none", zorder=3,
    ))
    ax.plot([x, x + w], [y + h - header_h, y + h - header_h],
            color="#59636E", linewidth=0.75, zorder=4)
    ax.text(x + w / 2, y + h - header_h / 2, title,
            ha="center", va="center", fontsize=12, weight="bold",
            color="#20252B", zorder=5)
    body_y = y + h - header_h - 0.22
    ax.text(x + 0.22, body_y, body,
            ha="left", va="top", fontsize=body_size, linespacing=1.32,
            color="#20252B", zorder=5)
    if status:
        ax.text(x + w / 2, y + 0.20, status,
                ha="center", va="center", fontsize=8.7, weight="bold",
                color="#20252B",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": color,
                      "edgecolor": "#59636E", "linewidth": 0.7},
                zorder=6)


def add_arrow(ax, start, end, *, lw=1.25, color="#59636E", mutation=13,
              connection="arc3,rad=0", zorder=1):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color, connectionstyle=connection,
        shrinkA=0, shrinkB=0, zorder=zorder,
    ))


def add_source(ax):
    points = [(2.45, 9.08), (13.55, 9.08), (13.92, 9.67), (2.82, 9.67)]
    ax.add_patch(Polygon(points, closed=True, facecolor="#EEF1F4",
                          edgecolor="#48515A", linewidth=1.15, zorder=2))
    ax.text(8.18, 9.39,
            "F 题题面 + A/B/C 附件（使用无隐藏字段版数据说明）",
            ha="center", va="center", fontsize=11.0, weight="bold",
            color="#20252B", zorder=3)


def build_figure():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(16, 10), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Shared input to Q1 and Q2, with the C1–C8 branch routed around the Q2 card.
    add_source(ax)
    ax.plot([8.18, 8.18], [9.08, 8.70], color="#59636E", linewidth=1.1, zorder=1)
    ax.plot([3.85, 12.15], [8.70, 8.70], color="#59636E", linewidth=1.1, zorder=1)
    add_arrow(ax, (3.85, 8.70), (3.85, 8.20))
    add_arrow(ax, (12.15, 8.70), (12.15, 8.20))
    ax.plot([12.15, 15.72], [8.70, 8.70], color="#59636E", linewidth=1.1, zorder=1)
    ax.plot([15.72, 15.72], [8.70, 3.18], color="#59636E", linewidth=1.1, zorder=1)
    add_arrow(ax, (15.72, 3.18), (15.50, 3.18))
    ax.text(15.84, 6.35, "C1–C8", rotation=90, ha="center", va="center",
            fontsize=8.3, color="#59636E")

    q1 = (0.50, 5.35, 6.70, 2.85)
    q2 = (8.80, 5.35, 6.70, 2.85)
    q3 = (0.50, 1.75, 6.70, 2.85)
    q4 = (8.80, 1.75, 6.70, 2.85)

    add_card(
        ax, *q1[:4], "Q1 质量信号",
        "• 272,505 行正式操作分接口\n"
        "• q_huber 为项目选定操作分；q_equal 作敏感性\n"
        "• Q1.2 冲突诊断；Q1.3 配比比较有适用边界\n"
        "• 30 条译文辅助双人核查未区分候选；并非全量效度验证",
        "#E8F1F8", status="FROZEN_WITH_LIMITATIONS", body_size=9.0,
    )
    add_card(
        ax, *q2[:4], "Q2 标度律与配比效应",
        "• B1 M0：L = E + A N^(−α) + B D^(−β)\n"
        "• 1,176 个检查点，8 条 Pythia 训练轨迹\n"
        "• A4/A5 p+Q 受限敏感性使用 Q1 q_huber\n"
        "• p Gate 失败：经核验的 p–Loss 行级连接为 0\n"
        "• Q Gate 失败：Q_covered(p) 由 p 推导\n"
        "• R² = 0.999999817（训练内拟合）",
        "#FFF2D7", status="M1/M2 = NOT_ESTIMATED", body_size=8.3,
    )
    add_card(
        ax, *q3[:4], "Q3 预算配置",
        "• 输入 Q1 的 Q0 与 Q2 的 M0\n"
        "• 3 档预算 × 5 档上下文，共 15 个情景\n"
        "• 只在 1,176 个实测 N/D 候选点中选择\n"
        "• β_Q 阈值是条件敏感性，未由数据估计",
        "#E8F3EC", status="固定 Q0 的有限网格参考优化", body_size=9.0,
    )

    # Q4 is split into its measured decomposition, pre-fit gate, and forecast-support outcomes.
    x, y, w, h = q4
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.10",
        facecolor="#FFFFFF", edgecolor="#48515A", linewidth=1.15, zorder=2,
    ))
    header_h = 0.48
    ax.add_patch(Rectangle((x, y + h - header_h), w, header_h,
                           facecolor="#F0EAF6", edgecolor="none", zorder=3))
    ax.plot([x, x + w], [y + h - header_h, y + h - header_h],
            color="#59636E", linewidth=0.75, zorder=4)
    ax.text(x + 0.24, y + h - header_h / 2, "Q4 技术演进与前沿",
            ha="left", va="center", fontsize=12, weight="bold",
            color="#20252B", zorder=5)

    q4_lines = (
        "候选 1,819 个唯一名称；严格版本确认 0\n"
        "共同支持分解 n=645\n"
        "Δsize −2.141（95% CI −2.342 至 −1.939）\n"
        "Δnon-size +3.326（95% CI −1.012 至 7.665）\n"
        "非规模及观测变化区间含 0 → 贡献份额留空\n"
        "12/24 月年度回测起点=0 → 不估计数值预测"
    )
    ax.text(x + 0.24, y + h - 0.70, q4_lines,
            ha="left", va="top", fontsize=8.0, linespacing=1.28,
            color="#20252B", zorder=5)

    # Loss-to-benchmark feasibility decision and blocked outcome.
    diamond = [(13.92, 3.93), (14.90, 3.46), (13.92, 2.99), (12.94, 3.46)]
    ax.add_patch(Polygon(diamond, closed=True, facecolor="#F8F5FA",
                         edgecolor="#59636E", linewidth=1.0, zorder=5))
    ax.text(13.92, 3.46, "Loss 可比性\n门控？", ha="center", va="center",
            fontsize=8.2, color="#20252B", zorder=6)
    blocked = FancyBboxPatch(
        (12.64, 1.93), 2.56, 0.87,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        facecolor="#F4F6F8", edgecolor="#59636E", linewidth=0.9, zorder=5,
    )
    ax.add_patch(blocked)
    ax.text(13.92, 2.36, "门控未通过\nBLOCKED_NO_FIT\nfit_executed = false\n无桥接系数",
            ha="center", va="center", fontsize=8.2, weight="bold",
            color="#20252B", zorder=6)
    add_arrow(ax, (13.92, 2.99), (13.92, 2.80), lw=1.0, mutation=11,
              zorder=6.5)

    # Verified dependencies: Q1 Q0 and Q2 M0 feed the Q3 reference grid;
    # Q2 loss candidates reach Q4's gate only.
    add_arrow(ax, (7.20, 6.36), (8.80, 6.36), lw=1.0, mutation=11,
              zorder=4.5)
    ax.text(8.00, 6.58, "q_huber：仅 A4/A5 敏感性", ha="center", va="center",
            fontsize=7.2, color="#46515B")
    add_arrow(ax, (3.85, 5.35), (3.85, 4.60))
    ax.text(3.55, 4.98, "Q0", ha="right", va="center", fontsize=8.2,
            color="#46515B")
    add_arrow(ax, (8.80, 5.35), (7.20, 4.65))
    ax.text(7.92, 5.15, "M0", ha="center", va="center", fontsize=8.2,
            color="#46515B", rotation=-29)
    add_arrow(ax, (14.85, 5.35), (13.92, 3.93), lw=1.0, mutation=11,
              zorder=4.5)
    ax.text(14.95, 4.89, "Q2 Loss", ha="center", va="center", fontsize=7.5,
            color="#46515B")

    # Q3 and Q4 produce the bounded cross-question answer.
    ax.plot([3.85, 3.85], [1.75, 1.42], color="#59636E", linewidth=1.1, zorder=1)
    ax.plot([12.15, 12.15], [1.75, 1.42], color="#59636E", linewidth=1.1, zorder=1)
    ax.plot([3.85, 12.15], [1.42, 1.42], color="#59636E", linewidth=1.1, zorder=1)
    add_arrow(ax, (8.00, 1.42), (8.00, 1.18))
    output = FancyBboxPatch(
        (3.15, 0.35), 9.70, 0.83,
        boxstyle="round,pad=0.025,rounding_size=0.12",
        facecolor="#EEF1F4", edgecolor="#48515A", linewidth=1.15, zorder=2,
    )
    ax.add_patch(output)
    ax.text(8.00, 0.77,
            "整题输出：逐问报告可识别结果，并保留未估计、门控阻断与适用范围",
            ha="center", va="center", fontsize=10.2, weight="bold",
            color="#20252B", zorder=3)

    fig.suptitle("F 题 Q1–Q4 当前模型、数据依赖与结果边界（截至 2026-09-26）",
                 x=0.5, y=0.988, fontsize=14, weight="bold", color="#20252B")
    fig.text(0.5, 0.012,
             "Q4 分解为描述性关联；Loss–Benchmark 门控先于拟合；12/24 月预测仅在年度历史支持充分时估计。",
             ha="center", va="bottom", fontsize=8.0, color="#59636E")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.035)
    return fig


def main():
    fig = build_figure()
    try:
        for out_dir in OUT_DIRS:
            out_dir.mkdir(parents=True, exist_ok=True)
            base = out_dir / "flow_overall_model"
            fig.savefig(base.with_suffix(".svg"), format="svg", bbox_inches="tight")
            fig.savefig(base.with_suffix(".png"), format="png", dpi=301,
                        bbox_inches="tight", pil_kwargs={"compress_level": 6})
            fig.savefig(base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")

            gray_path = out_dir / "grayscale_previews" / "flow_overall_model_grayscale.png"
            gray_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(base.with_suffix(".png")) as image:
                image.convert("L").save(gray_path, dpi=(301, 301), optimize=True)
    finally:
        plt.close(fig)

    print(f"Generated synchronized F-wide flowcharts in {len(OUT_DIRS)} current figure folders.")


if __name__ == "__main__":
    main()
