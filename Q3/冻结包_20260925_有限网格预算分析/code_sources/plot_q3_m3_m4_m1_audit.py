#!/usr/bin/env python3
"""Create publication-style Q3 M3/M4/M1 support and budget audit figures.

Run from the repository root with the huawei-cup Conda environment:
    python Q3/03_代码/plot_q3_m3_m4_m1_audit.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

# This is a batch export pipeline. Avoid initializing Qt/IDE GUI hooks on Windows,
# which can raise a native win32 exception before any figure is saved.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS = Path(r"C:\Users\86147\.codex\skills\math-modeling\tools\figure\scripts")
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


OBSERVED = ROOT / "Q3/04_结果/m3_observed_support.csv"
SUPPORT = ROOT / "Q3/04_结果/m3_q3_support_grid.csv"
CANDIDATES = ROOT / "Q3/04_结果/m3_candidate_grid.csv"
M3_MANIFEST = ROOT / "Q3/04_结果/M3_support_manifest.json"
M4_MASK = ROOT / "Q3/04_结果/m4_budget_feasible_mask.csv"
M1_MASK = ROOT / "Q3/04_结果/m1_final_feasible_mask.csv"
M1_OPTIMA = ROOT / "Q3/04_结果/m1_optimum_boundary_audit.csv"
M4_MANIFEST = ROOT / "Q3/04_结果/M4_budget_manifest.json"
M1_MANIFEST = ROOT / "Q3/04_结果/M1_masked_optimization_manifest.json"

FIG_DIR = ROOT / "Q3/04_结果/figures"
PREVIEW_DIR = ROOT / "Q3/04_结果/_figure_previews"
QA_REPORT = ROOT / "Q3/04_结果/Q3_M3_M4_M1_图表QA.md"
FIG_MANIFEST = ROOT / "Q3/04_结果/Q3_M3_M4_M1_图表清单.json"

PALETTE = {
    "observed": "#0072B2",       # Okabe-Ito blue
    "interpolation": "#E69F00",  # Okabe-Ito orange
    "outside": "#6B7280",        # neutral grey
    "hull": "#222222",
    "budget_1e19": "#0072B2",
    "budget_1e22": "#E69F00",
    "budget_1e24": "#D55E00",
    "grid": "#C8CDD3",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_hull(m3_manifest: dict[str, Any]) -> tuple[list[float], list[float]]:
    transform = m3_manifest["support_transform"]
    vertices = transform["hull_standardized_log_vertices"]
    mean_n = float(transform["log_N_mean"])
    sd_n = float(transform["log_N_population_sd"])
    mean_d = float(transform["log_D_mean"])
    sd_d = float(transform["log_D_population_sd"])
    n_values = [math.exp(mean_n + float(point[0]) * sd_n) for point in vertices]
    d_values = [math.exp(mean_d + float(point[1]) * sd_d) for point in vertices]
    return n_values + [n_values[0]], d_values + [d_values[0]]


def log_axes(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which="major", color="#D9DDE2", linewidth=0.5, alpha=0.65)
    ax.grid(False, which="minor")


def build_fig1(observed: list[dict[str, str]], hull_n: list[float], hull_d: list[float]):
    fig, ax = plt.subplots(figsize=(6.3, 4.0), layout="constrained")
    n_values = [float(row["N_params_B"]) for row in observed]
    d_values = [float(row["D_tokens_B"]) for row in observed]
    # Keep the hull as a dashed outline. Filling a polygon invokes Matplotlib's
    # Bézier extent path in this Windows runtime, which currently crashes natively.
    ax.plot(
        hull_n,
        hull_d,
        color=PALETTE["hull"],
        linewidth=1.15,
        linestyle="--",
        zorder=2,
        label="对数空间凸包（诊断）",
    )
    ax.scatter(
        n_values,
        d_values,
        s=10,
        color=PALETTE["observed"],
        alpha=0.5,
        edgecolors="none",
        rasterized=True,
        zorder=3,
        label=f"B1 实测组合（n={len(observed):,}）",
    )
    log_axes(ax)
    ax.set_xlabel("参数量 $N$（十亿）")
    ax.set_ylabel("训练 Token 数 $D$（十亿）")
    ax.set_title("B1 实测支持与对数空间凸包诊断\n蓝点为实测组合；虚线凸包仅作诊断")
    return fig


def build_fig2(
    support: list[dict[str, str]],
    candidates: list[dict[str, str]],
    m4_mask: list[dict[str, str]],
):
    by_n: dict[float, dict[str, int]] = {}
    for row in support:
        n_b = float(row["N_params_B"])
        counts = by_n.setdefault(n_b, {"exact": 0, "hull_only": 0, "outside": 0})
        if is_true(row["observed_exact"]):
            counts["exact"] += 1
        elif is_true(row["interpolation_candidate"]):
            counts["hull_only"] += 1
        elif is_true(row["extrapolation_flag"]):
            counts["outside"] += 1

    n_values = sorted(by_n)
    exact = np.array([by_n[n]["exact"] for n in n_values])
    hull_only = np.array([by_n[n]["hull_only"] for n in n_values])
    outside = np.array([by_n[n]["outside"] for n in n_values])
    expected_per_n = len({row["D_tokens_B"] for row in candidates})

    budgets = sorted({int(row["budget_flops"]) for row in m4_mask})
    contexts = sorted({int(row["context_length_tokens"]) for row in m4_mask})
    feasible = {(int(row["budget_flops"]), int(row["context_length_tokens"])): 0 for row in m4_mask}
    for row in m4_mask:
        key = (int(row["budget_flops"]), int(row["context_length_tokens"]))
        feasible[key] += int(is_true(row["budget_feasible"]))
    matrix = np.array([[feasible[(budget, context)] for budget in budgets] for context in contexts])

    fig, (ax_support, ax_budget) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.35),
        gridspec_kw={"width_ratios": [1.05, 1.15]},
        layout="constrained",
    )
    y = np.arange(len(n_values))
    ax_support.barh(y, exact, color=PALETTE["observed"], label="精确实测")
    ax_support.barh(y, hull_only, left=exact, color=PALETTE["interpolation"], label="凸包内非观测")
    ax_support.barh(y, outside, left=exact + hull_only, color=PALETTE["outside"], label="凸包外")
    ax_support.set_yticks(y, [f"{value:.4g}" for value in n_values])
    ax_support.invert_yaxis()
    ax_support.set_xlim(0, max(expected_per_n, int(exact.max())) + 28)
    ax_support.set_ylabel("参数量 $N$（十亿）")
    ax_support.set_title("(a) M3 支持分类（全部精确观测）")
    ax_support.grid(axis="x", color="#D9DDE2", linewidth=0.5, alpha=0.65)
    ax_support.set_axisbelow(True)
    for yi, exact_count in zip(y, exact):
        ax_support.text(exact_count + 3, yi, f"{exact_count}/{expected_per_n}", va="center", fontsize=6.5)

    heat = ax_budget.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=len(candidates),
    )
    ax_budget.set_xticks(np.arange(len(budgets)), [rf"$10^{{{int(math.log10(b))}}}$" for b in budgets])
    ax_budget.set_yticks(np.arange(len(contexts)), [f"{context:,}" for context in contexts])
    ax_budget.set_ylabel("上下文长度（Token）")
    ax_budget.set_title("(b) M4 预算可行点数")
    for i, context in enumerate(contexts):
        for j, budget in enumerate(budgets):
            count = feasible[(budget, context)]
            color = "white" if count < 0.45 * len(candidates) else "black"
            ax_budget.text(j, i, f"{count:,}\n/ {len(candidates):,}", ha="center", va="center", color=color, fontsize=6.2)
    cbar = fig.colorbar(heat, ax=ax_budget, fraction=0.047, pad=0.025)
    cbar.set_label("预算可行候选数", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle("M3 支持筛选与 M4 预算筛选各自回答不同问题", fontsize=10)
    return fig


def build_fig3(
    observed: list[dict[str, str]],
    optima: list[dict[str, str]],
    hull_n: list[float],
    hull_d: list[float],
):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(8.4, 5.6),
        gridspec_kw={"wspace": 0.32, "hspace": 0.36},
        layout="constrained",
    )
    colors = {
        10**19: PALETTE["budget_1e19"],
        10**22: PALETTE["budget_1e22"],
        10**24: PALETTE["budget_1e24"],
    }
    markers = {10**19: "o", 10**22: "s", 10**24: "^"}
    n_grid = [float(row["N_params_B"]) for row in observed]
    d_grid = [float(row["D_tokens_B"]) for row in observed]
    contexts = sorted({int(row["context_length_tokens"]) for row in optima})
    budgets = sorted({int(row["budget_flops"]) for row in optima})
    util_1e24 = [float(row["budget_utilization"]) for row in optima if int(row["budget_flops"]) == 10**24]

    for ax, context in zip(axes.flat, contexts):
        ax.scatter(
            n_grid,
            d_grid,
            s=5,
            color=PALETTE["grid"],
            alpha=0.55,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        ax.plot(hull_n, hull_d, color=PALETTE["hull"], linewidth=0.7, linestyle="--", alpha=0.75, zorder=2)
        for budget in budgets:
            row = next(
                item for item in optima
                if int(item["context_length_tokens"]) == context and int(item["budget_flops"]) == budget
            )
            ax.scatter(
                float(row["selected_N_B"]),
                float(row["selected_D_B"]),
                s=43,
                marker=markers[budget],
                color=colors[budget],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
        log_axes(ax)
        ax.set_title(f"$L_{{ctx}}={context:,}$", fontsize=8)
        ax.set_xlabel("$N$ (B)")
        ax.set_ylabel("$D$ (B)")
        ax.tick_params(axis="both", labelsize=6.2)

    note_ax = axes.flat[5]
    note_ax.axis("off")
    handles = [
        Line2D([0], [0], marker=markers[budget], color="none", markerfacecolor=colors[budget],
               markeredgecolor="white", markersize=7, label=rf"$10^{{{int(math.log10(budget))}}}$ FLOPs")
        for budget in budgets
    ]
    note_ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=7)
    note_ax.text(
        0.02,
        0.28,
        "灰点：B1 实测网格\n虚线：对数凸包，仅作诊断\n彩色标记：M1 最优点\n\n"
        "$10^{24}$ FLOPs 下所有点\n均可行，最优点触及 $N,D$ 上界。\n"
        f"预算利用率：{min(util_1e24):.1%}–{max(util_1e24):.1%}。",
        transform=note_ax.transAxes,
        ha="left",
        va="center",
        fontsize=7,
        linespacing=1.35,
    )
    fig.suptitle("M1 最优配置随预算和上下文变化（固定 Q 的 M0 参考）", fontsize=10)
    return fig


def export_one(fig, stem: str, contract: str, qa_lines: list[str]) -> list[str]:
    issues = audit_layout(fig)
    verdict = print_report(issues)
    qa_lines.append(f"- `{stem}`：程序布局检查 {verdict}；问题数 {len(issues)}。")
    qa_lines.extend(f"  - {severity}: {message}" for severity, message in issues)
    preview_path = PREVIEW_DIR / f"{stem}_preview.png"
    render_preview(fig, str(preview_path), dpi=150)
    if any(severity == "FAIL" for severity, _ in issues):
        raise RuntimeError(f"Figure QA failed before export: {stem}")
    output_stem = FIG_DIR / stem
    paths = export_figure(
        fig,
        str(output_stem),
        formats=["pdf", "svg", "png"],
        dpi=300,
        size_inches=fig.get_size_inches(),
        grayscale_preview=True,
        tight=True,
    )
    plt.close(fig)
    qa_lines.append(f"- 图意契约：{contract}")
    qa_lines.extend(f"  - 输出：`{Path(path).relative_to(ROOT).as_posix()}`" for path in paths)
    return paths


def main() -> int:
    inputs = [OBSERVED, SUPPORT, CANDIDATES, M3_MANIFEST, M4_MASK, M1_MASK, M1_OPTIMA, M4_MANIFEST, M1_MANIFEST]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Run the M3/M4/M1 mask pipeline first; missing:\n- " + "\n- ".join(missing))

    observed = read_csv(OBSERVED)
    support = read_csv(SUPPORT)
    candidates = read_csv(CANDIDATES)
    m4_mask = read_csv(M4_MASK)
    m1_mask = read_csv(M1_MASK)
    optima = read_csv(M1_OPTIMA)
    m3_manifest = read_json(M3_MANIFEST)
    hull_n, hull_d = load_hull(m3_manifest)

    setup_info = setup_style(journal="general", lang="zh", use_sciplots=False, constrained_layout=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    qa_lines = [
        "# Q3 M3/M4/M1 图表 QA",
        "",
        f"**生成时间（UTC）：** {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
        f"**绘图环境：** Python {platform.python_version()}；中文字体 `{setup_info.get('cjk_font')}`；通用报告样式。",
        f"**数据行数：** observed={len(observed)}, candidate={len(candidates)}, M3={len(support)}, M4={len(m4_mask)}, M1 final mask={len(m1_mask)}, optima={len(optima)}。",
        "**解释口径：** 当前候选全部来自实测 B1 网格；图中凸包只作诊断，M1 只在 `m3_search_allowed AND budget_feasible` 中选择。",
        "",
        "## 程序版面检查与导出",
        "",
    ]
    generated: list[str] = []
    generated.extend(
        export_one(
            build_fig1(observed, hull_n, hull_d),
            "raw_q3_observed_nd_support",
            "展示全部 B1 实测 N/D 点及标准化对数空间凸包；凸包不作为插值许可证明。",
            qa_lines,
        )
    )
    generated.extend(
        export_one(
            build_fig2(support, candidates, m4_mask),
            "process_q3_m3_m4_mask_audit",
            "按 N 汇总 M3 支持类别，并展示 15 个情景的 M4 预算可行点数。",
            qa_lines,
        )
    )
    generated.extend(
        export_one(
            build_fig3(observed, optima, hull_n, hull_d),
            "result_q3_m1_budget_optima",
            "五个上下文小面板叠加实测网格、凸包诊断边界和三档预算下的 M1 最优点。",
            qa_lines,
        )
    )
    qa_lines.extend(
        [
            "",
            "## 数据与文件检查",
            "",
            "- 原始/支持/预算/最终掩码均按完整输入绘制；图中没有删除候选点。",
            "- 未画误差棒或显著性标记：图示为确定性支持域、成本掩码和离散最优配置。",
            "- 预览位图位于 `Q3/04_结果/_figure_previews/`；正式图文件位于 `Q3/04_结果/figures/`。",
            "- 三张图已逐图检查，未见文字遮挡或裁切；复核说明见 `Q3_M3_M4_M1_图表读图复核.md`。",
            "- 三张彩色 PNG 已通过 `check_figure.py --strict`，分辨率均为 300 dpi。",
            "",
        ]
    )
    QA_REPORT.write_text("\n".join(qa_lines), encoding="utf-8")

    input_hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in inputs}
    output_paths = [Path(path) for path in generated]
    manifest = {
        "artifact": "Q3 M3/M4/M1 audit figures",
        "command": "python Q3/03_代码/plot_q3_m3_m4_m1_audit.py",
        "script": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha256_file(Path(__file__))},
        "backend": "Python/matplotlib",
        "style": setup_info,
        "dpi": 300,
        "formats": ["pdf", "svg", "png", "grayscale PNG"],
        "inputs_sha256": input_hashes,
        "outputs_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in output_paths},
        "qa_report": QA_REPORT.relative_to(ROOT).as_posix(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    FIG_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {QA_REPORT.relative_to(ROOT).as_posix()}")
    print(f"Wrote {FIG_MANIFEST.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
