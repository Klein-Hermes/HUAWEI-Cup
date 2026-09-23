#!/usr/bin/env python3
"""Create Q1.1 raw, process, result and model-flow figures from frozen artifacts.

This plotting program reads scalar tables and derived outputs only. It never
opens corpus content or interprets low-visibility PDF text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--skill-root", type=Path, default=Path(r"C:\Users\86147\.codex\skills\math-modeling"))
parser.add_argument("--matplotlib-path", type=Path, default=Path("tmp/q1_1_figure_deps"))
parser.add_argument("--output", type=Path, default=Path("figures/q1_1_model"))
args = parser.parse_args()
ROOT = args.project_root.resolve()
SKILL = args.skill_root.resolve()
MPL = args.matplotlib_path if args.matplotlib_path.is_absolute() else ROOT / args.matplotlib_path
OUT = args.output if args.output.is_absolute() else ROOT / args.output
OUT.mkdir(parents=True, exist_ok=True)
GRAY = OUT / "grayscale_previews"
GRAY.mkdir(parents=True, exist_ok=True)
MPLCONF = ROOT / "tmp" / "q1_1_mplconfig"
MPLCONF.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONF))
sys.path[:0] = [str(MPL), str(SKILL / "tools" / "figure" / "scripts"),
                str(SKILL / "references" / "roles" / "编程手" / "scripts")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from export_figure import export_figure
from setup_style import setup_style
from visual_qa import audit_layout
from figure_audit import audit_figure_directory

style_info = setup_style(journal="general", lang="zh", use_sciplots=False, constrained_layout=False)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": False})
COL = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "purple": "#CC79A7",
       "sky": "#56B4E9", "amber": "#E69F00", "gray": "#68717B", "dark": "#20252B",
       "light": "#E8EDF2", "red": "#B33A3A"}
PRE = ROOT / "results" / "q1_common_preprocess" / "v1"
RES = ROOT / "results" / "q1_1" / "v1"
AUDIT = ROOT / "results" / "q1_1" / "text_integrity_summary.json"
qa = []


def read_csv(path: Path, **kwargs):
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def short_label(value: str, limit: int = 17) -> str:
    value = str(value).replace("_", " ")
    return value if len(value) <= limit else value[:limit - 1] + "…"


def save(fig, stem: str, size: tuple[float, float], caption: str) -> None:
    fig.text(.01, .004, caption, ha="left", va="bottom", fontsize=7.5, color="#444444", wrap=True)
    fig.set_size_inches(*size)
    if stem == "result_q1_1_domain_rank_robustness":
        fig.subplots_adjust(left=.30, right=.92, bottom=.25, top=.88)
    elif stem == "process_q1_1_weight_calibration":
        fig.subplots_adjust(left=.29, right=.98, bottom=.15, top=.91, wspace=.28)
    elif stem == "result_q1_1_contamination_transfer_intervals":
        fig.subplots_adjust(left=.27, right=.98, bottom=.16, top=.81, wspace=.25)
    elif stem == "result_q1_1_unmarked_vs_full":
        fig.subplots_adjust(left=.25, right=.98, bottom=.18, top=.81, wspace=.25)
    else:
        fig.subplots_adjust(left=.13, right=.98, bottom=.20, top=.88)
    issues = audit_layout(fig)
    qa.append({"figure": stem, "issues": [{"severity": s, "message": m} for s, m in issues]})
    for severity, message in issues:
        print(f"VISUAL_QA {severity} {stem}: {message}")
    export_figure(fig, str(OUT / stem), formats=["svg", "png"], size_inches=size, dpi=301,
                  grayscale_preview=False)
    with Image.open(OUT / f"{stem}.png") as image:
        image.convert("L").save(GRAY / f"{stem}_grayscale.png", dpi=(301, 301))
    plt.close(fig)


contract = read_csv(ROOT / "results" / "q1_1" / "q1_1_input_contract.csv")
indicator_names = contract["indicator"].astype(str).tolist()
label_by_indicator = dict(zip(contract["indicator"].astype(str), contract["meaning"].astype(str)))
model_manifest = json.loads((RES / "manifest.json").read_text(encoding="utf-8"))
if model_manifest.get("status") != "computed_validation_pending_human_review":
    raise ValueError("Figures require the full Q1.1 run, not a smoke run")
unresolved = set(model_manifest["field_unit_scope"]["unresolved_unit_fields"])
pre_meta = json.loads((PRE / "manifest.json").read_text(encoding="utf-8"))
domain_summary = read_csv(RES / "domain_summary.csv")
domain_ci = read_csv(RES / "domain_confidence_intervals.csv")
same_domain = read_csv(RES / "same_domain_differences.csv")
contamination_transfer = read_csv(RES / "contamination_same_domain_intervals.csv")
unmarked_comparison = read_csv(RES / "a1_unmarked_vs_full.csv")
sensitivity = read_csv(RES / "score_sensitivity_domain.csv")
rank_sensitivity = read_csv(RES / "domain_rank_sensitivity.csv")
weights = read_csv(RES / "indicator_weights.csv")
audit = json.loads(AUDIT.read_text(encoding="utf-8"))
score_flags = read_csv(RES / "sample_scores.csv.gz", usecols=["dataset", "unicode_cf_cc_flag", "unicode_broad_flag"])


# Raw 1: actual record counts by dataset/source domain.
fig, ax = plt.subplots()
labels, counts = [], []
for dataset, name in (("a1", "A1"), ("a2", "A2"), ("a3", "A3")):
    summary = pre_meta["summaries"][dataset]
    for domain, count in summary["domains"].items():
        labels.append(f"{name} · {domain}")
        counts.append(int(count))
y = np.arange(len(labels))
ax.barh(y, counts, color=[COL["blue"] if x.startswith("A1") else COL["orange"] if x.startswith("A2") else COL["green"] for x in labels])
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("记录数")
ax.set_title("冻结输入规模按数据集与来源域分布")
for yi, value in zip(y, counts):
    ax.text(value + max(counts) * .006, yi, f"{value:,}", va="center", fontsize=7)
save(fig, "raw_q1_1_source_counts", (8.5, 5.8), "来源：冻结公共预处理 manifest；A1/A2/A3 按真实 source_domain 计数。")


# Raw 2: distributions of the 22 frozen A1 normalized indicators.
a1_path = PRE / "a1_preprocessed.csv.gz"
norm_fields = contract["norm_field"].astype(str).tolist()
a1_values = read_csv(a1_path, usecols=norm_fields)
box_data = [a1_values[field].dropna().to_numpy() for field in norm_fields]
fig, ax = plt.subplots()
bp = ax.boxplot(box_data, vert=False, showfliers=False, patch_artist=True,
                medianprops={"color": COL["dark"], "linewidth": 1.1},
                whiskerprops={"color": COL["gray"]}, capprops={"color": COL["gray"]})
for patch, field in zip(bp["boxes"], norm_fields):
    patch.set_facecolor(COL["red"] if field in {f"norm_{name}" for name in unresolved} else COL["sky"])
    patch.set_alpha(.78)
ax.set_yticks(np.arange(1, len(norm_fields) + 1), [short_label(label_by_indicator[name.removeprefix("norm_")]) for name in norm_fields])
ax.set_xlim(0, 1)
ax.set_xlabel("冻结 norm_* 值（A1 裁剪 ECDF 尺度）")
ax.set_title("A1 的 22 个标准化质量信号分布")
ax.plot([], [], color=COL["sky"], linewidth=7, label="其余字段")
ax.plot([], [], color=COL["red"], linewidth=7, label="7 个单位/语义未核实字段")
ax.legend(frameon=False, loc="lower right", fontsize=8)
save(fig, "raw_q1_1_indicator_distributions", (9.0, 7.0), "箱体为四分位距，须线为 1.5×IQR；离群点隐藏以便阅读但未从评分数据中删除。")


# Raw 3: separate historical, narrow and broad Unicode integrity counts; content availability.
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
left = axes[0]
labels = ["历史基线(1)", "窄口径\nCf/Cc", "宽口径(2)"]
a1_flags = score_flags[score_flags["dataset"] == "a1"]
values = [int(audit["report_baseline_invisible_rows_a1"]),
          int(a1_flags["unicode_cf_cc_flag"].sum()), int(a1_flags["unicode_broad_flag"].sum())]
bars = left.bar(labels, values, color=[COL["gray"], COL["amber"], COL["red"]])
left.set_ylabel("A1 命中记录数")
left.set_title("完整性标记计数（A1）")
for bar, value in zip(bars, values):
    left.text(bar.get_x() + bar.get_width()/2, value + 7, f"{value}", ha="center")
right = axes[1]
content_rows = [int(audit["datasets"][key]["rows"] - audit["datasets"][key]["missing_content_rows"]) for key in ("A1", "A2", "A3")]
bars = right.bar(["A1", "A2", "A3"], content_rows, color=[COL["blue"], COL["orange"], COL["green"]])
right.set_ylabel("带可审阅原文的记录数")
right.set_title("原文可用性门禁")
right.set_ylim(0, max(content_rows) * 1.2)
for bar, value in zip(bars, content_rows):
    right.text(bar.get_x() + bar.get_width()/2, value + max(content_rows)*.02, f"{value:,}", ha="center")
right.text(.5, .55, "A2/A3 原文与 ID 映射缺失", transform=right.transAxes, ha="center", color=COL["red"], fontsize=9)
fig.suptitle("文本污染审计边界：完整性标记不是恶意标签")
save(fig, "raw_q1_1_integrity_and_text_availability", (9.0, 4.6), "(1) 历史基线扫描口径当前不可复现；(2) 宽口径另含 Zl/Zp/Co/非字符。A1 标记不自动删除。")


# Process 1: show near-equal candidate weights and their completeness/stability inputs.
ordered_weights = weights.sort_values("weight", ascending=True).reset_index(drop=True)
fig, (axw, axp) = plt.subplots(1, 2, figsize=(10.0, 8.2), sharey=True,
                                gridspec_kw={"width_ratios": [1.1, 1.8]})
y = np.arange(len(ordered_weights))
colors = [COL["red"] if name in unresolved else COL["blue"] for name in ordered_weights["indicator"]]
axw.scatter(ordered_weights["weight"], y, color=colors, s=30, zorder=3)
axw.axvline(1 / len(ordered_weights), color=COL["gray"], linestyle="--", linewidth=.9, label="等权 1/22")
axw.set_yticks(y, [short_label(name, 23) for name in ordered_weights["indicator"]])
axw.invert_yaxis()
pad = max((ordered_weights["weight"].max() - ordered_weights["weight"].min()) * .18, .00005)
axw.set_xlim(ordered_weights["weight"].min() - pad, ordered_weights["weight"].max() + pad)
axw.set_xlabel("候选权重 w_j")
axw.set_title("最终权重")
axw.legend(frameon=False, fontsize=7, loc="lower right")
offset = .11
axp.scatter(ordered_weights["a1_completeness"], y - offset, color=colors, marker="o", s=22, label="A1 完整度 c_j")
axp.scatter(ordered_weights["domain_rank_stability"], y + offset, color=colors, marker="s", s=22, label="七域排序稳定性 S_j")
axp.set_xlim(.975, 1.005)
axp.set_xlabel("参数取值")
axp.set_title("A1 校准依据")
axp.legend(frameon=False, fontsize=7, loc="lower left")
fig.suptitle("A1 完整度与排序稳定性形成近等权候选", y=.98)
save(fig, "process_q1_1_weight_calibration", (10.0, 8.4), "红色为 7 个单位/语义未核实字段；完整度与排序稳定性接近 1，因此权重仅轻微偏离等权。权重不是人工重要性或效度。")


# Process 2: Huber loss and influence functions used by the candidate scorer.
residual = np.linspace(-1, 1, 1001)
delta = float(json.loads((RES / "manifest.json").read_text(encoding="utf-8"))["parameter_checks"]["all_a1_huber_delta"])
quadratic = .5 * residual ** 2
huber = np.where(np.abs(residual) <= delta, .5 * residual ** 2, delta * (np.abs(residual) - .5 * delta))
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
axes[0].plot(residual, quadratic, color=COL["gray"], label="平方损失")
axes[0].plot(residual, huber, color=COL["blue"], label="Huber 损失")
axes[0].axvline(delta, color=COL["red"], linestyle="--", linewidth=.8)
axes[0].axvline(-delta, color=COL["red"], linestyle="--", linewidth=.8)
axes[0].set_xlabel("指标残差 u")
axes[0].set_ylabel("ρδ(u)")
axes[0].set_title("位置估计的损失函数")
axes[0].legend(frameon=False)
axes[1].plot(residual, residual, color=COL["gray"], label="平方损失影响 ψ(u)=u")
axes[1].plot(residual, np.clip(residual, -delta, delta), color=COL["green"], label="Huber 影响 ψδ(u)")
axes[1].axhline(delta, color=COL["red"], linestyle=":", linewidth=.8)
axes[1].axhline(-delta, color=COL["red"], linestyle=":", linewidth=.8)
axes[1].set_xlabel("指标残差 u")
axes[1].set_ylabel("估计影响")
axes[1].set_title("大残差贡献封顶")
axes[1].legend(frameon=False, fontsize=7)
fig.suptitle(f"A1 冻结 Huber 阈值 δ = {delta:.4f}")
save(fig, "process_q1_1_huber_location", (9.0, 4.4), "函数用于降低大残差的影响；它不识别恶意文本，也不等于自动清洗。")


# Process 3: per-domain Huber score changes under contamination and unit scenarios.
scenario_names = ["unicode_strict_clean_a1", "unicode_broad_clean_a1", "unit_flagged_a1_excluded", "exclude_7_unit_ambiguous_fields"]
scenario_labels = ["窄 Unicode 校准排除", "宽 Unicode 校准排除", "单位标记行校准排除", "评分剔除 7 个未核实字段"]
a1_changes = sensitivity[(sensitivity["dataset"] == "a1") & (sensitivity["candidate"] == "q_huber") & sensitivity["scenario"].isin(scenario_names)].copy()
fig, ax = plt.subplots()
domains = sorted(a1_changes["source_domain"].unique())
colors = [COL["blue"], COL["orange"], COL["green"], COL["red"]]
width = .19
x = np.arange(len(domains))
for j, (scenario, label, color) in enumerate(zip(scenario_names, scenario_labels, colors)):
    part = a1_changes[a1_changes["scenario"] == scenario].set_index("source_domain").reindex(domains)
    ax.bar(x + (j - 1.5) * width, part["delta_from_all_a1"].to_numpy(), width,
           label=label, color=color, alpha=.85)
ax.axhline(0, color=COL["dark"], linewidth=.8)
ax.set_xticks(x, domains, rotation=30, ha="right")
ax.set_ylabel("相对全量校准的域分变化")
ax.set_title("污染与单位假设下 A1 域分的变化")
ax.legend(frameon=False, fontsize=7, ncol=2)
save(fig, "process_q1_1_contamination_sensitivity", (9.2, 5.0), "Unicode 场景仅排除 A1 参数校准行；单位字段场景另行剔除信号。命中不等于恶意。")


# Result 1: domain locations and 95% record-bootstrap intervals.
plot_data = domain_summary.merge(domain_ci, on=["dataset", "source_domain", "candidate"], how="left")
plot_data = plot_data[plot_data["scenario"] == "all_a1"].copy()
data_order = [("a1", d) for d in sorted(plot_data.loc[plot_data.dataset == "a1", "source_domain"].unique())]
data_order += [("a2", "arxiv"), ("a3", "github")]
fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.6), sharey=True)
for ax, model, title, color in zip(axes, ("q_equal", "q_huber"), ("等权候选", "稳定性加权 Huber 候选"), (COL["blue"], COL["orange"])):
    part = plot_data[plot_data["candidate"] == model]
    for yi, (dataset, domain) in enumerate(data_order):
        row = part[(part.dataset == dataset) & (part.source_domain == domain)].iloc[0]
        ax.errorbar(float(row.domain_score), yi,
                    xerr=[[float(row.domain_score - row.ci_low)], [float(row.ci_high - row.domain_score)]],
                    fmt="o", color=color, capsize=2.5, markersize=4)
    ax.set_title(title)
    ax.set_xlabel("域级 Huber 位置及 95% 区间")
    ax.set_xlim(0, 1)
axes[0].set_yticks(np.arange(len(data_order)), [f"{ds.upper()} · {domain}" for ds, domain in data_order])
axes[0].invert_yaxis()
fig.suptitle("Q1.1 候选质量分的域级结果")
save(fig, "result_q1_1_domain_scores", (10.0, 5.8), "记录 bootstrap 区间固定 A1 权重与阈值，不包含参数估计不确定性；A2/A3 原文尚未验证。")


# Result 2: extensions against same-domain A1; all displayed intervals include zero.
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6), sharex=True)
for ax, model, title, color in zip(axes, ("q_equal", "q_huber"), ("等权候选", "加权 Huber 候选"), (COL["blue"], COL["orange"])):
    part = same_domain[same_domain["candidate"] == model]
    y = np.arange(len(part))
    ax.errorbar(part["difference_extension_minus_a1"], y,
                xerr=[part["difference_extension_minus_a1"] - part["ci_low"], part["ci_high"] - part["difference_extension_minus_a1"]],
                fmt="o", color=color, capsize=3)
    ax.axvline(0, color=COL["dark"], linewidth=.8, linestyle="--")
    ax.set_yticks(y, [f"{row.extension_dataset.upper()} · {row.source_domain}" for row in part.itertuples()])
    ax.set_title(title)
    ax.set_xlabel("扩展集减 A1 同域分差及 95% 区间")
fig.suptitle("扩展集同域迁移比较")
save(fig, "result_q1_1_same_domain_transfer", (9.0, 4.7), "区间覆盖 0 表示当前记录 bootstrap 未显示稳定方向差；不能替代 A2/A3 原文效度验证。")


# Result 3: A1 domain rank shifts under major contamination/unit assumptions.
rank_part = rank_sensitivity[(rank_sensitivity["candidate"] == "q_huber") & rank_sensitivity["scenario"].isin(scenario_names)]
rank_matrix = rank_part.pivot(index="scenario", columns="source_domain", values="rank_shift").reindex(index=scenario_names)
fig, ax = plt.subplots()
rank_cmap = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0, 1, 7)))
rank_norm = BoundaryNorm(np.arange(-3.5, 4.5, 1), rank_cmap.N)
im = ax.pcolormesh(np.arange(rank_matrix.shape[1] + 1), np.arange(rank_matrix.shape[0] + 1),
                   rank_matrix.to_numpy(), cmap=rank_cmap, norm=rank_norm,
                   edgecolors="white", linewidth=.8, shading="flat")
ax.invert_yaxis()
ax.set_yticks(np.arange(len(scenario_names)) + .5, scenario_labels)
ax.set_xticks(np.arange(len(rank_matrix.columns)) + .5, rank_matrix.columns, rotation=30, ha="right")
ax.set_title("A1 七域 Huber 分数名次变化")
ax.set_xlabel("来源域")
ax.set_ylabel("敏感性场景")
for i in range(rank_matrix.shape[0]):
    for j in range(rank_matrix.shape[1]):
        value = rank_matrix.iloc[i, j]
        ax.text(j + .5, i + .5, f"{value:+.0f}", ha="center", va="center", fontsize=8,
                color="white" if abs(value) >= 2 else COL["dark"])
cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.03, boundaries=np.arange(-3.5, 4.5, 1),
                  ticks=np.arange(-3, 4), spacing="uniform")
cb.set_label("名次变化（正值 = 排名下降）")
save(fig, "result_q1_1_domain_rank_robustness", (9.5, 4.8), "名次按分数从高到低排序（1 为最高分）；正变化表示名次下降。")


# Result 4: scenario-specific A2/A3 same-domain differences and record-bootstrap CIs.
scenario_labels = {
    "unicode_strict_clean_a1": "窄 Unicode",
    "unicode_broad_clean_a1": "宽 Unicode",
    "unit_flagged_a1_excluded": "单位标记",
    "exclude_7_unit_ambiguous_fields": "排除 7 字段",
}
scenario_colors = {
    "unicode_strict_clean_a1": COL["blue"],
    "unicode_broad_clean_a1": COL["orange"],
    "unit_flagged_a1_excluded": COL["green"],
    "exclude_7_unit_ambiguous_fields": COL["red"],
}
transfer_plot = contamination_transfer[contamination_transfer["run_status"] == "computed"].copy()
fig, axes = plt.subplots(1, 2, figsize=(10.0, 7.2), sharex=True)
for ax, candidate, title in zip(axes, ("q_equal", "q_huber"), ("等权候选", "加权 Huber 候选")):
    part = transfer_plot[transfer_plot["candidate"] == candidate].copy()
    part["scenario_order"] = part["scenario"].map({name: i for i, name in enumerate(scenario_labels)})
    part["dataset_order"] = part["extension_dataset"].map({"a2": 0, "a3": 1})
    part = part.sort_values(["scenario_order", "dataset_order"]).reset_index(drop=True)
    y = np.arange(len(part))
    colors = [scenario_colors[name] for name in part["scenario"]]
    ax.hlines(y, part["ci_low"], part["ci_high"], color=COL["gray"], linewidth=.9, zorder=2)
    ax.vlines(part["ci_low"], y - .08, y + .08, color=COL["gray"], linewidth=.8, zorder=2)
    ax.vlines(part["ci_high"], y - .08, y + .08, color=COL["gray"], linewidth=.8, zorder=2)
    ax.scatter(part["difference_extension_minus_a1"], y, c=colors, s=24, zorder=3)
    ax.axvline(0, color=COL["dark"], linewidth=.8, linestyle="--")
    ax.set_yticks(y, [f"{scenario_labels[row.scenario]} · {row.extension_dataset.upper()}" for row in part.itertuples()])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("扩展集减 A1 同域分差及 95% 区间")
scenario_handles = [Line2D([], [], color=scenario_colors[name], marker="o", linestyle="none", label=label)
                    for name, label in scenario_labels.items()]
fig.legend(handles=scenario_handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=4,
           frameon=False, fontsize=8)
fig.suptitle("污染与单位假设下的扩展集同域迁移比较")
save(fig, "result_q1_1_contamination_transfer_intervals", (10.0, 7.4),
     "A2-arxiv/A1-arxiv 与 A3-github/A1-github 分开比较；每种敏感性场景使用其 A1 重估域阈值并冻结到对应扩展域。区间为独立记录 bootstrap，不能代替扩展集原文验证。")


# Result 5: A1 Unicode-unmarked subset versus the full domain estimate.
unmarked_labels = {"unicode_strict_clean_a1": "窄口径", "unicode_broad_clean_a1": "宽口径"}
unmarked_colors = {"unicode_strict_clean_a1": COL["blue"], "unicode_broad_clean_a1": COL["orange"]}
fig, axes = plt.subplots(1, 2, figsize=(10.0, 7.8), sharex=True, sharey=True)
for ax, candidate, title in zip(axes, ("q_equal", "q_huber"), ("等权候选", "加权 Huber 候选")):
    part = unmarked_comparison[unmarked_comparison["candidate"] == candidate].copy()
    part["scenario_order"] = part["scenario"].map({name: i for i, name in enumerate(unmarked_labels)})
    part = part.sort_values(["scenario_order", "source_domain"]).reset_index(drop=True)
    y = np.arange(len(part))
    colors = [unmarked_colors[name] for name in part["scenario"]]
    ax.hlines(y, part["ci_low"], part["ci_high"], color=COL["gray"], linewidth=.9, zorder=2)
    ax.vlines(part["ci_low"], y - .08, y + .08, color=COL["gray"], linewidth=.8, zorder=2)
    ax.vlines(part["ci_high"], y - .08, y + .08, color=COL["gray"], linewidth=.8, zorder=2)
    ax.scatter(part["difference_unmarked_minus_full"], y, c=colors, s=24, zorder=3)
    ax.axvline(0, color=COL["dark"], linewidth=.8, linestyle="--")
    ax.set_yticks(y, [f"{unmarked_labels[row.scenario]} · {row.source_domain}" for row in part.itertuples()])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("未标记子集减全样本域分及配对 95% 区间")
unmarked_handles = [Line2D([], [], color=unmarked_colors[name], marker="o", linestyle="none", label=label)
                    for name, label in unmarked_labels.items()]
fig.legend(handles=unmarked_handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=2,
           frameon=False, fontsize=8)
fig.suptitle("A1 未标记记录子集与全样本域分比较")
save(fig, "result_q1_1_unmarked_vs_full", (10.0, 8.0),
     "严格 Cf/Cc 与宽 Unicode 标记口径分别计算。未标记样本重抽结果在子集和全样本估计中共享；表格另报告七域排序与宏平均变化。排除标记校准不证明上游预计算信号已去污染。")


# Q1.1-only model flow: contaminated text and PDF instructions stay outside scoring.
fig, ax = plt.subplots(figsize=(9.0, 8.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")


def box(x, y, w, h, text, face, edge=COL["blue"], fontsize=9, style="round"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10" if style == "round" else "square,pad=0.03",
                           facecolor=face, edgecolor=edge, linewidth=1.15)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return (x+w/2, y+h/2)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.0, color=COL["gray"], connectionstyle="arc3,rad=0"))


p1 = box(2.1, 10.7, 5.8, .8, "冻结 A1 / A2 / A3 标量与 22 指标合同\n保留 ID、域、缺失/异常/单位标记", "#EAF2F8")
p2 = box(2.1, 9.1, 5.8, .95, "污染与完整性门禁\nPDF 低可见度指令和 JSONL 文本只作审计数据\nA1 Unicode 命中不自动删行；A2/A3 原文映射缺失", "#FFF5E5", edge=COL["amber"], fontsize=8.5)
p3 = box(2.1, 7.5, 5.8, .85, "A1 参数校准\n缺失度 × 七域重抽样排序稳定性估权；A1 估计并冻结 Huber δ", "#EAF2F8")
p4a = box(.8, 5.7, 3.8, .9, "等权样本分 Q_equal\n有效指标透明平均", "#E9F4EF", edge=COL["green"])
p4b = box(5.4, 5.7, 3.8, .9, "加权 Huber 样本分 Q_huber\n按有效权重重新归一", "#E9F4EF", edge=COL["green"])
p5 = box(2.1, 4.0, 5.8, .85, "域级稳健位置 + 记录 bootstrap 区间\nA1 七域；A2-arxiv/A1-arxiv；A3-github/A1-github", "#EAF2F8")
p6 = box(.7, 2.1, 4.2, 1.0, "全量 / 窄宽 Unicode / 单位标记校准\n权重、Huber δ、7 字段剔除敏感性", "#FFF5E5", edge=COL["amber"], fontsize=8.5)
p7 = box(5.1, 2.1, 4.2, 1.0, "A1 原文双人盲评包\n人工评分待回填；A2/A3 原文验证待补", "#F8ECEC", edge=COL["red"], fontsize=8.5)
p8 = box(2.0, .35, 6.0, .9, "当前交付：候选评分与稳健性结果已计算\n整体效度验证 / 最终唯一 Q 仍待人工与原文验证", "#E8EDF2", edge=COL["dark"], fontsize=9)
for source, target in ((p1, p2), (p2, p3), (p3, p4a), (p3, p4b), (p4a, p5), (p4b, p5), (p5, p6), (p5, p7), (p6, p8), (p7, p8)):
    arrow(source[0], source[1] - .48, target[0], target[1] + .48)
ax.set_title("F 题 Q1.1 建模与验证数据流", pad=14, fontsize=13)
save(fig, "flow_q1_1_model", (9.0, 8.6), "范围限定为 Q1.1；Q1.2、Q1.3 与题目分析报告保持独立。")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


(OUT / "visual_qa.json").write_text(json.dumps({"style": style_info, "figures": qa}, ensure_ascii=False, indent=2), encoding="utf-8")
issues = [{"figure": item["figure"], **issue} for item in qa for issue in item["issues"]]
figure_audit_report = audit_figure_directory(OUT, min_dpi=300, min_per_category=3, questions=("q1",))
figure_audit_path = OUT / "figure_audit_q1.json"
figure_audit_path.write_text(json.dumps(figure_audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
figure_manifest = {
    "schema_version": "1.0",
    "scope": "F Q1.1 only",
    "plot_script": {"file": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
    "input_manifest_sha256": sha256(RES / "manifest.json"),
    "figure_count": 12,
    "categories": {"raw": 3, "process": 3, "result": 5, "flow": 1},
    "dpi": 301,
    "svg_text_as_text": True,
    "programmatic_visual_qa": "PASS" if not issues else "WARN",
    "figure_audit_ok": figure_audit_report["ok"],
    "figure_audit_warnings": len(figure_audit_report["issues"]),
    "figure_audit_report": figure_audit_report,
    "figure_qa": qa,
    "outputs_sha256": {},
}
for path in sorted(OUT.rglob("*")):
    if path.is_file() and path.name != "figure_manifest.json":
        figure_manifest["outputs_sha256"][path.relative_to(OUT).as_posix()] = sha256(path)
(OUT / "figure_manifest.json").write_text(json.dumps(figure_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"figures": 12, "visual_qa": figure_manifest["programmatic_visual_qa"],
                  "figure_audit_ok": figure_audit_report["ok"],
                  "output": str(OUT), "qa_issues": len(issues),
                  "audit_issues": len(figure_audit_report["issues"])}, ensure_ascii=False, indent=2))
