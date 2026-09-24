"""Create Q1.3 raw-data, process, and result figures.

The plots are generated from the hash-bound Q1.3 CSV inputs and run outputs.
This script requires Python with matplotlib, NumPy, and Pillow. It uses the
project's selected Python backend and the figure-skill QA/export helpers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "q1_3_model"
RESULTS = ROOT / "results" / "q1_3" / "v1"
DATA = ROOT / "中文题目" / "F题" / "real_attachments" / "A_data_value" / "regmix_tables"
FIGURE_SKILL = Path(r"C:\Users\86147\.codex\skills\math-modeling\tools\figure\scripts")
FIGURE_AUDIT_SKILL = Path(r"C:\Users\86147\.codex\skills\math-modeling\references\roles\编程手\scripts")

# The workspace already contains this CPython 3.12-compatible plotting bundle.
# A normal Python environment with matplotlib installed does not need this path.
LOCAL_PLOT_DEPS = ROOT / "tmp" / "q1_2_figure_deps"
if "Q13_PLOT_DEPENDENCIES" in os.environ:
    sys.path.insert(0, os.environ["Q13_PLOT_DEPENDENCIES"])
elif LOCAL_PLOT_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_PLOT_DEPS))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "q1_3_mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(FIGURE_SKILL))
sys.path.insert(0, str(FIGURE_AUDIT_SKILL))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from export_figure import export_figure
from setup_style import setup_style
from visual_qa import audit_layout, print_report, render_preview
from figure_audit import audit_figure_directory


PALETTE = {
    "simplex_ridge": "#0072B2",
    "simplex_huber_ridge_sensitivity": "#009E73",
    "quadratic_ridge_sensitivity": "#D55E00",
    "mean": "#777777",
    "reference_ols": "#E69F00",
}
MODEL_LABELS = {
    "simplex_ridge": "Simplex Ridge",
    "simplex_huber_ridge_sensitivity": "Huber Ridge",
    "quadratic_ridge_sensitivity": "Quadratic sensitivity",
    "mean": "Training mean",
    "reference_ols": "Reference OLS",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_domain(name: str) -> str:
    name = name.replace("train_the_pile_", "").replace("metric/the_pile_", "")
    name = name.replace("_val_loss", "")
    return name.replace("_", " ")


def export_one(fig, name: str, size: tuple[float, float], audit: dict[str, Any]) -> None:
    base = OUT / name
    if name != "result_q1_3_test_1m_observed_predicted":
        # Resolve labels inside the canvas before auditing; constrained layout
        # can leave axis labels a few pixels outside the nominal canvas even
        # though bbox_inches='tight' would later crop them in the saved file.
        fig.set_constrained_layout(False)
        fig.tight_layout(rect=(0.03, 0.04, 0.98, 0.96), h_pad=0.8, w_pad=0.7)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    preview = OUT / "previews" / f"{name}_preview.png"
    render_preview(fig, str(preview), dpi=150)
    paths = export_figure(
        fig,
        str(base),
        formats=["pdf", "svg", "png"],
        size_inches=size,
        dpi=300,
        grayscale_preview=True,
    )
    gray_path = OUT / f"{name}_grayscale.png"
    if gray_path.exists():
        gray_dir = OUT / "grayscale"
        gray_dir.mkdir(exist_ok=True)
        gray_target = gray_dir / gray_path.name
        gray_path.replace(gray_target)
        from PIL import Image
        metadata_target = gray_target.with_name(f"{gray_target.stem}_metadata.png")
        with Image.open(gray_target) as gray_image:
            gray_image.save(metadata_target, dpi=(300, 300))
        metadata_target.replace(gray_target)
        paths = [str(gray_target) if Path(p) == gray_path else p for p in paths]
    audit[name] = {
        "verdict": verdict,
        "issues": [{"severity": sev, "message": message} for sev, message in issues],
        "preview": preview.relative_to(ROOT).as_posix(),
        "outputs": [Path(p).relative_to(ROOT).as_posix() for p in paths],
        "size_inches": list(size),
        "png_dpi": 300,
    }
    plt.close(fig)


def make_profile(mixture_rows: list[dict[str, str]], loss_rows: list[dict[str, str]],
                 pcols: list[str], ycols: list[str], p_raw: np.ndarray,
                 y: np.ndarray) -> dict[str, Any]:
    row_sums = p_raw.sum(axis=1)
    zero_counts = np.sum(p_raw == 0.0, axis=1)
    p = p_raw / row_sums[:, None]
    pc = (p - p.mean(axis=0)) / p.std(axis=0, ddof=1)
    yc = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
    correlation = pc.T @ yc / (len(p) - 1)
    return {
        "training_rows": len(mixture_rows),
        "n_mixture_columns": len(pcols),
        "n_loss_columns": len(ycols),
        "index_order_matches": [r["index"] for r in mixture_rows] == [r["index"] for r in loss_rows],
        "mixture_columns": [
            {
                "name": column,
                "min": float(np.min(p_raw[:, j])),
                "max": float(np.max(p_raw[:, j])),
                "mean": float(np.mean(p_raw[:, j])),
                "zero_rows": int(np.sum(p_raw[:, j] == 0.0)),
            }
            for j, column in enumerate(pcols)
        ],
        "loss_columns": [
            {
                "name": column,
                "min": float(np.min(y[:, j])),
                "max": float(np.max(y[:, j])),
                "mean": float(np.mean(y[:, j])),
                "sd": float(np.std(y[:, j], ddof=1)),
            }
            for j, column in enumerate(ycols)
        ],
        "raw_row_sum": {
            "min": float(row_sums.min()),
            "max": float(row_sums.max()),
            "count_abs_deviation_gt_0_001": int(np.sum(np.abs(row_sums - 1.0) > 0.001 + 1e-12)),
            "n": int(len(row_sums)),
        },
        "zero_count_per_recipe_histogram": {
            str(i): int(np.sum(zero_counts == i)) for i in range(p_raw.shape[1] + 1)
        },
        "closed_share_to_loss_pearson_r": {
            pcols[i]: {ycols[j]: float(correlation[i, j]) for j in range(len(ycols))}
            for i in range(len(pcols))
        },
        "interpretation": "Descriptive training-data profile; correlations do not identify causal effects.",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "previews").mkdir(parents=True, exist_ok=True)
    setup_info = setup_style(journal="general", lang="zh", serif_for_zh=False, constrained_layout=False)
    plt.rcParams["axes.unicode_minus"] = False

    mix = read_csv(DATA / "train_mixture_1m.csv")
    loss = read_csv(DATA / "train_pile_loss_1m.csv")
    pcols = [key for key in mix[0] if key != "index"]
    ycols = [key for key in loss[0] if key != "index"]
    if len(mix) != len(loss) or [r["index"] for r in mix] != [r["index"] for r in loss]:
        raise ValueError("A4/A5 row counts or index order do not match")
    x = np.asarray([[float(row[c]) for c in pcols] for row in mix], dtype=float)
    y = np.asarray([[float(row[c]) for c in ycols] for row in loss], dtype=float)
    sums = x.sum(axis=1)
    p = x / sums[:, None]
    profile = make_profile(mix, loss, pcols, ycols, x, y)
    profile["input_sha256"] = {
        "train_mixture_1m.csv": file_sha256(DATA / "train_mixture_1m.csv"),
        "train_pile_loss_1m.csv": file_sha256(DATA / "train_pile_loss_1m.csv"),
    }
    (OUT / "profile_q1_3_data.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_audit: dict[str, Any] = {}
    width, height = 7.2, 4.8

    # Raw 1: A4 closure deviations, preserving all recipes.
    fig, ax = plt.subplots(figsize=(width, height))
    dev = sums - 1.0
    bins = np.linspace(float(dev.min()) - 0.0002, float(dev.max()) + 0.0002, 25)
    ax.hist(dev, bins=bins, color="#0072B2", edgecolor="white", linewidth=0.5)
    for threshold in (-0.001, 0.0, 0.001):
        ax.axvline(threshold, color="#D55E00" if threshold else "#333333",
                   linestyle="--" if threshold else "-", linewidth=1.0)
    ax.set_xlabel("原始配比行和 - 1")
    ax.set_ylabel("配方数（n = 512）")
    ax.set_title("A4 配比行和偏差分布；虚线为 ±0.001 审计阈值")
    ax.grid(axis="y", alpha=0.2)
    export_one(fig, "raw_q1_3_row_sum_deviation", (width, height), figure_audit)

    # Raw 2: sparsity profile by recipe.
    fig, ax = plt.subplots(figsize=(width, height))
    zero_counts = np.sum(x == 0.0, axis=1)
    bins = np.arange(-0.5, len(pcols) + 1.5, 1)
    counts, edges = np.histogram(zero_counts, bins=bins)
    ax.bar(edges[:-1] + 0.5, counts, width=0.82, color="#56B4E9", edgecolor="white")
    ax.set_xticks(np.arange(0, len(pcols) + 1, 2))
    ax.set_xlabel("每条配方中的零配比项数（17 域）")
    ax.set_ylabel("配方数（n = 512）")
    ax.set_title("A4 零配比稀疏度；511/512 条配方至少含一个零")
    ax.grid(axis="y", alpha=0.2)
    export_one(fig, "raw_q1_3_zero_count", (width, height), figure_audit)

    # Raw 3: cross-correlation matrix between closed shares and observed Loss.
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    corr = np.asarray([[profile["closed_share_to_loss_pearson_r"][pcols[i]][ycols[j]]
                        for j in range(len(ycols))] for i in range(len(pcols))])
    im = ax.pcolormesh(np.arange(len(ycols) + 1), np.arange(len(pcols) + 1), corr,
                       cmap="RdBu_r", vmin=-1, vmax=1, shading="flat", linewidth=0)
    ax.set_xlim(0, len(ycols))
    ax.set_ylim(len(pcols), 0)
    ax.set_xticks(np.arange(len(ycols)) + 0.5,
                  [compact_domain(c) for c in ycols], rotation=70, ha="right")
    ax.set_yticks(np.arange(len(pcols)) + 0.5, [compact_domain(c) for c in pcols])
    ax.set_xlabel("13 个观测 Loss 目标")
    ax.set_ylabel("17 个闭合配比域")
    ax.set_title("A4/A5 配比—Loss Pearson 相关（描述性，不作因果解释）")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Pearson r")
    export_one(fig, "raw_q1_3_share_loss_correlation", (10.5, 6.2), figure_audit)

    # Process 1: lambda selection curves; zero is retained using symlog.
    fig, ax = plt.subplots(figsize=(width, height))
    lambda_specs = [
        ("lambda_selection_closed.csv", "simplex_ridge", "线性 Ridge", "o"),
        ("lambda_selection_huber_sensitivity.csv", "simplex_huber_ridge_sensitivity", "Huber Ridge", "s"),
        ("lambda_selection_quadratic_sensitivity.csv", "quadratic_ridge_sensitivity", "二次敏感性", "^"),
    ]
    for filename, model, label, marker in lambda_specs:
        rows = read_csv(RESULTS / filename)
        lam = np.asarray([float(r["lambda"]) for r in rows])
        score = np.asarray([float(r["standardized_mse"]) for r in rows])
        order = np.argsort(lam)
        ax.plot(lam[order], score[order], marker=marker, color=PALETTE[model],
                linewidth=1.5, markersize=4, label=label)
        best = int(np.argmin(score))
        ax.scatter([lam[best]], [score[best]], s=45, facecolors="none",
                   edgecolors=PALETTE[model], linewidths=1.2, zorder=4)
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlabel("Ridge 正则强度 λ（对称对数轴，含 λ = 0）")
    ax.set_ylabel("分组 CV standardized MSE")
    ax.set_title("A4/A5 训练内正则强度选择；空心点为各模型最低分")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    export_one(fig, "process_q1_3_lambda_cv", (width, height), figure_audit)

    # Process 2: training-CV candidate comparison.
    fig, ax = plt.subplots(figsize=(width, 3.6))
    rows = read_csv(RESULTS / "full_training_model_selection.csv")
    display = {
        "mean": "训练均值基线",
        "reference_ols": "参考域 OLS",
        "simplex_ridge": "单纯形 Ridge（主选）",
        "simplex_huber_ridge_sensitivity": "Huber 稳健敏感性",
        "quadratic_ridge_sensitivity": "二次非线性敏感性",
    }
    ys = np.arange(len(rows))
    for i, row in enumerate(rows):
        name = row["candidate_model"]
        selected = row["selected_on_full_training_cv"].casefold() == "true"
        ax.scatter(float(row["grouped_cv_standardized_mse"]), i,
                   color=PALETTE.get(name, "#777777"), marker="*" if selected else "o",
                   s=100 if selected else 55, zorder=3)
    ax.set_yticks(ys, [display.get(r["candidate_model"], r["candidate_model"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Grouped-CV standardized MSE（越低越好）")
    ax.set_title("A4/A5 完整训练集 CV 模型比较；星号为当前主模型")
    ax.grid(axis="x", alpha=0.25)
    export_one(fig, "process_q1_3_model_selection", (width, 3.6), figure_audit)

    # Process 3: all Huber weights, with target-specific boxplots and jittered points.
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    weight_rows = read_csv(RESULTS / "huber_training_residual_weights.csv")
    weight_targets = list(dict.fromkeys(row["target"] for row in weight_rows))
    weight_values = [np.asarray([float(r["irls_weight"]) for r in weight_rows if r["target"] == t])
                     for t in weight_targets]
    positions = np.arange(1, len(weight_targets) + 1)
    ax.boxplot(weight_values, positions=positions, widths=0.5, showfliers=False,
               medianprops={"color": "#222222", "linewidth": 1.1},
               boxprops={"color": "#0072B2"}, whiskerprops={"color": "#0072B2"},
               capprops={"color": "#0072B2"})
    rng = np.random.default_rng(20260923)
    for pos, vals in zip(positions, weight_values):
        jitter = rng.uniform(-0.17, 0.17, len(vals))
        ax.scatter(pos + jitter, vals, s=8, color="#009E73", alpha=0.34, linewidths=0)
    ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--", label="未降权 = 1")
    ax.set_xticks(positions, [compact_domain(t) for t in weight_targets], rotation=65, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Huber IRLS 权重（0–1）")
    ax.set_xlabel("Loss 目标（每目标 n = 512）")
    ax.set_title("Huber 对大条件残差的降权分布；低权重不等于污染判定")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    export_one(fig, "process_q1_3_huber_weights", (10.0, 5.0), figure_audit)

    # Result 1: held-out 1M observed-vs-predicted relationship, all 13 targets.
    fig, axes = plt.subplots(4, 4, figsize=(11.0, 8.6))
    pred_rows = [r for r in read_csv(RESULTS / "holdout_predictions.csv")
                 if r["split"] == "test_1m" and r["variant"] == "closed"
                 and r["model"] in {"simplex_ridge", "simplex_huber_ridge_sensitivity",
                                    "quadratic_ridge_sensitivity"}]
    target_order = list(dict.fromkeys(r["target"] for r in pred_rows))
    markers = {"simplex_ridge": "o", "simplex_huber_ridge_sensitivity": "s",
               "quadratic_ridge_sensitivity": "^"}
    for i, target in enumerate(target_order):
        ax = axes.flat[i]
        target_rows = [r for r in pred_rows if r["target"] == target]
        observed = np.asarray([float(r["observed_or_estimated_loss"]) for r in target_rows])
        lo = float(min(observed.min(), min(float(r["predicted_loss"]) for r in target_rows)))
        hi = float(max(observed.max(), max(float(r["predicted_loss"]) for r in target_rows)))
        pad = max((hi - lo) * 0.04, 1e-6)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="#555555", linewidth=0.8, linestyle="--")
        for model in markers:
            rs = [r for r in target_rows if r["model"] == model]
            ax.scatter([float(r["observed_or_estimated_loss"]) for r in rs],
                       [float(r["predicted_loss"]) for r in rs],
                       s=7, alpha=0.35, color=PALETTE[model], marker=markers[model],
                       label=MODEL_LABELS[model] if i == 0 else None, linewidths=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(compact_domain(target), fontsize=8)
        if i // 4 == 3:
            ax.set_xlabel("Observed Loss")
        if i % 4 == 0:
            ax.set_ylabel("Predicted Loss")
    for ax in axes.flat[len(target_order):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    # Reserve a dedicated header band for title and legend. The default
    # constrained layout let both overlap the first row of facet titles.
    fig.set_constrained_layout(False)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90), h_pad=0.8, w_pad=0.6)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.958))
    fig.suptitle("冻结 A6/A7（1M）预测—观测关系；各面板 n = 256", y=0.995, fontsize=11)
    export_one(fig, "result_q1_3_test_1m_observed_predicted", (11.0, 8.6), figure_audit)

    # Result 2: descriptive cross-scale transfer error, no uncertainty claims.
    fig, ax = plt.subplots(figsize=(width, height))
    metric_rows = [r for r in read_csv(RESULTS / "holdout_metrics.csv")
                   if r["variant"] == "closed" and r["split"] in {"test_1m", "test_60m", "test_1b"}
                   and r["model"] in {"simplex_ridge", "simplex_huber_ridge_sensitivity",
                                      "quadratic_ridge_sensitivity"}]
    scales = ["test_1m", "test_60m", "test_1b"]
    scale_labels = ["1M same-scale", "60M transfer", "1B transfer"]
    model_order = ["simplex_ridge", "simplex_huber_ridge_sensitivity", "quadratic_ridge_sensitivity"]
    offsets = np.linspace(-0.16, 0.16, len(model_order))
    for model, offset in zip(model_order, offsets):
        vals = []
        for split in scales:
            group = [float(r["rmse"]) for r in metric_rows if r["split"] == split and r["model"] == model]
            vals.append(float(np.mean(group)))
        ax.scatter(np.arange(len(scales)) + offset, vals, s=60, color=PALETTE[model],
                   marker={"simplex_ridge": "o", "simplex_huber_ridge_sensitivity": "s",
                           "quadratic_ridge_sensitivity": "^"}[model],
                   label=MODEL_LABELS[model], zorder=3)
    ax.set_xticks(np.arange(len(scales)), scale_labels)
    ax.set_ylabel("13-target macro RMSE（Loss units）")
    ax.set_ylim(bottom=0)
    ax.set_title("冻结集跨规模误差；A12–A15 估计表不作为真实测试集")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    export_one(fig, "result_q1_3_cross_scale_rmse", (width, height), figure_audit)

    # Result 3: primary simplex-ridge substitution coefficient matrix.
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    params = read_csv(RESULTS / "parameters_selected_closed.csv")
    target_order = [r["target"] for r in params]
    coefficient_columns = [f"coef::{column}" for column in pcols]
    beta = np.asarray([[float(row[c]) for c in coefficient_columns] for row in params]).T
    limit = float(np.max(np.abs(beta)))
    im = ax.pcolormesh(np.arange(len(target_order) + 1), np.arange(len(pcols) + 1), beta,
                       cmap="RdBu_r", vmin=-limit, vmax=limit, shading="flat", linewidth=0)
    ax.set_xlim(0, len(target_order))
    ax.set_ylim(len(pcols), 0)
    ax.set_xticks(np.arange(len(target_order)) + 0.5,
                  [compact_domain(t) for t in target_order],
                  rotation=70, ha="right")
    ax.set_yticks(np.arange(len(pcols)) + 0.5, [compact_domain(c) for c in pcols])
    ax.set_xlabel("Loss target")
    ax.set_ylabel("配比域")
    ax.set_title("主模型 simplex Ridge 系数；系数表示单纯形内的配比替代关联")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Ridge coefficient (Loss units per share)")
    export_one(fig, "result_q1_3_substitution_coefficients", (10.5, 6.2), figure_audit)

    profile["figure_style"] = setup_info
    profile["runtime"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                          "matplotlib": matplotlib.__version__}
    (OUT / "profile_q1_3_data.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "visual_qa.json").write_text(json.dumps(figure_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [name for name, item in figure_audit.items() if item["verdict"] == "FAIL"]
    qa_status = "PASS" if all(item["verdict"] == "PASS" for item in figure_audit.values()) else "WARN"
    figure_audit_report = audit_figure_directory(OUT, min_dpi=300, questions=("q1",))
    figure_audit_path = OUT / "figure_audit_q1.json"
    figure_audit_path.write_text(
        json.dumps(figure_audit_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contract_path = OUT / "图表契约.md"
    figure_outputs = [
        {"file": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path)}
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "figure_manifest.json"
    ]
    figure_manifest = {
        "schema_version": "1.0",
        "status": "RENDERED_QA_PASS_AUDIT_OK_PENDING_FINAL_P2",
        "plot_script": {
            "file": Path(__file__).relative_to(ROOT).as_posix(),
            "sha256": file_sha256(Path(__file__)),
        },
        "contract": {
            "file": contract_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(contract_path),
        },
        "runtime": profile["runtime"],
        "figure_count": len(figure_audit),
        "categories": {"raw": 3, "process": 3, "result": 3},
        "programmatic_visual_qa": qa_status,
        "figure_audit_ok": figure_audit_report["ok"],
        "figure_audit_warnings": len(figure_audit_report["issues"]),
        "figure_audit_report": figure_audit_report,
        "figure_qa": figure_audit,
        "outputs": figure_outputs,
    }
    figure_manifest_path = OUT / "figure_manifest.json"
    figure_manifest_path.write_text(json.dumps(figure_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    model_manifest_path = RESULTS / "repro_manifest.json"
    if model_manifest_path.exists():
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        model_manifest["status"] = "MODEL_FIT_AND_FIGURES_COMPLETED_PENDING_FINAL_P2"
        model_manifest["figure_artifacts"] = {
            "manifest": {
                "file": figure_manifest_path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(figure_manifest_path),
            },
            "programmatic_visual_qa": qa_status,
            "figure_audit": {
                "file": figure_audit_path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(figure_audit_path),
                "ok": figure_audit_report["ok"],
                "warnings": len(figure_audit_report["issues"]),
            },
            "count": len(figure_audit),
            "runtime": profile["runtime"],
        }
        pending = [item for item in model_manifest.get("pending", [])
                   if not item.startswith("Q1.3 figure set")]
        if "Final P2 programming gate after figures and visualization QA" not in pending:
            pending.append("Final P2 programming gate after figures and visualization QA")
        model_manifest["pending"] = pending
        model_manifest.setdefault("verification", {})["visual_qa"] = qa_status
        model_manifest["verification"]["figure_audit"] = (
            "PASS_WITH_WARNINGS" if figure_audit_report["ok"] and figure_audit_report["issues"]
            else "PASS" if figure_audit_report["ok"] else "FAIL"
        )
        output_records = {item["file"]: item for item in model_manifest.get("outputs", [])}
        for item in figure_outputs:
            output_records[item["file"]] = item
        output_records[figure_manifest_path.relative_to(ROOT).as_posix()] = {
            "file": figure_manifest_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(figure_manifest_path),
        }
        for output_name, output_record in output_records.items():
            output_path = ROOT / Path(output_name)
            if output_path.is_file():
                output_record["sha256"] = file_sha256(output_path)
        model_manifest["outputs"] = [output_records[key] for key in sorted(output_records)]
        model_manifest_path.write_text(json.dumps(model_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(figure_audit), "failed": failed,
                      "visual_qa": qa_status, "output": str(OUT),
                      "runtime": profile["runtime"]}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
