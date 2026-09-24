#!/usr/bin/env python3
"""Rebuild and package the current-data Q2 closeout without refitting M0.

The package runner refreshes only deterministic post-processing products:
M0 effect tables, the evidence/identifiability closeout, four M0 effect figures,
input inventory, README, and run log. It never refits the scaling model or
re-samples the saved cluster-bootstrap fits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "q2_scaling_baseline" / "v1"
EFFECTS = ROOT / "results" / "q2_scaling_baseline" / "v1"
Q1_FINAL = ROOT / "results" / "q1_final"
Q1_3 = ROOT / "results" / "q1_3"
Q1_3_PQ = ROOT / "results" / "q1_3" / "pq_extension_v1"
Q2 = ROOT / "Q2"
Q2_BASELINE = Q2 / "03_结果" / "经典ScalingLaw基线" / "v1"
Q2_P_AUDIT = Q2 / "03_结果" / "p配比可行性审计" / "v1"
Q2_B2 = Q2 / "03_结果" / "B2半合成压力测试" / "v1"
Q2_CLOSEOUT = Q2 / "03_结果" / "综合收口" / "v1"
Q2_FIGURES = Q2 / "04_图表" / "q2_m0_effects"
RESULTS_CLOSEOUT = ROOT / "results" / "q2_closeout" / "v1"
DOCS = ROOT / "docs"
FIGURES = ROOT / "figures" / "q2_m0_effects"
PACKAGE_FILES = [
    "README_Q2_FINAL.md",
    "README_inputs.md",
    "inputs_manifest.csv",
    "q2_m0_effects_figure_contract.csv",
    "q2_m0_effects_distribution_summary.csv",
    "q2_m0_effects_summary_readme.md",
    "run_log.txt",
]
MIRRORED_CLOSEOUT_FILES = [
    "q2_evidence_matrix.csv",
    "q2_identifiability_gates.csv",
    "q2_model_status.csv",
    "q2_q_incremental_effect_gate.md",
    "q2_stage_conclusion.md",
    "q2_closeout_manifest.json",
    "q2_final_package_manifest.json",
]
FIGURE_NAMES = [
    "m0_marginal_effect_N",
    "m0_marginal_effect_D",
    "m0_elasticity_N",
    "m0_elasticity_D",
]
EXPECTED_VALIDATION_N = {"B2": 1029, "B3": 4000, "B4": 57, "B5": 44}
BLUE = "#0072B2"  # Okabe-Ito blue; all categories also have direct labels/panels.
GRAY = "#5B6573"
BAND = "#B8D4E6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(content)
    temp.replace(path)


def run_child(script: str) -> dict:
    command = [sys.executable, str(ROOT / script)]
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    record = {
        "script": script,
        "command": subprocess.list2cmdline(command),
        "exit_code": result.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return record


def configure_plotting() -> tuple[dict, object | None, object | None, object | None, object | None]:
    """Use the project's figure helpers when available; retain a local fallback."""
    helper_dir = Path.home() / ".codex" / "skills" / "math-modeling" / "tools" / "figure" / "scripts"
    setup_style = export_figure = audit_layout = print_report = finalize_figure = None
    if helper_dir.is_dir():
        sys.path.insert(0, str(helper_dir))
        try:
            from setup_style import setup_style as _setup_style
            from export_figure import export_figure as _export_figure
            from visual_qa import audit_layout as _audit_layout, print_report as _print_report
            from layout_tools import finalize_figure as _finalize_figure
            setup_style, export_figure = _setup_style, _export_figure
            audit_layout, print_report = _audit_layout, _print_report
            finalize_figure = _finalize_figure
        except Exception:
            setup_style = export_figure = audit_layout = print_report = finalize_figure = None

    if setup_style is not None:
        style_info = setup_style(journal="general", lang="en", use_sciplots=False)
    else:
        matplotlib.rcParams.update({
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        })
        style_info = {"journal": "general", "lang": "en", "sciplots_used": False, "fallback": True}
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    return style_info, export_figure, audit_layout, print_report, finalize_figure


def export_plot(fig, basename: Path, export_figure, size: tuple[float, float]) -> list[Path]:
    basename.parent.mkdir(parents=True, exist_ok=True)
    if export_figure is not None:
        names = export_figure(
            fig,
            str(basename),
            formats=["pdf", "svg", "png"],
            dpi=300,
            size_inches=size,
            grayscale_preview=True,
            tight=False,
        )
        paths = [Path(name) for name in names]
        # Pillow's grayscale conversion drops PNG resolution metadata; restore
        # the same 300-DPI tag as the color preview for a clean export audit.
        for path in paths:
            if path.name.endswith("_grayscale.png"):
                from PIL import Image
                temp = path.with_name(path.stem + ".dpi.tmp.png")
                with Image.open(path) as preview:
                    preview.save(temp, dpi=(300, 300))
                temp.replace(path)
        return paths
    fig.set_size_inches(*size)
    paths = [basename.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    fig.savefig(paths[0], bbox_inches=None)
    fig.savefig(paths[1], bbox_inches=None)
    fig.savefig(paths[2], dpi=300, bbox_inches=None)
    gray_path = basename.with_name(basename.name + "_grayscale.png")
    from PIL import Image
    Image.open(paths[2]).convert("L").save(gray_path)
    return [*paths, gray_path]


def qa_figure(fig, name: str, audit_layout, print_report, finalize_figure) -> str:
    if audit_layout is None:
        return "PASS_BASIC_LAYOUT_ONLY"
    # visual_qa renders at 100 DPI and measures against fig.bbox; align the
    # figure DPI first so its clipping audit compares the same pixel scale.
    fig.set_dpi(100)
    if finalize_figure is not None:
        finalize_figure(fig, prefer="constrained")
    else:
        fig.canvas.draw()
    issues = audit_layout(fig)
    verdict = print_report(issues)
    if isinstance(verdict, str) and verdict.upper().startswith("FAIL"):
        raise RuntimeError(f"Figure layout QA failed for {name}: {issues}")
    if any(str(severity).lower() in {"fail", "error", "critical"} for severity, _ in issues):
        raise RuntimeError(f"Figure layout QA failed for {name}: {issues}")
    return "PASS" if not issues else "WARN_REVIEWED"


def run_figure_compliance_check() -> dict:
    checker = Path.home() / ".codex" / "skills" / "math-modeling" / "tools" / "figure" / "scripts" / "check_figure.py"
    if not checker.is_file():
        return {"status": "CHECKER_UNAVAILABLE", "exit_code": None, "command": ""}
    command = [
        sys.executable, str(checker), "--min-dpi", "300", "--strict",
        str(FIGURES / "*.pdf"), str(FIGURES / "*.svg"), str(FIGURES / "*.png"),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Strict figure-file QA failed:\n{result.stdout}\n{result.stderr}")
    return {
        "status": "PASS" if "[WARN]" not in result.stdout else "PASS_WITH_WARNINGS",
        "exit_code": result.returncode,
        "command": subprocess.list2cmdline(command),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def finite_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values found in {name}: {columns}")


def render_effect_figures(style_info: dict) -> tuple[list[dict], list[Path]]:
    checkpoint_path = EFFECTS / "q2_m0_marginal_effects_by_checkpoint.csv"
    size_path = EFFECTS / "q2_m0_marginal_effects_by_size.csv"
    checkpoints = pd.read_csv(checkpoint_path)
    sizes = pd.read_csv(size_path).sort_values("N_params_B").reset_index(drop=True)
    if len(checkpoints) != 1176 or checkpoints["model_repo"].nunique() != 8:
        raise ValueError("M0 effect plot inputs must cover 1,176 checkpoints and eight tracks")
    if not (checkpoints.groupby("model_repo").size() == 147).all() or len(sizes) != 8:
        raise ValueError("M0 effect plot inputs do not match the frozen 8 x 147 trajectory structure")
    effect_cols = [
        "dLoss_dN_per_B_params", "dLoss_dN_per_B_params_lower_95", "dLoss_dN_per_B_params_upper_95",
        "dLoss_dD_per_B_tokens", "dLoss_dD_per_B_tokens_lower_95", "dLoss_dD_per_B_tokens_upper_95",
        "elasticity_N", "elasticity_N_lower_95", "elasticity_N_upper_95",
        "elasticity_D", "elasticity_D_lower_95", "elasticity_D_upper_95",
    ]
    finite_columns(checkpoints, effect_cols, "checkpoint effects")
    finite_columns(sizes, [
        "dLoss_dN_per_B_params_median", "dLoss_dN_per_B_params_median_lower_95", "dLoss_dN_per_B_params_median_upper_95",
        "elasticity_N_median", "elasticity_N_median_lower_95", "elasticity_N_median_upper_95",
    ], "track effect summary")

    style_info, export_figure, audit_layout, print_report, finalize_figure = configure_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_records: list[dict] = []
    generated: list[Path] = []

    def save(fig, base: str, claim: str, chart_type: str, xaxis: str, yaxis: str, n_note: str, size: tuple[float, float] = (7.2, 4.5)):
        qa = qa_figure(fig, base, audit_layout, print_report, finalize_figure)
        paths = export_plot(fig, FIGURES / base, export_figure, size)
        generated.extend(paths)
        figure_records.append({
            "figure": base,
            "claim": claim,
            "chart_type": chart_type,
            "data_sources": "q2_m0_marginal_effects_by_checkpoint.csv; q2_m0_marginal_effects_by_size.csv",
            "x_axis": xaxis,
            "y_axis": yaxis,
            "interval": "95% percentile interval from 1,000 saved complete-track cluster-bootstrap fits; 8 independent trajectory clusters",
            "sample_definition": n_note,
            "qa": qa,
            "style": json.dumps(style_info, ensure_ascii=False),
        })
        plt.close(fig)

    # N derivative depends only on N under the frozen additive M0. Plot the
    # positive magnitude on a log axis; the signed derivative is negative.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    n = sizes["N_params_B"].to_numpy(float)
    center = -sizes["dLoss_dN_per_B_params_median"].to_numpy(float)
    low = -sizes["dLoss_dN_per_B_params_median_upper_95"].to_numpy(float)
    high = -sizes["dLoss_dN_per_B_params_median_lower_95"].to_numpy(float)
    ax.errorbar(n, center, yerr=np.vstack([center - low, high - center]), fmt="o", color=BLUE,
                ecolor=GRAY, elinewidth=1.15, capsize=3.2, markersize=5.5)
    for x, y, label in zip(n, center, sizes["B12_model_size"].astype(str)):
        ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, color=GRAY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (B parameters; log scale)")
    ax.set_ylabel("-dLoss/dN (Loss/B params; log)")
    ax.set_title("M0 marginal effect of model size")
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.55, alpha=0.8)
    ax.text(0.01, 0.015, "Whiskers: 95% bootstrap CI for track-median effect; signed dLoss/dN is negative.",
            transform=ax.transAxes, fontsize=8, color=GRAY, va="bottom")
    save(fig, "m0_marginal_effect_N", "The magnitude of the negative N marginal effect falls as observed model size grows.",
         "log-log dot plot with asymmetric 95% CI", "N (billion parameters; observed eight sizes)",
         "-dLoss/dN magnitude (Loss / billion parameters)", "8 model-size trajectories; 147 checkpoints per trajectory; inference clusters=8")

    # Under additive M0, the D derivative is independent of N. Deduplicate the
    # shared observed D grid instead of treating eight repeated tracks as 8x n.
    repeated = checkpoints.groupby("D_tokens_B")[[
        "dLoss_dD_per_B_tokens", "dLoss_dD_per_B_tokens_lower_95", "dLoss_dD_per_B_tokens_upper_95"
    ]].agg(["min", "max"])
    max_spread = float((repeated.xs("max", axis=1, level=1) - repeated.xs("min", axis=1, level=1)).abs().to_numpy().max())
    if max_spread > 1e-10:
        raise ValueError(f"D-only M0 derivative differs across N at a shared D checkpoint: spread={max_spread}")
    dcurve = checkpoints.sort_values("D_tokens_B").drop_duplicates("D_tokens_B").copy()
    dcurve["effect_mag"] = -dcurve["dLoss_dD_per_B_tokens"]
    dcurve["ci_low_mag"] = -dcurve["dLoss_dD_per_B_tokens_upper_95"]
    dcurve["ci_high_mag"] = -dcurve["dLoss_dD_per_B_tokens_lower_95"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    dx = dcurve["D_tokens_B"].to_numpy(float)
    dy = dcurve["effect_mag"].to_numpy(float)
    ax.plot(dx, dy, color=BLUE, linewidth=1.6)
    ax.fill_between(dx, dcurve["ci_low_mag"].to_numpy(float), dcurve["ci_high_mag"].to_numpy(float),
                    color=BAND, alpha=0.72, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D (B tokens; log scale)")
    ax.set_ylabel("-dLoss/dD (Loss/B tokens; log)")
    ax.set_title("M0 marginal effect of training tokens")
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.55, alpha=0.8)
    ax.text(0.01, 0.015, f"Band: pointwise 95% bootstrap CI; {len(dcurve)} distinct observed D checkpoints; the derivative is N-invariant in M0.",
            transform=ax.transAxes, fontsize=8, color=GRAY, va="bottom")
    save(fig, "m0_marginal_effect_D", "The magnitude of the negative D marginal effect declines across observed training-token checkpoints.",
         "log-log line with pointwise 95% CI band", "D (billion tokens; observed support)",
         "-dLoss/dD magnitude (Loss / billion tokens)", f"{len(dcurve)} distinct D checkpoints shared by eight tracks; inference clusters=8")

    # N elasticity uses the within-track checkpoint median, with the saved
    # bootstrap interval for that median. Eight points are tracks, not 1,176 iid runs.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    eps_n = sizes["elasticity_N_median"].to_numpy(float)
    eps_n_lo = sizes["elasticity_N_median_lower_95"].to_numpy(float)
    eps_n_hi = sizes["elasticity_N_median_upper_95"].to_numpy(float)
    ax.errorbar(n, eps_n, yerr=np.vstack([eps_n - eps_n_lo, eps_n_hi - eps_n]), fmt="o", color=BLUE,
                ecolor=GRAY, elinewidth=1.15, capsize=3.2, markersize=5.5)
    for x, y, label in zip(n, eps_n, sizes["B12_model_size"].astype(str)):
        ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, color=GRAY)
    ax.set_xscale("log")
    ax.set_xlabel("N (B parameters; log scale)")
    ax.set_ylabel("N elasticity (dimensionless)")
    ax.set_title("M0 elasticity by model size")
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.55, alpha=0.8)
    ax.text(0.01, 0.015, "Point: median over 147 observed checkpoints within a track; whisker: 95% cluster-bootstrap CI.",
            transform=ax.transAxes, fontsize=8, color=GRAY, va="bottom")
    save(fig, "m0_elasticity_N", "The within-track median N elasticity becomes less negative at larger observed model sizes.",
         "log-x dot plot with asymmetric 95% CI", "N (billion parameters; observed eight sizes)",
         "Elasticity_N (dimensionless)", "8 model trajectories; track center is median across 147 checkpoints; inference clusters=8")

    # Eight panels avoid an unreadable eight-item legend. Same line/color/axes
    # across facets makes the size-specific elasticity shapes directly comparable.
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.4), sharex=True, sharey=True)
    y_min = float(checkpoints["elasticity_D_lower_95"].min())
    y_max = float(checkpoints["elasticity_D_upper_95"].max())
    pad = max((y_max - y_min) * 0.08, 0.003)
    for ax, row in zip(axes.flat, sizes.to_dict(orient="records")):
        track = checkpoints[checkpoints["model_repo"] == row["model_repo"]].sort_values("D_tokens_B")
        x = track["D_tokens_B"].to_numpy(float)
        center = track["elasticity_D"].to_numpy(float)
        lower = track["elasticity_D_lower_95"].to_numpy(float)
        upper = track["elasticity_D_upper_95"].to_numpy(float)
        ax.plot(x, center, color=BLUE, linewidth=1.25)
        ax.fill_between(x, lower, upper, color=BAND, alpha=0.72, linewidth=0)
        ax.set_xscale("log")
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(f"{row['B12_model_size']}  |  N={row['N_params_B']:.3g}B", fontsize=9)
        ax.grid(axis="y", color="#D8DEE6", linewidth=0.45, alpha=0.8)
        ax.tick_params(axis="both", labelsize=7)
    fig.supxlabel("Training tokens D (billion tokens; log scale)", fontsize=9)
    fig.supylabel("D elasticity of predicted Loss (dimensionless)", fontsize=9)
    fig.suptitle("M0 elasticity with respect to D by model size\nShading: pointwise 95% CI (1,000 bootstrap fits; 8 tracks)", fontsize=10)
    save(fig, "m0_elasticity_D", "D elasticity varies over training progress and model size within the observed B1 support.",
         "2x4 small-multiple trend panels with 95% CI bands", "D (billion tokens; observed support)",
         "Elasticity_D (dimensionless)", "8 tracks x 147 checkpoints; the 4,000 B3 interpolation points are not included", size=(7.2, 5.4))

    # Mirror the exact exported files and contract into the user-facing Q2 package.
    Q2_FIGURES.mkdir(parents=True, exist_ok=True)
    mirrored = []
    for path in generated:
        target = Q2_FIGURES / path.name
        shutil.copy2(path, target)
        mirrored.append(target)
    contract = pd.DataFrame(figure_records)
    contract_bytes = contract.to_csv(index=False, encoding="utf-8-sig", lineterminator="\n").encode("utf-8-sig")
    for path in [BASELINE / "q2_m0_effects_figure_contract.csv", Q2_BASELINE / "q2_m0_effects_figure_contract.csv", DOCS / "q2_m0_effects_figure_contract.csv"]:
        write_bytes(path, contract_bytes)
    for source, target in zip(generated, mirrored):
        assert sha256(source) == sha256(target), f"Figure mirror hash mismatch: {target}"
    return figure_records, [*generated, *mirrored]


def build_effect_distribution_summary() -> tuple[pd.DataFrame, str]:
    """Summarize checkpoint coverage separately from Bootstrap CI of medians."""
    checkpoint = pd.read_csv(EFFECTS / "q2_m0_marginal_effects_by_checkpoint.csv")
    track_ci = pd.read_csv(EFFECTS / "q2_m0_marginal_effects_by_size.csv").set_index("model_repo")
    grouped = checkpoint.groupby("model_repo", sort=False)
    summaries = []
    source_columns = {
        "N_params_B": "N_params_B",
        "D_tokens_B": "D_tokens_B",
        "observed_val_loss": "observed_val_loss",
        "dLoss_dN_per_B_params": "dLoss_dN_per_B_params",
        "dLoss_dD_per_B_tokens": "dLoss_dD_per_B_tokens",
        "elasticity_N": "elasticity_N",
        "elasticity_D": "elasticity_D",
    }
    bootstrap_columns = {
        "dLoss_dN_per_B_params": ("dLoss_dN_per_B_params_median_lower_95", "dLoss_dN_per_B_params_median_upper_95"),
        "dLoss_dD_per_B_tokens": ("dLoss_dD_per_B_tokens_median_lower_95", "dLoss_dD_per_B_tokens_median_upper_95"),
        "elasticity_N": ("elasticity_N_median_lower_95", "elasticity_N_median_upper_95"),
        "elasticity_D": ("elasticity_D_median_lower_95", "elasticity_D_median_upper_95"),
    }
    for model_repo, frame in grouped:
        row = {
            "model_repo": model_repo,
            "B12_model_size": str(frame["B12_model_size"].iloc[0]),
            "n_checkpoints": int(len(frame)),
        }
        for output_name, source_name in source_columns.items():
            values = frame[source_name].astype(float)
            row[f"{output_name}_min"] = float(values.min())
            row[f"{output_name}_q1"] = float(values.quantile(0.25))
            row[f"{output_name}_median"] = float(values.median())
            row[f"{output_name}_q3"] = float(values.quantile(0.75))
            row[f"{output_name}_max"] = float(values.max())
        ci_row = track_ci.loc[model_repo]
        for effect_name, (lower_col, upper_col) in bootstrap_columns.items():
            row[f"{effect_name}_track_median_bootstrap_95ci_low"] = float(ci_row[lower_col])
            row[f"{effect_name}_track_median_bootstrap_95ci_high"] = float(ci_row[upper_col])
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values("N_params_B_median").reset_index(drop=True)
    if len(summary) != 8 or not (summary["n_checkpoints"] == 147).all():
        raise ValueError("Effect distribution summary must contain eight 147-checkpoint tracks")
    numeric = summary.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Effect distribution summary contains non-finite values")

    table = [
        "| 轨迹 | N (B) | D 覆盖 (B tokens) | 观测 Loss 覆盖 | εN 中位数 [Bootstrap 95% CI] | εD 中位数 [Bootstrap 95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        table.append(
            f"| {row['B12_model_size']} | {row['N_params_B_median']:.6g} | "
            f"{row['D_tokens_B_min']:.6g}–{row['D_tokens_B_max']:.6g} | "
            f"{row['observed_val_loss_min']:.6g}–{row['observed_val_loss_max']:.6g} | "
            f"{row['elasticity_N_median']:.6g} [{row['elasticity_N_track_median_bootstrap_95ci_low']:.6g}, {row['elasticity_N_track_median_bootstrap_95ci_high']:.6g}] | "
            f"{row['elasticity_D_median']:.6g} [{row['elasticity_D_track_median_bootstrap_95ci_low']:.6g}, {row['elasticity_D_track_median_bootstrap_95ci_high']:.6g}] |"
        )
    report = """# Q2 M0 效应分布摘要

## 口径

本表按 8 条 B1 模型轨迹分别汇总每条 147 个观测检查点。N、D、观测 Loss 和四种效应的 `min/q1/median/q3/max` 描述该轨迹内实际检查点的覆盖分布；这些分位数不是置信区间。

`*_track_median_bootstrap_95ci_low/high` 是对已保存的 1,000 次完整轨迹 cluster Bootstrap 参数样本计算轨迹中位数后得到的百分位区间。只有 8 个独立轨迹 cluster，区间只作稳定性提示。

单位：N 为十亿参数；D 为十亿 Token；`dLoss_dN` 为 Loss/十亿参数；`dLoss_dD` 为 Loss/十亿 Token；弹性无量纲。弹性分母采用每个检查点的 M0 预测 Loss。负边际效应/弹性表示模型内预测 Loss 随 N/D 增长而下降，不构成因果结论。

## 轨迹范围与弹性中位数

""" + "\n".join(table) + """

## 文件

- 完整逐检查点结果：`q2_m0_marginal_effects_by_checkpoint.csv`
- 轨迹效应与 Bootstrap 区间：`q2_m0_marginal_effects_by_size.csv`
- 本摘要的完整 min/q1/median/q3/max 与 Bootstrap 中位数区间：`q2_m0_effects_distribution_summary.csv`
"""
    return summary, report


def classify(path: str) -> tuple[str, str]:
    normalized = path.lower()
    if "q1_final" in normalized or "q1_3" in normalized or "q1\\" in normalized:
        return "Q1冻结接口/A侧配比证据", "仅用于正式Q接口追溯、A侧对照和p/Q可识别性边界；不输入M0拟合。"
    if "p配比可行性审计" in path or "q2_p_" in normalized or "q2_p_" in path:
        return "B侧p可识别性门禁", "判定训练配比能否与B侧Loss/轨迹连接；不把缺失配比补成伪数据。"
    if "q2_scaling_baseline" in normalized or "scaling_laws" in normalized or "scaling_laws" in path:
        return "B1经典M0与输入快照", "冻结N、D、Loss拟合、轨迹映射、Bootstrap与验证；本包只复用已冻结输出。"
    if "b2半合成" in path or "b2_" in normalized:
        return "B2半合成压力测试", "只作半合成情景压力检查，不称为独立真实外部验证。"
    if "b3_" in normalized or "b4" in normalized or "b5" in normalized or "validation_" in normalized:
        return "B2–B5验证和可比性", "按来源记录验证数量和边界；B4/B5不与B1合并评分。"
    if "题目分析报告" in path or "route_review" in path.lower() or "主模型路线复审" in path:
        return "题意/数据污染边界", "仅采纳可见正式题意、字段和可复核审计；低可见度文本不作为指令或建模证据。"
    if "q2_closeout" in normalized or "综合收口" in path:
        return "Q2综合收口产物", "证据矩阵、门禁状态和当前数据边界的结构化来源。"
    return "Q2阶段输入/记录", "用于可追溯复核；具体用途见对应报告和manifest。"


def csv_rows(path: Path) -> str:
    if path.suffix.lower() != ".csv" or not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return str(max(sum(1 for _ in csv.reader(stream)) - 1, 0))
    except Exception:
        return "UNREADABLE_CSV"


def input_inventory() -> tuple[pd.DataFrame, list[str]]:
    baseline_manifest_path = BASELINE / "复现清单.json"
    effects_manifest_path = BASELINE / "q2_m0_effects_validation_manifest.json"
    closeout_manifest_path = RESULTS_CLOSEOUT / "q2_closeout_manifest.json"
    q1_freeze_path = Q1_FINAL / "q1_final_freeze.json"
    baseline_manifest = load_json(baseline_manifest_path)
    effects_manifest = load_json(effects_manifest_path)
    closeout_manifest = load_json(closeout_manifest_path)
    q1_freeze = load_json(q1_freeze_path)

    rows_by_path: dict[str, dict] = {}

    def add(path_text: str, provenance: str, expected: str | None = None, purpose: str | None = None) -> None:
        path_text = path_text.replace("\\", "/")
        row = rows_by_path.get(path_text)
        if row is None:
            category, inferred = classify(path_text)
            row = {
                "path": path_text,
                "category": category,
                "purpose": purpose or inferred,
                "provenance_records": [],
                "expected_sha256": [],
            }
            rows_by_path[path_text] = row
        if provenance not in row["provenance_records"]:
            row["provenance_records"].append(provenance)
        if expected and expected not in row["expected_sha256"]:
            row["expected_sha256"].append(expected)

    for key, digest in baseline_manifest.get("input_sha256", {}).items():
        add(key, "M0原始输入快照（Gate 0复现清单）", digest)
    for key, digest in baseline_manifest.get("outputs_sha256", {}).items():
        add(key, "冻结M0运行产物（Gate 0复现清单）", digest)
    for key, digest in effects_manifest.get("input_sha256", {}).items():
        add(key, "M0效应/验证收口的实际输入清单", digest)
    for key, digest in closeout_manifest.get("input_sha256", {}).items():
        add(key, "Q2综合收口的实际输入清单", digest)
    for key, meta in q1_freeze.get("outputs", {}).items():
        add(f"results/q1_final/{key}", "Q1最终冻结文件哈希", meta.get("sha256"))

    extras = [
        (q1_freeze_path, "Q1最终接口冻结", None),
        (Q1_FINAL / "q1_final_decision.md", "Q1最终接口冻结", None),
        (Q2 / "01_方案说明" / "题目分析报告.md", "当前权威题目分析/污染审计", None),
        (Q2 / "01_方案说明" / "Q2_主模型路线复审_2026-09-24.md", "Q2主模型层级复审", None),
        (Q2_BASELINE / "gate0_file_roles.csv", "B1–B5 Gate 0数据角色合同", None),
        (Q2_BASELINE / "gate0_b1_trajectory_mapping.csv", "B1轨迹分组映射", None),
        (Q2_BASELINE / "b3_validation_closeout.md", "B3来源性质与限制", None),
        (Q2_BASELINE / "b4_b5_loss_comparability_audit.md", "B4/B5可比性审计", None),
        (Q2_B2 / "b2_cerebras_stress_report.md", "B2半合成压力测试说明", None),
        (Q2_B2 / "b2_stress_metrics.csv", "B2情景压力指标", None),
        (Q2_B2 / "b2_endpoint_anchor_checks.csv", "B2端点锚点核验", None),
        (Q2_B2 / "b2_cerebras_pythia_calibrated_trajectories.csv", "B2半合成轨迹数据", None),
        (Q2_B2 / "generation_manifest.json", "B2生成参数和哈希", None),
        (Q2_P_AUDIT / "q2_p_audit_summary.json", "p门禁摘要", None),
        (Q2_P_AUDIT / "q2_p_feasibility_audit.md", "p门禁报告", None),
        (Q2_P_AUDIT / "q2_p_trajectory_mapping.csv", "p与训练轨迹的审计映射", None),
        (Q2_P_AUDIT / "q2_p_source_evidence_catalog.csv", "p来源证据目录", None),
        (Q2_P_AUDIT / "q2_p_audit_manifest.json", "p门禁输入输出哈希", None),
        (Q2_BASELINE / "q2_m0_marginal_effects_by_checkpoint.csv", "M0逐检查点效应表", None),
        (Q2_BASELINE / "q2_m0_marginal_effects_by_size.csv", "M0逐轨迹摘要表", None),
        (Q2_BASELINE / "q2_m0_marginal_effects_report.md", "M0效应报告", None),
        (Q2_BASELINE / "q2_validation_scope_closeout.md", "B2–B5验证边界报告", None),
        (Q2_CLOSEOUT / "q2_evidence_matrix.csv", "Q2阶段证据矩阵", None),
        (Q2_CLOSEOUT / "q2_identifiability_gates.csv", "p/Q可识别性门禁", None),
        (Q2_CLOSEOUT / "q2_model_status.csv", "Q2模型状态", None),
    ]
    for path, source, expected in extras:
        add(relative(path), source, expected)

    issues: list[str] = []
    records = []
    for relpath, row in sorted(rows_by_path.items()):
        path = ROOT / Path(relpath)
        exists = path.is_file()
        current_hash = sha256(path) if exists else ""
        expected_hashes = row["expected_sha256"]
        if not exists:
            status = "MISSING_AT_RECORDED_PATH" if expected_hashes else "NOT_FOUND"
            if expected_hashes:
                issues.append(f"missing frozen source: {relpath}")
        elif len(expected_hashes) > 1:
            status = "CONFLICTING_EXPECTED_HASH_RECORDS" if current_hash not in expected_hashes else "HASH_MATCH_WITH_MULTIPLE_RECORDS"
            if status == "CONFLICTING_EXPECTED_HASH_RECORDS":
                issues.append(f"conflicting frozen hashes: {relpath}")
        elif expected_hashes:
            status = "HASH_MATCH" if current_hash == expected_hashes[0] else "CHANGED_SINCE_FROZEN_SNAPSHOT"
            if status != "HASH_MATCH":
                issues.append(f"changed since frozen snapshot: {relpath}")
        else:
            status = "AVAILABLE_CURRENT_HASH_RECORDED"
        records.append({
            "path": relpath,
            "category": row["category"],
            "purpose": row["purpose"],
            "provenance_records": "; ".join(row["provenance_records"]),
            "current_status": status,
            "current_sha256": current_hash,
            "expected_sha256": ";".join(expected_hashes),
            "bytes": path.stat().st_size if exists else "",
            "csv_data_rows": csv_rows(path) if exists and path.suffix.lower() == ".csv" else "",
        })
    return pd.DataFrame(records), issues


def validate_current_evidence() -> dict:
    baseline_manifest = load_json(BASELINE / "复现清单.json")
    effect_manifest = load_json(BASELINE / "q2_m0_effects_validation_manifest.json")
    closeout_manifest = load_json(RESULTS_CLOSEOUT / "q2_closeout_manifest.json")
    q_freeze = load_json(Q1_FINAL / "q1_final_freeze.json")
    p_gate = load_json(Q2_P_AUDIT / "q2_p_audit_summary.json")
    fit = load_json(BASELINE / "fit_parameters.json")
    validation = pd.read_csv(BASELINE / "validation_metrics_by_source.csv").set_index("dataset")
    mapping = pd.read_csv(Q2_BASELINE / "gate0_b1_trajectory_mapping.csv")
    predictions = pd.read_csv(BASELINE / "b1_fitted_predictions.csv")
    effects = pd.read_csv(BASELINE / "q2_m0_marginal_effects_by_checkpoint.csv")
    effects_size = pd.read_csv(BASELINE / "q2_m0_marginal_effects_by_size.csv")
    statuses = pd.read_csv(RESULTS_CLOSEOUT / "q2_model_status.csv").set_index("module")
    gates = pd.read_csv(RESULTS_CLOSEOUT / "q2_identifiability_gates.csv").set_index("gate")

    if q_freeze.get("status") != "FROZEN_WITH_LIMITATIONS" or q_freeze.get("operational_q") != "q_huber" or q_freeze.get("sensitivity_q") != "q_equal":
        raise ValueError("Q1 formal/sensitivity Q interface is not the expected frozen q_huber/q_equal pair")
    for name, meta in q_freeze.get("outputs", {}).items():
        path = Q1_FINAL / name
        if not path.is_file() or sha256(path) != meta.get("sha256"):
            raise ValueError(f"Q1 frozen output hash mismatch: {path}")
    if p_gate.get("overall_p_only_data_gate") != "FAIL" or int(p_gate.get("verified_p_vectors_joined_to_loss", -1)) != 0:
        raise ValueError("Current B-side p Gate has changed; review before producing this package")
    if len(predictions) != 1176 or predictions["run_id"].nunique() != 1176:
        raise ValueError("Frozen B1 predictions no longer match 1,176 checkpoint rows")
    if len(mapping) != 8 or mapping["model_repo"].nunique() != 8 or not (mapping["B1_rows"].astype(int) == 147).all():
        raise ValueError("Frozen B1 trajectory mapping no longer matches 8 x 147")
    if len(effects) != 1176 or effects["model_repo"].nunique() != 8 or len(effects_size) != 8:
        raise ValueError("M0 effects tables do not cover the full B1 trajectory set")
    if {str(k): int(v) for k, v in validation["n"].items()} != EXPECTED_VALIDATION_N:
        raise ValueError("B2-B5 validation row counts changed")
    if statuses.loc["M1 (N,D,p)", "status"] != "NOT_ESTIMATED" or statuses.loc["M2 (N,D,p,Q)", "status"] != "NOT_ESTIMATED":
        raise ValueError("M1/M2 statuses changed; inspect before current-data closeout")
    if gates.loc["p-effect identifiability", "status"] != "FAIL_CURRENT_DATA" or gates.loc["Q incremental-effect identifiability", "status"] != "FAIL_CURRENT_EVIDENCE":
        raise ValueError("p/Q identifiability gates changed")
    if int(effect_manifest["checks"]["bootstrap_observed"]) != 1000 or int(effect_manifest["checks"]["trajectory_clusters"]) != 8:
        raise ValueError("The saved 1,000-fit/8-cluster Bootstrap contract changed")
    if closeout_manifest.get("verification", {}).get("pooled_validation_metric") is not None:
        raise ValueError("A pooled B2-B5 metric is present; validation contract requires source-specific reporting")
    for relpath, expected in baseline_manifest.get("outputs_sha256", {}).items():
        path = ROOT / Path(relpath)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen baseline output hash mismatch: {relpath}")
    return {
        "Q1_status": q_freeze["status"],
        "Q_operational": q_freeze["operational_q"],
        "Q_sensitivity": q_freeze["sensitivity_q"],
        "M0_status": "FROZEN; NO REFIT",
        "B1_checkpoints": int(len(predictions)),
        "B1_tracks": int(mapping["model_repo"].nunique()),
        "checkpoints_per_track": sorted(mapping["B1_rows"].astype(int).unique().tolist()),
        "cluster_bootstrap_fits": int(effect_manifest["checks"]["bootstrap_observed"]),
        "validation_rows": {str(k): int(v) for k, v in validation["n"].items()},
        "p_gate": str(gates.loc["p-effect identifiability", "status"]),
        "Q_gate": str(gates.loc["Q incremental-effect identifiability", "status"]),
        "M1": str(statuses.loc["M1 (N,D,p)", "status"]),
        "M2": str(statuses.loc["M2 (N,D,p,Q)", "status"]),
        "M0_R2": float(fit["metrics"]["r2"]),
    }


def build_readme(checks: dict, inventory_rows: int, input_issues: list[str]) -> str:
    issues = "；".join(input_issues) if input_issues else "无与冻结清单不一致的当前输入。"
    return f"""# README_Q2_FINAL：Q2 当前数据版本收口包

## 最终状态

当前数据版本记为 **FROZEN_WITH_LIMITATIONS**。经典 M0 `Loss=f(N,D)` 已冻结；M0 边际效应、弹性、Bootstrap 区间、B1–B5 分源验证边界及 p/Q 可识别性门禁已整理。B 侧 p 门禁为 `{checks['p_gate']}`，Q 独立增量门禁为 `{checks['Q_gate']}`；M1/M2 均为 `NOT_ESTIMATED`。这不是 p 或 Q 无效的结论。

Q1 冻结的操作性接口为 `{checks['Q_operational']}`，敏感性接口为 `{checks['Q_sensitivity']}`。这只是 Q1 的接口决定，不表示 B1–B5 已观测到与 Loss 行级匹配、且独立变化的 Q。`Q_covered(p)` 是 p 的派生特征，也不能证明独立 Q 增量。M0 本身不使用 Q/p。

## 按题目分析报告核对 Q2 要求

阶段结论与完整回答补充均按四项组织：广义标度律；边际效应和弹性；领域配比替代与互补；质量—规模等 Loss 条件及分来源验证。详见 `q2_stage_conclusion.md` 和 `Q2/03_结果/完整回答补充/v1/q2_full_answer_evidence_supplement.md`。该组织用于逐项对照题意，不把 Q2 重新命名为四个正式小问。

## 主要交付

- 阶段结论：`q2_stage_conclusion.md`
- A/B 证据矩阵：`q2_evidence_matrix.csv`
- p/Q 门禁：`q2_identifiability_gates.csv`、`q2_q_incremental_effect_gate.md`
- 模块状态：`q2_model_status.csv`
- M0 逐检查点效应：`q2_m0_marginal_effects_by_checkpoint.csv`（1,176 行）
- M0 轨迹摘要：`q2_m0_marginal_effects_by_size.csv`（8 条轨迹）
- M0 轨迹覆盖分布摘要：`q2_m0_effects_distribution_summary.csv`（含 N/D/Loss 与边际效应的 min/Q1/median/Q3/max，并区分 Bootstrap CI）
- 分布摘要口径说明：`q2_m0_effects_summary_readme.md`
- B2–B5 验证边界：`q2_validation_scope_closeout.md`
- M0 四类边际效应/弹性图：`Q2/04_图表/q2_m0_effects/`（PNG、SVG、PDF 和灰度预览）
- 图表契约：`q2_m0_effects_figure_contract.csv`
- 输入登记：`inputs_manifest.csv`，共 {inventory_rows} 个输入/来源记录；解释见 `README_inputs.md`
- 运行日志：`run_log.txt`
- 输入输出哈希：`q2_final_package_manifest.json`、`q2_closeout_manifest.json`、`q2_m0_effects_validation_manifest.json`

## 完整回答、图表与来源索引

- 完整回答及按题意逐项对照：`Q2/03_结果/完整回答补充/v1/q2_full_answer_evidence_supplement.md`；该报告含图表索引、数据投毒边界和 M1/M2 识别条件。
- 完整回答图表：`Q2/04_图表/q2_full_answer/`；图表合同和布局 QA：`Q2/03_结果/完整回答补充/v1/q2_full_answer_figures_contract.csv`、`q2_full_answer_figures_qa.json`。
- 图表数据/来源与复现哈希：`Q2/03_结果/完整回答补充/v1/repro_manifest.json`。
- B4/B5 原文口径和来源定位：`Q2/05_审计交付/论文网址清单.md`、`Q2/03_结果/经典ScalingLaw基线/v1/loss_protocol_evidence_sources.md`、`Q2/03_结果/经典ScalingLaw基线/v1/b4_b5_loss_comparability_audit.md`；引用时按这些文件区分原文指标与本地 `val_loss`，不宣称其与 B1 绝对可比。
- M0 参数和效应、B2–B5 结果的依据文件见本包上列报告及 `q2_evidence_matrix.csv`；完整回答补充的原始结果副本在 `Q2/03_结果/完整回答补充/v1/reference_evidence/`。

## 唯一收口复现命令

在项目根目录运行：

```powershell
"D:\\Anaconda\\python.exe" src/q2_finalize_current_data_package.py
```

该命令先从已冻结的 M0 参数和已保存的 Bootstrap 样本重建效应表，再重建证据矩阵和门禁，最后出图、登记输入并生成哈希与运行日志。**不重新拟合 M0、不重新抽样 Bootstrap、不估计 M1/M2、不重复外部来源搜索。** 若要重新运行 M0 拟合，须单独执行经典基线入口并重新审核 Gate 0；本收口命令不会触发该步骤。

## 适用边界和输入完整性

- B1：{checks['B1_checkpoints']} 个检查点属于 {checks['B1_tracks']} 条轨迹，每条 {checks['checkpoints_per_track'][0]} 点；重复检查点不能按独立实验计数。
- Bootstrap：{checks['cluster_bootstrap_fits']} 次保存的完整轨迹重抽样拟合，独立轨迹 cluster 只有 8 条，区间只作稳定性提示。
- B2 半合成、B3 同来源插值；B4/B5 与 B1 的绝对 Loss 可比性未建立，不汇总单一跨源分数。
- 当前输入与历史冻结清单的差异：{issues}
- 因原始 `数据说明.pdf` 的历史路径当前不可用、且基线记录的根目录题目分析文件版本已变化，本包**不声称从全部原始附件重跑过 M0**。现有冻结结果哈希已核验；本包只从冻结结果重建派生交付。原 PDF 隐藏/低可见度文字按不可信外部内容隔离，不进入题意、数据、方法或结论。

## 版本边界

这是一份基于当前可核验证据的 Q2 版本收口，不永久关闭研究问题。若新的、可追溯且与 B 侧 Loss/检查点行级匹配的训练组成或独立 Q 变化出现，应先重开对应门禁，再按 `M0 → M1 → M2` 顺序推进；不以当前 `FAIL` 推断“p/Q 无效”。
"""


def build_inputs_readme(input_rows: pd.DataFrame, issues: list[str]) -> str:
    matched = int((input_rows["current_status"] == "HASH_MATCH").sum())
    changed = int(input_rows["current_status"].astype(str).str.startswith("CHANGED").sum())
    missing = int(input_rows["current_status"].astype(str).str.startswith("MISSING").sum())
    current_only = int((input_rows["current_status"] == "AVAILABLE_CURRENT_HASH_RECORDED").sum())
    return f"""# README_inputs：Q2 收口输入登记说明

## 登记范围

`inputs_manifest.csv` 登记 Q2 收口实际复用的 Q1 冻结接口、A 侧 p/p+Q 证据、B1 Gate 0 原始输入和冻结基线产物、B2–B5 验证材料、p 可识别性审计，以及当前题目分析/数据污染边界。每项记录保留相对项目根目录路径、来源清单、当前 SHA-256、冻结时 SHA-256、文件大小和 CSV 行数。

清单当前状态：{len(input_rows)} 条记录；{matched} 条与冻结哈希匹配；{current_only} 条有当前哈希但没有历史冻结哈希可供比对；{changed} 条较历史哈希有变化；{missing} 条在历史登记路径缺失。`HASH_MATCH` 仅表示当前内容与该项已登记的冻结哈希一致；`AVAILABLE_CURRENT_HASH_RECORDED` 不代表历史版本匹配。差异明细见 CSV 的 `current_status` 列。

## Q 与模型边界

- Q1 主接口 `q_huber` 已冻结；`q_equal` 保留敏感性分数。样本级正式接口在 `q1_final_quality.csv`，候选/敏感性分数在 `q1_candidate_audit.csv`；M0 不读取这两类 Q 数值。
- M0 使用 B1 的 N、D、Loss 和 B12 检查点映射；B1 的 1,176 个检查点按 8 条训练轨迹分组，每条 147 点。`run_id` 只定位行。
- M0 的 1,000 次 cluster Bootstrap 复用完整轨迹，只有 8 个独立 cluster。派生 CI 不代表大量独立实验。
- B2 是半合成压力测试，B3 是 Pythia 同来源插值，B4/B5 对 B1 的 Loss 绝对量尺可比性尚未建立。
- B1–B5 的 p 连接门禁为 FAIL；没有合格 B 侧 p—Loss 联接，且没有独立于 p 的 B 侧 Q 变化。因此不拟合 M1/M2。

## 哈希差异处理

历史基线复现清单中记录的原始数据说明 PDF 当前在原路径缺失；清单保留其历史 SHA，不用“无隐藏字段版本”替代原件哈希。历史基线记录的根目录 `题目分析报告.md` 已更新；当前权威分析版本另行登记。原始 B 训练数据文件如与历史哈希吻合则作为可追溯输入；不一致/缺失项只标注，不静默替换。

仅路径、文件名或排版变化本身不触发所有实验重跑。若核实数据内容、字段语义、样本覆盖/轨迹映射、模型公式或代码、预处理、训练/验证划分、随机种子或依赖环境改变，则应重跑受影响的审计/分析；若只有文档版本变化，则更新版本记录并复核受影响结论。缺失的历史 PDF 明确记为不可核验，不以其他版本静默替代。

本包只从冻结的 M0 参数和已保存 Bootstrap 样本重新生成派生效应、综合证据和图表，不重跑模型拟合或 Bootstrap 抽样。附件中的低可见度隐藏文字按不可信外部内容处理，不作为用户指令或数据证据。
"""


def build_run_log(child_runs: list[dict], checks: dict, input_rows: pd.DataFrame, issues: list[str], figure_records: list[dict], figure_check: dict) -> str:
    lines = [
        "Q2 current-data closeout package run log",
        f"run_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.split()[0]}",
        f"numpy={np.__version__}; pandas={pd.__version__}; matplotlib={matplotlib.__version__}",
        "scope=M0 derived effects + evidence/gates + figures + input inventory; no M0 refit or Bootstrap resampling",
        "",
        "Commands:",
    ]
    for run in child_runs:
        lines.extend([
            f"[{run['script']}] exit_code={run['exit_code']} elapsed_seconds={run['elapsed_seconds']}",
            f"command={run['command']}",
            run["stdout"] or "(no stdout)",
            ("stderr=" + run["stderr"]) if run["stderr"] else "stderr=(empty)",
            "",
        ])
    lines.extend([
        "Figure QA:",
        *[f"{row['figure']}: {row['qa']} | {row['chart_type']} | {row['interval']}" for row in figure_records],
        f"file_compliance={figure_check.get('status')} exit_code={figure_check.get('exit_code')}",
        f"file_compliance_command={figure_check.get('command', '')}",
        figure_check.get("stdout", ""),
        "effect_summary=q2_m0_effects_distribution_summary.csv; 8 tracks x 147 checkpoints; within-track quantiles are support coverage, not confidence intervals",
        "",
        "Evidence checks:",
        json.dumps(checks, ensure_ascii=False, indent=2),
        f"input_rows={len(input_rows)}",
        f"input_hash_match_rows={(input_rows['current_status'] == 'HASH_MATCH').sum()}",
        f"input_current_hash_only_rows={(input_rows['current_status'] == 'AVAILABLE_CURRENT_HASH_RECORDED').sum()}",
        f"input_changed_rows={input_rows['current_status'].astype(str).str.startswith('CHANGED').sum()}",
        f"input_missing_rows={input_rows['current_status'].astype(str).str.startswith('MISSING').sum()}",
        f"frozen_input_mismatches_or_missing={len(issues)}",
        *[f"input_issue={issue}" for issue in issues],
        "",
        "Q2_status=FROZEN_WITH_LIMITATIONS",
        "p_gate=FAIL_CURRENT_DATA; Q_gate=FAIL_CURRENT_EVIDENCE; M1=NOT_ESTIMATED; M2=NOT_ESTIMATED",
    ])
    return "\n".join(lines) + "\n"


def create_package() -> dict:
    child_runs = [
        run_child("src/q2_m0_effects_validation_closeout.py"),
        run_child("src/q2_build_evidence_closeout.py"),
    ]
    checks = validate_current_evidence()
    input_rows, issues = input_inventory()
    figure_records, figure_paths = render_effect_figures({})
    effect_summary, effect_summary_readme = build_effect_distribution_summary()
    figure_check = run_figure_compliance_check()
    # Regenerate inventory/notes after all post-processors have refreshed their manifests.
    input_rows, issues = input_inventory()
    readme = build_readme(checks, len(input_rows), issues)
    inputs_readme = build_inputs_readme(input_rows, issues)
    run_log = build_run_log(child_runs, checks, input_rows, issues, figure_records, figure_check)
    input_csv = input_rows.to_csv(index=False, encoding="utf-8-sig", lineterminator="\n").encode("utf-8-sig")
    contract = pd.DataFrame(figure_records).to_csv(index=False, encoding="utf-8-sig", lineterminator="\n").encode("utf-8-sig")
    common_contents = {
        "README_Q2_FINAL.md": readme.encode("utf-8"),
        "README_inputs.md": inputs_readme.encode("utf-8"),
        "inputs_manifest.csv": input_csv,
        "q2_m0_effects_figure_contract.csv": contract,
        "q2_m0_effects_distribution_summary.csv": effect_summary.to_csv(index=False, encoding="utf-8-sig", float_format="%.12g", lineterminator="\n").encode("utf-8-sig"),
        "q2_m0_effects_summary_readme.md": effect_summary_readme.encode("utf-8"),
        "run_log.txt": run_log.encode("utf-8"),
    }
    destinations = [RESULTS_CLOSEOUT, Q2_CLOSEOUT, DOCS]
    outputs: dict[str, str] = {}
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in common_contents.items():
            path = destination / name
            write_bytes(path, content)
            outputs[relative(path)] = sha256(path)
        # Capture the evidence tables/reports alongside the new package docs.
        for name in [*MIRRORED_CLOSEOUT_FILES[:-1], "q2_closeout_manifest.json"]:
            path = destination / name
            if path.is_file():
                outputs[relative(path)] = sha256(path)

    script_copy = Q2 / "02_代码" / Path(__file__).name
    write_bytes(script_copy, Path(__file__).read_bytes())
    outputs[relative(script_copy)] = sha256(script_copy)
    for path in figure_paths:
        outputs[relative(path)] = sha256(path)
    q2_readme = Q2 / "README.md"
    if q2_readme.is_file():
        outputs[relative(q2_readme)] = sha256(q2_readme)

    package_manifest = {
        "task": "Q2 current-data closeout package",
        "status": "FROZEN_WITH_LIMITATIONS",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": f'"{sys.executable}" src/q2_finalize_current_data_package.py',
        "script": relative(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "executable": sys.executable},
        "execution": [{k: v for k, v in run.items() if k not in {"stdout", "stderr"}} for run in child_runs],
        "figure_compliance_audit": {k: v for k, v in figure_check.items() if k in {"status", "exit_code", "command", "stderr"}},
        "checks": checks,
        "input_inventory": {
            "rows": int(len(input_rows)),
            "manifest_sha256": sha256(RESULTS_CLOSEOUT / "inputs_manifest.csv"),
            "matching_frozen_hash_rows": int((input_rows["current_status"] == "HASH_MATCH").sum()),
            "current_hash_without_historical_expected_hash_rows": int((input_rows["current_status"] == "AVAILABLE_CURRENT_HASH_RECORDED").sum()),
            "changed_rows": int(input_rows["current_status"].astype(str).str.startswith("CHANGED").sum()),
            "missing_rows": int(input_rows["current_status"].astype(str).str.startswith("MISSING").sum()),
            "issues": issues,
        },
        "figures": figure_records,
        "output_sha256_excluding_this_manifest": outputs,
        "manifest_note": "Mirrored to results/q2_closeout/v1, Q2/03_结果/综合收口/v1, and docs; self-hash excluded.",
        "reproduction_boundary": "This command verifies the frozen M0 outputs and regenerates only deterministic derived tables/reports/figures. It does not rerun the B1 baseline fit. Historical source files that are missing or changed are explicitly recorded in inputs_manifest.csv.",
    }
    manifest_bytes = (json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for destination in destinations:
        write_bytes(destination / "q2_final_package_manifest.json", manifest_bytes)

    print("Q2 closeout package generated from frozen M0 outputs; no model refit or bootstrap resampling was run.")
    print(f"Input inventory rows: {len(input_rows)}; current hash/missing issues: {len(issues)}.")
    print(f"M0 plots: {len(figure_records)} figure families; {len(figure_paths)} PNG/SVG/PDF/grayscale files across results and Q2.")
    print(f"Q2 status: p={checks['p_gate']}; Q={checks['Q_gate']}; M1/M2=NOT_ESTIMATED.")
    return package_manifest


def main() -> int:
    create_package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
