"""Generate three non-redundant Q2 evidence-complement figures.

Only frozen result tables and documented semi-synthetic B6/B7 fits are used.
No model is refit and no raw input is modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "Q2" / "README.md").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the HUAWEI-Cup project root.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
Q2_ROOT = PROJECT_ROOT / "Q2"
OUT_DIR = Q2_ROOT / "04_图表" / "q2_evidence_complements"
BASELINE = Q2_ROOT / "03_结果" / "经典ScalingLaw基线" / "v1"
SUPPLEMENT = Q2_ROOT / "03_结果" / "完整回答补充" / "v1"
ATTACHMENTS = PROJECT_ROOT / "中文题目" / "F题" / "real_attachments" / "B_scaling_laws"
SKILL_ROOT = Path(
    os.environ.get(
        "CODEX_MATH_MODELING_SKILL_ROOT",
        str(Path.home() / ".codex" / "skills" / "math-modeling"),
    )
)
INPUTS = {
    "B1 fitted checkpoints": BASELINE / "b1_fitted_predictions.csv",
    "B3 interpolated checkpoints": BASELINE / "b3_predictions.csv",
    "B3 trajectory closeout": BASELINE / "b3_validation_closeout.md",
    "B10 M0 predictions": SUPPLEMENT / "b9_b10_m0_extrapolation.csv",
    "B10 support summary": SUPPLEMENT / "b9_b10_extrapolation_summary.json",
    "B6/B7 Q response fits": SUPPLEMENT / "b6_b7_q_model_supplement.csv",
    "B6/B7 equal-Loss reference": SUPPLEMENT / "b6_b7_q_n_equal_loss_demo.csv",
    "B6 semi-synthetic source": ATTACHMENTS / "supplementary_NQ_experiment.csv",
    "B7 semi-synthetic source": ATTACHMENTS / "supplementary_NQ_experiment_expanded.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_figure_tools():
    scripts = SKILL_ROOT / "tools" / "figure" / "scripts"
    if not scripts.is_dir():
        raise FileNotFoundError(
            f"Figure skill helpers are missing at {scripts}. Set "
            "CODEX_MATH_MODELING_SKILL_ROOT to the installed skill root."
        )
    sys.path.insert(0, str(scripts))
    from setup_style import setup_style
    from export_figure import export_figure
    from visual_qa import audit_layout, render_preview
    return setup_style, export_figure, audit_layout, render_preview


def generate() -> dict[str, object]:
    for label, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input for {label}: {path}")

    setup_style, export_figure, audit_layout, render_preview = load_figure_tools()
    setup_style(
        journal="general", lang="zh", use_sciplots=False,
        serif_for_zh=False, constrained_layout=False,
    )
    from utils.plot_style import PALETTE

    plt.rcParams["axes.unicode_minus"] = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(tempfile.gettempdir()) / "q2_evidence_complement_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    contracts: list[dict[str, str]] = []
    qa_records: list[dict[str, object]] = []

    def save_figure(fig, figure_id, category, claim, evidence, warning, size):
        layout = audit_layout(fig)
        failures = [str(item) for item in layout if str(item).upper().startswith("FAIL")]
        if failures:
            plt.close(fig)
            raise RuntimeError(f"Figure layout QA failed for {figure_id}: {failures}")
        preview = preview_dir / f"{figure_id}_preview.png"
        render_preview(fig, str(preview), dpi=160)
        exports = export_figure(
            fig, str(OUT_DIR / figure_id), formats=["pdf", "svg", "png"],
            size_inches=size, dpi=300, grayscale_preview=True, tight=True,
        )
        plt.close(fig)
        contracts.append({
            "figure_id": figure_id, "category": category, "claim": claim,
            "evidence": evidence, "warning": warning,
        })
        qa_records.append({
            "figure_id": figure_id, "layout_issues": [str(item) for item in layout],
            "preview": str(preview), "exports": exports,
        })

    # 1. B3 validation at the trajectory level, rather than treating 4,000
    # interpolated checkpoints as 4,000 independent training runs.
    b3 = pd.read_csv(INPUTS["B3 interpolated checkpoints"])
    required = {"trajectory", "N_params_B", "observed_val_loss", "predicted_val_loss"}
    if not required.issubset(b3.columns):
        raise ValueError(f"B3 table lacks required fields: {sorted(required - set(b3.columns))}")
    b3["residual"] = b3["predicted_val_loss"] - b3["observed_val_loss"]
    by_track = (
        b3.groupby("trajectory", sort=False)
        .agg(
            N_params_B=("N_params_B", "median"), n=("residual", "size"),
            rmse=("residual", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            bias=("residual", "mean"),
        )
        .sort_values("N_params_B")
        .reset_index()
    )
    if len(by_track) != 8 or not (by_track["n"] == 500).all():
        raise ValueError("B3 must contain eight Pythia trajectories with 500 points each.")
    labels = [f"{value:.3f}B" for value in by_track["N_params_B"]]
    y = np.arange(len(by_track))
    fig, (ax_rmse, ax_bias) = plt.subplots(
        1, 2, figsize=(10.2, 5.1), sharey=True, gridspec_kw={"width_ratios": [1, 1]}
    )
    ax_rmse.scatter(by_track["rmse"], y, s=42, color=PALETTE["primary"],
                    edgecolor="white", linewidth=0.6, zorder=3)
    ax_rmse.set_xlim(0, max(0.0045, float(by_track["rmse"].max()) * 1.12))
    ax_rmse.set_xlabel("逐轨迹 RMSE（Loss）")
    ax_rmse.set_title("误差幅度")
    ax_rmse.grid(axis="x", color="#D9DEE5", linewidth=0.7)
    for x, yi in zip(by_track["rmse"], y):
        ax_rmse.annotate(f"{x:.4f}", (x, yi), xytext=(5, 0),
                         textcoords="offset points", va="center", fontsize=7.2)
    ax_bias.scatter(by_track["bias"], y, s=42, color=PALETTE["contrast"],
                    edgecolor="white", linewidth=0.6, zorder=3)
    ax_bias.axvline(0, color=PALETTE["dark"], linewidth=0.9, linestyle="--")
    ax_bias.set_xlim(min(-0.0025, float(by_track["bias"].min()) * 1.12),
                     max(0.0003, float(by_track["bias"].max()) * 0.7))
    ax_bias.set_xlabel("逐轨迹偏差：预测 - 观测（Loss）")
    ax_bias.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.4f}".replace("-0.0000", "0.0000"))
    )
    ax_bias.set_title("系统偏差")
    ax_bias.grid(axis="x", color="#D9DEE5", linewidth=0.7)
    for x, yi in zip(by_track["bias"], y):
        ax_bias.annotate(f"{x:.4f}", (x, yi), xytext=(5, 0),
                         textcoords="offset points", va="center", fontsize=7.2)
    ax_bias.set_yticks(y, labels)
    ax_bias.invert_yaxis()
    ax_bias.set_ylabel("Pythia 轨迹参数规模 N")
    fig.suptitle("B3 同来源插值检查：按 8 条轨迹展示误差", fontsize=12, fontweight="bold")
    fig.text(
        0.5, 0.015,
        "每行是一条完整 Pythia 参数轨迹（每条 500 个插值点）；点为轨迹级描述统计，不是独立训练重复。",
        ha="center", va="bottom", fontsize=8, color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.105, 0.99, 0.92))
    save_figure(
        fig, "result_q2_b3_track_level_errors", "result",
        "B3 总体指标拆分到 8 条 Pythia 参数轨迹后，轨迹间 RMSE 和系统偏差相近。",
        "冻结 B1 参数生成的 b3_predictions.csv；按 trajectory 汇总 8 组，每组 500 个插值点。",
        "B3 是 Pythia 同来源插值检查；500 点/轨迹不是独立训练实验，图不提供总体推断区间。",
        (10.2, 5.1),
    )

    # 2. B10 estimated points relative to the actual B1 empirical N/D support.
    b1 = pd.read_csv(INPUTS["B1 fitted checkpoints"])
    b10 = pd.read_csv(INPUTS["B10 M0 predictions"])
    summary = json.loads(INPUTS["B10 support summary"].read_text(encoding="utf-8"))
    required_b10 = {"N_params_B", "D_tokens_B", "M0_predicted_loss"}
    if not required_b10.issubset(b10.columns):
        raise ValueError("B10 table lacks N, D or frozen M0 predictions.")
    if len(b10) != 128 or not np.isfinite(
        b10[["N_params_B", "D_tokens_B", "M0_predicted_loss"]].to_numpy(float)
    ).all():
        raise ValueError("B10 must contain 128 finite N/D/M0 prediction rows.")
    n_max, d_max = float(b1["N_params_B"].max()), float(b1["D_tokens_B"].max())
    n_out = int((b10["N_params_B"] > n_max).sum())
    d_out = int((b10["D_tokens_B"] > d_max).sum())
    if n_out != int(summary["n_N_above_B1_support"]) or d_out != int(summary["n_D_above_B1_support"]):
        raise ValueError("B10 support counts disagree with the frozen summary.")
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.scatter(
        b1["N_params_B"], b1["D_tokens_B"], s=17, color=PALETTE["primary"],
        alpha=0.23, linewidth=0, label=f"B1 训练检查点（n={len(b1):,}）",
        rasterized=True, zorder=1,
    )
    points = ax.scatter(
        b10["N_params_B"], b10["D_tokens_B"], c=b10["M0_predicted_loss"],
        cmap="cividis", s=34, alpha=0.82, edgecolor="white", linewidth=0.35,
        label=f"B10 估算表（n={len(b10)}）", zorder=3,
    )
    ax.axvline(n_max, color=PALETTE["dark"], linewidth=1.1, linestyle="--")
    ax.axhline(d_max, color=PALETTE["dark"], linewidth=1.1, linestyle="--")
    ax.text(n_max * 1.07, 0.12, f"B1 N 上界\n{n_max:.2f}B",
            rotation=90, ha="left", va="bottom", fontsize=7.5, color="#374151")
    ax.text(0.075, d_max * 1.12, f"B1 D 上界 {d_max:.1f}B",
            ha="left", va="bottom", fontsize=7.5, color="#374151")
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_label = FuncFormatter(
        lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:g}"
    )
    ax.xaxis.set_major_formatter(log_label)
    ax.yaxis.set_major_formatter(log_label)
    ax.set_xlim(0.05, max(15000.0, float(b10["N_params_B"].max()) * 1.18))
    ax.set_ylim(0.08, max(50000.0, float(b10["D_tokens_B"].max()) * 1.2))
    ax.set_xlabel("参数规模 N（十亿参数，log）")
    ax.set_ylabel("训练数据量 D（十亿 token，log）")
    ax.set_title("B10 大尺度估算点相对 B1 经验支持域的位置", fontsize=12, fontweight="bold")
    ax.grid(which="major", color="#D9DEE5", linewidth=0.7, alpha=0.8)
    ax.grid(which="minor", color="#E9EDF1", linewidth=0.45, alpha=0.65)
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.065),
               frameon=True, framealpha=0.95, fontsize=8)
    cbar = fig.colorbar(points, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("冻结 M0 预测 Loss（颜色仅表示预测值）")
    ax.text(
        0.02, 0.98,
        f"{n_out}/{len(b10)} 个 B10 点的 N 超出 B1 上界\n"
        f"{d_out}/{len(b10)} 个 B10 点的 D 超出 B1 上界",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": "#9CA3AF", "alpha": 0.94},
    )
    fig.text(
        0.5, 0.012,
        "B10 是估算/整理表，不是独立验证；点色为 M0 预测，不是观测 Loss。参数 Bootstrap 区间不覆盖外推与协议误差。",
        ha="center", va="bottom", fontsize=7.7, color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.17, 0.98, 0.96))
    save_figure(
        fig, "raw_q2_b10_extrapolation_support_map", "raw",
        "B10 的 128 条大尺度估算均超出 B1 的 N 上界，其中 102 条也超出 D 上界。",
        "B1 冻结拟合检查点 N/D 坐标、B10 M0 预测表及保存的支持域摘要。",
        "B10 是估算/整理表；位置图说明外推范围，不代表预测已验证。参数 Bootstrap 不含外推模型形式误差。",
        (8.6, 6.0),
    )

    # 3. Conditional equal-Loss curves on the semi-synthetic B6/B7 fits.
    fits = pd.read_csv(INPUTS["B6/B7 Q response fits"])
    demo = pd.read_csv(INPUTS["B6/B7 equal-Loss reference"]).set_index("dataset")
    b6 = pd.read_csv(INPUTS["B6 semi-synthetic source"])
    b7 = pd.read_csv(INPUTS["B7 semi-synthetic source"])
    q_levels = {
        "B6": set(b6["Q_score"].dropna().astype(float).unique()),
        "B7": set(b7["Q_score"].dropna().astype(float).unique()),
    }
    q0 = float(demo.loc["B6", "reference_Q_score"])
    n0 = float(demo.loc["B6", "reference_N_B"])
    d0 = float(demo.loc["B6", "reference_D_B"])
    if not all(
        np.isclose(demo.loc["B7", key], demo.loc["B6", key])
        for key in ("reference_Q_score", "reference_N_B", "reference_D_B")
    ):
        raise ValueError("B6/B7 equal-Loss reference points are no longer shared.")
    q_after = q0 + 0.1
    q_grid = np.linspace(q0, 1.0, 240)
    colors = {"B6": PALETTE["primary"], "B7": PALETTE["contrast"]}
    line_styles = {"B6": "-", "B7": "--"}
    fig, ax = plt.subplots(figsize=(8.7, 5.6))
    for dataset in ("B6", "B7"):
        row = fits.loc[(fits["dataset"] == dataset) & (fits["model"] == "M_Q_NDQ")]
        if len(row) != 1:
            raise ValueError(f"Expected one frozen M_Q_NDQ fit for {dataset}.")
        fit = row.iloc[0]
        E, A, alpha, B, beta, kappa = (
            float(fit[name])
            for name in ("E", "A", "alpha", "B", "beta", "kappa_Q_exponent")
        )
        ref = demo.loc[dataset]
        target_loss = float(ref["reference_fitted_Loss"])
        q_ref, n_ref, d_ref = (
            float(ref[name]) for name in ("reference_Q_score", "reference_N_B", "reference_D_B")
        )
        denominator = target_loss - E - B * d_ref ** (-beta) * q_grid ** (-kappa)
        if np.any(denominator <= 0):
            raise ValueError(f"{dataset} equal-Loss curve leaves the real solution domain.")
        n_required = (A / denominator) ** (1.0 / alpha)
        ax.plot(q_grid, n_required, color=colors[dataset], linestyle=line_styles[dataset],
                linewidth=2, label=f"{dataset} 半合成拟合曲面")
        ax.scatter(
            [q_after], [float(ref["N_equal_Loss_after_B"])], s=55,
            facecolor="white" if q_after not in q_levels[dataset] else colors[dataset],
            edgecolor=colors[dataset], linewidth=1.7,
            zorder=5 if dataset == "B6" else 4,
        )
        ax.scatter([q_ref], [n_ref], s=38, facecolor=colors[dataset],
                   edgecolor="white", linewidth=0.7, zorder=4)
    ax.axvline(q0, color="#6B7280", linewidth=0.9, linestyle=":")
    ax.axvline(q_after, color="#6B7280", linewidth=0.9, linestyle=":")
    ax.set_xlim(q0 - 0.015, 1.01)
    ax.set_ylim(0.61, 1.055)
    ax.set_xticks(np.arange(0.6, 1.01, 0.1))
    ax.set_xlabel("来源表中的 Q_score（拟合条件变量）")
    ax.set_ylabel("保持各自参考 Loss 所需的 N（十亿参数）")
    fig.suptitle("B6/B7 半合成拟合曲面上的条件等 Loss 关系",
                 y=0.975, fontsize=12, fontweight="bold")
    ax.grid(color="#D9DEE5", linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    note = (
        f"固定 D={d0:g}B，目标 Loss 取各自 N={n0:g}B、Q={q0:g} 的拟合值；"
        f"Q={q_after:.1f} 时 ΔN：B6 {demo.loc['B6', 'delta_N_B']:.3f}B（插值），"
        f"B7 {demo.loc['B7', 'delta_N_B']:.3f}B。"
    )
    fig.text(0.5, 0.925, note, ha="center", va="top", fontsize=7.8, color="#374151")
    fig.text(
        0.5, 0.012,
        "仅为 B6/B7 半合成拟合曲面的数学示例；B6 与 B7 并非独立复制，不能解释为真实训练资源替代率或 Q1 q_huber 效应。",
        ha="center", va="bottom", fontsize=7.4, color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.87))
    save_figure(
        fig, "result_q2_b6_b7_conditional_equal_loss", "result",
        "在 B6/B7 半合成拟合曲面、固定 D 和目标 Loss 的条件下，拟合模型给出 Q 提高时所需 N 的变化。",
        "已保存的 B6/B7 M_Q_NDQ 系数、共同等 Loss 参考点及各来源 Q_score 水平。",
        "半合成条件计算而非真实训练资源替代率；B6 的 Q=0.7 点为拟合插值，B6/B7 不能作为独立复制。",
        (8.7, 5.6),
    )

    contract_path = OUT_DIR / "figure_contract.csv"
    pd.DataFrame(contracts).to_csv(contract_path, index=False, encoding="utf-8-sig")
    qa_path = OUT_DIR / "figure_qa.json"
    qa_path.write_text(json.dumps({"figures": qa_records}, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(
        "# Q2 补充图：未覆盖证据视角\n\n"
        "本组补充三个现有图表尚未直接展示的视角，不增加新模型结论：\n\n"
        "1. result_q2_b3_track_level_errors：将 B3 总计 4,000 个插值点拆到 8 条 Pythia 轨迹展示，避免把检查点数当作独立实验数。\n"
        "2. raw_q2_b10_extrapolation_support_map：显示 B10 估算点相对 B1 经验 N/D 支持点的位置。它展示外推域，不是外部验证。\n"
        "3. result_q2_b6_b7_conditional_equal_loss：展示报告中的 B6/B7 半合成等 Loss 条件曲线；B6 的 Q=0.7 点为拟合插值。\n\n"
        "每图导出 PDF、SVG、PNG 和灰度 PNG。合同、布局 QA、哈希及复现命令见 figure_contract.csv、figure_qa.json、manifest.json。\n\n"
        "复现命令（项目根目录）：\n\n"
        "    D:\\\\Anaconda\\\\python.exe src/q2_evidence_complement_figures.py\n\n"
        "本组只消费已冻结结构化结果与可见审计材料；不读取隐藏文本，不拼接 A/B/半合成来源为真实联合样本。\n",
        encoding="utf-8",
    )
    input_hashes = {
        label: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)}
        for label, path in INPUTS.items()
    }
    output_files = sorted(
        path for path in OUT_DIR.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "status": "GENERATED_WITH_LIMITATIONS",
        "generated_utc_date": "2026-09-25",
        "reproduction_command": "D:\\Anaconda\\python.exe src/q2_evidence_complement_figures.py",
        "script": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "inputs": input_hashes,
        "outputs": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in output_files},
        "scope": [
            "No M0 refit, M1/M2 fit, new bootstrap, or raw data edit.",
            "B3 is a same-source interpolation check summarized by eight trajectories.",
            "B10 is an estimated/curated table outside B1 support, not external validation.",
            "B6/B7 equal-Loss curves are conditional semi-synthetic fit illustrations only.",
        ],
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_dir": str(OUT_DIR),
        "figures": [row["figure_id"] for row in contracts],
        "contract": str(contract_path),
        "qa": str(qa_path),
        "manifest": str(manifest_path),
    }


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
