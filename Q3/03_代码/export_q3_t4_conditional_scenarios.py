#!/usr/bin/env python3
"""Export task-card Q3-T4 tables and a conditional beta-Q decision map.

This is a derivative export only. It does not refit M0, estimate beta_Q, or
rerun the optimization; it normalizes the frozen Q3-T4 summary and envelope.

AI-assisted programming note (2026-09-25): OpenAI Codex, GPT-6, assisted with
export organization and plotting. The participating team must inspect and
disclose AI assistance under the current competition rules.

Run from the repository root:
    python Q3/03_代码/export_q3_t4_conditional_scenarios.py
Use --tables-only for the table-stage handoff before formal plotting.
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
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
from q3_repro_support import resolve_math_modeling_skill_root

SKILL_ROOT = resolve_math_modeling_skill_root()
FIGURE_SCRIPTS = SKILL_ROOT / "tools" / "figure" / "scripts"
if str(FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_style import choose_font  # noqa: E402


SUMMARY = Path("Q3/04_结果/Q_conditional_Q_break_even_summary.csv")
REGIMES = Path("Q3/04_结果/Q_conditional_Q_optimality_intervals.csv")
UPSTREAM_MANIFEST = Path("Q3/04_结果/Q3_conditional_Q_optimization_manifest.json")
THRESHOLD_OUT = Path("Q3/04_结果/q3_betaQ_threshold.csv")
REGIME_OUT = Path("Q3/04_结果/q3_conditional_regime.csv")
MANIFEST_OUT = Path("Q3/04_结果/q3_t4_conditional_beta_manifest.json")
FIGURE_STEM = Path("Q3/04_结果/figures/result_q3_t4_betaQ_scenario")
PREVIEW = Path("Q3/04_结果/_figure_previews/q3_t4_betaQ_scenario_preview.png")
FIGURE_SIZE = (7.2, 6.3)
DPI = 300
EXPECTED_SCENARIOS = 45
EXPECTED_REGIMES = 166
G_ORDER = ("exponential", "power4", "logarithmic")

THRESHOLD_FIELDS = [
    "scenario_id",
    "budget_flops",
    "context_length_tokens",
    "g_function",
    "q0_reference",
    "betaQ_identified",
    "betaQ_status",
    "betaQ_first_positive_Q",
    "betaQcrit_first_ND_reallocation",
    "betaQ_units",
    "fixed_Q0_baseline_run_id",
    "fixed_Q0_baseline_loss",
    "first_positive_Q_run_id",
    "first_positive_Q_N_B",
    "first_positive_Q_D_B",
    "first_positive_Q_level",
    "first_reallocated_run_id",
    "first_reallocated_N_B",
    "first_reallocated_D_B",
    "first_reallocated_Q",
    "conditional_envelope_segments",
    "positive_Q_segments_on_envelope",
    "B1_feasible_ND_points",
    "model_scope",
    "conditional_assumption",
]

REGIME_SOURCE_FIELDS = [
    "budget_flops",
    "context_length_tokens",
    "g_function",
    "beta_Q_segment",
    "beta_Q_lower",
    "beta_Q_lower_inclusive",
    "beta_Q_upper",
    "beta_Q_upper_exclusive",
    "beta_Q_units",
    "selected_run_id",
    "selected_N_B",
    "selected_D_B",
    "selected_Q",
    "Q_increment_from_Q0",
    "M0_base_loss",
    "conditional_loss_at_segment_start",
    "C_train_flops",
    "C_Q_flops",
    "C_attn_flops",
    "C_total_flops",
    "budget_utilization",
    "is_fixed_Q0_baseline",
    "p_setting",
    "conditional_assumption",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    with absolute.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scenario_key(row: dict[str, str]) -> tuple[int, int, str]:
    return int(row["budget_flops"]), int(row["context_length_tokens"]), row["g_function"]


def scenario_id(key: tuple[int, int, str]) -> str:
    budget, context, g_function = key
    budget_label = f"{budget:.0e}".replace("+", "")
    return f"B{budget_label}_C{context}_{g_function}"


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite numeric value, got {value!r}")
    return result


def is_infinite_upper(value: str) -> bool:
    return value.strip().lower() in {"infinity", "+infinity", "inf", "+inf"}


def validate_upstream(
    summary_rows: list[dict[str, str]], regime_rows: list[dict[str, str]], manifest: dict[str, Any]
) -> None:
    expected_hashes = manifest.get("outputs", {})
    for relative in (SUMMARY, REGIMES):
        expected = expected_hashes.get(relative.as_posix())
        actual = sha256(ROOT / relative)
        if expected != actual:
            raise ValueError(f"Upstream manifest hash mismatch for {relative}: {actual}")

    if len(summary_rows) != EXPECTED_SCENARIOS or len(regime_rows) != EXPECTED_REGIMES:
        raise ValueError(
            f"Unexpected upstream sizes: {len(summary_rows)} summary rows, "
            f"{len(regime_rows)} regime rows"
        )
    if not summary_rows or not regime_rows:
        raise ValueError("Upstream conditional Q outputs are empty")

    summary_by_key = {scenario_key(row): row for row in summary_rows}
    if len(summary_by_key) != len(summary_rows):
        raise ValueError("Duplicate scenario in the threshold summary")
    expected_keys = {
        (budget, context, g_function)
        for budget in sorted({key[0] for key in summary_by_key})
        for context in sorted({key[1] for key in summary_by_key})
        for g_function in G_ORDER
    }
    if set(summary_by_key) != expected_keys or len(expected_keys) != EXPECTED_SCENARIOS:
        raise ValueError("The expected budget × context × g(Q) scenario grid is incomplete")

    regime_groups: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in regime_rows:
        regime_groups[scenario_key(row)].append(row)
    if set(regime_groups) != set(summary_by_key):
        raise ValueError("Threshold and optimality-interval scenario keys do not match")

    for key, rows in regime_groups.items():
        rows.sort(key=lambda row: int(row["beta_Q_segment"]))
        summary = summary_by_key[key]
        baseline_id = int(summary["fixed_Q0_baseline_run_id"])
        if len(rows) != int(summary["conditional_envelope_segments"]):
            raise ValueError(f"Envelope segment count mismatch for {scenario_id(key)}")
        if int(rows[0]["beta_Q_segment"]) != 1 or finite_float(rows[0]["beta_Q_lower"]) != 0.0:
            raise ValueError(f"The first beta_Q interval does not begin at 0 for {scenario_id(key)}")

        previous_upper: float | None = None
        first_different: dict[str, str] | None = None
        for index, row in enumerate(rows):
            if int(row["beta_Q_segment"]) != index + 1:
                raise ValueError(f"Nonconsecutive interval numbering for {scenario_id(key)}")
            lower = finite_float(row["beta_Q_lower"])
            if index and previous_upper != lower:
                raise ValueError(f"Gap or overlap in beta_Q intervals for {scenario_id(key)}")
            upper_text = row["beta_Q_upper"].strip()
            if is_infinite_upper(upper_text):
                if index != len(rows) - 1:
                    raise ValueError(f"Infinite interval is not last for {scenario_id(key)}")
                previous_upper = math.inf
            else:
                upper = finite_float(upper_text)
                if upper <= lower:
                    raise ValueError(f"Nonpositive beta_Q interval width for {scenario_id(key)}")
                previous_upper = upper
            if first_different is None and int(row["selected_run_id"]) != baseline_id:
                first_different = row

        if not is_infinite_upper(rows[-1]["beta_Q_upper"]):
            raise ValueError(f"The last beta_Q interval is not open-ended for {scenario_id(key)}")

        summary_threshold = summary["beta_Q_first_ND_reallocation"].strip()
        if first_different is None:
            if summary_threshold:
                raise ValueError(f"Unexpected finite N/D threshold for {scenario_id(key)}")
        else:
            if not summary_threshold:
                raise ValueError(f"Missing N/D threshold for {scenario_id(key)}")
            if not math.isclose(
                finite_float(summary_threshold),
                finite_float(first_different["beta_Q_lower"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(f"N/D threshold does not match the first changed regime for {scenario_id(key)}")
            if int(summary["first_reallocated_run_id"]) != int(first_different["selected_run_id"]):
                raise ValueError(f"First changed N/D decision does not match for {scenario_id(key)}")


def build_tables(
    summary_rows: list[dict[str, str]], regime_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sort_key = lambda row: (*scenario_key(row)[:2], G_ORDER.index(row["g_function"]))
    thresholds: list[dict[str, Any]] = []
    for source in sorted(summary_rows, key=sort_key):
        key = scenario_key(source)
        finite = source["beta_Q_first_ND_reallocation"].strip() != ""
        row: dict[str, Any] = {field: "" for field in THRESHOLD_FIELDS}
        row.update(
            {
                "scenario_id": scenario_id(key),
                "budget_flops": source["budget_flops"],
                "context_length_tokens": source["context_length_tokens"],
                "g_function": source["g_function"],
                "q0_reference": source["q0_reference"],
                "betaQ_identified": "false",
                "betaQ_status": "finite_first_ND_reallocation" if finite else "no_finite_ND_reallocation",
                "betaQ_first_positive_Q": source["beta_Q_first_positive_Q"],
                "betaQcrit_first_ND_reallocation": source["beta_Q_first_ND_reallocation"],
                "betaQ_units": "validation-loss units per one Q-score unit",
                "fixed_Q0_baseline_run_id": source["fixed_Q0_baseline_run_id"],
                "fixed_Q0_baseline_loss": source["fixed_Q0_baseline_loss"],
                "first_positive_Q_run_id": source["first_positive_Q_run_id"],
                "first_positive_Q_N_B": source["first_positive_Q_N_B"],
                "first_positive_Q_D_B": source["first_positive_Q_D_B"],
                "first_positive_Q_level": source["first_positive_Q_level"],
                "first_reallocated_run_id": source["first_reallocated_run_id"],
                "first_reallocated_N_B": source["first_reallocated_N_B"],
                "first_reallocated_D_B": source["first_reallocated_D_B"],
                "first_reallocated_Q": source["first_reallocated_Q"],
                "conditional_envelope_segments": source["conditional_envelope_segments"],
                "positive_Q_segments_on_envelope": source["positive_Q_segments_on_envelope"],
                "B1_feasible_ND_points": source["B1_feasible_ND_points"],
                "model_scope": source["model_scope"],
                "conditional_assumption": "L_cond = L_M0(N,D) - beta_Q*(Q-Q0), beta_Q >= 0",
            }
        )
        thresholds.append(row)

    regimes: list[dict[str, Any]] = []
    for source in sorted(regime_rows, key=sort_key):
        key = scenario_key(source)
        row: dict[str, Any] = {
            "scenario_id": scenario_id(key),
            "q0_reference": "0.49847810484764643",
            "betaQ_identified": "false",
        }
        row.update({field: source[field] for field in REGIME_SOURCE_FIELDS})
        regimes.append(row)
    return thresholds, regimes


def setup_figure_style() -> None:
    from setup_style import setup_style

    setup_style(journal="general", lang="zh", use_sciplots=False)
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [choose_font("zh"), "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.grid"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"


def draw_figure(thresholds: list[dict[str, Any]]) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors, patches
    from matplotlib.patheffects import withStroke
    from PIL import Image
    from export_figure import export_figure
    from visual_qa import audit_layout, print_report, render_preview

    setup_figure_style()
    lookup = {
        (int(row["budget_flops"]), int(row["context_length_tokens"]), row["g_function"]): row
        for row in thresholds
    }
    budgets = sorted({int(row["budget_flops"]) for row in thresholds})
    contexts = sorted({int(row["context_length_tokens"]) for row in thresholds})
    finite_values = [
        float(row["betaQcrit_first_ND_reallocation"])
        for row in thresholds
        if row["betaQcrit_first_ND_reallocation"] != ""
    ]
    if len(finite_values) != 30:
        raise ValueError(f"Expected 30 finite thresholds, received {len(finite_values)}")

    log_values = np.log10(np.asarray(finite_values, dtype=float))
    norm = colors.Normalize(vmin=float(log_values.min()), vmax=float(log_values.max()))
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#e5e7eb")
    g_labels = {"exponential": "指数", "power4": "四次幂", "logarithmic": "对数"}
    ctx_labels = {2048: "2,048", 4096: "4,096", 8192: "8,192", 32768: "32,768", 131072: "131,072"}

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=FIGURE_SIZE,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )
    images = []
    for axis, budget in zip(axes, budgets):
        matrix = np.full((len(contexts), len(G_ORDER)), np.nan, dtype=float)
        panel_rows: dict[tuple[int, int], dict[str, Any]] = {}
        for row_index, context in enumerate(contexts):
            for col_index, g_function in enumerate(G_ORDER):
                row = lookup[(budget, context, g_function)]
                panel_rows[(row_index, col_index)] = row
                value = row["betaQcrit_first_ND_reallocation"]
                if value != "":
                    matrix[row_index, col_index] = math.log10(float(value))

        image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        images.append(image)
        axis.set_title(f"预算 $10^{{{int(math.log10(budget))}}}$ FLOPs", loc="left", pad=3, fontsize=9)
        axis.set_yticks(range(len(contexts)), [ctx_labels.get(value, f"{value:,}") for value in contexts])
        axis.set_ylabel("上下文长度（tokens）", fontsize=8)
        axis.tick_params(axis="both", length=0, labelsize=7.5)
        axis.set_xticks(range(len(G_ORDER)), [g_labels[g] for g in G_ORDER])
        axis.set_xlim(-0.5, len(G_ORDER) - 0.5)
        axis.set_ylim(len(contexts) - 0.5, -0.5)
        axis.set_xticks(np.arange(-0.5, len(G_ORDER), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(contexts), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.1)
        axis.tick_params(which="minor", bottom=False, left=False)
        for (row_index, col_index), row in panel_rows.items():
            value = row["betaQcrit_first_ND_reallocation"]
            if value == "":
                rectangle = patches.Rectangle(
                    (col_index - 0.5, row_index - 0.5), 1, 1,
                    facecolor="#e5e7eb", edgecolor="#a1a1aa", linewidth=0.35, hatch="///",
                    zorder=2,
                )
                axis.add_patch(rectangle)
                label = "无重分配"
                color = "#30343b"
            else:
                label = f"{float(value):.3g}"
                color = "white" if norm(math.log10(float(value))) < 0.52 else "#151515"
            axis.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.1 if value != "" else 6.1,
                color=color,
                path_effects=[withStroke(linewidth=1.0, foreground="black", alpha=0.16)]
                if value != ""
                else None,
                zorder=3,
            )
        for spine in axis.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("质量成本函数 $g(Q)$", fontsize=8.5, labelpad=5)
    fig.suptitle(
        "条件 N/D 重分配的 $\\beta_Q$ 临界值",
        fontsize=10.5,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.962,
        r"$\beta_Q$ 未由数据识别；条件断点仅适用于冻结的 Q2 B1/M0 支持域",
        ha="center",
        va="top",
        fontsize=6.5,
        color="#444444",
    )
    colorbar = fig.colorbar(images[-1], ax=list(axes), location="right", shrink=0.9, pad=0.025)
    tick_values = sorted({min(finite_values), 0.01, 0.03, 0.1, 0.3, max(finite_values)})
    tick_values = [value for value in tick_values if min(finite_values) <= value <= max(finite_values)]
    colorbar.set_ticks([math.log10(value) for value in tick_values])
    colorbar.set_ticklabels([f"{value:.2g}" for value in tick_values])
    colorbar.set_label("$\\beta_{Q,crit}$（对数色阶；交叉熵损失单位 / Q 单位）", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    PREVIEW_PATH = ROOT / PREVIEW
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    render_preview(fig, str(PREVIEW_PATH), dpi=150)
    print_report(audit_layout(fig))

    stem = ROOT / FIGURE_STEM
    stem.parent.mkdir(parents=True, exist_ok=True)
    exported = export_figure(
        fig,
        basename=str(stem),
        formats=["pdf", "svg", "png"],
        size_inches=FIGURE_SIZE,
        dpi=DPI,
        grayscale_preview=False,
        tight=False,
    )
    color_png = stem.with_suffix(".png")
    gray_png = stem.with_name(stem.name + "_grayscale.png")
    Image.open(color_png).convert("L").save(gray_png)
    exported.append(str(gray_png))
    plt.close(fig)
    return [Path(path) for path in exported] + [PREVIEW]


def write_manifest(phase: str, outputs: list[Path]) -> None:
    script = Path(__file__).resolve()
    upstream = ROOT / UPSTREAM_MANIFEST
    manifest = {
        "artifact": "Q3-T4 conditional beta_Q threshold and regime export",
        "phase": phase,
        "conditional_response": "L_cond = L_M0(N,D) - beta_Q*(Q-Q0), beta_Q >= 0",
        "beta_Q_identified": False,
        "beta_Q_interpretation": "conditional break-even coefficient, not estimated from Q2",
        "p_setting": "excluded from the objective and decision variables",
        "q0_reference": 0.49847810484764643,
        "scenario_count": EXPECTED_SCENARIOS,
        "conditional_regime_count": EXPECTED_REGIMES,
        "finite_first_ND_reallocation_threshold_count": 30,
        "no_finite_first_ND_reallocation_count": 15,
        "figure_contract": {
            "core_claim": "The first conditional N/D reallocation threshold varies across budget, context and g(Q); some B1-supported scenarios have no finite reallocation boundary.",
            "figure": FIGURE_STEM.as_posix(),
            "encoding": "Three budget panels; rows are context lengths; columns are g(Q); finite beta_Q thresholds use a log10 cividis scale; no finite threshold is shown as a hatched gray cell.",
            "annotation": "Subtitle states beta_Q is unidentified and results are limited to the frozen Q2 B1/M0 support domain.",
            "size_inches": list(FIGURE_SIZE),
            "dpi": DPI,
        },
        "run": f"{Path(__file__).resolve().relative_to(ROOT).as_posix()}",
        "script": {
            "path": script.relative_to(ROOT).as_posix(),
            "sha256": sha256(script),
        },
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "reproduction_command": f'& "{Path(sys.executable).resolve().as_posix()}" "Q3/03_代码/export_q3_t4_conditional_scenarios.py"',
        "code_dependencies_sha256": {
            "Q3/03_代码/q3_repro_support.py": sha256(ROOT / "Q3/03_代码/q3_repro_support.py"),
            "utils/plot_style.py": sha256(ROOT / "utils/plot_style.py"),
        },
        "figure_tools": {
            "skill_root": str(SKILL_ROOT),
            "files_sha256": {
                path.relative_to(SKILL_ROOT).as_posix(): sha256(path)
                for path in sorted(FIGURE_SCRIPTS.glob("*.py"))
            },
        },
        "packages": {},
        "inputs": {
            SUMMARY.as_posix(): sha256(ROOT / SUMMARY),
            REGIMES.as_posix(): sha256(ROOT / REGIMES),
            UPSTREAM_MANIFEST.as_posix(): sha256(upstream),
        },
        "upstream_model_manifest": json.loads(upstream.read_text(encoding="utf-8")),
        "outputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in outputs if path.is_file()},
        "claim_boundary": "Results are conditional on a linear Q-to-Loss response, frozen Q1 Q0, frozen Q2 M0 and the B1 observed support grid; not an empirical Q effect or full Q/p optimum.",
    }
    try:
        import matplotlib
        import numpy
        import pandas
        from PIL import __version__ as pillow_version

        manifest["packages"] = {
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "pillow": pillow_version,
        }
    except ImportError:
        manifest["packages"] = {"note": "Figure dependencies are loaded only in full-figure mode."}
    manifest_path = ROOT / MANIFEST_OUT
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {MANIFEST_OUT.as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Write the two standardized CSVs without generating the formal figure.",
    )
    args = parser.parse_args()

    summary_rows = read_csv(SUMMARY)
    regime_rows = read_csv(REGIMES)
    upstream_manifest = json.loads((ROOT / UPSTREAM_MANIFEST).read_text(encoding="utf-8"))
    validate_upstream(summary_rows, regime_rows, upstream_manifest)
    threshold_rows, conditional_regime_rows = build_tables(summary_rows, regime_rows)
    write_csv(THRESHOLD_OUT, threshold_rows)
    write_csv(REGIME_OUT, conditional_regime_rows)
    print(f"Wrote {len(threshold_rows)} scenario threshold rows to {THRESHOLD_OUT.as_posix()}")
    print(f"Wrote {len(conditional_regime_rows)} piecewise decision regions to {REGIME_OUT.as_posix()}")

    outputs = [ROOT / THRESHOLD_OUT, ROOT / REGIME_OUT]
    if args.tables_only:
        write_manifest("tables_ready", outputs)
        return 0

    figure_outputs = draw_figure(threshold_rows)
    outputs.extend(ROOT / path if not path.is_absolute() else path for path in figure_outputs)
    write_manifest("complete", outputs)
    print("The beta_Q values remain conditional break-even thresholds; none was estimated from Q2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
