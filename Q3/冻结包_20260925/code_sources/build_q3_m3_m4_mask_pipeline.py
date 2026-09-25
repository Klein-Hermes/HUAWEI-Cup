#!/usr/bin/env python3
"""Build separated Q3 M3 support and M4 budget masks.

Current default domain is the frozen B1/M0 observed N/D grid. Convex-hull
membership and nearest-observation distance are diagnostics only. Under this
default contract, only an exact observed N/D pair is allowed into M1's search.

Run from the repository root:
    python Q3/03_代码/build_q3_m3_m4_mask_pipeline.py

An optional candidate CSV can be supplied with --candidate-grid. It must have
grid_id, N_params_B, and D_tokens_B columns. Unsupported rows remain excluded
from M1 unless the support policy is deliberately revised with evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
Q2_GRID = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q2_PARAMS = Path("Q2/03_结果/经典ScalingLaw基线/v1/fit_parameters.json")
C7_INPUT = Path(
    "中文题目/F题/real_attachments/C_efficiency_evolution/"
    "model_architecture_metadata.csv"
)

OUT_OBSERVED = Path("Q3/04_结果/m3_observed_support.csv")
OUT_CANDIDATES = Path("Q3/04_结果/m3_candidate_grid.csv")
OUT_SUPPORT = Path("Q3/04_结果/m3_q3_support_grid.csv")
OUT_BUDGET_MASK = Path("Q3/04_结果/m4_budget_feasible_mask.csv")
OUT_M3_REPORT = Path("Q3/04_结果/M3_support_domain_audit.md")
OUT_M3_MANIFEST = Path("Q3/04_结果/M3_support_manifest.json")
OUT_M4_REPORT = Path("Q3/04_结果/M4_budget_feasibility_audit.md")
OUT_M4_MANIFEST = Path("Q3/04_结果/M4_budget_manifest.json")

ETA = Fraction(1, 5000)
TRAINING_COEFFICIENT = 6
UNIT_SCALE = 10**9
BUDGETS_FLOPS = (10**19, 10**22, 10**24)
EXACT_MATCH_REL_TOL = 1e-12
EXACT_MATCH_ABS_TOL = 1e-12
GEOMETRY_TOL = 1e-11
PREDICTION_TOLERANCE = 1e-8
SUPPORT_POLICY = "exact_observed_only; hull and distance are diagnostics"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty output: {relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def predict_loss(n_b: float, d_b: float, parameters: dict[str, float]) -> float:
    return (
        parameters["E"]
        + parameters["A"] * n_b ** (-parameters["alpha"])
        + parameters["B"] * d_b ** (-parameters["beta"])
    )


def load_observed_support(
    grid_path: Path,
    parameter_path: Path,
) -> list[dict[str, Any]]:
    grid_rows = read_csv(grid_path)
    fit = read_json(parameter_path)
    require(fit.get("fit_scope") == "B1 only", "Expected the frozen B1-only M0 fit")
    params = {key: float(value) for key, value in fit["parameters"].items()}
    require(len(grid_rows) == 1176, f"Expected 1,176 B1 rows, found {len(grid_rows)}")

    observed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_points: set[tuple[Fraction, Fraction]] = set()
    for row in grid_rows:
        run_id = str(row["run_id"])
        n_raw = row["N_params_B"]
        d_raw = row["D_tokens_B"]
        n_fraction = Fraction(n_raw)
        d_fraction = Fraction(d_raw)
        n_b, d_b = float(n_fraction), float(d_fraction)
        require(n_b > 0 and d_b > 0, f"Nonpositive N/D at run_id={run_id}")
        require(run_id not in seen_ids, f"Duplicate B1 run_id={run_id}")
        require((n_fraction, d_fraction) not in seen_points, f"Duplicate B1 N/D point at {run_id}")
        seen_ids.add(run_id)
        seen_points.add((n_fraction, d_fraction))

        observed_loss = float(row["observed_val_loss"])
        supplied_prediction = float(row["predicted_val_loss"])
        computed_prediction = predict_loss(n_b, d_b, params)
        require(
            abs(computed_prediction - supplied_prediction) <= PREDICTION_TOLERANCE,
            f"M0 prediction mismatch at run_id={run_id}",
        )
        observed.append(
            {
                "run_id": run_id,
                "N_params_B": n_b,
                "D_tokens_B": d_b,
                "N_params_raw": n_raw,
                "D_tokens_raw": d_raw,
                "observed_val_loss": observed_loss,
                "m0_predicted_val_loss": computed_prediction,
                "_n_fraction": n_fraction,
                "_d_fraction": d_fraction,
            }
        )

    n_values = {point["_n_fraction"] for point in observed}
    d_values = {point["_d_fraction"] for point in observed}
    require(len(n_values) == 8 and len(d_values) == 147, "Expected the 8 x 147 B1 support grid")
    require(len(observed) == len(n_values) * len(d_values), "B1 support grid is not a complete Cartesian grid")
    return observed


def load_candidates(
    candidate_path: Path | None,
    observed: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if candidate_path is None:
        rows = [
            {
                "grid_id": point["run_id"],
                "source_run_id": point["run_id"],
                "N_params_B": point["N_params_raw"],
                "D_tokens_B": point["D_tokens_raw"],
            }
            for point in observed
        ]
        source = "Q2 B1 observed 1,176-point grid; matches the existing M0 search domain"
    else:
        candidate_path = candidate_path.resolve() if candidate_path.is_absolute() else ROOT / candidate_path
        require(candidate_path.is_file(), f"Candidate grid does not exist: {candidate_path}")
        rows = read_csv(candidate_path)
        source = relative(candidate_path)
        require(rows, "Candidate grid is empty")

    observed_lookup = {
        (point["_n_fraction"], point["_d_fraction"]): point for point in observed
    }
    seen_ids: set[str] = set()
    candidates = []
    for row in rows:
        grid_id = str(row.get("grid_id", "")).strip()
        require(grid_id, "Candidate rows must provide a nonempty grid_id")
        require(grid_id not in seen_ids, f"Duplicate grid_id={grid_id}")
        seen_ids.add(grid_id)
        n_raw = str(row["N_params_B"])
        d_raw = str(row["D_tokens_B"])
        n_fraction = Fraction(n_raw)
        d_fraction = Fraction(d_raw)
        n_b, d_b = float(n_fraction), float(d_fraction)
        require(n_b > 0 and d_b > 0, f"Candidate {grid_id} has nonpositive N/D")
        exact_point = observed_lookup.get((n_fraction, d_fraction))
        candidates.append(
            {
                "grid_id": grid_id,
                "source_run_id": str(row.get("source_run_id", row.get("run_id", ""))),
                "N_params_B": n_b,
                "D_tokens_B": d_b,
                "N_params_raw": n_raw,
                "D_tokens_raw": d_raw,
                "N_parameters": int(n_fraction * UNIT_SCALE),
                "D_tokens": int(d_fraction * UNIT_SCALE),
                "observed_exact": exact_point is not None,
                "matched_run_id": exact_point["run_id"] if exact_point else "",
                "observed_val_loss": exact_point["observed_val_loss"] if exact_point else "",
                # Do not publish an M0 prediction for unsupported points: doing
                # so would make an unchecked interpolation/extrapolation easy to use.
                "m0_predicted_val_loss": (
                    exact_point["m0_predicted_val_loss"] if exact_point else ""
                ),
                "_n_fraction": n_fraction,
                "_d_fraction": d_fraction,
            }
        )
    return candidates, source


def standardize_log_points(
    observed: list[dict[str, Any]],
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    log_n = [math.log(point["N_params_B"]) for point in observed]
    log_d = [math.log(point["D_tokens_B"]) for point in observed]
    mean_n, mean_d = statistics.fmean(log_n), statistics.fmean(log_d)
    sd_n, sd_d = statistics.pstdev(log_n), statistics.pstdev(log_d)
    require(sd_n > 0 and sd_d > 0, "Cannot standardize a constant log N or log D coordinate")
    points = [((x - mean_n) / sd_n, (y - mean_d) / sd_d) for x, y in zip(log_n, log_d)]
    metadata = {
        "log_N_mean": mean_n,
        "log_N_population_sd": sd_n,
        "log_D_mean": mean_d,
        "log_D_population_sd": sd_d,
    }
    return points, metadata


def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    require(len(unique) >= 3, "At least three unique observed coordinates are required for a 2D hull")
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= GEOMETRY_TOL:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= GEOMETRY_TOL:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    require(len(hull) >= 3, "Observed support is geometrically degenerate")
    return hull


def inside_convex_polygon(point: tuple[float, float], hull: list[tuple[float, float]]) -> bool:
    # Monotone-chain output is counter-clockwise. Edge points count as inside.
    return all(cross(hull[i], hull[(i + 1) % len(hull)], point) >= -GEOMETRY_TOL for i in range(len(hull)))


def classify_support(
    candidates: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    standardized_observed, transform = standardize_log_points(observed)
    hull = convex_hull(standardized_observed)
    obs_x = [point["N_params_B"] for point in observed]
    obs_y = [point["D_tokens_B"] for point in observed]
    ranges = {
        "N_min_B": min(obs_x),
        "N_max_B": max(obs_x),
        "D_min_B": min(obs_y),
        "D_max_B": max(obs_y),
    }
    obs_z = standardized_observed
    result = []
    for candidate in candidates:
        x = (math.log(candidate["N_params_B"]) - transform["log_N_mean"]) / transform["log_N_population_sd"]
        y = (math.log(candidate["D_tokens_B"]) - transform["log_D_mean"]) / transform["log_D_population_sd"]
        point = (x, y)
        in_n_range = ranges["N_min_B"] - EXACT_MATCH_ABS_TOL <= candidate["N_params_B"] <= ranges["N_max_B"] + EXACT_MATCH_ABS_TOL
        in_d_range = ranges["D_min_B"] - EXACT_MATCH_ABS_TOL <= candidate["D_tokens_B"] <= ranges["D_max_B"] + EXACT_MATCH_ABS_TOL
        in_hull = inside_convex_polygon(point, hull)
        nearest_distance = min(math.hypot(x - ox, y - oy) for ox, oy in obs_z)
        exact = bool(candidate["observed_exact"])
        interpolation_candidate = in_hull and not exact
        extrapolation = not in_hull
        # Frozen main-analysis rule: only actual observed pairs enter M1.
        allowed = exact
        if exact:
            support_class = "observed_exact"
        elif interpolation_candidate:
            support_class = "hull_only_interpolation_candidate"
        elif in_n_range and in_d_range:
            support_class = "inside_range_box_outside_hull"
        else:
            support_class = "outside_observed_hull"
        result.append(
            {
                "grid_id": candidate["grid_id"],
                "N_params_B": candidate["N_params_B"],
                "D_tokens_B": candidate["D_tokens_B"],
                "inside_N_range": in_n_range,
                "inside_D_range": in_d_range,
                "inside_range_box": in_n_range and in_d_range,
                "observed_exact": exact,
                "inside_convex_hull": in_hull,
                "interpolation_candidate": interpolation_candidate,
                "extrapolation_flag": extrapolation,
                "nearest_observed_distance_standardized_log": nearest_distance,
                "support_class": support_class,
                "m3_search_allowed": allowed,
                "support_policy": SUPPORT_POLICY,
            }
        )
    return result, {**transform, **ranges, "hull_vertex_count": len(hull), "hull_standardized_log_vertices": hull}


def cost_for(candidate: dict[str, Any], context: int) -> tuple[Fraction, Fraction, Fraction]:
    n_abs = candidate["N_parameters"]
    d_abs = candidate["D_tokens"]
    nd = n_abs * d_abs
    train = Fraction(TRAINING_COEFFICIENT * nd)
    quality = Fraction(0)  # Current M0 branch fixes Q=Q0.
    attention = ETA * nd * context
    return train, quality, attention


def make_budget_mask(
    candidates: list[dict[str, Any]],
    contexts: list[int],
    budgets: list[int],
) -> list[dict[str, Any]]:
    mask_rows = []

    for context in contexts:
        for budget in budgets:
            for candidate in candidates:
                train, quality, attention = cost_for(candidate, context)
                total = train + quality + attention
                budget_feasible = total <= budget
                mask_rows.append(
                    {
                        "scenario_id": f"C{budget}_L{context}",
                        "budget_flops": budget,
                        "context_length_tokens": context,
                        "grid_id": candidate["grid_id"],
                        "source_run_id": candidate["source_run_id"],
                        "N_params_B": candidate["N_params_B"],
                        "D_tokens_B": candidate["D_tokens_B"],
                        "budget_feasible": budget_feasible,
                        "C_train_flops": int(train),
                        "C_Q_flops": int(quality),
                        "C_attn_flops": float(attention),
                        "C_total_flops": float(total),
                        "budget_slack_flops": float(Fraction(budget) - total),
                    }
                )
    return mask_rows


def public_observed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": point["run_id"],
            "run_id": point["run_id"],
            "N_params_B": point["N_params_B"],
            "D_tokens_B": point["D_tokens_B"],
            "observed_val_loss": point["observed_val_loss"],
            "m0_predicted_val_loss": point["m0_predicted_val_loss"],
        }
        for point in rows
    ]


def public_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "grid_id": item["grid_id"],
            "source_run_id": item["source_run_id"],
            "N_params_B": item["N_params_B"],
            "D_tokens_B": item["D_tokens_B"],
            "N_parameters": item["N_parameters"],
            "D_tokens": item["D_tokens"],
            "observed_exact": item["observed_exact"],
            "matched_run_id": item["matched_run_id"],
            "observed_val_loss": item["observed_val_loss"],
            "m0_predicted_val_loss": item["m0_predicted_val_loss"],
        }
        for item in rows
    ]


def render_m3_report(
    observed: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    support_meta: dict[str, Any],
    candidate_source: str,
) -> str:
    exact_count = sum(bool(row["observed_exact"]) for row in support_rows)
    hull_count = sum(bool(row["inside_convex_hull"]) for row in support_rows)
    interpolation_count = sum(bool(row["interpolation_candidate"]) for row in support_rows)
    extrapolation_count = sum(bool(row["extrapolation_flag"]) for row in support_rows)
    allowed_count = sum(bool(row["m3_search_allowed"]) for row in support_rows)
    lines = [
        "# Q3 M3 支持域审计",
        "",
        f"**生成时间（UTC）：** {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
        f"**候选域来源：** `{candidate_source}`  ",
        f"**M3 主分析口径：** `{SUPPORT_POLICY}`  ",
        "",
        f"- 实际观测点：{len(observed):,}；候选点：{len(candidates):,}。",
        f"- 候选中精确匹配观测组合：{exact_count:,}；凸包内：{hull_count:,}；凸包内非观测插值候选：{interpolation_count:,}；凸包外：{extrapolation_count:,}。",
        f"- 当前允许 M1 搜索的候选点：{allowed_count:,}。凸包和标准化对数距离仅作诊断，不自动开放插值。",
        f"- 凸包在标准化 $(\\log N_B,\\log D_B)$ 空间中构造；每轴使用观测支持的总体标准差标准化。凸包顶点数：{support_meta['hull_vertex_count']}。",
        f"- 最近点距离保存在 `m3_q3_support_grid.csv`，不设未经验证的距离阈值。",
        "",
        "当前主支路沿用实测离散域。若未来开放插值，需另提供候选网格、插值验证证据和预先冻结的接受规则；凸包内本身不足以放行。",
        "",
        "复现命令：`python Q3/03_代码/build_q3_m3_m4_mask_pipeline.py --component m3`。",
        "",
    ]
    return "\n".join(lines)


def render_m4_report(mask_rows: list[dict[str, Any]], candidate_count: int, candidate_source: str) -> str:
    scenario_counts: dict[tuple[int, int], int] = {}
    for row in mask_rows:
        key = (int(row["budget_flops"]), int(row["context_length_tokens"]))
        scenario_counts[key] = scenario_counts.get(key, 0) + int(bool(row["budget_feasible"]))
    lines = [
        "# Q3 M4 预算可行性审计",
        "",
        f"**生成时间（UTC）：** {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
        f"**候选域来源：** `{candidate_source}`  ",
        "**职责：** 仅计算给定候选点在预算与上下文情景下是否可负担，不读取 M3 支持掩码，也不运行 M1 优化。",
        "**范围：** 固定 Q=Q0，因此当前 M0 支路的 $C_Q=0$；N、D 按绝对参数数和 Token 数计入 FLOPs。",
        "",
        f"- 掩码行数：{len(mask_rows):,}（{candidate_count:,} 个候选 × {len(scenario_counts)} 个预算/上下文情景）。",
        "- 成本：$C_{\\mathrm{train}}=6ND$，$C_{\\mathrm{attn}}=2\\times10^{-4}NDL_{ctx}$。",
        "",
        "| 上下文 | 预算 FLOPs | 预算可行点 |",
        "|---:|---:|---:|",
    ]
    for (budget, context), count in sorted(scenario_counts.items(), key=lambda item: (item[0][1], item[0][0])):
        lines.append(f"| {context:,} | {budget:.0e} | {count:,}/{candidate_count:,} |")
    lines.extend(
        [
            "",
            "预算可行不代表 M3 支持。M1 需要读取本表与 M3 支持表，单独计算两者交集后再优化。",
            "",
            "复现命令：`python Q3/03_代码/build_q3_m3_m4_mask_pipeline.py --component m4`。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("m3", "m4", "both"),
        default="both",
        help="Build M3 support, M4 budget, or both. Run m3 and m4 separately for parallel execution.",
    )
    parser.add_argument(
        "--candidate-grid",
        type=Path,
        help="Optional CSV with grid_id,N_params_B,D_tokens_B; default is the actual B1 grid.",
    )
    return parser.parse_args()


def load_cost_candidates(candidate_path: Path | None) -> tuple[list[dict[str, Any]], str]:
    if candidate_path is None:
        rows = read_csv(ROOT / Q2_GRID)
        source = "Q2 B1 observed 1,176-point grid; matches the existing M0 search domain"
    else:
        candidate_path = candidate_path.resolve() if candidate_path.is_absolute() else ROOT / candidate_path
        require(candidate_path.is_file(), f"Candidate grid does not exist: {candidate_path}")
        rows = read_csv(candidate_path)
        source = relative(candidate_path)
    require(rows, "M4 candidate grid is empty")

    candidates = []
    seen: set[str] = set()
    for row in rows:
        grid_id = str(row.get("grid_id", row.get("run_id", ""))).strip()
        require(grid_id, "M4 candidate rows require grid_id or run_id")
        require(grid_id not in seen, f"Duplicate M4 candidate grid_id={grid_id}")
        seen.add(grid_id)
        n_raw, d_raw = str(row["N_params_B"]), str(row["D_tokens_B"])
        n_fraction, d_fraction = Fraction(n_raw), Fraction(d_raw)
        require(n_fraction > 0 and d_fraction > 0, f"M4 candidate has nonpositive N/D: {grid_id}")
        candidates.append(
            {
                "grid_id": grid_id,
                "source_run_id": str(row.get("source_run_id", row.get("run_id", grid_id))),
                "N_params_B": float(n_fraction),
                "D_tokens_B": float(d_fraction),
                "N_parameters": int(n_fraction * UNIT_SCALE),
                "D_tokens": int(d_fraction * UNIT_SCALE),
            }
        )
    return candidates, source


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.component in {"m3", "both"}:
        m3_inputs = [ROOT / Q2_GRID, ROOT / Q2_PARAMS]
        missing = [str(path) for path in m3_inputs if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required M3 inputs missing:\n- " + "\n- ".join(missing))
        observed = load_observed_support(ROOT / Q2_GRID, ROOT / Q2_PARAMS)
        candidates, candidate_source = load_candidates(args.candidate_grid, observed)
        support_rows, support_meta = classify_support(candidates, observed)
        observed_public = public_observed(observed)
        candidate_public = public_candidates(candidates)
        write_csv(ROOT / OUT_OBSERVED, observed_public)
        write_csv(ROOT / OUT_CANDIDATES, candidate_public)
        write_csv(ROOT / OUT_SUPPORT, support_rows)
        (ROOT / OUT_M3_REPORT).write_text(
            render_m3_report(observed_public, candidates, support_rows, support_meta, candidate_source),
            encoding="utf-8",
        )
        input_hashes = {relative(path): sha256_file(path) for path in m3_inputs}
        if args.candidate_grid is not None:
            resolved = args.candidate_grid.resolve() if args.candidate_grid.is_absolute() else ROOT / args.candidate_grid
            input_hashes[relative(resolved)] = sha256_file(resolved)
        m3_outputs = [OUT_OBSERVED, OUT_CANDIDATES, OUT_SUPPORT, OUT_M3_REPORT]
        m3_manifest = {
            "artifact": "Q3 M3 observed support and candidate support diagnostics",
            "command": "python Q3/03_代码/build_q3_m3_m4_mask_pipeline.py --component m3",
            "script": {"path": relative(Path(__file__)), "sha256": sha256_file(Path(__file__))},
            "python_version": platform.python_version(),
            "inputs_sha256": input_hashes,
            "outputs_sha256": {relative(ROOT / path): sha256_file(ROOT / path) for path in m3_outputs},
            "policy": {
                "M3": SUPPORT_POLICY,
                "convex_hull_space": "standardized log(N_params_B), log(D_tokens_B)",
                "nearest_distance": "Euclidean distance in standardized log coordinates; diagnostic only",
                "exact_match_rel_tol": EXACT_MATCH_REL_TOL,
                "exact_match_abs_tol": EXACT_MATCH_ABS_TOL,
                "interpolation_acceptance_threshold": None,
            },
            "support_transform": support_meta,
            "scope": {
                "observed_support_points": len(observed),
                "candidate_points": len(candidates),
                "candidate_grid_source": candidate_source,
                "claim_boundary": "support diagnostics only; no interpolation is inferred",
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_manifest(ROOT / OUT_M3_MANIFEST, m3_manifest)
        print(
            "M3: observed={}, candidates={}, exact={}, hull-only={}, outside={}, allowed={}".format(
                len(observed),
                len(candidates),
                sum(bool(row["observed_exact"]) for row in support_rows),
                sum(bool(row["interpolation_candidate"]) for row in support_rows),
                sum(bool(row["extrapolation_flag"]) for row in support_rows),
                sum(bool(row["m3_search_allowed"]) for row in support_rows),
            )
        )
        for path in m3_outputs + [OUT_M3_MANIFEST]:
            print(f"Wrote {path.as_posix()}")

    if args.component in {"m4", "both"}:
        m4_candidates, candidate_source = load_cost_candidates(args.candidate_grid)
        m4_inputs = [ROOT / Q2_GRID, ROOT / C7_INPUT]
        missing = [str(path) for path in m4_inputs if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required M4 inputs missing:\n- " + "\n- ".join(missing))
        context_rows = read_csv(ROOT / C7_INPUT)
        contexts = sorted({int(row["max_position_embeddings"]) for row in context_rows})
        require(contexts and all(length > 0 for length in contexts), "C7 contexts must be positive")
        require(len(contexts) == 5, "Expected five C7 context lengths")
        budgets = list(BUDGETS_FLOPS)
        budget_rows = make_budget_mask(m4_candidates, contexts, budgets)
        write_csv(ROOT / OUT_BUDGET_MASK, budget_rows)
        (ROOT / OUT_M4_REPORT).write_text(
            render_m4_report(budget_rows, len(m4_candidates), candidate_source),
            encoding="utf-8",
        )
        input_hashes = {relative(path): sha256_file(path) for path in m4_inputs}
        if args.candidate_grid is not None:
            resolved = args.candidate_grid.resolve() if args.candidate_grid.is_absolute() else ROOT / args.candidate_grid
            input_hashes[relative(resolved)] = sha256_file(resolved)
        m4_outputs = [OUT_BUDGET_MASK, OUT_M4_REPORT]
        m4_manifest = {
            "artifact": "Q3 M4 candidate budget feasibility mask",
            "command": "python Q3/03_代码/build_q3_m3_m4_mask_pipeline.py --component m4",
            "script": {"path": relative(Path(__file__)), "sha256": sha256_file(Path(__file__))},
            "python_version": platform.python_version(),
            "inputs_sha256": input_hashes,
            "outputs_sha256": {relative(ROOT / path): sha256_file(ROOT / path) for path in m4_outputs},
            "policy": {
                "cost_formula": "C_train=6ND; C_Q=0 at fixed Q=Q0; C_attn=eta*N*D*L_ctx",
                "eta": float(ETA),
                "training_cost_coefficient": TRAINING_COEFFICIENT,
                "budgets_flops": budgets,
                "M3_dependency": "none; M4 emits budget_feasible only",
            },
            "scope": {
                "candidate_points": len(m4_candidates),
                "candidate_grid_source": candidate_source,
                "context_lengths": contexts,
                "claim_boundary": "budget affordability only; does not imply model support",
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_manifest(ROOT / OUT_M4_MANIFEST, m4_manifest)
        print(f"M4: candidates={len(m4_candidates)}, mask_rows={len(budget_rows)}, scenarios={len(budgets) * len(contexts)}")
        for path in m4_outputs + [OUT_M4_MANIFEST]:
            print(f"Wrote {path.as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, ZeroDivisionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
