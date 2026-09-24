#!/usr/bin/env python3
"""Generate a versioned Cerebras-GPT semi-synthetic transfer stress set.

The generated paths are calibrated to published Cerebras-GPT terminal Pile
test cross-entropies and inherit their within-run shape from the frozen B1
Pythia scaling fit. They are deliberately not represented as observed runs or
independent external validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
B1_DATA_REL = "中文题目/F题/real_attachments/B_scaling_laws/pythia_training_log_existing.csv"
B2_LOCAL_BASELINE_REL = "中文题目/F题/real_attachments/B_scaling_laws/scaling_baseline.csv"
B1_OUTPUT_DIR = Path("results/q2_scaling_baseline/v1")
OUTPUT_DIR = Path("results/q2_b2_semisynthetic/v1")
PAPER_URL = "https://arxiv.org/html/2304.03208v1"
PAPER_VERSION = "arXiv:2304.03208v1, 2023-04-06"
CALIBRATION_RULE = (
    "For each Cerebras size, target_loss(D)=published_Pile_test_xent + "
    "slope_multiplier*(frozen_B1_Pythia_loss(N,D)-frozen_B1_Pythia_loss(N,D_end)) "
    "+ family_offset + endpoint-centered_AR1_noise. D_end uses the published "
    "per-model token count. The terminal Cerebras anchor is never included in "
    "the interior stress metrics."
)
CHECKPOINTS_DEFAULT = 147
MIN_PROGRESS_FRACTION = 0.10
STRESS_SCENARIOS = (
    {"name": "calibrated_pythia_shape", "family_offset": 0.0, "slope_multiplier": 1.0, "noise_sd": 0.0, "rho": 0.0},
    {"name": "family_offset_minus_0p15", "family_offset": -0.15, "slope_multiplier": 1.0, "noise_sd": 0.0, "rho": 0.0},
    {"name": "family_offset_plus_0p15", "family_offset": 0.15, "slope_multiplier": 1.0, "noise_sd": 0.0, "rho": 0.0},
    {"name": "shallow_progress_0p75", "family_offset": 0.0, "slope_multiplier": 0.75, "noise_sd": 0.0, "rho": 0.0},
    {"name": "steep_progress_1p25", "family_offset": 0.0, "slope_multiplier": 1.25, "noise_sd": 0.0, "rho": 0.0},
    {"name": "ar1_noise_sd_0p02_rho_0p70", "family_offset": 0.0, "slope_multiplier": 1.0, "noise_sd": 0.02, "rho": 0.70},
)


@dataclass(frozen=True)
class CerebrasAnchor:
    model_label: str
    N_params_B: float
    D_end_tokens_B: float
    pile_test_xent: float
    training_flops: float


# Table 1 training-token counts and Table 8 Pile-test xent values in the
# Cerebras-authored paper. FLOPs are recorded for provenance, not used to
# construct a fabricated compute-time or hardware-utilization field.
CEREBRAS_ANCHORS = (
    CerebrasAnchor("111M", 0.111, 2.2, 2.608, 2.6e18),
    CerebrasAnchor("256M", 0.256, 5.1, 2.349, 1.3e19),
    CerebrasAnchor("590M", 0.590, 11.8, 2.181, 6.1e19),
    CerebrasAnchor("1.3B", 1.300, 26.3, 1.997, 2.8e20),
    CerebrasAnchor("2.7B", 2.700, 53.0, 1.834, 1.1e21),
    CerebrasAnchor("6.7B", 6.700, 133.2, 1.704, 6.3e21),
    CerebrasAnchor("13B", 13.000, 257.1, 1.572, 2.3e22),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_b1_frozen(project_root: Path = PROJECT_ROOT) -> dict:
    """Require the documented B1 freeze, passing Gate 0, and hash-closed run."""
    root = project_root.resolve()
    b1_dir = root / B1_OUTPUT_DIR
    manifest_path = b1_dir / "复现清单.json"
    report_path = b1_dir / "q2_scaling_baseline_report.md"
    params_path = b1_dir / "fit_parameters.json"
    predictions_path = b1_dir / "b1_fitted_predictions.csv"
    data_path = root / B1_DATA_REL
    gate_path = b1_dir / "gate0_audit.json"
    snapshot_path = b1_dir / "gate0_frozen_input_snapshot.json"
    required = (manifest_path, report_path, params_path, predictions_path, data_path, gate_path, snapshot_path)
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("B1 freeze gate: required evidence missing: " + ", ".join(missing))

    manifest = _read_json(manifest_path)
    gate = _read_json(gate_path)
    snapshot = _read_json(snapshot_path)
    report = report_path.read_text(encoding="utf-8")
    if "冻结" not in report or "基线已拟合" not in report:
        raise RuntimeError("B1 freeze gate: baseline report does not record a fitted frozen reference branch")
    if gate.get("status") != "PASS":
        raise RuntimeError("B1 freeze gate: Gate 0 is not PASS")

    data_hash = sha256(data_path)
    data_key = B1_DATA_REL
    expected_input_hash = manifest.get("input_sha256", {}).get(data_key)
    if not expected_input_hash or expected_input_hash != data_hash:
        raise RuntimeError("B1 freeze gate: current Pythia B1 input differs from the frozen run manifest")
    if gate.get("input_sha256", {}).get(data_key) != data_hash:
        raise RuntimeError("B1 freeze gate: Gate 0 input hash differs from the current Pythia B1 input")
    if snapshot.get("input_sha256", {}).get(data_key) != data_hash:
        raise RuntimeError("B1 freeze gate: frozen Gate 0 snapshot differs from the current Pythia B1 input")

    output_hashes = manifest.get("outputs_sha256", {})
    pinned_files = {
        "results/q2_scaling_baseline/v1/fit_parameters.json": params_path,
        "results/q2_scaling_baseline/v1/b1_fitted_predictions.csv": predictions_path,
        "results/q2_scaling_baseline/v1/q2_scaling_baseline_report.md": report_path,
    }
    for relative, path in pinned_files.items():
        if output_hashes.get(relative) != sha256(path):
            raise RuntimeError(f"B1 freeze gate: frozen output hash mismatch for {relative}")

    fit = _read_json(params_path)
    if fit.get("fit_scope") != "B1 only":
        raise RuntimeError("B1 freeze gate: fitted reference is not explicitly restricted to B1")
    parameters = fit.get("parameters", {})
    expected_names = {"E", "A", "alpha", "B", "beta"}
    if set(parameters) != expected_names or not all(math.isfinite(float(parameters[key])) for key in expected_names):
        raise RuntimeError("B1 freeze gate: invalid or incomplete frozen parameter vector")

    return {
        "status": "PASS",
        "freeze_evidence": "Q2 classic baseline report + hash-closed reproduction manifest",
        "b1_input_sha256": data_hash,
        "b1_fit_parameters_sha256": sha256(params_path),
        "b1_fitted_predictions_sha256": sha256(predictions_path),
        "b1_gate0_status": gate["status"],
        "b1_parameters": parameters,
        "b1_fit_scope": fit["fit_scope"],
    }


def predict_scaling(parameters: dict[str, float], n_params_b, d_tokens_b) -> np.ndarray:
    """Evaluate the frozen five-parameter Pythia reference law."""
    n = np.asarray(n_params_b, dtype=float)
    d = np.asarray(d_tokens_b, dtype=float)
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(d)) or np.any(n <= 0) or np.any(d <= 0):
        raise ValueError("N_params_B and D_tokens_B must be finite and strictly positive")
    e, a, alpha, b, beta = (float(parameters[name]) for name in ("E", "A", "alpha", "B", "beta"))
    if not all(math.isfinite(value) for value in (e, a, alpha, b, beta)):
        raise ValueError("all frozen Pythia parameters must be finite")
    if a <= 0 or alpha <= 0 or b <= 0 or beta <= 0:
        raise ValueError("A, alpha, B, and beta must be strictly positive")
    return e + a * np.power(n, -alpha) + b * np.power(d, -beta)


def _ar1_endpoint_centered(rng: np.random.Generator, size: int, sd: float, rho: float) -> np.ndarray:
    if size < 2 or not 0 <= rho < 1 or sd < 0:
        raise ValueError("AR(1) noise requires size >= 2, sd >= 0, and 0 <= rho < 1")
    innovations = rng.normal(size=size)
    values = np.empty(size, dtype=float)
    values[0] = innovations[0]
    scale = math.sqrt(1.0 - rho * rho)
    for index in range(1, size):
        values[index] = rho * values[index - 1] + scale * innovations[index]
    values -= values[-1]  # Preserve the published terminal anchor by construction.
    observed_sd = float(np.std(values[:-1], ddof=1))
    if observed_sd > 0:
        values *= sd / observed_sd
    return values


def generate_trajectories(
    parameters: dict[str, float], *, seed: int = 20260924, checkpoints: int = CHECKPOINTS_DEFAULT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible, endpoint-calibrated synthetic transfer paths."""
    if checkpoints < 3:
        raise ValueError("checkpoints must be at least 3 so the endpoint can be excluded")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    records: list[dict] = []
    anchor_records: list[dict] = []
    for model_index, anchor in enumerate(CEREBRAS_ANCHORS):
        d_end = anchor.D_end_tokens_B
        d_grid = np.geomspace(d_end * MIN_PROGRESS_FRACTION, d_end, checkpoints)
        pythia_curve = predict_scaling(parameters, anchor.N_params_B, d_grid)
        pythia_end = float(predict_scaling(parameters, anchor.N_params_B, d_end))
        pythia_delta = pythia_curve - pythia_end

        for scenario_index, scenario in enumerate(STRESS_SCENARIOS):
            rng = np.random.default_rng(np.random.SeedSequence([seed, model_index, scenario_index]))
            noise = _ar1_endpoint_centered(rng, checkpoints, scenario["noise_sd"], scenario["rho"])
            target = (
                anchor.pile_test_xent
                + float(scenario["slope_multiplier"]) * pythia_delta
                + float(scenario["family_offset"])
                + noise
            )
            run_id = f"cerebras_{anchor.model_label}__{scenario['name']}"
            for checkpoint_index in range(checkpoints):
                records.append({
                    "run_id": run_id,
                    "scenario": scenario["name"],
                    "model_label": anchor.model_label,
                    "N_params_B": anchor.N_params_B,
                    "D_tokens_B": float(d_grid[checkpoint_index]),
                    "tokens_per_parameter": float(d_grid[checkpoint_index] / anchor.N_params_B),
                    "synthetic_checkpoint": checkpoint_index + 1,
                    "target_loss": float(target[checkpoint_index]),
                    "published_endpoint_pile_test_xent": anchor.pile_test_xent,
                    "frozen_pythia_reference_loss": float(pythia_curve[checkpoint_index]),
                    "family_offset": float(scenario["family_offset"]),
                    "slope_multiplier": float(scenario["slope_multiplier"]),
                    "ar1_noise": float(noise[checkpoint_index]),
                    "is_terminal_checkpoint": checkpoint_index == checkpoints - 1,
                    "matches_published_anchor": bool(
                        checkpoint_index == checkpoints - 1
                        and abs(float(target[checkpoint_index]) - anchor.pile_test_xent) < 1e-12
                    ),
                    "response_protocol": "synthetic path anchored to published Cerebras Pile-test xent; comparability to B1 val_loss unverified",
                    "trajectory_status": "semi-synthetic; not an observed Cerebras training run",
                })
            anchor_records.append({
                "scenario": scenario["name"],
                "model_label": anchor.model_label,
                "N_params_B": anchor.N_params_B,
                "D_end_tokens_B": anchor.D_end_tokens_B,
                "training_tokens_per_parameter": anchor.D_end_tokens_B / anchor.N_params_B,
                "published_endpoint_pile_test_xent": anchor.pile_test_xent,
                "generated_terminal_loss": float(target[-1]),
                "endpoint_difference": float(target[-1] - anchor.pile_test_xent),
                "terminal_is_calibration_anchor_position": True,
                "terminal_matches_published_anchor": abs(float(target[-1]) - anchor.pile_test_xent) < 1e-12,
                "terminal_excluded_from_stress_score": True,
            })

    trajectories = pd.DataFrame.from_records(records)
    anchors = pd.DataFrame.from_records(anchor_records)
    if not np.isfinite(trajectories["target_loss"].to_numpy(dtype=float)).all():
        raise RuntimeError("generated target contains NaN or infinity")
    expected_rows = len(CEREBRAS_ANCHORS) * len(STRESS_SCENARIOS) * checkpoints
    if len(trajectories) != expected_rows:
        raise RuntimeError(f"generated {len(trajectories)} rows; expected {expected_rows}")
    return trajectories, anchors


def score_interior_stress(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Compare B1 reference predictions on non-anchor points; diagnostic only."""
    scored = trajectories.loc[~trajectories["is_terminal_checkpoint"]].copy()
    scored["prediction_error"] = scored["frozen_pythia_reference_loss"] - scored["target_loss"]
    rows = []
    for scenario, group in scored.groupby("scenario", sort=False):
        error = group["prediction_error"].to_numpy(dtype=float)
        rows.append({
            "scenario": scenario,
            "n_interior_points": int(len(group)),
            "n_trajectory_clusters": int(group["run_id"].nunique()),
            "rmse_diagnostic_only": float(np.sqrt(np.mean(error ** 2))),
            "mae_diagnostic_only": float(np.mean(np.abs(error))),
            "bias_prediction_minus_synthetic_target": float(np.mean(error)),
            "max_abs_error_diagnostic_only": float(np.max(np.abs(error))),
            "endpoint_used_for_calibration_excluded": True,
            "independent_external_validation": False,
            "comparability_caveat": "B1 val_loss evaluation protocol is not established as identical to published Cerebras Pile test xent.",
        })
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _relative(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def run_generation(*, seed: int = 20260924, checkpoints: int = CHECKPOINTS_DEFAULT, project_root: Path = PROJECT_ROOT) -> dict:
    """Run only after the current B1 freeze artifacts pass the hash gate."""
    root = project_root.resolve()
    freeze = verify_b1_frozen(root)
    fit = _read_json(root / B1_OUTPUT_DIR / "fit_parameters.json")
    trajectories, anchors = generate_trajectories(fit["parameters"], seed=seed, checkpoints=checkpoints)
    metrics = score_interior_stress(trajectories)

    out_dir = root / OUTPUT_DIR
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite versioned B2 output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    paths = {
        "trajectories": out_dir / "b2_cerebras_pythia_calibrated_trajectories.csv",
        "metrics": out_dir / "b2_stress_metrics.csv",
        "anchors": out_dir / "b2_endpoint_anchor_checks.csv",
        "report": out_dir / "b2_cerebras_stress_report.md",
    }
    trajectories.to_csv(paths["trajectories"], index=False, encoding="utf-8-sig", float_format="%.10g")
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig", float_format="%.10g")
    anchors.to_csv(paths["anchors"], index=False, encoding="utf-8-sig", float_format="%.10g")

    report_lines = [
        "# B2 Cerebras 半合成轨迹压力测试",
        "",
        "**性质：半合成压力测试，不是 Cerebras 原始训练日志，也不是独立外部验证。**",
        "",
        f"- B1 冻结门禁：{freeze['status']}；参数文件 SHA-256：`{freeze['b1_fit_parameters_sha256']}`。",
        f"- 轨迹：{len(CEREBRAS_ANCHORS)} 个 Cerebras-GPT 参数规模 × {len(STRESS_SCENARIOS)} 个情景 × {checkpoints} 个检查点。",
        f"- 校准：`{CALIBRATION_RULE}`",
        "- 正式报告分数只用非终点合成检查点；终点用于锚定，已从误差统计中排除。",
        "- 终点位置标记不代表每个情景都等于论文锚点：族偏移情景的终点保留 ±0.15 偏移；逐点是否匹配锚点见锚点审计表。",
        "- Cerebras 论文报告的是 Pile test xent（nats/token）；B1 的 `val_loss` 是否同一评测协议未证实，因此所有误差只作压力诊断。",
        "- 本运行不改写题目附件中的 `cerebras_training_log.csv`，也不回写或重拟合 B1。题目目录的 `scaling_baseline.csv` 与论文逐模型终点不一致，已排除于校准；其 SHA-256 写入清单。",
        "",
        "## 情景诊断",
        "",
        metrics.to_markdown(index=False),
        "",
        "## 再现",
        "",
        "```powershell",
        f'"{sys.executable}" src/f_q2_b2_semisynthetic.py --seed {seed}',
        "```",
        "",
        "## 来源",
        "",
        f"Cerebras-GPT 原始模型规格、各模型训练 token 数与 Pile 测试损失取自 [{PAPER_VERSION}]({PAPER_URL})。",
        "",
    ]
    paths["report"].write_text("\n".join(report_lines), encoding="utf-8")

    source_paths = {
        "B1_Pythia_training_log": root / B1_DATA_REL,
        "B1_fit_parameters": root / B1_OUTPUT_DIR / "fit_parameters.json",
        "B1_fit_manifest": root / B1_OUTPUT_DIR / "复现清单.json",
        "B1_fit_predictions": root / B1_OUTPUT_DIR / "b1_fitted_predictions.csv",
        "B1_Gate0_snapshot": root / B1_OUTPUT_DIR / "gate0_frozen_input_snapshot.json",
    }
    excluded_source_path = root / B2_LOCAL_BASELINE_REL
    payload = {
        "task": "B2 Cerebras-GPT semi-synthetic trajectory stress test",
        "status": "COMPLETED_SEMISYNTHETIC_STRESS_TEST_NOT_EXTERNAL_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reproduction_command": f'"{sys.executable}" src/f_q2_b2_semisynthetic.py --seed {seed}',
        "seed": seed,
        "checkpoint_count_per_trajectory": checkpoints,
        "progress_fraction": [MIN_PROGRESS_FRACTION, 1.0],
        "calibration_rule": CALIBRATION_RULE,
        "b1_freeze_evidence": freeze,
        "cerebras_source": {"paper_version": PAPER_VERSION, "url": PAPER_URL, "reported_metric": "Pile test cross-entropy in nats/token, GPT-2 vocabulary corrected"},
        "anchors": [asdict(anchor) for anchor in CEREBRAS_ANCHORS],
        "scenarios": list(STRESS_SCENARIOS),
        "inputs_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "audited_but_excluded_sources_sha256": {
            B2_LOCAL_BASELINE_REL: {
                "sha256": sha256(excluded_source_path),
                "reason": "Its seven Cerebras rows use D=2050B and loss 2.95..2.00, inconsistent with the paper's model-specific training tokens and Pile test xent; not used for calibration.",
            }
        },
        "outputs_sha256": {name: sha256(path) for name, path in paths.items()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "executable": sys.executable},
        "limits": [
            "Generated trajectories are not observed Cerebras checkpoints.",
            "Published Cerebras terminal points were used for calibration and are excluded from interior error scores.",
            "The Pythia val_loss and Cerebras Pile test xent evaluation protocols are not proven identical.",
            "The source attachment's older 14–2050B token paths were left unchanged and are not reproduced as authentic Cerebras schedules.",
        ],
    }
    manifest_path = out_dir / "generation_manifest.json"
    _write_json(manifest_path, payload)
    print(json.dumps({"status": payload["status"], "output_dir": _relative(root, out_dir), "manifest": _relative(root, manifest_path), "rows": len(trajectories), "seed": seed}, ensure_ascii=False, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--checkpoints", type=int, default=CHECKPOINTS_DEFAULT)
    args = parser.parse_args()
    try:
        run_generation(seed=args.seed, checkpoints=args.checkpoints)
    except Exception as exc:
        print(f"B2 generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
