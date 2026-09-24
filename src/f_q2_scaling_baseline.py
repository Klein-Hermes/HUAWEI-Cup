#!/usr/bin/env python3
"""Fit and audit the classical Q2 loss scaling reference model.

The only fitted model is L = E + A*N**(-alpha) + B*D**(-beta), fitted on
B1. B2-B5 are held out and reported separately by source. B6-B10 are outside
this classical baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACHMENTS = PROJECT_ROOT / "中文题目" / "F题" / "real_attachments"
DATA_DIR = ATTACHMENTS / "B_scaling_laws"
OUTPUT_DIR = PROJECT_ROOT / "results" / "q2_scaling_baseline" / "v1"
FIGURE_DIR = PROJECT_ROOT / "figures" / "q2_scaling_baseline"
FIGURE_PREVIEW_DIR = OUTPUT_DIR / "figure_previews"
SEED_DEFAULT = 20260923
BOOTSTRAP_REPLICATES = 1000
ROLE_CONTRACT_VERSION = "q2-classic-baseline-gate0-v1"
PARAMETER_NAMES = ("E", "A", "alpha", "B", "beta")
MODEL_FORMULA = "L = E + A*N_params_B**(-alpha) + B*D_tokens_B**(-beta)"
STRUCTURAL_MISSING_B2 = ("gpu_days", "step_time_ms", "grad_norm_avg")
BOOTSTRAP_CI = (2.5, 97.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _finite_series(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def _size_b(model_size: str) -> float:
    value = model_size.strip().lower()
    if value.endswith("m"):
        return float(value[:-1]) / 1000.0
    if value.endswith("b"):
        return float(value[:-1])
    raise ValueError(f"Unknown B12 model_size label: {model_size!r}")


def _input_paths() -> dict[str, list[Path]]:
    b3_paths = sorted((DATA_DIR / "training_trajectories").glob("*.csv"))
    return {
        "B1": [DATA_DIR / "pythia_training_log_existing.csv"],
        "B2": [DATA_DIR / "cerebras_training_log.csv"],
        "B3": b3_paths,
        "B4": [DATA_DIR / "scaling_baseline.csv"],
        "B5": [DATA_DIR / "published_scaling_data.csv"],
        "B12": [DATA_DIR / "pythia_checkpoint_index.csv"],
        "data_description": [PROJECT_ROOT / "中文题目" / "F题" / "数据说明.pdf"],
        "question_analysis": [PROJECT_ROOT / "题目分析报告.md"],
    }


def _hash_inputs(input_paths: dict[str, list[Path]]) -> dict[str, str]:
    hashes = {}
    for paths in input_paths.values():
        for path in paths:
            if path.is_file():
                hashes[_rel(path)] = _sha256(path)
    return dict(sorted(hashes.items()))


def _role_table() -> list[dict[str, str]]:
    return [
        {"dataset": "B1", "role": "主拟合数据", "origin": "Pythia training log", "use": "仅用于经典 Loss=f(N,D) 拟合、诊断和按轨迹重抽样", "independent_real_validation": "不适用"},
        {"dataset": "B2", "role": "族外验证轨迹", "origin": "Cerebras 半合成/校准轨迹", "use": "留出验证；按参数规模分层", "independent_real_validation": "否"},
        {"dataset": "B3", "role": "轨迹验证", "origin": "由 Pythia 轨迹插值得到", "use": "留出验证；按插值轨迹分层", "independent_real_validation": "否"},
        {"dataset": "B4", "role": "跨族验证", "origin": "真实开放模型族标度数据", "use": "留出验证；按模型族分层", "independent_real_validation": "是；Loss 口径可比性未由字段证明"},
        {"dataset": "B5", "role": "文献验证", "origin": "公开文献汇总数据", "use": "留出验证；按模型族和来源分层", "independent_real_validation": "是；Loss 口径可比性未由字段证明"},
        {"dataset": "B12", "role": "B1 检查点索引", "origin": "Pythia checkpoint index", "use": "确认 B1 规模—仓库—step 轨迹映射", "independent_real_validation": "否"},
    ]


def run_gate0() -> dict:
    """Audit source roles, schemas, units, expected structures, and B1 clusters."""
    input_paths = _input_paths()
    missing_files = [str(path) for paths in input_paths.values() for path in paths if not path.is_file()]
    role_rows = _role_table()
    checks: list[dict] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "details": details})

    add_check("input_file_inventory", not missing_files, "all declared files exist" if not missing_files else "missing: " + "; ".join(missing_files))
    if missing_files:
        report = {"gate": "Gate 0", "status": "FAIL", "checks": checks, "missing_files": missing_files, "role_contract_version": ROLE_CONTRACT_VERSION}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(OUTPUT_DIR / "gate0_audit.json", report)
        _write_csv(OUTPUT_DIR / "gate0_file_roles.csv", pd.DataFrame(role_rows))
        return report

    print("Gate 0: loading B1-B5 and B12 source tables...", flush=True)
    try:
        b1 = pd.read_csv(input_paths["B1"][0])
        b2 = pd.read_csv(input_paths["B2"][0])
        b4 = pd.read_csv(input_paths["B4"][0])
        b5 = pd.read_csv(input_paths["B5"][0])
        b12 = pd.read_csv(input_paths["B12"][0])
        b3_parts = [(path, pd.read_csv(path)) for path in input_paths["B3"]]
    except Exception as exc:
        add_check("input_csv_readable", False, f"{type(exc).__name__}: {exc}")
        report = {"gate": "Gate 0", "role_contract_version": ROLE_CONTRACT_VERSION, "status": "FAIL", "checks": checks, "file_roles": role_rows, "input_sha256": _hash_inputs(input_paths)}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(OUTPUT_DIR / "gate0_audit.json", report)
        _write_csv(OUTPUT_DIR / "gate0_file_roles.csv", pd.DataFrame(role_rows))
        return report
    print("Gate 0: source tables loaded; checking schema, row counts, and B12 mapping...", flush=True)

    expected = {
        "B1": {"rows": 1176, "required": {"N_params_B", "D_tokens_B", "steps", "val_loss", "run_id"}},
        "B2": {"rows": 1029, "required": {"N_params_B", "D_tokens_B", "steps", "val_loss", "run_id", *STRUCTURAL_MISSING_B2}},
        "B4": {"rows": 57, "required": {"family", "N_params_B", "D_tokens_B", "val_loss", "is_converged"}},
        "B5": {"rows": 44, "required": {"family", "N_params_B", "D_tokens_B", "val_loss", "source", "is_converged"}},
    }
    dataframes = {"B1": b1, "B2": b2, "B4": b4, "B5": b5}
    schema_missing = {
        name: sorted(required - set(dataframes[name].columns))
        for name, required in ((name, expected[name]["required"]) for name in expected)
    }
    schema_missing["B12"] = sorted({"model_repo", "model_size", "step", "branch", "commit"} - set(b12.columns))
    schema_missing["B3"] = [f"{path.name}:{','.join(sorted({'N_params_B','D_tokens_B','val_loss','step','interpolated'} - set(frame.columns)))}" for path, frame in b3_parts if not {"N_params_B", "D_tokens_B", "val_loss", "step", "interpolated"}.issubset(frame.columns)]
    for name, missing_columns in schema_missing.items():
        add_check(f"{name}_required_schema", not missing_columns, "required columns present" if not missing_columns else "missing: " + "; ".join(missing_columns))
    if any(schema_missing.values()):
        report = {
            "gate": "Gate 0",
            "role_contract_version": ROLE_CONTRACT_VERSION,
            "status": "FAIL",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "file_roles": role_rows,
            "schema_missing": schema_missing,
            "input_sha256": _hash_inputs(input_paths),
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(OUTPUT_DIR / "gate0_audit.json", report)
        _write_csv(OUTPUT_DIR / "gate0_file_roles.csv", pd.DataFrame(role_rows))
        return report

    numeric_invalid: dict[str, list[str]] = {}
    numeric_contracts = {
        "B1": (b1, ("N_params_B", "D_tokens_B", "steps", "val_loss")),
        "B2": (b2, ("N_params_B", "D_tokens_B", "steps", "val_loss")),
        "B4": (b4, ("N_params_B", "D_tokens_B", "val_loss")),
        "B5": (b5, ("N_params_B", "D_tokens_B", "val_loss")),
        "B12": (b12, ("step",)),
    }
    for name, (frame, columns) in numeric_contracts.items():
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            invalid_count = int((~np.isfinite(values)).sum())
            if invalid_count:
                numeric_invalid.setdefault(name, []).append(f"{column}: nonnumeric/nonfinite={invalid_count}")
            if column in {"steps", "step"}:
                noninteger = int((np.isfinite(values) & (values != np.floor(values))).sum())
                if noninteger:
                    numeric_invalid.setdefault(name, []).append(f"{column}: noninteger={noninteger}")
        if {"N_params_B", "D_tokens_B"}.issubset(frame.columns):
            for column in ("N_params_B", "D_tokens_B"):
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                nonpositive = int((np.isfinite(values) & (values <= 0)).sum())
                if nonpositive:
                    numeric_invalid.setdefault(name, []).append(f"{column}: nonpositive={nonpositive}")
    for path, frame in b3_parts:
        issues = []
        for column in ("N_params_B", "D_tokens_B", "val_loss", "step", "interpolated"):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            invalid_count = int((~np.isfinite(values)).sum())
            if invalid_count:
                issues.append(f"{column}: nonnumeric/nonfinite={invalid_count}")
            if column == "step":
                noninteger = int((np.isfinite(values) & (values != np.floor(values))).sum())
                if noninteger:
                    issues.append(f"step: noninteger={noninteger}")
            if column in {"N_params_B", "D_tokens_B"}:
                nonpositive = int((np.isfinite(values) & (values <= 0)).sum())
                if nonpositive:
                    issues.append(f"{column}: nonpositive={nonpositive}")
        if issues:
            numeric_invalid[path.name] = issues
    add_check("numeric_contracts", not numeric_invalid, "all modeling fields are finite numeric values with positive N,D and integer steps" if not numeric_invalid else json.dumps(numeric_invalid, ensure_ascii=False, sort_keys=True))
    if numeric_invalid:
        report = {
            "gate": "Gate 0",
            "role_contract_version": ROLE_CONTRACT_VERSION,
            "status": "FAIL",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "file_roles": role_rows,
            "numeric_invalid": numeric_invalid,
            "input_sha256": _hash_inputs(input_paths),
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(OUTPUT_DIR / "gate0_audit.json", report)
        _write_csv(OUTPUT_DIR / "gate0_file_roles.csv", pd.DataFrame(role_rows))
        return report
    for name, frame in dataframes.items():
        required = expected[name]["required"]
        ok = len(frame) == expected[name]["rows"] and required.issubset(frame.columns)
        add_check(f"{name}_role_schema_rows", ok, f"rows={len(frame)} expected={expected[name]['rows']}; required_missing={sorted(required - set(frame.columns))}")
        core = [column for column in ("N_params_B", "D_tokens_B", "val_loss") if column in frame]
        positive = all((pd.to_numeric(frame[column], errors="coerce") > 0).all() for column in ("N_params_B", "D_tokens_B") if column in frame)
        finite = all(_finite_series(frame[column]) for column in core)
        add_check(f"{name}_core_values", positive and finite, f"positive_N_D={positive}; finite_core={finite}")

    b1_sizes = sorted(pd.to_numeric(b1["N_params_B"]).unique().tolist())
    b1_counts = b1.groupby("N_params_B", dropna=False).size()
    b1_checkpoint_counts = b1.groupby("N_params_B")["steps"].nunique()
    b1_structure_ok = len(b1_sizes) == 8 and bool((b1_counts == 147).all()) and bool((b1_checkpoint_counts == 147).all())
    add_check("B1_eight_trajectories_147_checkpoints", b1_structure_ok, f"sizes={len(b1_sizes)}; rows_per_size={sorted(b1_counts.unique().tolist())}; distinct_steps_per_size={sorted(b1_checkpoint_counts.unique().tolist())}")

    structural_missing = {column: int(b2[column].isna().sum()) if column in b2 else None for column in STRUCTURAL_MISSING_B2}
    structural_ok = all(value == len(b2) for value in structural_missing.values()) and all(column in b2 for column in STRUCTURAL_MISSING_B2)
    add_check("B2_structural_missing_fields", structural_ok, json.dumps(structural_missing, ensure_ascii=False, sort_keys=True))
    add_check("B2_seven_size_support", b2["N_params_B"].nunique() == 7, f"unique_sizes={b2['N_params_B'].nunique()}")

    b3_rows = [len(frame) for _, frame in b3_parts]
    b3_flags = [bool(pd.to_numeric(frame["interpolated"], errors="coerce").eq(1).all()) if "interpolated" in frame else False for _, frame in b3_parts]
    b3_ok = len(b3_parts) == 8 and b3_rows == [500] * 8 and all(b3_flags)
    add_check("B3_eight_interpolated_500_point_trajectories", b3_ok, f"files={len(b3_parts)}; rows={b3_rows}; all_interpolated={b3_flags}")
    for path, frame in b3_parts:
        required = {"N_params_B", "D_tokens_B", "val_loss", "step", "interpolated"}
        valid = required.issubset(frame.columns) and all(_finite_series(frame[column]) for column in ("N_params_B", "D_tokens_B", "val_loss")) and (frame["N_params_B"] > 0).all() and (frame["D_tokens_B"] > 0).all()
        add_check(f"B3_schema_values_{path.stem}", valid, f"rows={len(frame)}; required_missing={sorted(required - set(frame.columns))}")

    add_check("B4_twelve_families", b4["family"].nunique() == 12, f"unique_families={b4['family'].nunique()}")
    add_check("B5_nine_families", b5["family"].nunique() == 9, f"unique_families={b5['family'].nunique()}")
    add_check("B4_converged_flags", bool(pd.to_numeric(b4["is_converged"], errors="coerce").eq(1).all()), f"converged={int(pd.to_numeric(b4['is_converged'], errors='coerce').eq(1).sum())}/{len(b4)}")
    add_check("B5_converged_flags", bool(pd.to_numeric(b5["is_converged"], errors="coerce").eq(1).all()), f"converged={int(pd.to_numeric(b5['is_converged'], errors='coerce').eq(1).sum())}/{len(b5)}")

    index_columns = {"model_repo", "model_size", "step", "branch", "commit"}
    index_schema_ok = index_columns.issubset(b12.columns) and not b12.duplicated(["model_size", "step"]).any()
    add_check("B12_index_schema_unique", index_schema_ok, f"rows={len(b12)}; unique_sizes={b12['model_size'].nunique()}; duplicate_size_step={int(b12.duplicated(['model_size','step']).sum())}")
    try:
        index_sizes = sorted(b12["model_size"].astype(str).unique().tolist(), key=_size_b) if index_schema_ok else []
    except (TypeError, ValueError) as exc:
        index_sizes = []
        add_check("B12_model_size_labels", False, f"invalid model_size label: {exc}")
    size_mapping_rows: list[dict] = []
    mapping_ok = index_schema_ok
    for size in b1_sizes:
        if not np.isfinite(float(size)) or float(size) <= 0:
            mapping_ok = False
            continue
        candidates = [(abs(_size_b(label) - float(size)) / float(size), label) for label in index_sizes]
        candidates.sort()
        if not candidates or candidates[0][0] > 0.08 or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
            mapping_ok = False
            continue
        relative_error, label = candidates[0]
        indexed = b12.loc[b12["model_size"].astype(str) == label]
        model_repo = str(indexed["model_repo"].iloc[0])
        b1_group = b1.loc[b1["N_params_B"] == size]
        available_steps = set(pd.to_numeric(indexed["step"], errors="coerce").astype(int))
        observed_steps = set(pd.to_numeric(b1_group["steps"], errors="coerce").astype(int))
        missing_steps = sorted(observed_steps - available_steps)
        exact_repo = indexed["model_repo"].nunique() == 1 and bool((indexed["model_repo"] == model_repo).all())
        matched = len(observed_steps) - len(missing_steps)
        one_to_one = b1_group["N_params_B"].nunique() == 1 and exact_repo
        mapping_ok = mapping_ok and not missing_steps and one_to_one
        size_mapping_rows.append({
            "N_params_B": float(size),
            "B12_model_size": label,
            "model_repo": model_repo,
            "B1_rows": int(len(b1_group)),
            "B1_unique_steps": int(len(observed_steps)),
            "B12_available_steps": int(len(available_steps)),
            "matched_steps": int(matched),
            "missing_steps": json.dumps(missing_steps),
            "relative_nominal_size_difference": float(relative_error),
            "cluster_key": model_repo,
        })
    add_check("B1_B12_model_checkpoint_mapping", mapping_ok and len(size_mapping_rows) == 8, f"mapped_sizes={len(size_mapping_rows)}; matched_checkpoints={sum(row['matched_steps'] for row in size_mapping_rows)}/{len(b1)}; all_steps_present={all(not json.loads(row['missing_steps']) for row in size_mapping_rows)}")
    add_check("B1_run_id_not_cluster", b1["run_id"].nunique() == len(b1) and len(size_mapping_rows) == 8, f"run_id_unique_rows={b1['run_id'].nunique()}/{len(b1)}; frozen_cluster_key=model_repo via B12")

    # Use the standard-library implementation here: this is only a Gate 0
    # descriptive diagnostic and need not invoke a native BLAS routine.
    if (pd.to_numeric(b1["N_params_B"], errors="coerce") > 0).all() and (pd.to_numeric(b1["D_tokens_B"], errors="coerce") > 0).all() and _finite_series(b1["N_params_B"]) and _finite_series(b1["D_tokens_B"]):
        log_n_values = [math.log(float(value)) for value in b1["N_params_B"]]
        log_d_values = [math.log(float(value)) for value in b1["D_tokens_B"]]
        try:
            b1_log_corr = float(statistics.correlation(log_n_values, log_d_values))
        except statistics.StatisticsError:
            b1_log_corr = None
    else:
        b1_log_corr = None
    log_n_range = [float(b1["N_params_B"].min()), float(b1["N_params_B"].max())]
    log_d_range = [float(b1["D_tokens_B"].min()), float(b1["D_tokens_B"].max())]
    support = {
        "N_params_B_range": log_n_range,
        "D_tokens_B_range": log_d_range,
        "N_unique": int(b1["N_params_B"].nunique()),
        "D_unique": int(b1["D_tokens_B"].nunique()),
        "unique_N_D_pairs": int(b1[["N_params_B", "D_tokens_B"]].drop_duplicates().shape[0]),
        "log_N_log_D_pearson_r": b1_log_corr,
        "rows": int(len(b1)),
        "warning": "B1 has eight trajectory clusters; row count is not the independent-cluster count.",
    }
    support_ok = support["unique_N_D_pairs"] == len(b1) and support["D_unique"] == 147 and b1_log_corr is not None
    add_check("B1_N_D_crossed_support", support_ok, f"unique_pairs={support['unique_N_D_pairs']}; unique_D={support['D_unique']}; log_correlation={b1_log_corr if b1_log_corr is not None else 'unavailable'}")

    file_count = sum(len(paths) for paths in input_paths.values())
    print(f"Gate 0: structural checks complete; hashing {file_count} inputs...", flush=True)
    file_hashes = _hash_inputs(input_paths)
    passed = all(row["passed"] for row in checks)
    report = {
        "gate": "Gate 0",
        "role_contract_version": ROLE_CONTRACT_VERSION,
        "status": "PASS" if passed else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "file_roles": role_rows,
        "input_sha256": dict(sorted(file_hashes.items())),
        "b1_trajectory_mapping": size_mapping_rows,
        "b1_design_support": support,
        "loss_contract": {
            "response": "val_loss",
            "N_unit": "billions of parameters (N_params_B)",
            "D_unit": "billions of training tokens (D_tokens_B)",
            "loss_unit": "dataset-specific validation-loss units; the source CSVs do not state a shared tokenizer or evaluation-set protocol",
            "B1_role": "Pythia main fit",
            "validation_policy": "Fit only B1; keep B2-B5 separate. Source metadata does not establish one shared evaluation protocol, so do not pool validation metrics.",
            "B2": "same named response field; semisynthetic family-out trajectory, not independent real validation",
            "B3": "interpolated Pythia trajectory; not independent real validation",
            "B4_B5": "real cross-family/literature records, but tokenizer/evaluation-set protocol comparability is not established in these CSV fields",
        },
        "data_scope": "B6-B8 (Q supplements) and B9-B10 (large-model extrapolation) are excluded from the classical reference baseline.",
    }
    print(f"Gate 0: checks {sum(bool(row['passed']) for row in checks)}/{len(checks)}; saving audit records...", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_DIR / "gate0_audit.json", report)
    _write_csv(OUTPUT_DIR / "gate0_file_roles.csv", pd.DataFrame(role_rows))
    _write_csv(OUTPUT_DIR / "gate0_b1_trajectory_mapping.csv", pd.DataFrame(size_mapping_rows))
    _write_json(OUTPUT_DIR / "gate0_design_support.json", support)
    return report


def _snapshot_path() -> Path:
    return OUTPUT_DIR / "gate0_frozen_input_snapshot.json"


def _freeze_or_compare_snapshot(audit: dict, mode: str, refresh_snapshot: bool = False) -> tuple[bool, str]:
    path = _snapshot_path()
    if mode == "audit":
        if audit["status"] != "PASS":
            return False, "Gate 0 failed; snapshot was not changed"
        current = {"role_contract_version": ROLE_CONTRACT_VERSION, "input_sha256": audit["input_sha256"], "file_roles": audit["file_roles"]}
        if path.exists():
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            same = snapshot.get("role_contract_version") == current["role_contract_version"] and snapshot.get("input_sha256") == current["input_sha256"] and snapshot.get("file_roles") == current["file_roles"]
            if same:
                return True, "current inputs match the existing frozen snapshot"
            _write_json(OUTPUT_DIR / "gate0_latest_attempt.json", audit)
            if not refresh_snapshot:
                return False, "input hashes or roles differ from the frozen snapshot; review the source change, then pass --refresh-input-snapshot to audit mode to explicitly re-freeze"
        elif refresh_snapshot:
            return False, "cannot refresh: no prior frozen snapshot exists; run audit mode without the refresh flag first"
        current["frozen_by"] = "explicitly reviewed audit mode" if refresh_snapshot else "first passing audit"
        _write_json(path, current)
        return True, "passing Gate 0 snapshot frozen" if not refresh_snapshot else "passing Gate 0 snapshot explicitly refreshed"
    if audit["status"] != "PASS":
        return False, "Gate 0 role/schema/data checks failed"
    if not path.exists():
        _write_json(path, {"role_contract_version": ROLE_CONTRACT_VERSION, "input_sha256": audit["input_sha256"], "file_roles": audit["file_roles"], "frozen_by": "first full run"})
        return True, "first full run froze the passing Gate 0 input snapshot"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    same = snapshot.get("role_contract_version") == ROLE_CONTRACT_VERSION and snapshot.get("input_sha256") == audit["input_sha256"] and snapshot.get("file_roles") == audit["file_roles"]
    if not same:
        _write_json(OUTPUT_DIR / "gate0_latest_attempt.json", audit)
        return False, "input hashes or frozen file-role contract differ from gate0_frozen_input_snapshot.json; rerun --mode audit only after reviewing the source change"
    return True, "current Gate 0 inputs match the frozen snapshot"


def _predict(params: np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    e, a, alpha, b, beta = params
    return e + a * np.power(n, -alpha) + b * np.power(d, -beta)


def _initial_params(n: np.ndarray, d: np.ndarray, y: np.ndarray, variant: int) -> np.ndarray:
    min_y = float(np.min(y))
    e_fractions = (0.15, 0.35, 0.55, 0.75)
    exponent_seeds = (0.015, 0.035, 0.07, 0.14, 0.28, 0.5)
    splits = (0.2, 0.35, 0.5, 0.65, 0.8)
    e = min_y * e_fractions[variant % len(e_fractions)]
    alpha = exponent_seeds[(variant * 3 + 1) % len(exponent_seeds)]
    beta = exponent_seeds[(variant * 5 + 2) % len(exponent_seeds)]
    excess = max(float(np.median(y) - e), min_y * 0.02)
    split = splits[(variant * 2) % len(splits)]
    n_center = float(np.exp(np.mean(np.log(n))))
    d_center = float(np.exp(np.mean(np.log(d))))
    a = max(excess * split * n_center ** alpha, 1e-10)
    b = max(excess * (1.0 - split) * d_center ** beta, 1e-10)
    return np.array([e, a, alpha, b, beta], dtype=float)


def _fit_one(n: np.ndarray, d: np.ndarray, y: np.ndarray, x0: np.ndarray | None = None, max_nfev: int = 5000):
    min_y = float(np.min(y))
    lower = np.array([0.0, 1e-12, 1e-8, 1e-12, 1e-8])
    upper = np.array([max(min_y - max(1e-10, min_y * 1e-10), 1e-9), np.inf, np.inf, np.inf, np.inf])
    if x0 is None:
        x0 = _initial_params(n, d, y, 0)
    x0 = np.maximum(x0, lower + np.array([1e-12, 1e-12, 1e-9, 1e-12, 1e-9]))
    x0[0] = min(x0[0], upper[0] - max(1e-12, upper[0] * 1e-10))
    return least_squares(
        lambda params: _predict(params, n, d) - y,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=max_nfev,
    )


def fit_multistart(frame: pd.DataFrame, starts: int = 24, anchor: np.ndarray | None = None) -> tuple[np.ndarray, list[dict], object]:
    n = frame["N_params_B"].to_numpy(dtype=float)
    d = frame["D_tokens_B"].to_numpy(dtype=float)
    y = frame["val_loss"].to_numpy(dtype=float)
    candidates: list[np.ndarray] = []
    if anchor is not None:
        candidates.append(np.asarray(anchor, dtype=float).copy())
    for index in range(starts):
        candidates.append(_initial_params(n, d, y, index))
    outcomes: list[dict] = []
    best = None
    for index, initial in enumerate(candidates):
        try:
            result = _fit_one(n, d, y, initial)
            finite = bool(np.isfinite(result.x).all() and np.isfinite(result.fun).all())
            success = bool(result.success and finite)
            row = {
                "start_id": index,
                "success": success,
                "optimizer_status": int(result.status),
                "message": str(result.message),
                "cost_sse": float(np.dot(result.fun, result.fun)) if finite else None,
                "nfev": int(result.nfev),
                "optimality": float(result.optimality) if np.isfinite(result.optimality) else None,
                **{name: float(value) if np.isfinite(value) else None for name, value in zip(PARAMETER_NAMES, result.x)},
            }
            outcomes.append(row)
            if success and (best is None or row["cost_sse"] < best[0]):
                best = (row["cost_sse"], result)
        except Exception as exc:  # diagnostics record failed starts; one failure does not discard valid starts
            outcomes.append({"start_id": index, "success": False, "optimizer_status": -1, "message": f"{type(exc).__name__}: {exc}", "cost_sse": None, "nfev": None, "optimality": None})
    if best is None:
        raise RuntimeError("No bounded nonlinear least-squares start converged")
    return np.asarray(best[1].x, dtype=float), outcomes, best[1]


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed
    sse = float(np.sum(residual ** 2))
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = 1.0 - sse / denominator if denominator > 0 else None
    rho = spearmanr(observed, predicted).statistic if len(observed) > 2 else np.nan
    return {
        "n": int(len(observed)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_pred_minus_obs": float(np.mean(residual)),
        "r2": float(r2) if r2 is not None and np.isfinite(r2) else None,
        "spearman": float(rho) if np.isfinite(rho) else None,
        "max_abs_error": float(np.max(np.abs(residual))) if len(residual) else None,
    }


def _scaled_jacobian_diagnostics(result, params: np.ndarray) -> dict:
    jacobian = np.asarray(result.jac, dtype=float)
    scales = np.maximum(np.abs(params), np.array([0.1, 0.1, 0.05, 0.1, 0.05]))
    scaled = jacobian * scales[np.newaxis, :]
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = singular[0] * 1e-8 if singular.size and singular[0] > 0 else 0.0
    rank = int(np.sum(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else None
    return {
        "scaled_jacobian_singular_values": [float(value) for value in singular],
        "scaled_jacobian_rank": rank,
        "parameter_count": int(jacobian.shape[1]),
        "scaled_jacobian_condition_number": condition,
        "rank_deficient": rank < jacobian.shape[1],
        "condition_warning": bool(condition is None or condition > 1e8),
    }


def _multistart_diagnostics(outcomes: list[dict], best_params: np.ndarray) -> dict:
    successes = [row for row in outcomes if row.get("success") and row.get("cost_sse") is not None]
    if not successes:
        return {"success_count": 0, "attempt_count": len(outcomes), "near_optimal_count": 0, "relative_parameter_spread": {}, "unstable": True}
    best_sse = min(row["cost_sse"] for row in successes)
    near = [row for row in successes if row["cost_sse"] <= best_sse + max(1e-12, best_sse * 1e-5)]
    spreads = {}
    for name in PARAMETER_NAMES:
        values = np.array([row[name] for row in near if row.get(name) is not None], dtype=float)
        denom = max(abs(float(best_params[PARAMETER_NAMES.index(name)])), 1e-6)
        spreads[name] = float((np.max(values) - np.min(values)) / denom) if len(values) else None
    unstable = any(value is None or value > 0.5 for value in spreads.values())
    return {
        "attempt_count": len(outcomes),
        "success_count": len(successes),
        "success_fraction": float(len(successes) / len(outcomes)),
        "best_sse": float(best_sse),
        "near_optimal_count": len(near),
        "near_optimal_sse_relative_tolerance": 1e-5,
        "relative_parameter_spread": spreads,
        "unstable": bool(unstable),
    }


def _fit_reduced(frame: pd.DataFrame, omit: str) -> dict:
    n = frame["N_params_B"].to_numpy(dtype=float)
    d = frame["D_tokens_B"].to_numpy(dtype=float)
    y = frame["val_loss"].to_numpy(dtype=float)
    include_n = omit != "N"
    include_d = omit != "D"
    min_y = float(y.min())
    active = []
    if include_n:
        active.extend(["A", "alpha"])
    if include_d:
        active.extend(["B", "beta"])
    lower = np.array([0.0] + [1e-12 if name in {"A", "B"} else 1e-8 for name in active])
    upper = np.array([max(min_y - 1e-10, 1e-9)] + [np.inf] * len(active))

    def predict(theta):
        cursor = 1
        values = np.full_like(y, theta[0])
        if include_n:
            values += theta[cursor] * n ** (-theta[cursor + 1])
            cursor += 2
        if include_d:
            values += theta[cursor] * d ** (-theta[cursor + 1])
        return values

    outcomes = []
    for start in range(8):
        full_start = _initial_params(n, d, y, start)
        init = [full_start[0]]
        if include_n:
            init += [full_start[1], full_start[2]]
        if include_d:
            init += [full_start[3], full_start[4]]
        try:
            fit = least_squares(lambda theta: predict(theta) - y, np.array(init), bounds=(lower, upper), x_scale="jac", max_nfev=5000)
            sse = float(np.dot(fit.fun, fit.fun))
            if fit.success and np.isfinite(sse):
                outcomes.append((sse, fit, predict(fit.x)))
        except Exception:
            continue
    if not outcomes:
        return {"omitted_term": omit, "status": "failed"}
    sse, fit, predicted = min(outcomes, key=lambda row: row[0])
    values = dict(zip(["E"] + active, [float(value) for value in fit.x]))
    expanded = {name: values.get(name) for name in PARAMETER_NAMES}
    return {"omitted_term": omit, "status": "diagnostic_only", "parameters": expanded, "metrics": _metrics(y, predicted), "sse": sse, "successful_starts": len(outcomes), "total_starts": 8}


def _bootstrap(frame: pd.DataFrame, params: np.ndarray, seed: int, replicates: int) -> tuple[pd.DataFrame, dict]:
    clusters = sorted(frame["_cluster"].unique().tolist())
    grouped = {cluster: frame.loc[frame["_cluster"] == cluster].copy() for cluster in clusters}
    rng = np.random.default_rng(seed)
    estimates: list[dict] = []
    failures = 0
    progress_every = max(1, replicates // 10)
    for replicate in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[cluster] for cluster in sampled], ignore_index=True)
        successful = None
        # Warm-start plus two deterministic alternate starts limit compute while allowing recovery from a local basin.
        starts = [params]
        for variant in (3, 11):
            alt = _initial_params(sample["N_params_B"].to_numpy(dtype=float), sample["D_tokens_B"].to_numpy(dtype=float), sample["val_loss"].to_numpy(dtype=float), variant)
            starts.append(alt)
        for initial in starts:
            try:
                fit = _fit_one(sample["N_params_B"].to_numpy(dtype=float), sample["D_tokens_B"].to_numpy(dtype=float), sample["val_loss"].to_numpy(dtype=float), initial, max_nfev=2500)
                if fit.success and np.isfinite(fit.x).all() and np.isfinite(fit.fun).all():
                    if successful is None or np.dot(fit.fun, fit.fun) < successful[0]:
                        successful = (float(np.dot(fit.fun, fit.fun)), fit.x.copy())
            except Exception:
                continue
        if successful is None:
            failures += 1
        else:
            estimates.append({"replicate": replicate + 1, "sampled_cluster_sequence": ";".join(map(str, sampled)), **{name: float(value) for name, value in zip(PARAMETER_NAMES, successful[1])}, "sse": successful[0]})
        if (replicate + 1) % progress_every == 0 or replicate + 1 == replicates:
            print(f"Bootstrap progress: {replicate + 1}/{replicates}; successful={len(estimates)}", flush=True)
    estimate_frame = pd.DataFrame(estimates)
    intervals = {}
    if not estimate_frame.empty:
        for name in PARAMETER_NAMES:
            intervals[name] = {
                "estimate": float(params[PARAMETER_NAMES.index(name)]),
                "lower_95": float(np.percentile(estimate_frame[name], BOOTSTRAP_CI[0])),
                "upper_95": float(np.percentile(estimate_frame[name], BOOTSTRAP_CI[1])),
                "bootstrap_median": float(np.median(estimate_frame[name])),
            }
    else:
        for name in PARAMETER_NAMES:
            intervals[name] = {
                "estimate": float(params[PARAMETER_NAMES.index(name)]),
                "lower_95": None,
                "upper_95": None,
                "bootstrap_median": None,
            }
    summary = {
        "method": "cluster bootstrap by the eight B12-confirmed Pythia model_repo trajectories; sample eight clusters with replacement, retaining all rows within selected trajectories",
        "cluster_count": len(clusters),
        "requested_replicates": replicates,
        "successful_replicates": len(estimates),
        "failed_replicates": failures,
        "success_fraction": float(len(estimates) / replicates) if replicates else None,
        "interval_percentiles": list(BOOTSTRAP_CI),
        "intervals": intervals,
        "limitation": "Only eight independent trajectory clusters are available; percentile intervals are sensitive to the small number of clusters and are not high-precision population intervals.",
    }
    return estimate_frame, summary


def _leave_one_size_out(frame: pd.DataFrame, starts: int = 12) -> pd.DataFrame:
    rows = []
    sizes = sorted(frame["N_params_B"].unique())
    for index, size in enumerate(sizes):
        train = frame.loc[frame["N_params_B"] != size].copy()
        test = frame.loc[frame["N_params_B"] == size].copy()
        params, outcomes, _ = fit_multistart(train, starts=starts)
        prediction = _predict(params, test["N_params_B"].to_numpy(dtype=float), test["D_tokens_B"].to_numpy(dtype=float))
        metrics = _metrics(test["val_loss"].to_numpy(dtype=float), prediction)
        rows.append({
            "held_out_N_params_B": float(size),
            "held_out_rows": int(len(test)),
            "successful_starts": sum(bool(row["success"]) for row in outcomes),
            **{name: float(value) for name, value in zip(PARAMETER_NAMES, params)},
            **{f"holdout_{key}": value for key, value in metrics.items() if key != "n"},
        })
        print(f"Leave-one-size-out progress: {index + 1}/{len(sizes)}", flush=True)
    return pd.DataFrame(rows)


def _standardize_validation(frame: pd.DataFrame, dataset: str, source: str, strata: str) -> pd.DataFrame:
    result = frame.copy()
    result["dataset"] = dataset
    result["source_role"] = source
    if "source" in result.columns:
        result["literature_source"] = result["source"].astype(str)
    result["stratum"] = result[strata].astype(str) if strata in result.columns else "all"
    result["source_row"] = np.arange(1, len(result) + 1)
    result["observed_val_loss"] = pd.to_numeric(result["val_loss"], errors="coerce")
    result["predicted_val_loss"] = _predict(
        _ACTIVE_PARAMS,
        pd.to_numeric(result["N_params_B"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(result["D_tokens_B"], errors="coerce").to_numpy(dtype=float),
    )
    result["residual_pred_minus_obs"] = result["predicted_val_loss"] - result["observed_val_loss"]
    result["abs_error"] = result["residual_pred_minus_obs"].abs()
    return result


_ACTIVE_PARAMS: np.ndarray


def _validate_all(params: np.ndarray) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict]:
    global _ACTIVE_PARAMS
    _ACTIVE_PARAMS = params
    b2 = pd.read_csv(DATA_DIR / "cerebras_training_log.csv")
    b4 = pd.read_csv(DATA_DIR / "scaling_baseline.csv")
    b5 = pd.read_csv(DATA_DIR / "published_scaling_data.csv")
    sources: dict[str, pd.DataFrame] = {
        "B2": _standardize_validation(b2, "B2", "Cerebras semisynthetic family-out", "N_params_B"),
        "B4": _standardize_validation(b4, "B4", "real cross-family", "family"),
        "B5": _standardize_validation(b5, "B5", "published literature", "family"),
    }
    b3_rows = []
    for path in sorted((DATA_DIR / "training_trajectories").glob("*.csv")):
        frame = pd.read_csv(path)
        frame["trajectory"] = path.stem
        b3_rows.append(frame)
    b3 = pd.concat(b3_rows, ignore_index=True)
    sources["B3"] = _standardize_validation(b3, "B3", "interpolated Pythia trajectories", "trajectory")
    summary_rows = []
    strata_rows = []
    comparability = {
        "B2": "separate: semisynthetic; same field name does not establish independent real validation",
        "B3": "separate: interpolation from Pythia trajectories; not independent real validation",
        "B4": "separate: real cross-family values; CSV lacks evaluation protocol needed to establish direct Loss-scale comparability",
        "B5": "separate: literature values from multiple sources; CSV lacks common evaluation protocol",
    }
    for dataset in ("B2", "B3", "B4", "B5"):
        frame = sources[dataset]
        metrics = _metrics(frame["observed_val_loss"].to_numpy(dtype=float), frame["predicted_val_loss"].to_numpy(dtype=float))
        summary_rows.append({"dataset": dataset, "role": comparability[dataset], **metrics})
        for stratum, group in frame.groupby("stratum", dropna=False):
            strata_rows.append({"dataset": dataset, "stratum": str(stratum), **_metrics(group["observed_val_loss"].to_numpy(dtype=float), group["predicted_val_loss"].to_numpy(dtype=float))})
        if dataset == "B5":
            for source, group in frame.groupby("literature_source", dropna=False):
                strata_rows.append({"dataset": "B5_source", "stratum": str(source), **_metrics(group["observed_val_loss"].to_numpy(dtype=float), group["predicted_val_loss"].to_numpy(dtype=float))})
    validation_info = {"source_comparability": comparability, "pooled_validation_metric": None, "reason_no_pooled_metric": "The source roles and evaluation protocols differ or are undocumented; all B2-B5 metrics are reported by source only."}
    # Keep every source column beside the generated predictions so results remain
    # traceable to the original validation records (for example B2 run_id and B3
    # interpolated flags). The per-source files avoid a lossy cross-source schema.
    source_predictions = {key: value.copy() for key, value in sources.items()}
    return source_predictions, pd.DataFrame(summary_rows), pd.DataFrame(strata_rows), validation_info


def _load_figure_helpers():
    skill_root = Path.home() / ".codex" / "skills" / "math-modeling" / "tools" / "figure" / "scripts"
    if not skill_root.is_dir():
        raise RuntimeError(f"Figure skill helpers not found at {skill_root}")
    helpers = {}
    for name, filename in (("style", "setup_style.py"), ("export", "export_figure.py")):
        path = skill_root / filename
        spec = importlib.util.spec_from_file_location(f"q2_{name}_helper", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load figure helper: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helpers[name] = module
    helpers["helper_paths"] = {name: str(skill_root / filename) for name, filename in (("style", "setup_style.py"), ("export", "export_figure.py"))}
    return helpers


def _export_figure(fig, basename: Path, helpers: dict) -> None:
    helpers["export"].export_figure(fig, str(basename), formats=["svg", "png"], size_inches=tuple(float(value) for value in fig.get_size_inches()), dpi=300, grayscale_preview=False, tight=True)
    plt.close(fig)
    try:
        from PIL import Image
        gray_dir = FIGURE_PREVIEW_DIR / "grayscale"
        gray_dir.mkdir(parents=True, exist_ok=True)
        Image.open(str(basename) + ".png").convert("L").save(gray_dir / (basename.name + "_grayscale.png"), dpi=(300, 300))
    except ImportError:
        warnings.warn("Pillow is unavailable; grayscale previews were skipped.")


def _make_flowchart(helpers: dict) -> None:
    from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, label, face="#f4f6f8", round_box=False):
        if round_box:
            patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12", linewidth=1.2, edgecolor="#3b4652", facecolor=face)
        else:
            patch = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.08", linewidth=1.2, edgecolor="#3b4652", facecolor=face)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, wrap=True)

    def arrow(x1, y1, x2, y2, label=None, label_xy=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#3b4652"))
        if label and label_xy:
            ax.text(label_xy[0], label_xy[1], label, ha="center", va="center", fontsize=8)

    box(0.4, 5.35, 2.1, 0.85, "Inputs: B1-B5 + B12", "#edf3f8", True)
    box(3.3, 5.35, 2.2, 0.85, "Gate 0: roles, units,\nrows, checkpoint map")
    diamond = Polygon([[7.0, 5.78], [8.2, 6.42], [9.4, 5.78], [8.2, 5.14]], closed=True, edgecolor="#3b4652", facecolor="#fff7e6", linewidth=1.2)
    ax.add_patch(diamond)
    ax.text(8.2, 5.78, "Gate 0\npassed?", ha="center", va="center", fontsize=9)
    box(10.0, 5.35, 1.7, 0.85, "Stop and save\ndiagnostics", "#fbeeee", True)
    box(3.3, 3.7, 2.2, 0.85, "Fit B1: bounded\nfixed multi-start LS")
    box(6.2, 3.7, 2.3, 0.85, "Check residuals,\nterms, Jacobian")
    box(9.2, 3.7, 2.4, 0.85, "Bootstrap 8 tracks;\nleave one size out")
    box(3.3, 1.95, 2.2, 0.85, "Predict B2-B5\nwithout refitting")
    box(6.2, 1.95, 2.3, 0.85, "Report each source\nand strata separately")
    box(9.2, 1.95, 2.4, 0.85, "Freeze classical\nreference + limits", "#edf3f8", True)
    arrow(2.5, 5.78, 3.3, 5.78)
    arrow(5.5, 5.78, 7.0, 5.78)
    arrow(9.4, 5.78, 10.0, 5.78, "no", (9.7, 6.08))
    arrow(8.2, 5.14, 4.4, 4.55, "yes", (6.0, 4.8))
    arrow(5.5, 4.12, 6.2, 4.12)
    arrow(8.5, 4.12, 9.2, 4.12)
    arrow(10.4, 3.7, 4.4, 2.8)
    arrow(5.5, 2.38, 6.2, 2.38)
    arrow(8.5, 2.38, 9.2, 2.38)
    ax.set_title("Q2 classical baseline workflow", fontsize=12, pad=12)
    ax.text(6, 0.85, "B6-B8 (Q) and B9-B10 (large-scale extrapolation) remain outside this baseline.", ha="center", fontsize=8, color="#4b5563")
    _export_figure(fig, FIGURE_DIR / "flow_q2_model", helpers)


def make_figures(b1_raw: pd.DataFrame, b1_predictions: pd.DataFrame, starts: pd.DataFrame, bootstrap: pd.DataFrame, validation: dict[str, pd.DataFrame], loo: pd.DataFrame, params: np.ndarray) -> tuple[list[dict], dict]:
    helpers = _load_figure_helpers()
    style_info = helpers["style"].setup_style(journal="general", lang="en", use_sciplots=False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    contracts: list[dict] = []

    def save(fig, name: str, category: str, claim: str, chart_type: str, sources: str, caveat: str = ""):
        _export_figure(fig, FIGURE_DIR / name, helpers)
        contracts.append({"figure": name, "category": category, "claim": claim, "chart_type": chart_type, "data_sources": sources, "caveat": caveat})

    n = b1_raw["N_params_B"].to_numpy(dtype=float)
    d = b1_raw["D_tokens_B"].to_numpy(dtype=float)
    y = b1_raw["val_loss"].to_numpy(dtype=float)

    fig, ax = plt.subplots()
    ax.hist(n, bins=np.geomspace(n.min(), n.max(), 13), color="#0072B2", edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions, log scale)")
    ax.set_ylabel("Checkpoint rows")
    ax.set_title("B1 parameter-size coverage")
    ax.grid(axis="y", alpha=0.2)
    save(fig, "raw_q2_parameter_coverage", "raw", "Show the eight observed Pythia model sizes and their repeated checkpoints.", "log-scale histogram", "B1", "Rows are repeated checkpoints within eight model trajectories.")

    fig, ax = plt.subplots()
    sorted_d = np.sort(d)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.step(sorted_d, cdf, where="post", color="#009E73", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens (billions, log scale)")
    ax.set_ylabel("Empirical cumulative proportion")
    ax.set_title("B1 token coverage")
    ax.grid(alpha=0.2)
    save(fig, "raw_q2_token_ecdf", "raw", "Show how B1 checkpoints cover the training-token range.", "empirical CDF", "B1", "The checkpoint CDF is not an independent-sample distribution.")

    fig, ax = plt.subplots()
    scatter = ax.scatter(n, d, c=y, cmap="viridis", s=18, alpha=0.68, edgecolors="none", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (billions, log scale)")
    ax.set_ylabel("Training tokens (billions, log scale)")
    ax.set_title("B1 crossed N-D support")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Observed validation loss")
    save(fig, "raw_q2_N_D_support", "raw", "Show joint N-D support and the observed loss gradient.", "colored scatter", "B1")

    successful_starts = starts.loc[starts["success"]].sort_values("cost_sse").copy()
    best_sse = float(successful_starts["cost_sse"].min())
    successful_starts["delta_sse_1e18"] = (successful_starts["cost_sse"] - best_sse) * 1e18
    fig, ax = plt.subplots()
    ax.scatter(np.arange(1, len(successful_starts) + 1), successful_starts["delta_sse_1e18"], color="#0072B2", s=22, alpha=0.8)
    ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
    ax.set_xlabel("Successful start, sorted by SSE")
    ax.set_ylabel(r"SSE above minimum ($\times 10^{-18}$)")
    ax.set_title("Bounded multi-start fit outcomes")
    ax.grid(alpha=0.2)
    save(fig, "process_q2_multistart_cost", "process", "Check whether fixed initial values converge to comparable training minima.", "ranked point plot", "B1 fit starts")

    fig, ax = plt.subplots()
    ax.scatter(b1_predictions["predicted_val_loss"], b1_predictions["residual_pred_minus_obs"], c=b1_predictions["N_params_B"], cmap="plasma", s=14, alpha=0.55, rasterized=True)
    ax.axhline(0, color="#222222", linewidth=1, linestyle="--")
    ax.set_xlabel("Fitted validation loss")
    ax.set_ylabel("Residual (prediction - observation)")
    ax.set_title("B1 residual structure")
    ax.grid(alpha=0.2)
    save(fig, "process_q2_residual_structure", "process", "Reveal residual bias or curvature across the fitted loss range.", "residual scatter", "B1")

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.4), constrained_layout=True)
    if len(bootstrap):
        axes_flat = list(axes.flat)
        for ax, name, estimate in zip(axes_flat, PARAMETER_NAMES, params):
            values = bootstrap[name].to_numpy(dtype=float)
            relative = 100 * (values / estimate - 1)
            low, high = np.percentile(relative, BOOTSTRAP_CI)
            ax.hlines(0, low, high, color="#0072B2", linewidth=2)
            ax.plot([low, high], [0, 0], marker="|", linestyle="none", color="#0072B2", markersize=8)
            ax.scatter([0], [0], color="#D55E00", s=24, zorder=3, label="B1 estimate")
            ax.set_title(name)
            ax.set_yticks([])
            ax.set_xlabel("Change from B1 fit (%)")
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
            ax.grid(axis="x", alpha=0.2)
        axes_flat[-1].axis("off")
    else:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No successful\nbootstrap fits", ha="center", va="center")
            ax.set_axis_off()
    fig.suptitle("Trajectory-cluster bootstrap intervals (95%)")
    save(fig, "process_q2_cluster_bootstrap", "process", "Display percentile intervals for parameter percentage changes from the B1 estimate across resampled complete model trajectories.", "five-parameter relative interval panels", "B1 cluster bootstrap", "Eight clusters limit interval reliability; each parameter panel has its own horizontal scale.")

    fig, ax = plt.subplots()
    ax.scatter(b1_predictions["observed_val_loss"], b1_predictions["predicted_val_loss"], c=b1_predictions["N_params_B"], cmap="viridis", s=17, alpha=0.65, rasterized=True)
    low = min(float(b1_predictions["observed_val_loss"].min()), float(b1_predictions["predicted_val_loss"].min()))
    high = max(float(b1_predictions["observed_val_loss"].max()), float(b1_predictions["predicted_val_loss"].max()))
    ax.plot([low, high], [low, high], color="#D55E00", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed B1 validation loss")
    ax.set_ylabel("Fitted B1 validation loss")
    ax.set_title("B1 fitted reference model")
    ax.grid(alpha=0.2)
    save(fig, "result_q2_B1_observed_vs_fitted", "result", "Show B1 fitted values against observations for the frozen classic form.", "parity scatter", "B1", "In-sample fit only; not an external validation result.")

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    validation_titles = {
        "B2": "B2: semisynthetic family-out",
        "B3": "B3: interpolated Pythia paths",
        "B4": "B4: real cross-family",
        "B5": "B5: published literature",
    }
    for ax, key in zip(axes.flat, ("B2", "B3", "B4", "B5")):
        frame = validation[key]
        obs = frame["observed_val_loss"].to_numpy(dtype=float)
        pred = frame["predicted_val_loss"].to_numpy(dtype=float)
        ax.scatter(obs, pred, color="#0072B2" if key in {"B2", "B3"} else "#E69F00", s=13 if len(frame) > 100 else 24, alpha=0.55, rasterized=len(frame) > 100)
        lo, hi = min(float(obs.min()), float(pred.min())), max(float(obs.max()), float(pred.max()))
        ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1)
        ax.set_title(validation_titles[key], fontsize=9)
        ax.set_xlabel("Observed loss")
        ax.set_ylabel("B1 model prediction")
        ax.grid(alpha=0.18)
    fig.suptitle("Held-out source checks (panels use source-specific scales)")
    save(fig, "result_q2_validation_by_source", "result", "Show predictions for each held-out source without pooling or retuning.", "four-panel parity scatter", "B2, B3, B4, B5", "B2/B3 are not independent real validation; B4/B5 evaluation comparability is undocumented; no pooled metric is reported.")

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.0), constrained_layout=True)
    axes_list = list(axes.flat)
    for ax, name in zip(axes_list, PARAMETER_NAMES):
        estimate = params[PARAMETER_NAMES.index(name)]
        relative_change = 100 * (loo[name].to_numpy(dtype=float) / estimate - 1)
        ax.plot(np.arange(len(loo)), relative_change, marker="o", color="#0072B2", linewidth=1.2)
        ax.axhline(0, color="#D55E00", linestyle="--", linewidth=1, label="Full B1 fit")
        ax.set_xticks(np.arange(len(loo)))
        ax.set_xticklabels([f"{value:.3g}" for value in loo["held_out_N_params_B"]], rotation=40, ha="right", fontsize=7)
        ax.set_xlabel("Omitted N (billions)")
        ax.set_ylabel("Change from B1 fit (%)")
        ax.set_title(name)
        ax.grid(alpha=0.18)
    axes_list[-1].axis("off")
    axes_list[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Leave-one-model-size-out parameter sensitivity")
    save(fig, "result_q2_leave_one_size_out", "result", "Show the percentage change in each fitted parameter when one entire model-size trajectory is excluded.", "small-multiple relative parameter paths", "B1 leave-one-size-out", "Each omitted size is a whole 147-checkpoint trajectory; the horizontal zero line is the full B1 fit.")

    _make_flowchart(helpers)
    _write_csv(OUTPUT_DIR / "figure_contract.csv", pd.DataFrame(contracts))
    return contracts, {"style": style_info, "helpers": helpers["helper_paths"]}


def _save_validation_outputs(validation: dict[str, pd.DataFrame], summary: pd.DataFrame, strata: pd.DataFrame) -> None:
    for dataset, frame in validation.items():
        _write_csv(OUTPUT_DIR / f"{dataset.lower()}_predictions.csv", frame)
    _write_csv(OUTPUT_DIR / "validation_metrics_by_source.csv", summary)
    _write_csv(OUTPUT_DIR / "validation_metrics_by_stratum.csv", strata)


def _build_report(audit: dict, params: np.ndarray, metrics: dict, diagnostics: dict, bootstrap_summary: dict, loo: pd.DataFrame, validation_metrics: pd.DataFrame, status: str, seed: int) -> str:
    parameter_lines = "\n".join(f"- `{name}` = {value:.10g}" for name, value in zip(PARAMETER_NAMES, params))
    def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
        selected = frame[columns] if columns else frame
        headers = [str(column) for column in selected.columns]
        values = [[("" if pd.isna(value) else f"{value:.5g}" if isinstance(value, (int, float, np.number)) else str(value)) for value in row] for row in selected.itertuples(index=False, name=None)]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in values)
        return "\n".join(lines)
    validation_table = markdown_table(validation_metrics) if len(validation_metrics) else "No validation results."
    loo_rmse = markdown_table(loo, ["held_out_N_params_B", "holdout_rmse", "holdout_mae"])
    audit_passes = sum(bool(check["passed"]) for check in audit["checks"])
    return f"""# Q2 经典 Scaling Law 基线运行报告

## 状态

**{status}**。此产物冻结的是经典参考支路 `Loss=f(N,D)`，不代表完整 Q2 已完成。Q2 后续仍需使用正式冻结的质量分数 Q 与领域配比 p 扩展模型。

Gate 0：{audit_passes}/{len(audit['checks'])} 项通过；B1 通过 B12 映射到 8 条 Pythia 模型轨迹，每条 147 个检查点。`run_id` 每行唯一，未用作 cluster。

## 模型与拟合

拟合形式：

`{MODEL_FORMULA}`

只在 B1 上做有界非线性最小二乘。参数约束为 `0 <= E < min(B1 val_loss)` 且 `A, alpha, B, beta > 0`；固定多初值拟合 {diagnostics['multistart']['attempt_count']} 次。

{parameter_lines}

训练指标：RMSE={metrics['rmse']:.7g}，MAE={metrics['mae']:.7g}，R²={metrics['r2']:.7g}，Spearman={metrics['spearman']:.7g}。这些是 B1 拟合内指标。

优化诊断：成功初值 {diagnostics['multistart']['success_count']}/{diagnostics['multistart']['attempt_count']}；近最优解数量 {diagnostics['multistart']['near_optimal_count']}；缩放 Jacobian 秩 {diagnostics['jacobian']['scaled_jacobian_rank']}/{diagnostics['jacobian']['parameter_count']}，条件数 {diagnostics['jacobian']['scaled_jacobian_condition_number']}。可辨识性状态：`{'存在警告' if diagnostics['identifiability_warning'] else '未触发预设警告'}`。详细初值结果和删项诊断分别见 `multistart_fits.csv`、`dropped_term_diagnostics.json`。

## 轨迹 Bootstrap 与留一规模分析

对 B1 的 8 个完整模型轨迹做 {bootstrap_summary['requested_replicates']} 次 cluster Bootstrap，成功 {bootstrap_summary['successful_replicates']} 次、失败 {bootstrap_summary['failed_replicates']} 次。区间见 `cluster_bootstrap_intervals.json` 和 `cluster_bootstrap_estimates.csv`。**只有 8 个 cluster，区间对少量轨迹高度敏感，应视作稳定性提示而非高精度总体区间。**

留一参数规模敏感性（每次整条 147 点轨迹留出）：

{loo_rmse}

## B2–B5 留出验证

基线在读取验证误差之前只由 B1 拟合。验证集没有回流拟合或调参，且不汇总成一个跨来源分数。B2 是半合成 Cerebras 轨迹；B3 是 Pythia 插值轨迹；它们不作为独立真实验证。B4 是真实跨族数据；B5 是多来源文献数据，但这两个 CSV 不提供统一 tokenizer/评测集协议，Loss 量尺可比性未被证实。

{validation_table}

按参数规模、轨迹、模型族和 B5 文献来源的分层指标见 `validation_metrics_by_stratum.csv`。预测逐点见 `b2_predictions.csv` 至 `b5_predictions.csv`。

## 结论边界

- B6–B8 含 Q 的半合成补充数据不进入经典基线；不估计 Q、p、质量—规模替代条件或 Q3 配置。
- B9–B10 不进入拟合，只留给未来大尺度外推分析。
- B2/B3 支持合成迁移/轨迹形状检查，不支持独立真实外部验证结论。
- B4/B5 的误差分源报告；没有统一 Loss 量尺证据，不合并评分。
- B1 只有八条模型轨迹；残差行数 1,176 不能解释为 1,176 个独立训练实验。

## 复现

唯一复现命令（使用本次验证通过的解释器）：

```powershell
"{sys.executable}" src/f_q2_scaling_baseline.py --mode full --seed {seed}
```

本次实际运行环境：Python {platform.python_version()}，解释器 `{sys.executable}`。当前主机裸命令 `python` 会解析到独立 Python 3.10；其 SciPy 线性代数调用曾触发 Windows 异常 `0xc06d007f`。复现时请使用上面的解释器路径，或先切换到已验证可工作的 Python/SciPy 环境。输入 SHA-256、Python/依赖版本和产物清单见 `复现清单.json`。原始附件保持只读。
"""


def _make_contact_sheet() -> None:
    from matplotlib.image import imread
    paths = sorted(path for path in FIGURE_DIR.glob("*.png") if not path.name.endswith("_grayscale.png"))
    if not paths:
        return
    columns = 2
    rows = math.ceil(len(paths) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(9, rows * 2.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for index, path in enumerate(paths):
        ax = axes.flat[index]
        ax.imshow(imread(path))
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    FIGURE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PREVIEW_DIR / "contact_sheet.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_full(seed: int, replicates: int) -> int:
    started = time.time()
    audit = run_gate0()
    allowed, snapshot_note = _freeze_or_compare_snapshot(audit, "full")
    print(f"Gate 0: {audit['status']} ({snapshot_note})", flush=True)
    if not allowed:
        print("Stopped before fitting: " + snapshot_note, file=sys.stderr)
        return 2

    b1 = pd.read_csv(DATA_DIR / "pythia_training_log_existing.csv")
    mapping = {row["N_params_B"]: row["cluster_key"] for row in audit["b1_trajectory_mapping"]}
    b1["_cluster"] = b1["N_params_B"].map(mapping)
    if b1["_cluster"].isna().any() or b1["_cluster"].nunique() != 8:
        print("Stopped: B1 cluster mapping is not complete and one-to-one.", file=sys.stderr)
        return 2

    print("Fitting B1 using bounded multi-start nonlinear least squares...", flush=True)
    try:
        params, start_rows, best_fit = fit_multistart(b1, starts=24)
    except RuntimeError as exc:
        failure = {"status": "数据不足", "reason": str(exc), "gate0_status": audit["status"], "fit_scope": "B1 only", "seed": seed}
        _write_json(OUTPUT_DIR / "fit_failure.json", failure)
        _write_json(OUTPUT_DIR / "fit_summary.json", failure)
        (OUTPUT_DIR / "q2_scaling_baseline_report.md").write_text("# Q2 经典 Scaling Law 基线\n\n**数据不足。** Gate 0 通过，但固定多初值有界最小二乘未能产生任何有效解。未运行 Bootstrap、验证或正式图表。详见 `fit_failure.json`。\n", encoding="utf-8")
        return 2
    if not np.isfinite(params).all() or not (0 <= params[0] < b1["val_loss"].min()) or not all(params[index] > 0 for index in (1, 2, 3, 4)):
        print("Stopped: fitted parameters violate the predeclared constraints.", file=sys.stderr)
        return 2

    predicted = _predict(params, b1["N_params_B"].to_numpy(dtype=float), b1["D_tokens_B"].to_numpy(dtype=float))
    if not np.isfinite(predicted).all():
        print("Stopped: B1 predictions are not all finite.", file=sys.stderr)
        return 2
    b1_predictions = b1.drop(columns=["_cluster"]).copy()
    b1_predictions["observed_val_loss"] = b1_predictions["val_loss"].to_numpy(dtype=float)
    b1_predictions["predicted_val_loss"] = predicted
    b1_predictions["residual_pred_minus_obs"] = predicted - b1_predictions["val_loss"].to_numpy(dtype=float)
    b1_predictions["abs_error"] = np.abs(b1_predictions["residual_pred_minus_obs"])
    _write_csv(OUTPUT_DIR / "b1_fitted_predictions.csv", b1_predictions)
    starts_frame = pd.DataFrame(start_rows)
    _write_csv(OUTPUT_DIR / "multistart_fits.csv", starts_frame)

    metrics = _metrics(b1["val_loss"].to_numpy(dtype=float), predicted)
    jacobian = _scaled_jacobian_diagnostics(best_fit, params)
    multistart = _multistart_diagnostics(start_rows, params)
    boundary = {
        "E_at_zero_or_near": bool(params[0] <= max(1e-8, b1["val_loss"].min() * 1e-6)),
        "E_at_upper_or_near": bool(b1["val_loss"].min() - params[0] <= max(1e-8, b1["val_loss"].min() * 1e-6)),
        "alpha_near_lower": bool(params[2] <= 1e-5),
        "beta_near_lower": bool(params[4] <= 1e-5),
    }
    identifiability_warning = bool(jacobian["rank_deficient"] or jacobian["condition_warning"] or multistart["unstable"] or boundary["E_at_upper_or_near"] or boundary["alpha_near_lower"] or boundary["beta_near_lower"])
    diagnostics = {"jacobian": jacobian, "multistart": multistart, "boundary": boundary, "identifiability_warning": identifiability_warning}
    _write_json(OUTPUT_DIR / "fit_parameters.json", {"formula": MODEL_FORMULA, "parameters": dict(zip(PARAMETER_NAMES, map(float, params))), "constraints": {"E": "0 <= E < min(B1 val_loss)", "A_alpha_B_beta": "strictly positive"}, "fit_scope": "B1 only", "metrics": metrics, "diagnostics": diagnostics})

    print("Running term-deletion diagnostics and leave-one-size-out fits...", flush=True)
    dropped = [_fit_reduced(b1, "N"), _fit_reduced(b1, "D")]
    _write_json(OUTPUT_DIR / "dropped_term_diagnostics.json", {"scope": "diagnostic only; not a model search", "results": dropped})
    loo = _leave_one_size_out(b1, starts=12)
    _write_csv(OUTPUT_DIR / "leave_one_size_out.csv", loo)

    print(f"Running {replicates} trajectory-cluster bootstrap replicates...", flush=True)
    bootstrap_estimates, bootstrap_summary = _bootstrap(b1, params, seed, replicates)
    _write_csv(OUTPUT_DIR / "cluster_bootstrap_estimates.csv", bootstrap_estimates)
    _write_json(OUTPUT_DIR / "cluster_bootstrap_intervals.json", bootstrap_summary)
    interval_rows = [{"parameter": name, **values} for name, values in bootstrap_summary["intervals"].items()]
    _write_csv(OUTPUT_DIR / "cluster_bootstrap_intervals.csv", pd.DataFrame(interval_rows))

    print("Scoring held-out B2-B5 data without refitting...", flush=True)
    validation, validation_summary, validation_strata, validation_info = _validate_all(params)
    all_predictions_finite = all(_finite_series(frame["predicted_val_loss"]) for frame in validation.values())
    _save_validation_outputs(validation, validation_summary, validation_strata)
    _write_json(OUTPUT_DIR / "validation_comparability.json", validation_info)
    if not all_predictions_finite:
        print("Stopped: one or more validation predictions are non-finite.", file=sys.stderr)
        return 2

    if bootstrap_summary["successful_replicates"] == 0:
        status = "数据不足（轨迹 Bootstrap 未产生有效拟合区间）"
    elif identifiability_warning:
        status = "基线已拟合、验证受限（参数可辨识性诊断触发警告）"
    elif validation_info["source_comparability"] and validation_summary["dataset"].isin(["B4", "B5"]).any():
        status = "基线已拟合、验证受限（B2/B3 非独立真实验证；B4/B5 Loss 可比性未证实）"
    else:
        status = "基线已拟合、验证受限"

    contract_rows, figure_style = make_figures(b1, b1_predictions, starts_frame, bootstrap_estimates, validation, loo, params)
    figure_counts = {category: sum(row["category"] == category for row in contract_rows) for category in ("raw", "process", "result")}
    if any(count < 3 for count in figure_counts.values()):
        raise RuntimeError(f"Figure contract incomplete: {figure_counts}")
    _make_contact_sheet()

    _write_json(OUTPUT_DIR / "fit_summary.json", {
        "status": status,
        "formula": MODEL_FORMULA,
        "parameters": dict(zip(PARAMETER_NAMES, map(float, params))),
        "B1_metrics": metrics,
        "diagnostics": diagnostics,
        "bootstrap": bootstrap_summary,
        "validation": validation_summary.to_dict(orient="records"),
        "validation_source_comparability": validation_info,
        "validation_predictions_all_finite": all_predictions_finite,
        "figure_counts": figure_counts,
        "figure_style": figure_style["style"],
        "elapsed_seconds_before_report": time.time() - started,
        "seed": seed,
    })
    report = _build_report(audit, params, metrics, diagnostics, bootstrap_summary, loo, validation_summary, status, seed)
    (OUTPUT_DIR / "q2_scaling_baseline_report.md").write_text(report, encoding="utf-8")

    outputs = sorted(path for path in OUTPUT_DIR.rglob("*") if path.is_file() and path.name != "复现清单.json")
    outputs.extend(sorted(path for path in FIGURE_DIR.iterdir() if path.is_file()))
    manifest = {
        "task": "Q2 classical scaling-law reference baseline",
        "status": status,
        "reproduction_command": f'"{sys.executable}" src/f_q2_scaling_baseline.py --mode full --seed {seed}',
        "mode": "full",
        "seed": seed,
        "bootstrap_replicates": replicates,
        "role_contract_version": ROLE_CONTRACT_VERSION,
        "input_sha256": audit["input_sha256"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "executable": sys.executable, "runtime_note": "The standalone Python 3.10 interpreter on this workstation is known to fail inside SciPy native linear algebra with Windows exception 0xc06d007f; reproduce with this manifest's executable or another verified Python/SciPy runtime."},
        "figure_helpers": figure_style["helpers"],
        "figure_style": figure_style["style"],
        "figure_contract": contract_rows,
        "figure_counts": figure_counts,
        "outputs_sha256": {_rel(path): _sha256(path) for path in outputs if path.is_file()},
        "limits": ["B1 has only eight trajectory clusters.", "B2/B3 are not independent real validations.", "B4/B5 evaluation protocol comparability is undocumented.", "This is not the generalized Q2 model with Q and p."],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(OUTPUT_DIR / "复现清单.json", manifest)
    print(f"Full run complete: {status}", flush=True)
    print(f"Results: {_rel(OUTPUT_DIR)}", flush=True)
    print(f"Figures: {_rel(FIGURE_DIR)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and fit the Q2 classical Loss=f(N,D) baseline")
    parser.add_argument("--mode", choices=("audit", "full"), required=True)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES, help="For review/debug runs; the reproducible production run uses 1000")
    parser.add_argument("--refresh-input-snapshot", action="store_true", help="Audit mode only: explicitly accept and freeze reviewed input hash/role changes")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    if args.refresh_input_snapshot and args.mode != "audit":
        parser.error("--refresh-input-snapshot is only valid with --mode audit")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        audit = run_gate0()
        allowed, note = _freeze_or_compare_snapshot(audit, "audit", refresh_snapshot=args.refresh_input_snapshot)
        print(json.dumps({"gate0_status": audit["status"], "snapshot": note, "checks": audit["checks"]}, ensure_ascii=False, indent=2))
        return 0 if allowed else 2
    return run_full(args.seed, args.bootstrap_replicates)


if __name__ == "__main__":
    raise SystemExit(main())
