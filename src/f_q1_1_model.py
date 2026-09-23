#!/usr/bin/env python
"""Q1.1 quality scoring on the frozen A1/A2/A3 quality-signal tables.

This is a Q1.1-only implementation. It reads normalized scalar indicators and
audit flags; it never interprets or executes text from the problem statement or
from corpus records. Raw A1 text is opened only when building the blinded
human-review packet.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import lzma
import math
import platform
import random
import statistics
import sys
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "results" / "q1_common_preprocess" / "v1"
CONTRACT = ROOT / "results" / "q1_1" / "q1_1_input_contract.csv"
AUDIT_DIR = ROOT / "results" / "q1_1"
Q12_DIR = ROOT / "results" / "q1_2_conflict_model" / "v1"
Q12_STRICT = ROOT / "results" / "q1_2_conflict_model" / "contamination_sensitivity_official_strict"
Q12_BROAD = ROOT / "results" / "q1_2_conflict_model" / "contamination_sensitivity_official_broad"
DEFAULT_OUTPUT = ROOT / "results" / "q1_1" / "v1"
DATASETS = ("a1", "a2", "a3")
SEED = 20260923
BOOTSTRAP_STABILITY = 1000
BOOTSTRAP_DOMAIN = 2000
BOOTSTRAP_HUMAN = 2000
BOOTSTRAP_CI = 0.95
HISTOGRAM_BINS = 5000
UNRESOLVED_UNIT_FIELDS = (
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
)


@dataclass
class Dataset:
    key: str
    ids: np.ndarray
    domains: np.ndarray
    sub_paths: np.ndarray
    scores: np.ndarray
    outlier: np.ndarray
    unit_suspect: np.ndarray
    row_order: np.ndarray
    input_rows: int
    scanned_domain_counts: Dict[str, int]
    narrow_unicode: np.ndarray
    broad_unicode: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, label: str) -> int:
    offset = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + offset) % (2**32 - 1)


def parse_float(value: Optional[str]) -> float:
    if value is None or not str(value).strip():
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def parse_bool(value: Optional[str]) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def quantile(values: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q, method="linear")) if len(x) else float("nan")


def robust_scale(values: np.ndarray) -> Tuple[float, str]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 0.0, "no_finite_values"
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    if mad > 1e-12:
        return mad, "1.4826*MAD"
    iqr = float((np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")) / 1.349)
    if iqr > 1e-12:
        return iqr, "IQR/1.349"
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return (sd, "SD") if sd > 1e-12 else (0.0, "zero_scale")


def derive_delta(scores: np.ndarray) -> Tuple[float, float, str]:
    with np.errstate(all="ignore"):
        row_median = np.nanmedian(scores, axis=1)
    residuals = (scores - row_median[:, None]).ravel()
    residuals = residuals[np.isfinite(residuals)]
    scale, method = robust_scale(residuals)
    return 1.345 * scale, scale, method


def rank_average(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    rx = rank_average(np.asarray(x, dtype=float)[mask])
    ry = rank_average(np.asarray(y, dtype=float)[mask])
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def domain_groups(domains: np.ndarray) -> Dict[str, np.ndarray]:
    return {str(name): np.flatnonzero(domains == name) for name in sorted(set(domains.tolist()))}


def derive_weights(
    scores: np.ndarray,
    domains: np.ndarray,
    indicators: Sequence[str],
    repetitions: int,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """A1-only missingness × domain-rank stability weights from the approved plan."""
    groups = domain_groups(domains)
    full = np.vstack([np.nanmedian(scores[idx, :], axis=0) for idx in groups.values()])
    completeness = np.mean(np.isfinite(scores), axis=0)
    stability = np.full(len(indicators), 0.5, dtype=float)
    valid = np.zeros(len(indicators), dtype=int)
    undefined = np.zeros(len(indicators), dtype=int)
    fallback = ["insufficient_bootstrap_repetitions"] * len(indicators)
    if repetitions >= 2:
        rng = np.random.default_rng(seed)
        draws = np.full((repetitions, len(indicators)), np.nan)
        for b in range(repetitions):
            boot = []
            for indices in groups.values():
                chosen = indices[rng.integers(0, len(indices), size=len(indices))]
                boot.append(np.nanmedian(scores[chosen, :], axis=0))
            boot = np.vstack(boot)
            for j in range(len(indicators)):
                fv = full[:, j]
                if np.isfinite(fv).sum() < 3 or np.ptp(fv[np.isfinite(fv)]) == 0:
                    undefined[j] += 1
                    continue
                rho = spearman(full[:, j], boot[:, j])
                if math.isfinite(rho):
                    draws[b, j] = (1.0 + rho) / 2.0
                    valid[j] += 1
                else:
                    undefined[j] += 1
        for j in range(len(indicators)):
            fv = full[:, j]
            finite = fv[np.isfinite(fv)]
            if len(finite) < 3 or np.ptp(finite) == 0:
                stability[j] = 0.5
                fallback[j] = "constant_or_insufficient_a1_domain_medians"
            elif valid[j] / repetitions < 0.95:
                stability[j] = 0.5
                fallback[j] = "valid_bootstrap_rho_below_95_percent"
            else:
                stability[j] = float(np.nanmedian(draws[:, j]))
                fallback[j] = "none"
    raw = np.sqrt(np.clip(completeness, 0.0, 1.0) * np.clip(stability, 0.0, 1.0))
    weights = raw / raw.sum() if raw.sum() > 0 else np.full(len(indicators), 1.0 / len(indicators))
    rows = []
    for j, name in enumerate(indicators):
        rows.append({
            "indicator": name,
            "a1_completeness": float(completeness[j]),
            "domain_rank_stability": float(stability[j]),
            "valid_bootstrap_rho_count": int(valid[j]),
            "undefined_bootstrap_rho_count": int(undefined[j]),
            "stability_fallback_reason": fallback[j],
            "weight": float(weights[j]),
        })
    return weights, rows


def huber_sample_scores(scores: np.ndarray, weights: np.ndarray, delta: float) -> np.ndarray:
    valid = np.isfinite(scores)
    counts = valid.sum(axis=1)
    low = np.zeros(scores.shape[0], dtype=float)
    high = np.ones(scores.shape[0], dtype=float)
    if delta <= 1e-12:
        w = np.where(valid, scores, 0.0) @ weights
        m = valid @ weights
        result = np.full(scores.shape[0], np.nan)
        ok = m > 0
        result[ok] = w[ok] / m[ok]
        return result
    for _ in range(48):
        mid = (low + high) / 2.0
        residual = np.where(valid, scores - mid[:, None], 0.0)
        psi = np.clip(residual, -delta, delta)
        root_score = np.sum(psi * weights[None, :], axis=1)
        move_low = root_score > 0
        low[move_low] = mid[move_low]
        high[~move_low] = mid[~move_low]
    result = (low + high) / 2.0
    result[counts == 0] = np.nan
    return result


def equal_scores(scores: np.ndarray) -> np.ndarray:
    valid = np.isfinite(scores)
    count = valid.sum(axis=1)
    out = np.full(scores.shape[0], np.nan)
    ok = count > 0
    out[ok] = np.nansum(scores[ok], axis=1) / count[ok]
    return out


def weighted_mean_scores(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(scores)
    mass = valid @ weights
    num = np.where(valid, scores, 0.0) @ weights
    out = np.full(scores.shape[0], np.nan)
    ok = mass > 0
    out[ok] = num[ok] / mass[ok]
    return out


def huber_location(values: np.ndarray, delta: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    if delta <= 1e-12:
        return float(np.median(x))
    lo, hi = 0.0, 1.0
    for _ in range(48):
        mid = (lo + hi) / 2.0
        score = float(np.clip(x - mid, -delta, delta).sum())
        if score > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def read_contract() -> Tuple[List[str], Dict[str, str]]:
    with CONTRACT.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["indicator"] for row in rows]
    norm = {row["indicator"]: row["norm_field"] for row in rows}
    if len(names) != 22 or len(set(names)) != 22:
        raise ValueError("Q1.1 输入合同应含 22 个互异指标")
    if not set(UNRESOLVED_UNIT_FIELDS).issubset(names):
        raise ValueError("合同中缺少计划标记为单位待核验的字段")
    return names, norm


def read_integrity_audit() -> Tuple[dict, dict, dict, dict]:
    summary_path = AUDIT_DIR / "text_integrity_summary.json"
    audit_path = AUDIT_DIR / "text_integrity_audit.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual = sha256_file(audit_path)
    if actual != summary.get("record_audit_sha256"):
        raise ValueError("原文 Unicode 逐记录审计表哈希与 summary 不符")
    with audit_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    a1 = [row for row in rows if row["dataset"] == "A1"]
    if len(a1) != 51230:
        raise ValueError(f"A1 Unicode 审计记录数异常：{len(a1)}")
    broad: Dict[str, bool] = {}
    narrow: Dict[str, bool] = {}
    content_hash: Dict[str, str] = {}
    for row in a1:
        key = row["sample_key_sha256"]
        if key in broad:
            raise ValueError("Unicode 审计样本键重复")
        broad[key] = parse_bool(row["unicode_integrity_flag"])
        narrow[key] = int(row["format_char_count"] or 0) > 0 or int(row["other_control_count"] or 0) > 0
        content_hash[key] = row["content_sha256"]
    if sum(narrow.values()) != 428 or sum(broad.values()) != 463:
        raise ValueError("当前严格/宽 Unicode 规则命中数与冻结审计基线不符")
    return summary, broad, narrow, content_hash


def identity_hash(dataset: str, sample_id: str) -> str:
    return hashlib.sha256(f"{dataset}\0{sample_id if sample_id is not None else ''}".encode("utf-8")).hexdigest()


def read_dataset(
    key: str,
    indicators: Sequence[str],
    norm_fields: Mapping[str, str],
    expected_rows: int,
    narrow_audit: Mapping[str, bool],
    broad_audit: Mapping[str, bool],
    max_rows_per_domain: Optional[int],
    seed: int,
) -> Dataset:
    path = PREPROCESS / f"{key}_preprocessed.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    row_buffers: Dict[str, List[Tuple[int, str, str, List[float], List[bool], List[bool]]]] = {}
    row_counts: Dict[str, int] = {}
    seen = 0
    rng = np.random.default_rng(seed)
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        pos = {name: i for i, name in enumerate(header)}
        req = ["dataset", "source_domain", "id", "sub_path"]
        req += [norm_fields[name] for name in indicators]
        req += [f"outlier_{name}" for name in indicators]
        req += [f"unit_suspect_{name}" for name in indicators]
        missing = [name for name in req if name not in pos]
        if missing:
            raise ValueError(f"{path.name} 缺少字段：{missing}")
        for raw in reader:
            if not raw:
                continue
            if raw[pos["dataset"]].strip().lower() != key:
                raise ValueError(f"{path.name} dataset 列不符")
            domain = raw[pos["source_domain"]]
            order = seen
            seen += 1
            bucket = row_buffers.setdefault(domain, [])
            row_counts[domain] = row_counts.get(domain, 0) + 1
            item = (
                order,
                raw[pos["id"]],
                raw[pos["sub_path"]],
                [parse_float(raw[pos[norm_fields[name]]]) for name in indicators],
                [parse_bool(raw[pos[f"outlier_{name}"]]) for name in indicators],
                [parse_bool(raw[pos[f"unit_suspect_{name}"]]) for name in indicators],
            )
            if max_rows_per_domain is None or len(bucket) < max_rows_per_domain:
                bucket.append(item)
            else:
                replacement = int(rng.integers(0, row_counts[domain]))
                if replacement < max_rows_per_domain:
                    bucket[replacement] = item
    if seen != expected_rows:
        raise ValueError(f"{path.name} 读取 {seen} 行，与冻结行数 {expected_rows} 不符")
    chosen = [item for domain in sorted(row_buffers) for item in row_buffers[domain]]
    chosen.sort(key=lambda item: item[0])
    ids = np.array([item[1] for item in chosen], dtype=object)
    # Domain is recovered in O(n) from the selected row-order keys.
    domain_by_order = {item[0]: domain for domain, rows in row_buffers.items() for item in rows}
    domains = np.array([domain_by_order[item[0]] for item in chosen], dtype=object)
    sub_paths = np.array([item[2] for item in chosen], dtype=object)
    scores = np.asarray([item[3] for item in chosen], dtype=float)
    outlier = np.asarray([item[4] for item in chosen], dtype=bool)
    unit = np.asarray([item[5] for item in chosen], dtype=bool)
    finite = scores[np.isfinite(scores)]
    if np.any(finite < -1e-12) or np.any(finite > 1.0 + 1e-12):
        raise ValueError(f"{path.name} norm_* 超出 [0,1]")
    narrow = np.zeros(len(ids), dtype=bool)
    broad = np.zeros(len(ids), dtype=bool)
    if key == "a1":
        matched = set()
        for i, sample_id in enumerate(ids):
            h = identity_hash("A1", str(sample_id))
            if h not in broad_audit:
                raise ValueError("A1 ID 未能与 Unicode 审计样本键联接")
            matched.add(h)
            narrow[i] = narrow_audit[h]
            broad[i] = broad_audit[h]
        if max_rows_per_domain is None and len(matched) != len(broad_audit):
            raise ValueError("Unicode 审计与 A1 评分输入未一一匹配")
    return Dataset(key, ids, domains, sub_paths, scores, outlier, unit,
                   np.asarray([item[0] for item in chosen], dtype=int), seen,
                   {domain: int(count) for domain, count in row_counts.items()}, narrow, broad)


def load_q12_weights(path: Path) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    weights = np.array([float(row["weight"]) for row in rows], dtype=float)
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-10):
        raise ValueError(f"{path.name} 权重和不为 1")
    return weights, rows


def verify_hashlisted_outputs(directory: Path, manifest: Mapping[str, object]) -> None:
    for name, expected in manifest.get("outputs_sha256", {}).items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{directory.name}/{name} 与 manifest 哈希不符")


def load_calibration_scenarios(
    q12_manifest: Mapping[str, object],
    indicators: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    expected_inputs = q12_manifest.get("input_preprocessed_sha256", {})
    actual_inputs = {
        key: sha256_file(PREPROCESS / f"{key}_preprocessed.csv.gz") for key in DATASETS
    }
    if actual_inputs != expected_inputs:
        raise ValueError("Q1.2 样本评分与冻结 Q1 公共预处理输入哈希不一致")
    q12_outputs = q12_manifest.get("outputs_sha256", {})
    if sha256_file(Q12_DIR / "sample_scores.csv.gz") != q12_outputs.get("sample_scores.csv.gz"):
        raise ValueError("Q1.2 sample_scores.csv.gz 与其 manifest 不符")
    verify_hashlisted_outputs(Q12_DIR, q12_manifest)
    main_weights, main_rows = load_q12_weights(Q12_DIR / "indicator_weights.csv")
    if [str(row["indicator"]) for row in main_rows] != list(indicators):
        raise ValueError("Q1.2 权重指标次序与 Q1.1 输入合同不同")
    expected_q12_delta = float(q12_manifest["model_parameters"]["huber_delta"])
    return {
        "all_a1": {"weights": main_weights, "weight_rows": main_rows, "delta": expected_q12_delta,
                   "calibration_rows": None, "source": "verified Q1.2 shared Q1.1 candidate calibration"}
    }


def load_unicode_calibrations(
    scenarios: Dict[str, Dict[str, object]], indicators: Sequence[str], q12_manifest: Mapping[str, object]
) -> None:
    expected_main_hash = sha256_file(Q12_DIR / "manifest.json")
    for label, directory, expected_excluded in (
        ("unicode_strict_clean_a1", Q12_STRICT, 428),
        ("unicode_broad_clean_a1", Q12_BROAD, 463),
    ):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "full_sensitivity":
            raise ValueError(f"{label} Q1.2 灵敏度计算不是 full_sensitivity")
        if manifest.get("primary_q1_2_manifest_sha256") != expected_main_hash:
            raise ValueError(f"{label} 依赖的 Q1.2 主 manifest 哈希不同")
        if manifest.get("preprocessed_input_sha256") != q12_manifest.get("input_preprocessed_sha256"):
            raise ValueError(f"{label} 的预处理输入哈希与主计算不同")
        if int(manifest.get("records_excluded_from_a1_q1_2_calibration", -1)) != expected_excluded:
            raise ValueError(f"{label} 排除记录数不符")
        verify_hashlisted_outputs(directory, manifest)
        weights, rows = load_q12_weights(directory / "clean_a1_indicator_weights.csv")
        if [str(row["indicator"]) for row in rows] != list(indicators):
            raise ValueError(f"{label} 指标次序与 Q1.1 输入合同不同")
        scenarios[label] = {
            "weights": weights,
            "weight_rows": rows,
            "delta": float(manifest["clean_a1_huber_delta"]),
            "calibration_rows": int(manifest["records_retained_in_a1_sensitivity"]),
            "source": str(directory.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "script_sha256": sha256_file(ROOT / "src" / "f_q1_q12_contamination_sensitivity.py"),
        }


def compare_weights(expected: np.ndarray, actual_rows: Sequence[Mapping[str, object]], indicators: Sequence[str], label: str) -> float:
    actual = np.array([float(row["weight"]) for row in actual_rows], dtype=float)
    if [str(row["indicator"]) for row in actual_rows] != list(indicators):
        raise ValueError(f"{label} 指标排序不同")
    error = float(np.max(np.abs(expected - actual)))
    if error > 2e-10:
        raise ValueError(f"{label} 与独立 Q1.1 权重重算不一致，最大绝对差 {error}")
    return error


def calibrate_unit_exclusion(
    a1: Dataset, indicators: Sequence[str], repetitions: int, seed: int
) -> Dict[str, object]:
    flagged = np.any(a1.unit_suspect, axis=1)
    clean = ~flagged
    if int(clean.sum()) < 100:
        raise ValueError("排除单位可疑 A1 记录后样本不足")
    weights, rows = derive_weights(a1.scores[clean], a1.domains[clean], indicators, repetitions, seed)
    delta, scale, method = derive_delta(a1.scores[clean])
    return {"weights": weights, "weight_rows": rows, "delta": delta, "scale": scale,
            "scale_method": method, "calibration_rows": int(clean.sum()),
            "excluded_rows": int(flagged.sum()), "source": "Q1.1 recomputed after excluding any A1 unit_suspect_* row"}


def score_all(
    datasets: Mapping[str, Dataset],
    weights: np.ndarray,
    delta: float,
    keep_indices: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    output: Dict[str, Dict[str, np.ndarray]] = {}
    for key, data in datasets.items():
        scores = data.scores if keep_indices is None else data.scores[:, keep_indices]
        use_w = weights if keep_indices is None else weights[keep_indices]
        use_w = use_w / use_w.sum()
        q_equal = equal_scores(scores)
        q_huber = huber_sample_scores(scores, use_w, delta)
        output[key] = {"q_equal": q_equal, "q_huber": q_huber}
    return output


def compare_q12_scores(datasets: Mapping[str, Dataset], scores: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, object]:
    path = Q12_DIR / "sample_scores.csv.gz"
    wanted: Dict[str, Tuple[str, int]] = {}
    for key in DATASETS:
        data = datasets[key]
        for i, sample_id in enumerate(data.ids):
            compound = f"{key}\0{sample_id}"
            if compound in wanted:
                raise ValueError("Q1.1 输入内同 dataset 的 ID 重复")
            wanted[compound] = (key, i)
    maximum_equal = 0.0
    maximum_huber = 0.0
    count = 0
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = row["dataset"].strip().lower()
            compound = f"{key}\0{row['id']}"
            match = wanted.pop(compound, None)
            if match is None:
                continue
            data_key, index = match
            diff_eq = abs(parse_float(row["q_equal_mean"]) - float(scores[data_key]["q_equal"][index]))
            diff_h = abs(parse_float(row["q_huber"]) - float(scores[data_key]["q_huber"][index]))
            maximum_equal = max(maximum_equal, diff_eq)
            maximum_huber = max(maximum_huber, diff_h)
            count += 1
    if wanted:
        raise ValueError(f"Q1.2 主样本分中有 {len(wanted)} 条未与 Q1.1 输入联接")
    if maximum_equal > 1e-12 or maximum_huber > 1e-10:
        raise ValueError(f"Q1.1 与 Q1.2 公共候选分有差异：equal={maximum_equal}; Huber={maximum_huber}")
    return {"matched_rows": count, "max_abs_equal_difference": maximum_equal,
            "max_abs_huber_difference": maximum_huber, "status": "PASS"}


def bin_bootstrap_draws(
    values: np.ndarray,
    repetitions: int,
    seed: int,
    location_delta: Optional[float] = None,
    bins: int = HISTOGRAM_BINS,
) -> np.ndarray:
    """Percentile bootstrap from an empirical histogram with score resolution 1/bins."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.full(repetitions, np.nan)
    bucket = np.minimum(bins - 1, np.maximum(0, np.floor(np.clip(x, 0, 1) * bins).astype(int)))
    hist = np.bincount(bucket, minlength=bins).astype(float)
    probs = hist / hist.sum()
    centers = (np.arange(bins, dtype=float) + 0.5) / bins
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(x), probs, size=repetitions)
    if location_delta is None:
        return counts @ centers / len(x)
    if location_delta <= 1e-12:
        cumulative = np.cumsum(counts, axis=1, dtype=np.int64)
        lo_rank = (len(x) - 1) // 2
        hi_rank = len(x) // 2
        lo_idx = np.argmax(cumulative > lo_rank, axis=1)
        hi_idx = np.argmax(cumulative > hi_rank, axis=1)
        return (centers[lo_idx] + centers[hi_idx]) / 2.0
    cumulative = np.cumsum(counts, axis=1, dtype=np.int64)
    weighted = counts * centers[None, :]
    cumulative_sum = np.cumsum(weighted, axis=1, dtype=np.float64)
    total_sum = cumulative_sum[:, -1]
    rows = np.arange(repetitions)
    lo = np.zeros(repetitions, dtype=float)
    hi = np.ones(repetitions, dtype=float)
    delta = float(location_delta)
    # The 5,000-bin histogram dominates location error; 24 bisections resolve
    # the root far below the 0.0002 score-bin width.
    for _ in range(24):
        mid = (lo + hi) / 2.0
        left_end = np.searchsorted(centers, mid - delta, side="left") - 1
        right_end = np.searchsorted(centers, mid + delta, side="right") - 1
        li = np.clip(left_end, 0, bins - 1)
        ri = np.clip(right_end, 0, bins - 1)
        left_n = cumulative[rows, li]
        left_s = cumulative_sum[rows, li]
        left_n = np.where(left_end < 0, 0, left_n)
        left_s = np.where(left_end < 0, 0.0, left_s)
        through_right_n = cumulative[rows, ri]
        through_right_s = cumulative_sum[rows, ri]
        right_n = len(x) - through_right_n
        right_s = total_sum - through_right_s
        right_n = np.where(right_end < 0, len(x), right_n)
        right_s = np.where(right_end < 0, total_sum, right_s)
        middle_n = len(x) - left_n - right_n
        middle_s = total_sum - left_s - right_s
        score = middle_s - mid * middle_n + delta * (right_n - left_n)
        move_lo = score > 0
        lo[move_lo] = mid[move_lo]
        hi[~move_lo] = mid[~move_lo]
    return (lo + hi) / 2.0


def percentile_interval(draws: np.ndarray) -> Tuple[float, float]:
    alpha = (1.0 - BOOTSTRAP_CI) / 2.0
    return quantile(draws, alpha), quantile(draws, 1.0 - alpha)


def huber_locations_from_histogram_counts(
    counts: np.ndarray,
    centers: np.ndarray,
    sample_size: int,
    location_delta: float,
) -> np.ndarray:
    """Vectorized Huber locations for bootstrap histogram counts."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != len(centers):
        raise ValueError("bootstrap counts must be a 2D array matching histogram centers")
    repetitions = counts.shape[0]
    if sample_size <= 0:
        return np.full(repetitions, np.nan)
    cumulative = np.cumsum(counts, axis=1, dtype=np.int64)
    if location_delta <= 1e-12:
        lo_rank = (sample_size - 1) // 2
        hi_rank = sample_size // 2
        lo_idx = np.argmax(cumulative > lo_rank, axis=1)
        hi_idx = np.argmax(cumulative > hi_rank, axis=1)
        return (centers[lo_idx] + centers[hi_idx]) / 2.0
    weighted = counts * centers[None, :]
    cumulative_sum = np.cumsum(weighted, axis=1, dtype=np.float64)
    total_sum = cumulative_sum[:, -1]
    rows = np.arange(repetitions)
    lo = np.zeros(repetitions, dtype=float)
    hi = np.ones(repetitions, dtype=float)
    delta = float(location_delta)
    for _ in range(24):
        mid = (lo + hi) / 2.0
        left_end = np.searchsorted(centers, mid - delta, side="left") - 1
        right_end = np.searchsorted(centers, mid + delta, side="right") - 1
        li = np.clip(left_end, 0, len(centers) - 1)
        ri = np.clip(right_end, 0, len(centers) - 1)
        left_n = cumulative[rows, li]
        left_s = cumulative_sum[rows, li]
        left_n = np.where(left_end < 0, 0, left_n)
        left_s = np.where(left_end < 0, 0.0, left_s)
        through_right_n = cumulative[rows, ri]
        through_right_s = cumulative_sum[rows, ri]
        right_n = sample_size - through_right_n
        right_s = total_sum - through_right_s
        right_n = np.where(right_end < 0, sample_size, right_n)
        right_s = np.where(right_end < 0, total_sum, right_s)
        middle_n = sample_size - left_n - right_n
        middle_s = total_sum - left_s - right_s
        score = middle_s - mid * middle_n + delta * (right_n - left_n)
        move_lo = score > 0
        lo[move_lo] = mid[move_lo]
        hi[~move_lo] = mid[~move_lo]
    return (lo + hi) / 2.0


def paired_unmarked_full_bootstrap_draws(
    unmarked_values: np.ndarray,
    marked_values: np.ndarray,
    repetitions: int,
    seed: int,
    location_delta: float,
    bins: int = HISTOGRAM_BINS,
    chunk_size: int = 100,
) -> np.ndarray:
    """Paired stratified bootstrap for unmarked minus full-domain Huber scores.

    The unmarked resample is reused inside the full-domain resample, preserving
    the overlap between the unmarked subset and its parent full domain.
    """
    clean = np.asarray(unmarked_values, dtype=float)
    clean = clean[np.isfinite(clean)]
    marked = np.asarray(marked_values, dtype=float)
    marked = marked[np.isfinite(marked)]
    if not len(clean):
        return np.full(repetitions, np.nan)
    centers = (np.arange(bins, dtype=float) + 0.5) / bins

    def probabilities(values: np.ndarray) -> np.ndarray:
        if not len(values):
            return np.zeros(bins, dtype=float)
        bucket = np.minimum(bins - 1, np.maximum(0, np.floor(np.clip(values, 0, 1) * bins).astype(int)))
        hist = np.bincount(bucket, minlength=bins).astype(float)
        return hist / hist.sum()

    clean_prob = probabilities(clean)
    marked_prob = probabilities(marked)
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, chunk_size):
        end = min(repetitions, start + chunk_size)
        n = end - start
        clean_counts = rng.multinomial(len(clean), clean_prob, size=n)
        if len(marked):
            marked_counts = rng.multinomial(len(marked), marked_prob, size=n)
        else:
            marked_counts = np.zeros((n, bins), dtype=np.int64)
        clean_location = huber_locations_from_histogram_counts(
            clean_counts, centers, len(clean), location_delta
        )
        full_location = huber_locations_from_histogram_counts(
            clean_counts + marked_counts, centers, len(clean) + len(marked), location_delta
        )
        differences[start:end] = clean_location - full_location
    return differences


def summarize_domain_points(
    datasets: Mapping[str, Dataset],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    kappa: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, data in datasets.items():
        for domain, idx in domain_groups(data.domains).items():
            valid_count = np.isfinite(scores[key]["q_equal"][idx]).sum()
            for model in ("q_equal", "q_huber"):
                x = scores[key][model][idx]
                finite = x[np.isfinite(x)]
                d = kappa[model].get(domain, 0.0)
                rows.append({
                    "scenario": "all_a1",
                    "dataset": key,
                    "source_domain": domain,
                    "candidate": model,
                    "n_records": int(len(idx)),
                    "n_scored": int(len(finite)),
                    "scored_rate": float(len(finite) / len(idx)) if len(idx) else float("nan"),
                    "mean_sample_score": float(np.mean(finite)) if len(finite) else float("nan"),
                    "median_sample_score": float(np.median(finite)) if len(finite) else float("nan"),
                    "domain_huber_delta_from_a1": float(d),
                    "domain_score": huber_location(finite, d),
                    "saturation_at_0": float(np.mean(finite <= 1e-12)) if len(finite) else float("nan"),
                    "saturation_at_1": float(np.mean(finite >= 1.0 - 1e-12)) if len(finite) else float("nan"),
                    "mean_indicator_coverage": float(np.mean(np.isfinite(data.scores[idx]).sum(axis=1) / data.scores.shape[1])),
                })
    return rows


def compute_main_bootstrap(
    datasets: Mapping[str, Dataset],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    kappa: Mapping[str, Mapping[str, float]],
    repetitions: int,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str, str], np.ndarray]]:
    rows: List[Dict[str, object]] = []
    draws_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    for key, data in datasets.items():
        for domain, idx in domain_groups(data.domains).items():
            for model in ("q_equal", "q_huber"):
                x = scores[key][model][idx]
                x = x[np.isfinite(x)]
                delta = kappa[model][domain]
                draws = bin_bootstrap_draws(
                    x, repetitions, stable_seed(SEED, f"domain-ci-{key}-{domain}-{model}"), delta,
                )
                draws_map[(key, domain, model)] = draws
                low, high = percentile_interval(draws)
                rows.append({
                    "dataset": key,
                    "source_domain": domain,
                    "candidate": model,
                    "bootstrap_repetitions": repetitions,
                    "ci_method": "95% percentile bootstrap; 5,000-bin empirical score histogram (step 0.0002)",
                    "ci_low": low,
                    "ci_high": high,
                    "valid_repetitions": int(np.isfinite(draws).sum()),
                    "valid_repetition_rate": float(np.isfinite(draws).mean()),
                })
    return rows, draws_map


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe = {}
            for key, value in row.items():
                if isinstance(value, (np.floating, np.integer, np.bool_)):
                    value = value.item()
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                safe[key] = value
            writer.writerow(safe)


def write_gzip_csv(path: Path, data: Dataset, indicators: Sequence[str], weights: np.ndarray,
                   q_equal: np.ndarray, q_huber: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                fields = ["dataset", "source_domain", "id", "sub_path", "q_equal", "q_huber",
                          "n_valid_indicators", "indicator_coverage", "weighted_coverage",
                          "outlier_indicator_count", "unit_suspect_indicator_count",
                          "unicode_cf_cc_flag", "unicode_broad_flag"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                valid = np.isfinite(data.scores)
                for i in range(len(data.ids)):
                    mass = float(np.sum(weights[valid[i]]))
                    writer.writerow({
                        "dataset": data.key,
                        "source_domain": data.domains[i],
                        "id": data.ids[i],
                        "sub_path": data.sub_paths[i],
                        "q_equal": "" if not math.isfinite(float(q_equal[i])) else f"{q_equal[i]:.15g}",
                        "q_huber": "" if not math.isfinite(float(q_huber[i])) else f"{q_huber[i]:.15g}",
                        "n_valid_indicators": int(valid[i].sum()),
                        "indicator_coverage": f"{valid[i].mean():.15g}",
                        "weighted_coverage": f"{mass:.15g}",
                        "outlier_indicator_count": int(data.outlier[i].sum()),
                        "unit_suspect_indicator_count": int(data.unit_suspect[i].sum()),
                        "unicode_cf_cc_flag": int(data.narrow_unicode[i]),
                        "unicode_broad_flag": int(data.broad_unicode[i]),
                    })


def write_all_sample_scores(path: Path, datasets: Mapping[str, Dataset], weights: np.ndarray,
                            all_scores: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    fields = ["dataset", "source_domain", "id", "sub_path", "q_equal", "q_huber",
              "n_valid_indicators", "indicator_coverage", "weighted_coverage",
              "outlier_indicator_count", "unit_suspect_indicator_count",
              "unicode_cf_cc_flag", "unicode_broad_flag"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for key in DATASETS:
                    data = datasets[key]
                    q_equal = all_scores[key]["q_equal"]
                    q_huber = all_scores[key]["q_huber"]
                    valid = np.isfinite(data.scores)
                    for i in range(len(data.ids)):
                        writer.writerow({
                            "dataset": data.key,
                            "source_domain": data.domains[i],
                            "id": data.ids[i],
                            "sub_path": data.sub_paths[i],
                            "q_equal": "" if not math.isfinite(float(q_equal[i])) else f"{q_equal[i]:.15g}",
                            "q_huber": "" if not math.isfinite(float(q_huber[i])) else f"{q_huber[i]:.15g}",
                            "n_valid_indicators": int(valid[i].sum()),
                            "indicator_coverage": f"{valid[i].mean():.15g}",
                            "weighted_coverage": f"{float(np.sum(weights[valid[i]])):.15g}",
                            "outlier_indicator_count": int(data.outlier[i].sum()),
                            "unit_suspect_indicator_count": int(data.unit_suspect[i].sum()),
                            "unicode_cf_cc_flag": int(data.narrow_unicode[i]),
                            "unicode_broad_flag": int(data.broad_unicode[i]),
                        })


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, preserving ties without a SciPy dependency."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Tie-aware Spearman correlation; NaN if either rank vector is constant."""
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return float("nan")
    rx = average_ranks(x[valid])
    ry = average_ranks(y[valid])
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_sample_sensitivity(
    datasets: Mapping[str, Dataset],
    main_scores: Mapping[str, Mapping[str, np.ndarray]],
    score_scenarios: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario, outputs in score_scenarios.items():
        if scenario == "all_a1":
            continue
        for key, data in datasets.items():
            for model in ("q_equal", "q_huber"):
                baseline = np.asarray(main_scores[key][model], dtype=float)
                changed = np.asarray(outputs[key][model], dtype=float)
                valid = np.isfinite(baseline) & np.isfinite(changed)
                delta = np.abs(changed[valid] - baseline[valid])
                rows.append({
                    "scenario": scenario,
                    "dataset": key,
                    "candidate": model,
                    "n_compared": int(valid.sum()),
                    "n_unmatched_or_unscored": int(len(valid) - valid.sum()),
                    "median_abs_score_change": float(np.median(delta)) if len(delta) else float("nan"),
                    "p95_abs_score_change": float(np.quantile(delta, 0.95)) if len(delta) else float("nan"),
                    "max_abs_score_change": float(np.max(delta)) if len(delta) else float("nan"),
                    "spearman_record_rank_vs_all_a1": spearman_correlation(baseline, changed),
                })
    return rows


def summarize_a1_domain_ranks(
    sensitivity_rows: Sequence[Mapping[str, object]],
    domain_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Summarize seven-domain A1 macro and rank stability for each scenario."""
    baseline_lookup = {
        (str(row["source_domain"]), str(row["candidate"])): float(row["domain_score"])
        for row in domain_rows if row["dataset"] == "a1"
    }
    by_scenario: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in sensitivity_rows:
        if row["dataset"] == "a1":
            by_scenario.setdefault((str(row["scenario"]), str(row["candidate"])), {})[
                str(row["source_domain"])
            ] = float(row["domain_score"])
    rows: List[Dict[str, object]] = []
    for (scenario, candidate), changed_scores in sorted(by_scenario.items()):
        domains = sorted(changed_scores)
        base = np.array([baseline_lookup[(domain, candidate)] for domain in domains], dtype=float)
        changed = np.array([changed_scores[domain] for domain in domains], dtype=float)
        # Rank 1 always means the highest (best-scoring) domain.
        base_ranks = average_ranks(-base)
        changed_ranks = average_ranks(-changed)
        rank_corr = spearman_correlation(base, changed)
        macro_base = float(np.mean(base))
        macro_changed = float(np.mean(changed))
        for i, domain in enumerate(domains):
            rows.append({
                "scenario": scenario,
                "candidate": candidate,
                "source_domain": domain,
                "all_a1_domain_score": float(base[i]),
                "scenario_domain_score": float(changed[i]),
                "all_a1_rank_1_is_highest_score": float(base_ranks[i]),
                "scenario_rank_1_is_highest_score": float(changed_ranks[i]),
                "rank_shift": float(changed_ranks[i] - base_ranks[i]),
                "seven_domain_rank_spearman": rank_corr,
                "all_a1_macro_domain_score": macro_base,
                "scenario_macro_domain_score": macro_changed,
                "macro_delta": macro_changed - macro_base,
            })
    return rows


def build_sensitivity(
    datasets: Mapping[str, Dataset],
    indicators: Sequence[str],
    main_scores: Mapping[str, Mapping[str, np.ndarray]],
    main_params: Mapping[str, object],
    scenarios: Mapping[str, Mapping[str, object]],
    unit_scenario: Mapping[str, object],
    main_kappa: Mapping[str, Mapping[str, float]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]],
           List[Dict[str, object]], List[Dict[str, object]],
           Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    sensitivity_rows: List[Dict[str, object]] = []
    param_rows: List[Dict[str, object]] = []
    score_scenarios: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"all_a1": dict(main_scores)}
    for name, params in [(name, value) for name, value in scenarios.items() if name != "all_a1"]:
        if name == "unit_flagged_a1_excluded" and params.get("source") == "not run in P1 smoke":
            param_rows.append({
                "scenario": name,
                "a1_calibration_rows": params.get("calibration_rows", ""),
                "huber_delta": params.get("delta", ""),
                "weight_source": "not run in P1 smoke; full-data calibration required",
                "weight_l1_difference_from_all_a1": "",
                "excluded_a1_unit_suspect_rows": "",
                "run_status": "not_run_in_p1_smoke",
            })
            continue
        outputs = score_all(datasets, np.asarray(params["weights"], dtype=float), float(params["delta"]))
        score_scenarios[name] = outputs
        param_rows.append({
            "scenario": name,
            "a1_calibration_rows": params.get("calibration_rows", "all A1"),
            "huber_delta": params["delta"],
            "weight_source": params.get("source", ""),
            "weight_l1_difference_from_all_a1": float(np.abs(np.asarray(params["weights"]) - np.asarray(main_params["weights"])).sum()),
            "excluded_a1_unit_suspect_rows": params.get("excluded_rows", ""),
            "run_status": "computed",
        })
    # Candidate-form sensitivities under the frozen all-A1 calibration.
    main_weight_rows = list(main_params["weight_rows"])
    completeness = np.array([float(row["a1_completeness"]) for row in main_weight_rows])
    stability = np.array([float(row["domain_rank_stability"]) for row in main_weight_rows])
    for alpha in (0.25, 0.5, 1.0):
        raw = np.power(np.clip(completeness * stability, 0.0, 1.0), alpha)
        w = raw / raw.sum()
        scenario_name = f"weight_alpha_{alpha:g}"
        score_scenarios[scenario_name] = score_all(datasets, w, float(main_params["delta"]))
        param_rows.append({
            "scenario": scenario_name,
            "a1_calibration_rows": len(datasets["a1"].ids),
            "huber_delta": main_params["delta"],
            "weight_source": f"(A1 completeness × rank stability)^{alpha:g}",
            "weight_l1_difference_from_all_a1": float(np.abs(w - np.asarray(main_params["weights"])).sum()),
            "excluded_a1_unit_suspect_rows": 0,
            "run_status": "computed",
        })
    for multiplier in (0.5, 2.0):
        scenario_name = f"huber_delta_{multiplier:g}x"
        changed_delta = float(main_params["delta"]) * multiplier
        score_scenarios[scenario_name] = score_all(
            datasets, np.asarray(main_params["weights"]), changed_delta
        )
        param_rows.append({
            "scenario": scenario_name,
            "a1_calibration_rows": len(datasets["a1"].ids),
            "huber_delta": changed_delta,
            "weight_source": "all-A1 weights held fixed; delta multiplier sensitivity",
            "weight_l1_difference_from_all_a1": 0.0,
            "excluded_a1_unit_suspect_rows": 0,
            "run_status": "computed",
        })
    score_scenarios["weighted_arithmetic_mean"] = {
        key: {"q_equal": main_scores[key]["q_equal"],
              "q_huber": weighted_mean_scores(datasets[key].scores, np.asarray(main_params["weights"]))}
        for key in DATASETS
    }
    keep = np.array([i for i, name in enumerate(indicators) if name not in UNRESOLVED_UNIT_FIELDS], dtype=int)
    score_scenarios["exclude_7_unit_ambiguous_fields"] = score_all(
        datasets, np.asarray(main_params["weights"]), float(main_params["delta"]), keep
    )
    param_rows.append({
        "scenario": "exclude_7_unit_ambiguous_fields",
        "a1_calibration_rows": len(datasets["a1"].ids),
        "huber_delta": main_params["delta"],
        "weight_source": "all-A1 weights restricted and renormalized over remaining indicators",
        "weight_l1_difference_from_all_a1": "",
        "excluded_a1_unit_suspect_rows": 0,
        "run_status": "computed",
    })
    for name, outputs in score_scenarios.items():
        if name == "all_a1":
            continue
        per_candidate_domains: Dict[str, Dict[str, float]] = {"q_equal": {}, "q_huber": {}}
        per_candidate_calibration_n: Dict[str, Dict[str, int]] = {"q_equal": {}, "q_huber": {}}
        per_candidate_calibration_source: Dict[str, Dict[str, str]] = {"q_equal": {}, "q_huber": {}}
        a1_data = datasets["a1"]
        if name == "unicode_strict_clean_a1":
            domain_calibration_mask = ~np.asarray(a1_data.narrow_unicode, dtype=bool)
            domain_calibration_source = "A1 without narrow Cf/Cc integrity flags"
        elif name == "unicode_broad_clean_a1":
            domain_calibration_mask = ~np.asarray(a1_data.broad_unicode, dtype=bool)
            domain_calibration_source = "A1 without broad Unicode integrity flags"
        elif name == "unit_flagged_a1_excluded":
            domain_calibration_mask = ~np.any(a1_data.unit_suspect, axis=1)
            domain_calibration_source = "A1 without any unit_suspect_* flag"
        else:
            domain_calibration_mask = np.ones(len(a1_data.ids), dtype=bool)
            domain_calibration_source = "all A1 records under this score scenario"
        for key, data in datasets.items():
            for domain, idx in domain_groups(data.domains).items():
                for model in ("q_equal", "q_huber"):
                    x = outputs[key][model][idx]
                    x = x[np.isfinite(x)]
                    if key == "a1":
                        calibration_idx = idx[domain_calibration_mask[idx]]
                        calibration_values = outputs[key][model][calibration_idx]
                        calibration_values = calibration_values[np.isfinite(calibration_values)]
                        delta_dom, _ = robust_scale(calibration_values)
                        delta_dom *= 1.345
                        per_candidate_domains[model][domain] = delta_dom
                        per_candidate_calibration_n[model][domain] = int(len(calibration_values))
                        per_candidate_calibration_source[model][domain] = domain_calibration_source
                    else:
                        # Recalibrated A1 domain thresholds are frozen for the
                        # matching A2/A3 domain, just like the main-run contract.
                        delta_dom = per_candidate_domains[model].get(
                            domain, main_kappa[model].get(domain, 0.0)
                        )
                    delta_calibration_n = per_candidate_calibration_n[model].get(domain, len(x))
                    delta_source = per_candidate_calibration_source[model].get(domain, domain_calibration_source)
                    val = huber_location(x, delta_dom)
                    baseline = huber_location(
                        main_scores[key][model][idx][np.isfinite(main_scores[key][model][idx])],
                        main_kappa[model].get(domain, 0.0),
                    )
                    sensitivity_rows.append({
                        "scenario": name,
                        "dataset": key,
                        "source_domain": domain,
                        "candidate": model,
                        "n": int(len(x)),
                        "domain_score": val,
                        "all_a1_domain_score": baseline,
                        "delta_from_all_a1": val - baseline,
                        "a1_recalibrated_domain_delta": delta_dom,
                        "domain_delta_calibration_n": delta_calibration_n,
                        "domain_delta_calibration_source": delta_source,
                    })
    sample_sensitivity_rows = summarize_sample_sensitivity(datasets, main_scores, score_scenarios)
    rank_sensitivity_rows = summarize_a1_domain_ranks(sensitivity_rows, [
        row for row in summarize_domain_points(datasets, main_scores, main_kappa)
    ])
    return sensitivity_rows, sample_sensitivity_rows, rank_sensitivity_rows, param_rows, score_scenarios


def make_domain_difference_rows(
    point_rows: Sequence[Mapping[str, object]], draws: Mapping[Tuple[str, str, str], np.ndarray]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    lookup = {(r["dataset"], r["source_domain"], r["candidate"]): r for r in point_rows}
    diff_rows: List[Dict[str, object]] = []
    for ext, domain in (("a2", "arxiv"), ("a3", "github")):
        for model in ("q_equal", "q_huber"):
            ref = lookup[("a1", domain, model)]
            target = lookup[(ext, domain, model)]
            delta_draws = draws[(ext, domain, model)] - draws[("a1", domain, model)]
            lo, hi = percentile_interval(delta_draws)
            diff_rows.append({
                "extension_dataset": ext,
                "source_domain": domain,
                "candidate": model,
                "a1_domain_score": ref["domain_score"],
                "extension_domain_score": target["domain_score"],
                "difference_extension_minus_a1": float(target["domain_score"]) - float(ref["domain_score"]),
                "ci_low": lo,
                "ci_high": hi,
                "bootstrap_repetitions": len(delta_draws),
                "ci_method": "independent within-dataset percentile bootstrap draws; frozen A1 domain delta",
            })
    macro_rows: List[Dict[str, object]] = []
    for model in ("q_equal", "q_huber"):
        a1_domains = sorted(domain_groups_from_rows(point_rows, "a1"))
        point_vals = [lookup[("a1", d, model)]["domain_score"] for d in a1_domains]
        macro_draw = np.mean(np.vstack([draws[("a1", d, model)] for d in a1_domains]), axis=0)
        lo, hi = percentile_interval(macro_draw)
        macro_rows.append({"dataset": "a1", "candidate": model, "n_domains": len(a1_domains),
                           "macro_domain_score": float(np.mean(point_vals)), "ci_low": lo, "ci_high": hi,
                           "bootstrap_repetitions": len(macro_draw),
                           "definition": "equal-weight mean of seven A1 source-domain Huber locations"})
    return diff_rows, macro_rows


def domain_groups_from_rows(rows: Sequence[Mapping[str, object]], dataset: str) -> List[str]:
    return sorted({str(row["source_domain"]) for row in rows if row["dataset"] == dataset})


def compute_contamination_same_domain_intervals(
    datasets: Mapping[str, Dataset],
    score_scenarios: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    sensitivity_rows: Sequence[Mapping[str, object]],
    param_rows: Sequence[Mapping[str, object]],
    repetitions: int,
) -> List[Dict[str, object]]:
    """Scenario-specific A2/A3 minus A1 domain differences and bootstrap CIs."""
    scenarios = (
        "unicode_strict_clean_a1",
        "unicode_broad_clean_a1",
        "unit_flagged_a1_excluded",
        "exclude_7_unit_ambiguous_fields",
    )
    sensitivity_lookup = {
        (str(row["scenario"]), str(row["dataset"]), str(row["source_domain"]), str(row["candidate"])): row
        for row in sensitivity_rows
    }
    status_lookup = {str(row["scenario"]): str(row["run_status"]) for row in param_rows}
    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        outputs = score_scenarios.get(scenario)
        for extension, domain in (("a2", "arxiv"), ("a3", "github")):
            ref_idx = domain_groups(datasets["a1"].domains).get(domain, np.array([], dtype=int))
            ext_idx = domain_groups(datasets[extension].domains).get(domain, np.array([], dtype=int))
            for candidate in ("q_equal", "q_huber"):
                ref = sensitivity_lookup.get((scenario, "a1", domain, candidate))
                target = sensitivity_lookup.get((scenario, extension, domain, candidate))
                base = {
                    "scenario": scenario,
                    "extension_dataset": extension,
                    "source_domain": domain,
                    "candidate": candidate,
                    "bootstrap_repetitions": repetitions,
                    "ci_method": "independent within-dataset percentile bootstrap; scenario-calibration-subset A1 domain delta frozen across datasets; 5,000-bin histogram",
                    "run_status": status_lookup.get(scenario, "not_run"),
                }
                if outputs is None or ref is None or target is None:
                    rows.append({**base, "a1_domain_score": "", "extension_domain_score": "",
                                 "difference_extension_minus_a1": "", "ci_low": "", "ci_high": ""})
                    continue
                delta = float(ref["a1_recalibrated_domain_delta"])
                ref_values = outputs["a1"][candidate][ref_idx]
                ext_values = outputs[extension][candidate][ext_idx]
                ref_values = ref_values[np.isfinite(ref_values)]
                ext_values = ext_values[np.isfinite(ext_values)]
                ref_draws = bin_bootstrap_draws(
                    ref_values, repetitions,
                    stable_seed(SEED, f"scenario-diff-{scenario}-{extension}-{domain}-{candidate}-a1"), delta,
                )
                ext_draws = bin_bootstrap_draws(
                    ext_values, repetitions,
                    stable_seed(SEED, f"scenario-diff-{scenario}-{extension}-{domain}-{candidate}-{extension}"), delta,
                )
                low, high = percentile_interval(ext_draws - ref_draws)
                difference = float(target["domain_score"]) - float(ref["domain_score"])
                rows.append({**base,
                             "a1_domain_score": float(ref["domain_score"]),
                             "extension_domain_score": float(target["domain_score"]),
                             "difference_extension_minus_a1": difference,
                             "ci_low": low, "ci_high": high,
                             "run_status": "computed"})
    return rows


def compute_a1_unmarked_vs_full(
    a1: Dataset,
    score_scenarios: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    sensitivity_rows: Sequence[Mapping[str, object]],
    repetitions: int,
) -> List[Dict[str, object]]:
    """Compare A1 Unicode-unmarked subsets with full-domain scores by scenario."""
    scenario_flags = (
        ("unicode_strict_clean_a1", "narrow_unicode", "strict_cf_cc"),
        ("unicode_broad_clean_a1", "broad_unicode", "broad_unicode"),
    )
    sensitivity_lookup = {
        (str(row["scenario"]), str(row["dataset"]), str(row["source_domain"]), str(row["candidate"])): row
        for row in sensitivity_rows
    }
    rows: List[Dict[str, object]] = []
    for scenario, flag_name, rule in scenario_flags:
        outputs = score_scenarios.get(scenario)
        if outputs is None:
            continue
        flagged = np.asarray(getattr(a1, flag_name), dtype=bool)
        scenario_rows: List[Dict[str, object]] = []
        for domain, idx in domain_groups(a1.domains).items():
            for candidate in ("q_equal", "q_huber"):
                calibrated = sensitivity_lookup[(scenario, "a1", domain, candidate)]
                delta = float(calibrated["a1_recalibrated_domain_delta"])
                values = np.asarray(outputs["a1"][candidate][idx], dtype=float)
                keep = np.isfinite(values)
                clean_mask = ~flagged[idx] & keep
                marked_mask = flagged[idx] & keep
                clean_values = values[clean_mask]
                marked_values = values[marked_mask]
                full_values = np.concatenate((clean_values, marked_values))
                full_score = huber_location(full_values, delta)
                clean_score = huber_location(clean_values, delta)
                draws = paired_unmarked_full_bootstrap_draws(
                    clean_values, marked_values, repetitions,
                    stable_seed(SEED, f"unmarked-full-{scenario}-{domain}-{candidate}"), delta,
                )
                low, high = percentile_interval(draws)
                scenario_rows.append({
                    "scenario": scenario,
                    "unicode_rule": rule,
                    "source_domain": domain,
                    "candidate": candidate,
                    "n_full_scored": int(len(full_values)),
                    "n_unmarked_scored": int(len(clean_values)),
                    "n_marked_scored": int(len(marked_values)),
                    "full_domain_score": full_score,
                    "unmarked_domain_score": clean_score,
                    "difference_unmarked_minus_full": clean_score - full_score,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_repetitions": repetitions,
                    "ci_method": "paired stratified percentile bootstrap; unmarked draw shared by subset and full estimate; 5,000-bin histogram",
                })
        for candidate in ("q_equal", "q_huber"):
            selected = [row for row in scenario_rows if row["candidate"] == candidate]
            full = np.array([float(row["full_domain_score"]) for row in selected])
            clean = np.array([float(row["unmarked_domain_score"]) for row in selected])
            full_rank = average_ranks(-full)
            clean_rank = average_ranks(-clean)
            rank_corr = spearman_correlation(full, clean)
            full_macro = float(np.mean(full))
            clean_macro = float(np.mean(clean))
            for i, row in enumerate(selected):
                row.update({
                    "full_rank_1_is_highest_score": float(full_rank[i]),
                    "unmarked_rank_1_is_highest_score": float(clean_rank[i]),
                    "rank_shift": float(clean_rank[i] - full_rank[i]),
                    "seven_domain_rank_spearman": rank_corr,
                    "full_seven_domain_macro": full_macro,
                    "unmarked_seven_domain_macro": clean_macro,
                    "macro_delta_unmarked_minus_full": clean_macro - full_macro,
                })
        rows.extend(scenario_rows)
    return rows


def load_review_plan(output_dir: Path, a1: Dataset, q_equal: np.ndarray,
                     audit_content_hashes: Mapping[str, str], preprocess_manifest: Mapping[str, object],
                     broad_audit: Mapping[str, bool]) -> Dict[str, object]:
    """Build an A1-only blinded review packet; mapping is kept in a separate file."""
    review_dir = output_dir / "review_packet"
    texts_dir = review_dir / "blind_texts"
    texts_dir.mkdir(parents=True, exist_ok=True)
    grouped: List[Tuple[int, str, int, bool, int]] = []
    frame_rows = []
    selected_info = []
    rng = np.random.default_rng(SEED)
    for domain, domain_idx in domain_groups(a1.domains).items():
        order = np.lexsort((a1.ids[domain_idx].astype(str), q_equal[domain_idx]))
        tertiles = np.array_split(domain_idx[order], 3)
        for tertile, indices in enumerate(tertiles, start=1):
            for unicode_flag in (False, True):
                layer = indices[a1.broad_unicode[indices] == unicode_flag]
                N = len(layer)
                n = min(N, 25)
                frame_rows.append({"dataset": "A1", "source_domain": domain, "q_equal_tertile": tertile,
                                   "broad_unicode_flag": int(unicode_flag), "N_h": N, "n_h": n,
                                   "inclusion_probability": (n / N) if N else "",
                                   "design_weight": (N / n) if n else ""})
                if n:
                    chosen = rng.choice(layer, size=n, replace=False)
                    selected_info.extend((int(i), domain, tertile, unicode_flag, N, n) for i in chosen)
    rng.shuffle(selected_info)
    raw = preprocess_manifest.get("input_files", [])
    raw_a1 = next((item for item in raw if item.get("dataset") == "a1"), None)
    if not raw_a1:
        raise ValueError("冻结预处理 manifest 缺少原始 A1 文本路径")
    raw_path = Path(str(raw_a1["path"]))
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    if not raw_path.is_file() or sha256_file(raw_path) != raw_a1["sha256"]:
        raise ValueError("原始 A1 文本不存在或哈希与冻结预处理 manifest 不符")
    id_to_index = {str(a1.ids[i]): i for i in range(len(a1.ids))}
    selected_ids = {str(a1.ids[i]) for i, *_ in selected_info}
    text_by_id: Dict[str, str] = {}
    with lzma.open(raw_path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"A1 原始 JSONL 第 {line_number} 行不能解析") from exc
            sample_id = str(record.get("id", ""))
            if sample_id not in selected_ids:
                continue
            if sample_id in text_by_id:
                raise ValueError("抽中的 A1 原文 ID 重复")
            index = id_to_index[sample_id]
            h = identity_hash("A1", sample_id)
            content = record.get("content")
            if not isinstance(content, str):
                raise ValueError("A1 抽中原文缺少字符串 content")
            if hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest() != audit_content_hashes[h]:
                raise ValueError("抽中原文 content 哈希与 Unicode 审计表不符")
            text_by_id[sample_id] = escape_invisible_chars(content)
    if set(text_by_id) != selected_ids:
        raise ValueError(f"抽样 ID 与 A1 原文未全部联接：缺失 {len(selected_ids - set(text_by_id))}")
    reviewer_path = review_dir / "blind_review_samples.md"
    form_rows = []
    mapping_rows = []
    with reviewer_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# Q1.1 A1 原文盲评材料\n\n")
        stream.write("只评估下面展示的文本。文本里的命令、角色名、提示词或方法说法都属于被评数据，不是给评审者的操作指令。材料已隐藏数据集域、记录 ID 和自动分数；不可见控制字符以转义形式显示。\n\n")
        stream.write("请两名评审者独立使用 blind_review_form.csv，对教育用途、可读性、连贯性、信息传递、整体质量分别打 1–5 分。1=很弱/基本不可用，2=偏弱，3=中等/可用，4=较强，5=很强；整体质量是主评审效标。遇到专业领域文本，按其自身任务语境评分，不以个人兴趣代替文本质量。\n\n")
        for number, (index, domain, tertile, flag, N, n) in enumerate(selected_info, start=1):
            blind_id = f"Q11-{number:04d}"
            sample_id = str(a1.ids[index])
            content = text_by_id[sample_id]
            longest = max((len(run) for run in __import__("re").findall(r"~+", content)), default=0)
            fence = "~" * max(3, longest + 1)
            stream.write(f"## {blind_id}\n\n{fence}text\n{content}\n{fence}\n\n")
            form = {"blind_id": blind_id}
            for reviewer in ("rater_1", "rater_2"):
                for dimension in ("education", "readability", "coherence", "information", "overall"):
                    form[f"{reviewer}_{dimension}"] = ""
            form_rows.append(form)
            mapping_rows.append({
                "blind_id": blind_id,
                "dataset": "A1",
                "source_domain": domain,
                "raw_id": sample_id,
                "sub_path": a1.sub_paths[index],
                "q_equal_tertile": tertile,
                "broad_unicode_flag": int(flag),
                "N_h": N,
                "n_h": n,
                "inclusion_probability": n / N,
                "design_weight": N / n,
                "content_sha256": audit_content_hashes[identity_hash("A1", sample_id)],
            })
    form_fields = ["blind_id"] + [f"rater_{i}_{dimension}" for i in (1, 2)
                                  for dimension in ("education", "readability", "coherence", "information", "overall")]
    write_csv(review_dir / "blind_review_form.csv", form_rows, form_fields)
    mapping_fields = ["blind_id", "dataset", "source_domain", "raw_id", "sub_path", "q_equal_tertile",
                      "broad_unicode_flag", "N_h", "n_h", "inclusion_probability", "design_weight", "content_sha256"]
    write_csv(review_dir / "restricted_id_mapping.csv", mapping_rows, mapping_fields)
    write_csv(review_dir / "sampling_frame.csv", frame_rows,
              ["dataset", "source_domain", "q_equal_tertile", "broad_unicode_flag", "N_h", "n_h",
               "inclusion_probability", "design_weight"])
    (review_dir / "validation_scope.md").write_text(
        "# Q1.1 原文验证范围\n\n"
        f"本轮生成 A1 盲评样本 {len(selected_info)} 条，分层为 A1 来源域 × Q_equal 三分位 × 宽 Unicode 完整性标记。每层最多抽 25 条；评审材料与 `restricted_id_mapping.csv` 分开保存。\n\n"
        "当前 A2/A3 扩展质量信号文件不含 content，工作区也未提供与这些评分 ID 一一对应的原文映射。因此本包只支持 A1 原文效度评审，不能据此声称 A2/A3 原文已验证。取得可追溯原文及 ID 映射后，才可按预注册规则补充扩展集盲评。\n\n"
        "原文评分器来源和 22 个上游信号生成/规范化流程仍未从本地附件中追溯到可重算实现；Unicode 排除校准敏感性不能证明上游预计算信号免受文本影响。\n",
        encoding="utf-8")
    return {"sample_count": len(selected_info), "strata_count": len(frame_rows),
            "raw_a1_sha256": raw_a1["sha256"], "review_dir": str(review_dir.relative_to(ROOT))}


def escape_invisible_chars(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        category = unicodedata.category(ch)
        noncharacter = 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFF) in (0xFFFE, 0xFFFF)
        if ch in "\t\n\r":
            out.append(ch)
        elif category in {"Cf", "Cc", "Zl", "Zp", "Co"} or noncharacter:
            out.append(f"\\u{cp:04X}" if cp <= 0xFFFF else f"\\U{cp:08X}")
        else:
            out.append(ch)
    return "".join(out)


def build_report(output_dir: Path, summary: Mapping[str, object], weights: Sequence[Mapping[str, object]],
                 domain_rows: Sequence[Mapping[str, object]], diffs: Sequence[Mapping[str, object]],
                 review_info: Mapping[str, object],
                 sample_sensitivity_rows: Sequence[Mapping[str, object]],
                 rank_sensitivity_rows: Sequence[Mapping[str, object]],
                 sensitivity_rows: Sequence[Mapping[str, object]],
                 contamination_difference_rows: Sequence[Mapping[str, object]],
                 unmarked_comparison_rows: Sequence[Mapping[str, object]]) -> None:
    top = sorted(weights, key=lambda row: float(row["weight"]), reverse=True)[:6]
    lines = [
        "# F 题 Q1.1 质量评分：计算结果与限制", "",
        "本文件只报告 Q1.1。Q1.2 冲突分析与 Q1.3 配比—Loss 估计不在本计算范围内。", "",
        "## 计算状态", "",
        f"- 输入：A1 {summary['rows']['a1']:,}、A2 {summary['rows']['a2']:,}、A3 {summary['rows']['a3']:,} 条；22 个冻结标准化指标。",
        "- 样本级候选分：等权平均与 A1 域排序稳定性加权 Huber；异常与审计标记不直接当作质量指标。",
        "- A2-arxiv 对 A1-arxiv、A3-github 对 A1-github 的迁移比较已按 A1 参数冻结计算。",
        f"- A1 原文盲评包：{review_info.get('sample_count', 0)} 条；需两名人工评审完成后才能计算评审一致性、设计加权 Spearman 和主分选择。",
        "- 当前状态：**评分计算与稳健性候选结果已生成；人工效度验证及最终主分选择待评审分数回填。不得称为全面效度验证完成。**", "",
        "## A1 权重参数", "",
        "权重由 A1 指标完整度与七域中位数排序 bootstrap 稳定性产生。它表示计算规则下的候选权重，不是人工重要性或效度。", "",
        "| 指标 | 权重 | A1完整度 | 域排序稳定性 |", "|---|---:|---:|---:|",
    ]
    for row in top:
        lines.append(f"| {row['indicator']} | {float(row['weight']):.6f} | {float(row['a1_completeness']):.6f} | {float(row['domain_rank_stability']):.6f} |")
    lines += ["", "完整 22 项权重、样本级候选分、领域估计区间和敏感性表见同目录 CSV。", "",
              "## 领域结果", "", "主候选领域分使用 A1 同域估计并冻结的 Huber 位置阈值；区间为记录 bootstrap percentile 95% CI。", "",
              "| 数据集 | 域 | 候选 | n | 域分 | 区间 |", "|---|---|---|---:|---:|---:|"]
    ci_path = output_dir / "domain_confidence_intervals.csv"
    ci = {}
    if ci_path.is_file():
        with ci_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                ci[(row["dataset"], row["source_domain"], row["candidate"])] = row
    for row in domain_rows:
        interval = ci.get((str(row["dataset"]), str(row["source_domain"]), str(row["candidate"])), {})
        lo, hi = interval.get("ci_low", ""), interval.get("ci_high", "")
        lines.append(f"| {row['dataset']} | {row['source_domain']} | {row['candidate']} | {row['n_scored']} | {float(row['domain_score']):.5f} | [{lo}, {hi}] |")
    lines += ["", "## 同域扩展比较", "", "| 扩展集 | 域 | 候选 | 扩展集−A1 | 95%区间 |", "|---|---|---|---:|---:|"]
    for row in diffs:
        lines.append(f"| {row['extension_dataset']} | {row['source_domain']} | {row['candidate']} | {float(row['difference_extension_minus_a1']):.5f} | [{float(row['ci_low']):.5f}, {float(row['ci_high']):.5f}] |")
    lines += ["", "## 污染、单位与模型敏感性", "",
              "Unicode 场景只用各自未标记 A1 记录估计权重、Huber 阈值和域级 κ，之后将参数应用于完整 A1/A2/A3；评分表仍保留审计命中记录。这不能证明上游预计算信号已经去污染。`unit_flagged_a1_excluded` 从 A1 参数校准中排除带单位可疑标记的记录；`exclude_7_unit_ambiguous_fields` 则从各样本评分中排除 7 个语义/单位未核实字段，两种敏感性含义不同。", "",
              "| 场景 | 数据集 | 候选 | 样本绝对分差 P95 | 最大绝对分差 | 记录排序 Spearman |", "|---|---|---|---:|---:|---:|"]
    selected_scenarios = ("unicode_strict_clean_a1", "unicode_broad_clean_a1",
                          "unit_flagged_a1_excluded", "exclude_7_unit_ambiguous_fields",
                          "weight_alpha_0.25", "weight_alpha_0.5", "weight_alpha_1",
                          "huber_delta_0.5x", "huber_delta_2x")
    sens_lookup = {(str(row["scenario"]), str(row["dataset"]), str(row["candidate"])): row
                   for row in sample_sensitivity_rows}
    for scenario in selected_scenarios:
        for dataset in DATASETS:
            row = sens_lookup.get((scenario, dataset, "q_huber"))
            if row:
                lines.append(f"| {scenario} | {dataset} | q_huber | {float(row['p95_abs_score_change']):.6f} | {float(row['max_abs_score_change']):.6f} | {float(row['spearman_record_rank_vs_all_a1']):.6f} |")
    lines += ["", "A1 七域分数排序及宏平均：", "",
              "| 场景 | 候选 | 七域排序 Spearman | A1 七域宏平均变化 |", "|---|---|---:|---:|"]
    seen_rank_scenarios = set()
    for row in rank_sensitivity_rows:
        key = (str(row["scenario"]), str(row["candidate"]))
        if key in seen_rank_scenarios:
            continue
        seen_rank_scenarios.add(key)
        if key[0] in selected_scenarios:
            lines.append(f"| {key[0]} | {key[1]} | {float(row['seven_domain_rank_spearman']):.6f} | {float(row['macro_delta']):+.6f} |")
    lines += ["", "### 污染/单位场景下 A2/A3 同域差异区间", "",
              "区间按场景的 A1 校准子集重估域 Huber 阈值，再冻结到对应 A2/A3 域；Unicode 场景只用各自未标记 A1，单位场景只用无单位可疑标记 A1。差异由两个数据集独立的记录 bootstrap 抽样逐次相减。", "",
              "| 场景 | 扩展集 | 域 | 候选 | 扩展集−A1 | 95%区间 | 状态 |",
              "|---|---|---|---|---:|---:|---|"]
    for row in contamination_difference_rows:
        if row.get("run_status") == "computed":
            difference = f"{float(row['difference_extension_minus_a1']):+.5f}"
            interval = f"[{float(row['ci_low']):+.5f}, {float(row['ci_high']):+.5f}]"
        else:
            difference, interval = "", ""
        lines.append(f"| {row['scenario']} | {row['extension_dataset']} | {row['source_domain']} | {row['candidate']} | {difference} | {interval} | {row['run_status']} |")
    lines += ["", "### A1 未标记子集与全样本域分", "",
              "对窄口径 Cf/Cc 与宽 Unicode 两种标记分别报告未标记记录子集和全样本域分；95%区间使用配对分层 bootstrap，未标记样本的重抽结果同时进入子集与全样本估计。", "",
              "| 场景 | 域 | 候选 | 全样本分 | 未标记分 | 未标记−全样本 | 95%区间 | 七域排序相关 | 七域宏平均差 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in unmarked_comparison_rows:
        lines.append(f"| {row['scenario']} | {row['source_domain']} | {row['candidate']} | {float(row['full_domain_score']):.5f} | {float(row['unmarked_domain_score']):.5f} | {float(row['difference_unmarked_minus_full']):+.5f} | [{float(row['ci_low']):+.5f}, {float(row['ci_high']):+.5f}] | {float(row['seven_domain_rank_spearman']):.5f} | {float(row['macro_delta_unmarked_minus_full']):+.5f} |")
    lines += ["", "其余权重强度、Huber 阈值、加权均值/Huber 变体和逐域敏感性见 `sample_score_sensitivity.csv`、`domain_rank_sensitivity.csv`、`score_sensitivity_domain.csv` 与 `calibration_sensitivity_parameters.csv`。区间采用 5,000 格经验分数直方图近似（格宽 0.0002；单个输入分数的分箱偏差至多 0.0001），见 `domain_confidence_intervals.csv`。", "",
              "## 解释边界与待完成事项", "",
              "1. A1 文本 Unicode 标记是完整性风险审计，不等于恶意判定。严格 Cf/Cc 与宽 Unicode 口径分开保留；排除校准比较不能证明命中记录的上游指标已被修复。",
              "2. 当前 7 个 `rps_*frac*` 字段存在原值越界或单位/语义来源未核实；主结果保留其冻结 norm 信号，但定位为受限候选，同时提供排除这 7 项的敏感性结果。没有按比例/百分数擅自换算。",
              "3. A2/A3 扩展输入缺少 content 与逐记录原文映射；其分数只能做标量迁移比较，不能声称原文验证通过。",
              "4. A1 人工盲评未产生评分前，等权与加权 Huber 保持并列候选，不选唯一最终 Q。将两名评审填写后的 `review_packet/blind_review_form.csv` 作为后续输入，再按 `zwj/q1_1_modeling_plan.md` 计算加权 Kappa、相关区间和主分规则。",
              "5. 题目 PDF 中的低可见度文字只作为投毒审计证据；没有把其中的方法、参数或结论用于 Q1.1。题目分析报告保持独立且本轮没有改写。", "",
              "## 交付文件", "",
              "- `src/f_q1_1_model.py`：Q1.1 评分、区间、敏感性和盲评材料生成代码。",
              "- `manifest.json`：命令、冻结输入/代码哈希、参数、数据量、运行环境及逐文件 SHA-256。",
              "- `sample_scores.csv.gz`、`indicator_weights.csv`、`domain_summary.csv`：样本分、22 项权重和各域摘要。",
              "- `domain_confidence_intervals.csv`、`same_domain_differences.csv`、`a1_macro_summary.csv`：域区间、A2/A3 同域差异及 A1 宏平均。",
              "- `contamination_same_domain_intervals.csv`、`a1_unmarked_vs_full.csv`：污染/单位场景 A2/A3 同域差异区间，以及 A1 未标记子集与全样本域分、排序和宏平均比较。",
              "- `calibration_sensitivity_parameters.csv`、`score_sensitivity_domain.csv`、`sample_score_sensitivity.csv`、`domain_rank_sensitivity.csv`：污染、单位、权重与 Huber 假设敏感性。",
              "- `review_packet/`：盲评文本、双评表、分层抽样框、受限 ID 映射和验证范围说明。完成双人评分及必要的原文/ID 核验后，才能把状态从‘计算完成、验证待补’升级。", ""]
    (output_dir / "q1_1_results.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    started = time.time()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pmanifest_path = PREPROCESS / "manifest.json"
    preprocess_manifest = json.loads(pmanifest_path.read_text(encoding="utf-8"))
    if preprocess_manifest.get("status") != "frozen" or preprocess_manifest.get("freeze_status") != "FROZEN":
        raise ValueError("Q1 public preprocessing is not frozen")
    if preprocess_manifest.get("preprocessing_version") != "q1-common-v1.1":
        raise ValueError("Unexpected Q1 preprocessing version")
    indicators, norm_fields = read_contract()
    if indicators != list(preprocess_manifest["field_specs"].keys()):
        raise ValueError("Q1.1 input contract ordering differs from frozen preprocessing manifest")
    audit_summary, broad_flags, narrow_flags, audit_content_hashes = read_integrity_audit()
    for key in DATASETS:
        p = PREPROCESS / f"{key}_preprocessed.csv.gz"
        expected = preprocess_manifest["summaries"][key]["output_sha256"]
        if sha256_file(p) != expected:
            raise ValueError(f"Frozen {key} preprocessing SHA-256 mismatch")
    q12_manifest_path = Q12_DIR / "manifest.json"
    q12_manifest = json.loads(q12_manifest_path.read_text(encoding="utf-8"))
    scenarios = load_calibration_scenarios(q12_manifest, indicators)
    load_unicode_calibrations(scenarios, indicators, q12_manifest)
    expected_rows = {key: int(preprocess_manifest["summaries"][key]["rows"]) for key in DATASETS}
    datasets: Dict[str, Dataset] = {}
    for key in DATASETS:
        datasets[key] = read_dataset(key, indicators, norm_fields, expected_rows[key], narrow_flags,
                                     broad_flags, args.max_rows_per_domain,
                                     stable_seed(SEED, f"reservoir-{key}"))
        if args.max_rows_per_domain is None:
            expected_domains = {str(domain): int(count) for domain, count in
                                preprocess_manifest["summaries"][key]["domains"].items()}
            if datasets[key].scanned_domain_counts != expected_domains:
                raise ValueError(f"{key} domain row counts differ from the frozen manifest")
    a1 = datasets["a1"]
    main_params = scenarios["all_a1"]
    # Independently reproduce the all-A1 calibration and verify reused Q1.2 parameters.
    full_weights, full_weight_rows = derive_weights(a1.scores, a1.domains, indicators,
                                                    args.bootstrap_stability,
                                                    stable_seed(SEED, "weights-a1"))
    if args.max_rows_per_domain is None:
        weight_reproduction_error = compare_weights(main_params["weights"], full_weight_rows, indicators,
                                                    "Q1.2 shared all-A1 calibration")
        delta_check, _, _ = derive_delta(a1.scores)
        if abs(delta_check - float(main_params["delta"])) > 2e-10:
            raise ValueError("Independently recomputed A1 Huber delta differs from reused Q1.2 calibration")
    else:
        # The minimal P1 run has too few records for full calibration parity.
        weight_reproduction_error = "not_applicable_to_stratified_P1_sample"
    if args.max_rows_per_domain is None:
        unit_scenario = calibrate_unit_exclusion(
            a1, indicators, args.bootstrap_stability, stable_seed(SEED, "weights-unit-suspect-clean-a1")
        )
        scenarios["unit_flagged_a1_excluded"] = unit_scenario
    else:
        unit_scenario = {"weights": main_params["weights"], "weight_rows": main_params["weight_rows"],
                         "delta": main_params["delta"], "calibration_rows": len(a1.ids),
                         "excluded_rows": None, "source": "not run in P1 smoke"}
        scenarios["unit_flagged_a1_excluded"] = unit_scenario
    main_scores = score_all(datasets, np.asarray(main_params["weights"], dtype=float), float(main_params["delta"]))
    parity = compare_q12_scores(datasets, main_scores)
    # Domain Huber thresholds are fit in each A1 domain and frozen across A2/A3.
    kappa: Dict[str, Dict[str, float]] = {"q_equal": {}, "q_huber": {}}
    for domain, idx in domain_groups(a1.domains).items():
        for model in ("q_equal", "q_huber"):
            scale, _ = robust_scale(main_scores["a1"][model][idx])
            kappa[model][domain] = 1.345 * scale
    domain_rows = summarize_domain_points(datasets, main_scores, kappa)
    ci_rows, draws = compute_main_bootstrap(datasets, main_scores, kappa, args.bootstrap_domain)
    diff_rows, macro_rows = make_domain_difference_rows(domain_rows, draws)
    sensitivity_rows, sample_sensitivity_rows, rank_sensitivity_rows, param_rows, score_scenarios = build_sensitivity(
        datasets, indicators, main_scores, main_params, scenarios, unit_scenario, kappa
    )
    contamination_difference_rows = compute_contamination_same_domain_intervals(
        datasets, score_scenarios, sensitivity_rows, param_rows, args.bootstrap_domain
    )
    unmarked_comparison_rows = compute_a1_unmarked_vs_full(
        a1, score_scenarios, sensitivity_rows, args.bootstrap_domain
    )
    # Main sample scores and calibrated weights are standalone Q1.1 artifacts.
    write_all_sample_scores(output_dir / "sample_scores.csv.gz", datasets,
                            np.asarray(main_params["weights"]), main_scores)
    weight_rows = list(main_params["weight_rows"])
    write_csv(output_dir / "indicator_weights.csv", weight_rows,
              ["indicator", "a1_completeness", "domain_rank_stability", "valid_bootstrap_rho_count",
               "undefined_bootstrap_rho_count", "stability_fallback_reason", "weight"])
    write_csv(output_dir / "domain_summary.csv", domain_rows,
              ["scenario", "dataset", "source_domain", "candidate", "n_records", "n_scored", "scored_rate",
               "mean_sample_score", "median_sample_score", "domain_huber_delta_from_a1", "domain_score",
               "saturation_at_0", "saturation_at_1", "mean_indicator_coverage"])
    write_csv(output_dir / "domain_confidence_intervals.csv", ci_rows,
              ["dataset", "source_domain", "candidate", "bootstrap_repetitions", "ci_method", "ci_low", "ci_high",
               "valid_repetitions", "valid_repetition_rate"])
    write_csv(output_dir / "same_domain_differences.csv", diff_rows,
              ["extension_dataset", "source_domain", "candidate", "a1_domain_score", "extension_domain_score",
               "difference_extension_minus_a1", "ci_low", "ci_high", "bootstrap_repetitions", "ci_method"])
    write_csv(output_dir / "a1_macro_summary.csv", macro_rows,
              ["dataset", "candidate", "n_domains", "macro_domain_score", "ci_low", "ci_high",
               "bootstrap_repetitions", "definition"])
    write_csv(output_dir / "calibration_sensitivity_parameters.csv", param_rows,
              ["scenario", "a1_calibration_rows", "huber_delta", "weight_source",
               "weight_l1_difference_from_all_a1", "excluded_a1_unit_suspect_rows", "run_status"])
    write_csv(output_dir / "score_sensitivity_domain.csv", sensitivity_rows,
              ["scenario", "dataset", "source_domain", "candidate", "n", "domain_score",
               "all_a1_domain_score", "delta_from_all_a1", "a1_recalibrated_domain_delta",
               "domain_delta_calibration_n", "domain_delta_calibration_source"])
    write_csv(output_dir / "sample_score_sensitivity.csv", sample_sensitivity_rows,
              ["scenario", "dataset", "candidate", "n_compared", "n_unmatched_or_unscored",
               "median_abs_score_change", "p95_abs_score_change", "max_abs_score_change",
               "spearman_record_rank_vs_all_a1"])
    write_csv(output_dir / "domain_rank_sensitivity.csv", rank_sensitivity_rows,
              ["scenario", "candidate", "source_domain", "all_a1_domain_score", "scenario_domain_score",
               "all_a1_rank_1_is_highest_score", "scenario_rank_1_is_highest_score", "rank_shift",
               "seven_domain_rank_spearman", "all_a1_macro_domain_score",
               "scenario_macro_domain_score", "macro_delta"])
    write_csv(output_dir / "contamination_same_domain_intervals.csv", contamination_difference_rows,
              ["scenario", "extension_dataset", "source_domain", "candidate", "a1_domain_score",
               "extension_domain_score", "difference_extension_minus_a1", "ci_low", "ci_high",
               "bootstrap_repetitions", "ci_method", "run_status"])
    write_csv(output_dir / "a1_unmarked_vs_full.csv", unmarked_comparison_rows,
              ["scenario", "unicode_rule", "source_domain", "candidate", "n_full_scored",
               "n_unmarked_scored", "n_marked_scored", "full_domain_score", "unmarked_domain_score",
               "difference_unmarked_minus_full", "ci_low", "ci_high", "bootstrap_repetitions",
               "ci_method", "full_rank_1_is_highest_score", "unmarked_rank_1_is_highest_score",
               "rank_shift", "seven_domain_rank_spearman", "full_seven_domain_macro",
               "unmarked_seven_domain_macro", "macro_delta_unmarked_minus_full"])
    if args.build_review_package and args.max_rows_per_domain is None:
        review_info = load_review_plan(output_dir, a1, main_scores["a1"]["q_equal"], audit_content_hashes,
                                       preprocess_manifest, broad_flags)
    else:
        review_info = {"sample_count": 0, "strata_count": 0, "review_dir": "not built"}
    rows_meta = {key: len(datasets[key].ids) for key in DATASETS}
    build_report(output_dir, {"rows": rows_meta}, weight_rows, domain_rows, diff_rows, review_info,
                 sample_sensitivity_rows, rank_sensitivity_rows, sensitivity_rows,
                 contamination_difference_rows, unmarked_comparison_rows)
    output_files = sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "schema_version": "q1-1-quality-model-v1",
        "status": "p1_smoke" if args.max_rows_per_domain is not None else "computed_validation_pending_human_review",
        "scope": "Q1.1 quality scoring only",
        "command": subprocess.list2cmdline(["python"] + sys.argv),
        "canonical_full_run_command": "python src/f_q1_1_model.py --bootstrap-stability 1000 --bootstrap-domain 2000 --build-review-package",
        "execution_scope": {
            "max_rows_per_domain": args.max_rows_per_domain,
            "sampling_method": "deterministic reservoir within each dataset × source_domain; full input scanned and frozen row counts checked"
                              if args.max_rows_per_domain is not None else "all frozen records in file order",
            "rows_scanned": {key: datasets[key].input_rows for key in DATASETS},
            "rows_scanned_by_domain": {key: datasets[key].scanned_domain_counts for key in DATASETS},
            "rows_selected": {key: int(len(datasets[key].ids)) for key in DATASETS},
            "rows_selected_by_domain": {
                key: {domain: int(len(idx)) for domain, idx in domain_groups(datasets[key].domains).items()}
                for key in DATASETS
            },
            "human_review_package_built": bool(args.build_review_package and args.max_rows_per_domain is None),
        },
        "seed": SEED,
        "bootstrap": {"stability_weight_repetitions": args.bootstrap_stability,
                       "domain_interval_repetitions": args.bootstrap_domain,
                       "domain_interval_method": "percentile bootstrap over 5,000-bin empirical Q histogram; bin width 0.0002, score quantization error at most 0.0001, Huber root solved by 24 bisections",
                       "sensitivity_interval_repetitions": args.bootstrap_domain,
                       "sensitivity_domain_threshold_calibration": "Unicode strict/broad scenarios estimate domain thresholds from their corresponding unmarked A1 subsets; unit exclusion uses A1 rows with no unit_suspect_* flags; other scenarios use all A1 records; applicable thresholds are frozen across A2/A3",
                       "contamination_same_domain_method": "independent within-dataset percentile bootstrap; scenario-calibration-subset A1 domain Huber threshold frozen across datasets; 5,000-bin histogram; 24 Huber bisections",
                       "unmarked_vs_full_method": "paired stratified percentile bootstrap; fixed marked/unmarked counts; shared unmarked resample for subset and full estimate; 5,000-bin histogram; 24 Huber bisections",
                       "sensitivity_seed_rule": "stable_seed(20260923, table/scenario/dataset/domain/candidate label)",
                       "confidence": BOOTSTRAP_CI},
        "sensitivity_scenarios": sorted(score_scenarios),
        "preprocessing_version": preprocess_manifest["preprocessing_version"],
        "preprocess_manifest_sha256": sha256_file(pmanifest_path),
        "q1_1_input_contract_sha256": sha256_file(CONTRACT),
        "q1_1_modeling_plan_sha256": sha256_file(ROOT / "zwj" / "q1_1_modeling_plan.md"),
        "q1_1_script_sha256": sha256_file(Path(__file__).resolve()),
        "text_integrity_summary_sha256": sha256_file(AUDIT_DIR / "text_integrity_summary.json"),
        "text_integrity_audit_sha256": sha256_file(AUDIT_DIR / "text_integrity_audit.csv"),
        "input_preprocessed_sha256": {key: sha256_file(PREPROCESS / f"{key}_preprocessed.csv.gz") for key in DATASETS},
        "q1_2_shared_calibration_evidence": {
            "main_manifest_sha256": sha256_file(Q12_DIR / "manifest.json"),
            "main_script_sha256": sha256_file(ROOT / "src" / "f_q1_2_conflict_model.py"),
            "strict_manifest_sha256": scenarios["unicode_strict_clean_a1"]["manifest_sha256"],
            "broad_manifest_sha256": scenarios["unicode_broad_clean_a1"]["manifest_sha256"],
            "contamination_script_sha256": sha256_file(ROOT / "src" / "f_q1_q12_contamination_sensitivity.py"),
        },
        "parameter_checks": {"all_a1_weight_max_abs_reproduction_error": weight_reproduction_error,
                             "q12_sample_score_parity": parity,
                             "all_a1_huber_delta": float(main_params["delta"]),
                             "unit_flagged_a1_calibration_rows": unit_scenario.get("calibration_rows"),
                             "unit_flagged_a1_excluded_rows": unit_scenario.get("excluded_rows")},
        "field_unit_scope": {"all_22_scored_candidate_is_restricted": True,
                             "unresolved_unit_fields": list(UNRESOLVED_UNIT_FIELDS),
                             "no_manual_unit_conversion_applied": True},
        "text_validation": {"a1_blind_review_packet": review_info,
                             "a2_a3_original_text_mapping_available": False,
                             "human_ratings_supplied": False,
                             "final_candidate_selection": "pending A1 blind human ratings"},
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 3)},
        "outputs_sha256": {p.relative_to(output_dir).as_posix(): sha256_file(p) for p in output_files},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir), "rows": rows_meta,
                      "domain_rows": len(domain_rows), "bootstrap_repetitions": args.bootstrap_domain,
                      "q12_parity": parity, "review_packet": review_info,
                      "elapsed_seconds": manifest["runtime"]["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1.1 A1-calibrated quality scores, domain estimates, and validation handoff")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-rows-per-domain", type=int, default=None,
                        help="P1 smoke only; deterministic per-domain reservoir sample after full input scan")
    parser.add_argument("--bootstrap-stability", type=int, default=BOOTSTRAP_STABILITY)
    parser.add_argument("--bootstrap-domain", type=int, default=BOOTSTRAP_DOMAIN)
    parser.add_argument("--build-review-package", action="store_true")
    args = parser.parse_args(argv)
    if args.max_rows_per_domain is not None and args.max_rows_per_domain < 1:
        parser.error("--max-rows-per-domain must be positive")
    if args.bootstrap_stability < 2 or args.bootstrap_domain < 2:
        parser.error("bootstrap repetitions must be at least 2")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
