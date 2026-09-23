#!/usr/bin/env python
"""Q1.2 conflict diagnostics using the frozen A1-A3 public preprocessing.

The script never reads raw content to build indicators. It consumes the
q1-common-v1.1 ``norm_*`` columns and audit flags, estimates all thresholds on
A1, then applies those frozen rules to same-source-domain records in A2/A3.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESS_DIR = PROJECT_ROOT / "results" / "q1_common_preprocess" / "v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "q1_2_conflict_model" / "v1"
DATASETS = ("a1", "a2", "a3")
BOOTSTRAP_CI = 0.95
INDICATOR_COUNT = 22


@dataclass
class DatasetData:
    dataset: str
    ids: np.ndarray
    domains: np.ndarray
    scores: np.ndarray
    outlier: np.ndarray
    unit_suspect: np.ndarray
    input_rows: int


def _parse_float(value: str) -> float:
    if value is None or value.strip() == "":
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, preserving NaN positions."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if not len(valid_positions):
        return out
    order_local = np.argsort(values[valid_positions], kind="mergesort")
    ordered_positions = valid_positions[order_local]
    ordered_values = values[ordered_positions]
    start = 0
    while start < len(ordered_values):
        end = start + 1
        while end < len(ordered_values) and ordered_values[end] == ordered_values[start]:
            end += 1
        # Ranks are one-based; ties receive the mean of occupied ranks.
        out[ordered_positions[start:end]] = (start + 1 + end) / 2.0
        start = end
    return out


def _spearman(x: np.ndarray, y: np.ndarray, min_n: int = 3) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < min_n:
        return float("nan")
    rx = _rankdata_average(np.asarray(x)[mask])
    ry = _rankdata_average(np.asarray(y)[mask])
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    return float(np.quantile(values, q, method="linear"))


def _quantile_ci(draws: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    alpha = (1.0 - BOOTSTRAP_CI) / 2.0
    return _quantile(finite, alpha), _quantile(finite, 1.0 - alpha)


def _stable_seed(base_seed: int, label: str) -> int:
    offset = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + offset) % (2**32 - 1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(
    csv_path: Path,
    indicators: Sequence[str],
    expected_rows: int,
    max_rows_per_domain: Optional[int],
    seed: int,
) -> DatasetData:
    """Read needed columns; optional deterministic reservoir sample per domain."""
    with gzip.open(csv_path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        positions = {name: i for i, name in enumerate(header)}
        required = ["dataset", "source_domain", "id"]
        required += [f"norm_{name}" for name in indicators]
        required += [f"outlier_{name}" for name in indicators]
        required += [f"unit_suspect_{name}" for name in indicators]
        missing_columns = [name for name in required if name not in positions]
        if missing_columns:
            raise ValueError(f"{csv_path.name} 缺少列: {missing_columns}")

        idx_dataset = positions["dataset"]
        idx_domain = positions["source_domain"]
        idx_id = positions["id"]
        idx_norm = [positions[f"norm_{name}"] for name in indicators]
        idx_outlier = [positions[f"outlier_{name}"] for name in indicators]
        idx_suspect = [positions[f"unit_suspect_{name}"] for name in indicators]
        dataset_name = csv_path.name[:2].lower()
        seen_rows = 0

        if max_rows_per_domain is None:
            ids = np.empty(expected_rows, dtype=object)
            domains = np.empty(expected_rows, dtype=object)
            scores = np.full((expected_rows, len(indicators)), np.nan, dtype=np.float64)
            outlier = np.zeros((expected_rows, len(indicators)), dtype=bool)
            unit_suspect = np.zeros((expected_rows, len(indicators)), dtype=bool)
            for row in reader:
                if not row:
                    continue
                if seen_rows >= expected_rows:
                    raise ValueError(f"{csv_path.name} 行数超过冻结清单 {expected_rows}")
                if row[idx_dataset].strip().lower() != dataset_name:
                    raise ValueError(
                        f"{csv_path.name} 数据集列={row[idx_dataset]!r}，预期 {dataset_name!r}"
                    )
                ids[seen_rows] = row[idx_id]
                domains[seen_rows] = row[idx_domain]
                scores[seen_rows, :] = [_parse_float(row[i]) for i in idx_norm]
                outlier[seen_rows, :] = [_parse_bool(row[i]) for i in idx_outlier]
                unit_suspect[seen_rows, :] = [_parse_bool(row[i]) for i in idx_suspect]
                seen_rows += 1
            if seen_rows != expected_rows:
                raise ValueError(
                    f"{csv_path.name} 实际 {seen_rows} 行，与冻结清单 {expected_rows} 不符"
                )
            sampled = False
        else:
            rng = np.random.default_rng(seed)
            reservoirs: Dict[str, List[Tuple[str, List[float], List[bool], List[bool]]]] = {}
            counts: Dict[str, int] = {}
            for row in reader:
                if not row:
                    continue
                seen_rows += 1
                if row[idx_dataset].strip().lower() != dataset_name:
                    raise ValueError(
                        f"{csv_path.name} 数据集列={row[idx_dataset]!r}，预期 {dataset_name!r}"
                    )
                domain = row[idx_domain]
                bucket = reservoirs.setdefault(domain, [])
                counts[domain] = counts.get(domain, 0) + 1
                item = (
                    row[idx_id],
                    [_parse_float(row[i]) for i in idx_norm],
                    [_parse_bool(row[i]) for i in idx_outlier],
                    [_parse_bool(row[i]) for i in idx_suspect],
                )
                if len(bucket) < max_rows_per_domain:
                    bucket.append(item)
                else:
                    replacement = int(rng.integers(0, counts[domain]))
                    if replacement < max_rows_per_domain:
                        bucket[replacement] = item
            if seen_rows != expected_rows:
                raise ValueError(
                    f"{csv_path.name} 全量扫描 {seen_rows} 行，与冻结清单 {expected_rows} 不符"
                )
            row_count = sum(len(v) for v in reservoirs.values())
            ids = np.empty(row_count, dtype=object)
            domains = np.empty(row_count, dtype=object)
            scores = np.full((row_count, len(indicators)), np.nan, dtype=np.float64)
            outlier = np.zeros((row_count, len(indicators)), dtype=bool)
            unit_suspect = np.zeros((row_count, len(indicators)), dtype=bool)
            at = 0
            for domain in sorted(reservoirs):
                for rid, zrow, orow, urow in reservoirs[domain]:
                    ids[at] = rid
                    domains[at] = domain
                    scores[at, :] = zrow
                    outlier[at, :] = orow
                    unit_suspect[at, :] = urow
                    at += 1
            sampled = True

    finite = np.isfinite(scores)
    if np.any(scores[finite] < -1e-12) or np.any(scores[finite] > 1.0 + 1e-12):
        raise ValueError(f"{csv_path.name} 的 norm_* 超出 [0,1]")
    return DatasetData(
        dataset=dataset_name,
        ids=ids,
        domains=domains,
        scores=scores,
        outlier=outlier,
        unit_suspect=unit_suspect,
        input_rows=seen_rows,
    )


def conflict_intensity(scores: np.ndarray) -> np.ndarray:
    """Mean pairwise absolute gap per record, without constructing n x 22 x 22."""
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    m = valid.sum(axis=1)
    ordered = np.sort(np.where(valid, scores, np.inf), axis=1)
    positions = np.arange(1, scores.shape[1] + 1, dtype=float)[None, :]
    included = positions <= m[:, None]
    values = np.where(included, ordered, 0.0)
    coefficients = 2.0 * positions - m[:, None] - 1.0
    numerator = np.sum(np.where(included, coefficients * values, 0.0), axis=1)
    result = np.full(scores.shape[0], np.nan, dtype=float)
    enough = m >= 2
    result[enough] = numerator[enough] / (m[enough] * (m[enough] - 1.0) / 2.0)
    # Guard roundoff only; the underlying index is mathematically in [0,1].
    result[enough] = np.clip(result[enough], 0.0, 1.0)
    return result


def _domain_indices(domains: np.ndarray) -> Dict[str, np.ndarray]:
    return {str(d): np.flatnonzero(domains == d) for d in sorted(set(domains.tolist()))}


def _bootstrap_quantile_ci(
    values: np.ndarray, q: float, repetitions: int, seed: int
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or repetitions < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    n = len(values)
    for b in range(repetitions):
        sample = values[rng.integers(0, n, size=n)]
        draws[b] = _quantile(sample, q)
    return _quantile_ci(draws)


def _bootstrap_group_differences(
    reference: np.ndarray,
    extension: np.ndarray,
    threshold: float,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    ref = np.asarray(reference, dtype=float)
    ext = np.asarray(extension, dtype=float)
    ref = ref[np.isfinite(ref)]
    ext = ext[np.isfinite(ext)]
    if min(len(ref), len(ext)) < 2 or repetitions < 2:
        return {key: float("nan") for key in ("median_diff_low", "median_diff_high", "rate_diff_low", "rate_diff_high")}
    rng = np.random.default_rng(seed)
    median_diffs = np.empty(repetitions, dtype=float)
    rate_diffs = np.empty(repetitions, dtype=float)
    for b in range(repetitions):
        ref_sample = ref[rng.integers(0, len(ref), size=len(ref))]
        ext_sample = ext[rng.integers(0, len(ext), size=len(ext))]
        median_diffs[b] = np.median(ext_sample) - np.median(ref_sample)
        rate_diffs[b] = np.mean(ext_sample > threshold) - np.mean(ref_sample > threshold)
    md_lo, md_hi = _quantile_ci(median_diffs)
    rd_lo, rd_hi = _quantile_ci(rate_diffs)
    return {
        "median_diff_low": md_lo,
        "median_diff_high": md_hi,
        "rate_diff_low": rd_lo,
        "rate_diff_high": rd_hi,
    }


def wilson_interval(successes: int, n: int, confidence_z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    z2 = confidence_z * confidence_z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = confidence_z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    med = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - med)))
    if mad_scale > 1e-12:
        return mad_scale
    q25, q75 = np.quantile(values, [0.25, 0.75], method="linear")
    iqr_scale = float((q75 - q25) / 1.349)
    if iqr_scale > 1e-12:
        return iqr_scale
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return sd if sd > 1e-12 else 0.0


def derive_huber_delta(a1_scores: np.ndarray) -> Tuple[float, float, str]:
    row_median = np.nanmedian(a1_scores, axis=1)
    residuals = a1_scores - row_median[:, None]
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        return 0.0, 0.0, "no_valid_residuals"
    center = float(np.median(residuals))
    mad = float(1.4826 * np.median(np.abs(residuals - center)))
    if mad > 1e-12:
        return 1.345 * mad, mad, "1.4826*MAD"
    q25, q75 = np.quantile(residuals, [0.25, 0.75], method="linear")
    iqr_scale = float((q75 - q25) / 1.349)
    if iqr_scale > 1e-12:
        return 1.345 * iqr_scale, iqr_scale, "IQR/1.349"
    sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    if sd > 1e-12:
        return 1.345 * sd, sd, "SD"
    return 0.0, 0.0, "no_truncation"


def derive_reliability_weights(
    a1: DatasetData, indicators: Sequence[str], repetitions: int, seed: int
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Implement the A1-only domain-rank stability weight from Q1.1."""
    groups = _domain_indices(a1.domains)
    domain_names = list(groups)
    full_medians = np.vstack([np.nanmedian(a1.scores[idx, :], axis=0) for idx in groups.values()])
    completeness = np.mean(np.isfinite(a1.scores), axis=0)
    stabilities = np.full(len(indicators), 0.5, dtype=float)
    undefined_counts = np.zeros(len(indicators), dtype=int)
    valid_counts = np.zeros(len(indicators), dtype=int)
    fallback_reasons = ["insufficient_bootstrap_repetitions"] * len(indicators)
    if repetitions >= 2:
        rng = np.random.default_rng(seed)
        draws = np.full((repetitions, len(indicators)), np.nan, dtype=float)
        for b in range(repetitions):
            sample_medians = []
            for idx in groups.values():
                local = idx[rng.integers(0, len(idx), size=len(idx))]
                sample_medians.append(np.nanmedian(a1.scores[local, :], axis=0))
            boot_medians = np.vstack(sample_medians)
            for j in range(len(indicators)):
                full_values = full_medians[:, j]
                full_values = full_values[np.isfinite(full_values)]
                if len(full_values) < 3 or np.ptp(full_values) == 0:
                    undefined_counts[j] += 1
                    continue
                rho = _spearman(full_medians[:, j], boot_medians[:, j], min_n=3)
                if math.isfinite(rho):
                    draws[b, j] = (1.0 + rho) / 2.0
                    valid_counts[j] += 1
                else:
                    undefined_counts[j] += 1
        for j in range(len(indicators)):
            full_values = full_medians[:, j]
            full_values = full_values[np.isfinite(full_values)]
            if len(full_values) < 3 or np.ptp(full_values) == 0:
                stabilities[j] = 0.5
                fallback_reasons[j] = "constant_or_insufficient_a1_domain_medians"
            elif valid_counts[j] / repetitions < 0.95:
                stabilities[j] = 0.5
                fallback_reasons[j] = "valid_bootstrap_rho_below_95_percent"
            else:
                stabilities[j] = float(np.nanmedian(draws[:, j]))
                fallback_reasons[j] = "none"
    raw_weights = np.sqrt(np.clip(completeness, 0.0, 1.0) * np.clip(stabilities, 0.0, 1.0))
    if float(raw_weights.sum()) <= 0:
        weights = np.full(len(indicators), 1.0 / len(indicators))
    else:
        weights = raw_weights / raw_weights.sum()
    rows = []
    for j, name in enumerate(indicators):
        rows.append({
            "indicator": name,
            "a1_completeness": float(completeness[j]),
            "domain_rank_stability": float(stabilities[j]),
            "valid_bootstrap_rho_count": int(valid_counts[j]),
            "undefined_bootstrap_rho_count": int(undefined_counts[j]),
            "stability_fallback_reason": fallback_reasons[j],
            "weight": float(weights[j]),
        })
    return weights, rows


def huber_location(scores: np.ndarray, weights: np.ndarray, delta: float) -> np.ndarray:
    valid = np.isfinite(scores)
    counts = valid.sum(axis=1)
    if delta <= 1e-12:
        weighted = np.where(valid, scores, 0.0) @ weights
        mass = valid @ weights
        result = np.full(scores.shape[0], np.nan, dtype=float)
        good = mass > 0
        result[good] = weighted[good] / mass[good]
        return result
    low = np.zeros(scores.shape[0], dtype=float)
    high = np.ones(scores.shape[0], dtype=float)
    for _ in range(48):
        middle = (low + high) / 2.0
        residual = np.where(valid, scores - middle[:, None], 0.0)
        psi = np.clip(residual, -delta, delta)
        score = np.sum(psi * weights[None, :], axis=1)
        move_low = score > 0
        low[move_low] = middle[move_low]
        high[~move_low] = middle[~move_low]
    result = (low + high) / 2.0
    result[counts == 0] = np.nan
    return result


def _score_summary(scores: np.ndarray, weights: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_n = np.isfinite(scores).sum(axis=1)
    qeq = np.full(scores.shape[0], np.nan, dtype=float)
    rows_with_values = valid_n > 0
    qeq[rows_with_values] = np.nansum(scores[rows_with_values], axis=1) / valid_n[rows_with_values]
    qh = huber_location(scores, weights, delta)
    return qeq, qh, qh - qeq


def pair_diagnostics(
    data: DatasetData, indicators: Sequence[str]
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[Tuple[int, int], float]]]:
    output: List[Dict[str, object]] = []
    gap_by_domain: Dict[str, Dict[Tuple[int, int], float]] = {}
    for domain, idx in _domain_indices(data.domains).items():
        matrix = data.scores[idx, :]
        gaps: Dict[Tuple[int, int], float] = {}
        for j in range(len(indicators)):
            for k in range(j + 1, len(indicators)):
                x = matrix[:, j]
                y = matrix[:, k]
                valid = np.isfinite(x) & np.isfinite(y)
                n = int(valid.sum())
                gap = float(np.mean(np.abs(x[valid] - y[valid]))) if n else float("nan")
                rho = _spearman(x[valid], y[valid], min_n=3) if n else float("nan")
                gaps[(j, k)] = gap
                output.append({
                    "dataset": data.dataset,
                    "source_domain": domain,
                    "indicator_a": indicators[j],
                    "indicator_b": indicators[k],
                    "pair_n": n,
                    "mean_abs_gap": gap,
                    "spearman_rho": rho,
                })
        gap_by_domain[domain] = gaps
    return output, gap_by_domain


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_gzip_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _wilson_for_mask(flagged: np.ndarray) -> Tuple[float, float]:
    valid = np.asarray(flagged, dtype=bool)
    return wilson_interval(int(valid.sum()), len(valid))


def _domain_summary_rows(
    data: DatasetData,
    intensity: np.ndarray,
    qeq: np.ndarray,
    qh: np.ndarray,
    delta_q: np.ndarray,
    threshold_by_domain: Mapping[str, float],
    threshold_ci_by_domain: Mapping[str, Tuple[float, float]],
) -> List[Dict[str, object]]:
    rows = []
    for domain, idx in _domain_indices(data.domains).items():
        d = intensity[idx]
        finite = np.isfinite(d)
        d = d[finite]
        selected = idx[finite]
        tau = threshold_by_domain.get(domain, float("nan"))
        flagged = d > tau if math.isfinite(tau) else np.zeros(len(d), dtype=bool)
        rate_low, rate_high = wilson_interval(int(flagged.sum()), len(flagged))
        values = data.scores[selected, :]
        valid_values = values[np.isfinite(values)]
        n_zero = int(np.sum(valid_values <= 1e-12))
        n_one = int(np.sum(valid_values >= 1.0 - 1e-12))
        total_cells = int(valid_values.size)
        tau_low, tau_high = threshold_ci_by_domain.get(domain, (float("nan"), float("nan")))
        high_idx = selected[flagged]
        other_idx = selected[~flagged]

        def signed_delta_summary(group_idx: np.ndarray, suffix: str) -> Dict[str, object]:
            group_values = delta_q[group_idx]
            group_values = group_values[np.isfinite(group_values)]
            if not len(group_values):
                return {
                    f"delta_q_mean_{suffix}": float("nan"),
                    f"delta_q_median_{suffix}": float("nan"),
                    f"delta_q_q25_{suffix}": float("nan"),
                    f"delta_q_q75_{suffix}": float("nan"),
                    f"delta_q_positive_share_{suffix}": float("nan"),
                    f"delta_q_negative_share_{suffix}": float("nan"),
                }
            return {
                f"delta_q_mean_{suffix}": float(np.mean(group_values)),
                f"delta_q_median_{suffix}": float(np.median(group_values)),
                f"delta_q_q25_{suffix}": _quantile(group_values, 0.25),
                f"delta_q_q75_{suffix}": _quantile(group_values, 0.75),
                f"delta_q_positive_share_{suffix}": float(np.mean(group_values > 0)),
                f"delta_q_negative_share_{suffix}": float(np.mean(group_values < 0)),
            }
        rows.append({
            "dataset": data.dataset,
            "source_domain": domain,
            "n": int(len(idx)),
            "n_conflict_valid": int(len(d)),
            "mean_conflict_intensity": float(np.mean(d)) if len(d) else float("nan"),
            "median_conflict_intensity": float(np.median(d)) if len(d) else float("nan"),
            "p90_conflict_intensity": _quantile(d, 0.90),
            "p95_conflict_intensity": _quantile(d, 0.95),
            "p97_5_conflict_intensity": _quantile(d, 0.975),
            "a1_domain_threshold": tau,
            "threshold_ci_low": tau_low,
            "threshold_ci_high": tau_high,
            "high_conflict_n": int(flagged.sum()),
            "high_conflict_rate": float(flagged.mean()) if len(flagged) else float("nan"),
            "high_conflict_wilson_low": rate_low,
            "high_conflict_wilson_high": rate_high,
            "threshold_tie_rate": float(np.mean(d == tau)) if math.isfinite(tau) and len(d) else float("nan"),
            "norm_zero_cell_rate": n_zero / total_cells if total_cells else float("nan"),
            "norm_one_cell_rate": n_one / total_cells if total_cells else float("nan"),
            "q_equal_mean": float(np.nanmean(qeq[idx])) if np.isfinite(qeq[idx]).any() else float("nan"),
            "q_huber_mean": float(np.nanmean(qh[idx])) if np.isfinite(qh[idx]).any() else float("nan"),
            "delta_q_median": float(np.nanmedian(delta_q[idx])) if np.isfinite(delta_q[idx]).any() else float("nan"),
            "median_abs_delta_q_high_conflict": float(np.nanmedian(np.abs(delta_q[high_idx]))) if len(high_idx) and np.isfinite(delta_q[high_idx]).any() else float("nan"),
            "median_abs_delta_q_other": float(np.nanmedian(np.abs(delta_q[other_idx]))) if len(other_idx) and np.isfinite(delta_q[other_idx]).any() else float("nan"),
            **signed_delta_summary(high_idx, "high_conflict"),
            **signed_delta_summary(other_idx, "other"),
        })
    return rows


def _sensitivity_rows(
    a1: DatasetData,
    all_data: Sequence[DatasetData],
    variant_intensity: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, float, str], float]]:
    rows: List[Dict[str, object]] = []
    thresholds: Dict[Tuple[str, float, str], float] = {}
    a1_indices = _domain_indices(a1.domains)
    # Main-threshold alternatives: within-domain 90%, 95%, 97.5%, plus pooled A1 95%.
    main_a1 = variant_intensity["main"]["a1"]
    global_q95 = _quantile(main_a1, 0.95)
    for q in (0.90, 0.95, 0.975):
        for domain, idx in a1_indices.items():
            tau = _quantile(main_a1[idx], q)
            thresholds[("main", q, domain)] = tau
        for data in all_data:
            intensity = variant_intensity["main"][data.dataset]
            for domain, idx in _domain_indices(data.domains).items():
                tau = thresholds.get(("main", q, domain), float("nan"))
                d = intensity[idx]
                d = d[np.isfinite(d)]
                rows.append({
                    "variant": "main",
                    "threshold_rule": f"within_domain_q{q:g}",
                    "quantile": q,
                    "dataset": data.dataset,
                    "source_domain": domain,
                    "threshold": tau,
                    "n": int(len(d)),
                    "high_conflict_rate": float(np.mean(d > tau)) if len(d) and math.isfinite(tau) else float("nan"),
                })
    for data in all_data:
        d = variant_intensity["main"][data.dataset]
        for domain, idx in _domain_indices(data.domains).items():
            values = d[idx]
            values = values[np.isfinite(values)]
            rows.append({
                "variant": "main",
                "threshold_rule": "pooled_a1_q95",
                "quantile": 0.95,
                "dataset": data.dataset,
                "source_domain": domain,
                "threshold": global_q95,
                "n": int(len(values)),
                "high_conflict_rate": float(np.mean(values > global_q95)) if len(values) else float("nan"),
            })

    # Flag-based sensitivity: recompute A1 q95 after removing marked indicators.
    for variant in ("outlier_excluded", "unit_suspect_excluded", "both_flags_excluded"):
        for domain, idx in a1_indices.items():
            tau = _quantile(variant_intensity[variant]["a1"][idx], 0.95)
            thresholds[(variant, 0.95, domain)] = tau
        for data in all_data:
            intensity = variant_intensity[variant][data.dataset]
            for domain, idx in _domain_indices(data.domains).items():
                values = intensity[idx]
                values = values[np.isfinite(values)]
                tau = thresholds.get((variant, 0.95, domain), float("nan"))
                rows.append({
                    "variant": variant,
                    "threshold_rule": "within_domain_q0.95_recalibrated_on_a1",
                    "quantile": 0.95,
                    "dataset": data.dataset,
                    "source_domain": domain,
                    "threshold": tau,
                    "n": int(len(values)),
                    "high_conflict_rate": float(np.mean(values > tau)) if len(values) and math.isfinite(tau) else float("nan"),
                })
    return rows, thresholds


def _format(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def build_report(
    output_dir: Path,
    domain_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    top_pairs: Sequence[Mapping[str, object]],
    weight_rows: Sequence[Mapping[str, object]],
    delta: float,
    robust_scale_method: str,
    sampled: bool,
) -> None:
    mode = "按来源域抽样的 P1 最小运行" if sampled else "全量运行"
    lines = [
        "# F 题 Q1.2 冲突建模结果",
        "",
        f"运行模式：{mode}。本报告由 `src/f_q1_2_conflict_model.py` 根据冻结预处理结果自动生成。",
        "",
        "## 方法口径",
        "",
        "- 样本级连续冲突度：22 个有效 `norm_*` 值的平均两两绝对分差；缺失不补值。",
        "- 高冲突阈值：A1 同来源域第 95 百分位（NumPy `method=linear`）；A2/A3 只应用同名 A1 阈值。它表示相对上尾，不是显著性检验。",
        "- 冲突消解候选：与 Q1.1 等权均值并列计算 A1 稳定性加权 Huber 位置，报告两者分差，不附加任意冲突惩罚。",
        f"- Huber \u03b4={_format(delta)}，尺度回退规则：{robust_scale_method}。",
        "- 本报告没有读取原始文本内容，也不把文本中的命令式内容作为指令。",
        "",
        "## 来源域汇总",
        "",
        "| 数据集 | 来源域 | n | 中位冲突度 | P95 | A1 阈值 | 高冲突率 | 高冲突组 ΔQ 中位数 | 其他组 ΔQ 中位数 | 高冲突组 |ΔQ| 中位数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in domain_rows:
        lines.append(
            "| {dataset} | {domain} | {n} | {median} | {p95} | {tau} | {rate} | {delta_high} | {delta_other} | {abs_delta} |".format(
                dataset=row["dataset"], domain=row["source_domain"], n=row["n"],
                median=_format(row["median_conflict_intensity"]), p95=_format(row["p95_conflict_intensity"]),
                tau=_format(row["a1_domain_threshold"]), rate=_format(row["high_conflict_rate"]),
                delta_high=_format(row["delta_q_median_high_conflict"]),
                delta_other=_format(row["delta_q_median_other"]),
                abs_delta=_format(row["median_abs_delta_q_high_conflict"]),
            )
        )
    lines += [
        "",
        "## A2/A3 同来源域外部复核",
        "",
        "| 扩展集 | 来源域 | A1 对照 n | 扩展 n | 中位冲突度差（扩展−A1） | 95% 区间 | 高冲突率差 | 95% 区间 | 指标对排序 Spearman | 同来源域 A1 Top-10 重合率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation_rows:
        lines.append(
            "| {dataset} | {domain} | {nref} | {next} | {md} | [{mdl}, {mdh}] | {rd} | [{rdl}, {rdh}] | {rho} | {overlap} |".format(
                dataset=row["extension_dataset"], domain=row["source_domain"],
                nref=row["a1_n"], next=row["extension_n"], md=_format(row["median_conflict_diff"]),
                mdl=_format(row["median_diff_ci_low"]), mdh=_format(row["median_diff_ci_high"]),
                rd=_format(row["high_conflict_rate_diff"]), rdl=_format(row["rate_diff_ci_low"]),
                rdh=_format(row["rate_diff_ci_high"]), rho=_format(row["pair_rank_spearman"]),
                overlap=_format(row["top10_overlap_rate"]),
            )
        )
    lines += [
        "",
        "## A1 宏平均分差最大的指标对（正文摘要前 10 对）",
        "",
        "| 排名 | 指标对 | A1 七域宏平均分差 |",
        "|---:|---|---:|",
    ]
    for rank, row in enumerate(top_pairs, start=1):
        lines.append(f"| {rank} | {row['indicator_a']} × {row['indicator_b']} | {_format(row['macro_mean_abs_gap'])} |")
    lines += [
        "",
        "## 指标可靠性权重摘要",
        "",
        "权重由 Q1.1 预先定义的 A1 域排序稳定性公式产生；此处只列出权重最大的五项。完整权重表见 `indicator_weights.csv`。",
        "",
        "| 指标 | 权重 | 域排序稳定性 | 稳定性回退 |",
        "|---|---:|---:|---|",
    ]
    for row in sorted(weight_rows, key=lambda item: float(item["weight"]), reverse=True)[:5]:
        lines.append(f"| {row['indicator']} | {_format(row['weight'])} | {_format(row['domain_rank_stability'])} | {row['stability_fallback_reason']} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "ΔQ 定义为 Huber 分减等权均值，正值表示 Huber 分更高，负值表示更低；领域表同时给出高冲突组和其余样本的有符号中位数，完整均值、四分位数和方向比例见 `domain_summary.csv`。D 值和阈值只描述方向统一后的指标分歧。它们不能判定哪一项指标错，也不能证明原文有缺陷。若 A2/A3 的冲突度或主要指标对排序未复现，应报告迁移失败并收窄结论。最终质量分仍需依据 Q1.1 的盲评与敏感性证据选择；本次计算不单独确认最终 Q。",
        "",
    ]
    (output_dir / "q1_2_conflict_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    started = time.time()
    preprocess_dir = Path(args.preprocess_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    preprocess_manifest_path = preprocess_dir / "manifest.json"
    if not preprocess_manifest_path.exists():
        raise FileNotFoundError(f"缺少冻结预处理清单：{preprocess_manifest_path}")
    manifest = json.loads(preprocess_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("freeze_status") != "FROZEN":
        raise ValueError("公共预处理清单未标记为 FROZEN；停止 Q1.2 计算")
    if manifest.get("preprocessing_version") != "q1-common-v1.1":
        raise ValueError(f"预处理版本不符：{manifest.get('preprocessing_version')}")
    if manifest.get("audit_checks", {}).get("a2_a3_refit_calibration") is not False:
        raise ValueError("冻结清单未证明 A2/A3 未重拟合校准参数")
    indicators = list(manifest.get("field_specs", {}).keys())
    if len(indicators) != INDICATOR_COUNT:
        raise ValueError(f"预期 22 个质量指标，清单记录 {len(indicators)} 个")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng_seed = int(args.seed)
    expected_rows = {key: int(value["rows"]) for key, value in manifest["summaries"].items()}
    frozen_output_hashes: Dict[str, str] = {}
    datasets: Dict[str, DatasetData] = {}
    for key in DATASETS:
        path = preprocess_dir / f"{key}_preprocessed.csv.gz"
        summary = manifest["summaries"].get(key, {})
        expected_name = Path(str(summary.get("output", path.name))).name
        if expected_name != path.name:
            raise ValueError(f"{key} 预处理输出文件名与冻结清单不符：{expected_name}")
        expected_hash = str(summary.get("output_sha256", ""))
        if not expected_hash:
            raise ValueError(f"{key} 冻结清单缺少 output_sha256")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{key} 预处理文件 SHA-256 与冻结清单不符；停止 Q1.2 计算")
        frozen_output_hashes[key] = actual_hash
        data = _read_rows(
            path,
            indicators,
            expected_rows[key],
            args.max_rows_per_domain,
            _stable_seed(rng_seed, f"reservoir-{key}"),
        )
        datasets[key] = data
    # Confirm the files remained unchanged while the model was reading them.
    for key in DATASETS:
        path = preprocess_dir / f"{key}_preprocessed.csv.gz"
        if _sha256_file(path) != frozen_output_hashes[key]:
            raise ValueError(f"{key} 预处理文件在运行期间发生变化；停止并重跑 Q1.2")
    a1 = datasets["a1"]
    weights, weight_rows = derive_reliability_weights(
        a1, indicators, args.weight_bootstrap, _stable_seed(rng_seed, "weights-a1")
    )
    delta, robust_scale, scale_method = derive_huber_delta(a1.scores)

    intensity: Dict[str, np.ndarray] = {}
    variant_intensity: Dict[str, Dict[str, np.ndarray]] = {
        "main": {},
        "outlier_excluded": {},
        "unit_suspect_excluded": {},
        "both_flags_excluded": {},
    }
    q_equal: Dict[str, np.ndarray] = {}
    q_huber: Dict[str, np.ndarray] = {}
    delta_q: Dict[str, np.ndarray] = {}
    pair_rows: List[Dict[str, object]] = []
    pair_gaps: Dict[str, Dict[str, Dict[Tuple[int, int], float]]] = {}
    all_pair_stats: Dict[str, List[Dict[str, object]]] = {}

    for key in DATASETS:
        data = datasets[key]
        intensity[key] = conflict_intensity(data.scores)
        variant_intensity["main"][key] = intensity[key]
        outlier_scores = data.scores.copy()
        outlier_scores[data.outlier] = np.nan
        unit_scores = data.scores.copy()
        unit_scores[data.unit_suspect] = np.nan
        both_scores = data.scores.copy()
        both_scores[data.outlier | data.unit_suspect] = np.nan
        variant_intensity["outlier_excluded"][key] = conflict_intensity(outlier_scores)
        variant_intensity["unit_suspect_excluded"][key] = conflict_intensity(unit_scores)
        variant_intensity["both_flags_excluded"][key] = conflict_intensity(both_scores)
        q_equal[key], q_huber[key], delta_q[key] = _score_summary(data.scores, weights, delta)
        stats, gaps = pair_diagnostics(data, indicators)
        all_pair_stats[key] = stats
        pair_gaps[key] = gaps
        pair_rows.extend(stats)

    a1_domains = _domain_indices(a1.domains)
    threshold_by_domain: Dict[str, float] = {}
    threshold_ci: Dict[str, Tuple[float, float]] = {}
    threshold_rows: List[Dict[str, object]] = []
    for domain, idx in a1_domains.items():
        values = intensity["a1"][idx]
        values = values[np.isfinite(values)]
        tau = _quantile(values, 0.95)
        ci = _bootstrap_quantile_ci(
            values, 0.95, args.threshold_bootstrap, _stable_seed(rng_seed, f"threshold-{domain}")
        )
        threshold_by_domain[domain] = tau
        threshold_ci[domain] = ci
        threshold_rows.append({
            "variant": "main",
            "threshold_rule": "a1_source_domain_q95_linear",
            "source_domain": domain,
            "quantile": 0.95,
            "threshold": tau,
            "threshold_ci_low": ci[0],
            "threshold_ci_high": ci[1],
            "a1_n": int(len(values)),
            "a1_actual_exceedance_rate": float(np.mean(values > tau)) if len(values) else float("nan"),
            "a1_tie_rate": float(np.mean(values == tau)) if len(values) else float("nan"),
        })
    sensitivity_rows, sensitivity_thresholds = _sensitivity_rows(
        a1, [datasets[k] for k in DATASETS], variant_intensity
    )
    for (variant, q, domain), tau in sensitivity_thresholds.items():
        if variant == "main" and q == 0.95:
            continue
        threshold_rows.append({
            "variant": variant,
            "threshold_rule": f"a1_{'pooled' if domain == '__pooled__' else 'source_domain'}_q{q:g}",
            "source_domain": domain,
            "quantile": q,
            "threshold": tau,
            "threshold_ci_low": float("nan"),
            "threshold_ci_high": float("nan"),
            "a1_n": int(len(a1_domains.get(domain, np.array([], dtype=int)))) if domain != "__pooled__" else int(len(intensity["a1"])),
            "a1_actual_exceedance_rate": float("nan"),
            "a1_tie_rate": float("nan"),
        })

    # Pair ranking: macro-average A1 gaps choose the report's descriptive top 10.
    domain_gap_maps = pair_gaps["a1"]
    macro_gaps: Dict[Tuple[int, int], float] = {}
    for j in range(len(indicators)):
        for k in range(j + 1, len(indicators)):
            values = [domain_gap_maps[d].get((j, k), float("nan")) for d in a1_domains]
            finite = [v for v in values if math.isfinite(v)]
            macro_gaps[(j, k)] = float(np.mean(finite)) if finite else float("nan")
    ordered_pairs = sorted(
        macro_gaps,
        key=lambda pair: (
            -macro_gaps[pair] if math.isfinite(macro_gaps[pair]) else float("inf"),
            indicators[pair[0]], indicators[pair[1]],
        ),
    )
    top10_pairs = ordered_pairs[:10]
    top_pair_rows = [
        {"indicator_a": indicators[j], "indicator_b": indicators[k], "macro_mean_abs_gap": macro_gaps[(j, k)]}
        for j, k in top10_pairs
    ]
    ranks = {pair: rank + 1 for rank, pair in enumerate(ordered_pairs)}
    a1_macro_map = macro_gaps
    for row in pair_rows:
        pair = (indicators.index(str(row["indicator_a"])), indicators.index(str(row["indicator_b"])))
        row["a1_macro_rank"] = ranks[pair]

    # Pair-rank extension check uses the A1 same-source domain ranking.
    validation_rows: List[Dict[str, object]] = []
    for ext_key in ("a2", "a3"):
        ext = datasets[ext_key]
        for domain, ext_idx in _domain_indices(ext.domains).items():
            if domain not in a1_domains:
                continue
            ref_idx = a1_domains[domain]
            ref_d = intensity["a1"][ref_idx]
            ext_d = intensity[ext_key][ext_idx]
            tau = threshold_by_domain[domain]
            diff_ci = _bootstrap_group_differences(
                ref_d,
                ext_d,
                tau,
                args.validation_bootstrap,
                _stable_seed(rng_seed, f"validation-{ext_key}-{domain}"),
            )
            ref_gap = pair_gaps["a1"][domain]
            ext_gap = pair_gaps[ext_key][domain]
            pair_keys = sorted(ref_gap)
            rank_rho = _spearman(
                np.array([ref_gap[p] for p in pair_keys], dtype=float),
                np.array([ext_gap.get(p, float("nan")) for p in pair_keys], dtype=float),
                min_n=3,
            )
            ref_domain_top10 = sorted(
                ref_gap,
                key=lambda p: (-ref_gap[p], indicators[p[0]], indicators[p[1]]),
            )[:10]
            ext_top10 = sorted(
                ext_gap,
                key=lambda p: (-ext_gap[p], indicators[p[0]], indicators[p[1]]),
            )[:10]
            overlap_n = len(set(ref_domain_top10) & set(ext_top10))
            overlap = overlap_n / 10.0
            validation_rows.append({
                "extension_dataset": ext_key,
                "source_domain": domain,
                "a1_n": int(np.isfinite(ref_d).sum()),
                "extension_n": int(np.isfinite(ext_d).sum()),
                "a1_median_conflict": float(np.nanmedian(ref_d)),
                "extension_median_conflict": float(np.nanmedian(ext_d)),
                "median_conflict_diff": float(np.nanmedian(ext_d) - np.nanmedian(ref_d)),
                "median_diff_ci_low": diff_ci["median_diff_low"],
                "median_diff_ci_high": diff_ci["median_diff_high"],
                "a1_high_conflict_rate": float(np.mean(ref_d[np.isfinite(ref_d)] > tau)),
                "extension_high_conflict_rate": float(np.mean(ext_d[np.isfinite(ext_d)] > tau)),
                "high_conflict_rate_diff": float(
                    np.mean(ext_d[np.isfinite(ext_d)] > tau) - np.mean(ref_d[np.isfinite(ref_d)] > tau)
                ),
                "rate_diff_ci_low": diff_ci["rate_diff_low"],
                "rate_diff_ci_high": diff_ci["rate_diff_high"],
                "pair_rank_spearman": rank_rho,
                "top10_overlap_n": overlap_n,
                "top10_overlap_rate": overlap,
            })

    domain_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for key in DATASETS:
        data = datasets[key]
        rows = _domain_summary_rows(
            data,
            intensity[key],
            q_equal[key],
            q_huber[key],
            delta_q[key],
            threshold_by_domain,
            threshold_ci,
        )
        domain_rows.extend(rows)
        d = intensity[key]
        for i in range(len(data.ids)):
            domain = str(data.domains[i])
            tau = threshold_by_domain.get(domain, float("nan"))
            valid_n = int(np.isfinite(data.scores[i]).sum())
            sample_rows.append({
                "dataset": key,
                "source_domain": domain,
                "id": data.ids[i],
                "n_valid_indicators": valid_n,
                "conflict_intensity": d[i],
                "a1_domain_threshold": tau,
                "high_conflict": bool(math.isfinite(d[i]) and math.isfinite(tau) and d[i] > tau),
                "q_equal_mean": q_equal[key][i],
                "q_huber": q_huber[key][i],
                "delta_q_huber_minus_equal": delta_q[key][i],
                "outlier_indicator_count": int(data.outlier[i].sum()),
                "unit_suspect_indicator_count": int(data.unit_suspect[i].sum()),
            })

    _write_gzip_csv(
        output_dir / "sample_scores.csv.gz",
        sample_rows,
        ["dataset", "source_domain", "id", "n_valid_indicators", "conflict_intensity", "a1_domain_threshold", "high_conflict", "q_equal_mean", "q_huber", "delta_q_huber_minus_equal", "outlier_indicator_count", "unit_suspect_indicator_count"],
    )
    _write_csv(
        output_dir / "domain_summary.csv",
        domain_rows,
        list(domain_rows[0].keys()) if domain_rows else ["dataset", "source_domain"],
    )
    _write_csv(
        output_dir / "pair_diagnostics.csv",
        pair_rows,
        ["dataset", "source_domain", "indicator_a", "indicator_b", "pair_n", "mean_abs_gap", "spearman_rho", "a1_macro_rank"],
    )
    _write_csv(
        output_dir / "thresholds.csv",
        threshold_rows,
        ["variant", "threshold_rule", "source_domain", "quantile", "threshold", "threshold_ci_low", "threshold_ci_high", "a1_n", "a1_actual_exceedance_rate", "a1_tie_rate"],
    )
    _write_csv(
        output_dir / "threshold_sensitivity.csv",
        sensitivity_rows,
        ["variant", "threshold_rule", "quantile", "dataset", "source_domain", "threshold", "n", "high_conflict_rate"],
    )
    _write_csv(
        output_dir / "external_validation.csv",
        validation_rows,
        list(validation_rows[0].keys()) if validation_rows else ["extension_dataset", "source_domain"],
    )
    _write_csv(
        output_dir / "indicator_weights.csv",
        weight_rows,
        ["indicator", "a1_completeness", "domain_rank_stability", "valid_bootstrap_rho_count", "undefined_bootstrap_rho_count", "stability_fallback_reason", "weight"],
    )
    build_report(
        output_dir,
        domain_rows,
        validation_rows,
        top_pair_rows,
        weight_rows,
        delta,
        scale_method,
        args.max_rows_per_domain is not None,
    )

    input_hashes = frozen_output_hashes
    output_files = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    output_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output_files}
    run_manifest = {
        "schema_version": "q1-2-conflict-v1.1",
        "status": "sampled_p1" if args.max_rows_per_domain is not None else "full",
        "command": " ".join(sys.argv),
        "project_root": str(PROJECT_ROOT),
        "preprocessing_version": manifest["preprocessing_version"],
        "preprocess_manifest_sha256": hashlib.sha256(preprocess_manifest_path.read_bytes()).hexdigest(),
        "input_preprocessed_sha256": input_hashes,
        "rows_scanned": {key: datasets[key].input_rows for key in DATASETS},
        "rows_used": {key: int(len(datasets[key].ids)) for key in DATASETS},
        "indicators": indicators,
        "model_parameters": {
            "seed": rng_seed,
            "conflict_index": "row-wise mean pairwise absolute gap over finite norm_* scores",
            "threshold": "A1 source-domain q95, numpy method=linear",
            "threshold_bootstrap_repetitions": args.threshold_bootstrap,
            "validation_bootstrap_repetitions": args.validation_bootstrap,
            "q1_1_weight_bootstrap_repetitions": args.weight_bootstrap,
            "bootstrap_ci": BOOTSTRAP_CI,
            "bootstrap_ci_method": "percentile",
            "max_rows_per_domain": args.max_rows_per_domain,
            "huber_delta": delta,
            "huber_scale": robust_scale,
            "huber_scale_method": scale_method,
            "pair_report_top_n": 10,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "outputs_sha256": output_hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": run_manifest["status"],
        "output_dir": str(output_dir),
        "rows_used": run_manifest["rows_used"],
        "indicator_count": len(indicators),
        "thresholds": threshold_by_domain,
        "validation": validation_rows,
        "elapsed_seconds": run_manifest["runtime"]["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocess-dir", default=str(DEFAULT_PREPROCESS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--threshold-bootstrap", type=int, default=2000)
    parser.add_argument("--validation-bootstrap", type=int, default=2000)
    parser.add_argument("--weight-bootstrap", type=int, default=1000)
    parser.add_argument("--max-rows-per-domain", type=int, default=None,
                        help="P1 smoke mode: deterministic reservoir sample per dataset/source_domain")
    args = parser.parse_args(argv)
    for name in ("threshold_bootstrap", "validation_bootstrap", "weight_bootstrap"):
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} 至少为 2")
    if args.max_rows_per_domain is not None and args.max_rows_per_domain < 2:
        parser.error("--max-rows-per-domain 至少为 2")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
