"""Create exploratory profiles and interpretable waveform features.

Run after clean_2024_c_data.py. No classifier or loss model is fitted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "2024_C题"
OUTPUT = ROOT / "results" / "2024_C题_eda"
TRACE_COLS = [f"b_{i:04d}_t" for i in range(1024)]
INPUT_NAMES = {
    "train": "train.csv",
    "test_waveform": "test_waveform.csv",
    "test_loss": "test_loss.csv",
}


def extract_features(frame: pd.DataFrame) -> pd.DataFrame:
    signal = frame[TRACE_COLS].to_numpy(dtype=np.float64, copy=False)
    n = signal.shape[1]
    mean = signal.mean(axis=1)
    centered = signal - mean[:, None]
    span = np.ptp(signal, axis=1)
    if np.any(span <= 0):
        raise ValueError("Found a constant waveform")
    diff = np.roll(signal, -1, axis=1) - signal
    second = np.roll(diff, -1, axis=1) - diff
    spectrum = np.abs(np.fft.rfft(centered, axis=1))
    fundamental = spectrum[:, 1]
    if np.any(fundamental <= 1e-12):
        raise ValueError("Found a waveform without a usable fundamental harmonic")
    abs_diff = np.abs(diff) / span[:, None]
    result = frame[[
        "source_file", "source_row", "sample_id", "material", "temperature_c",
        "frequency_hz", "loss_w_m3", "waveform", "waveform_code",
    ]].copy()
    result["b_mean_t"] = mean
    result["b_peak_abs_t"] = np.max(np.abs(signal), axis=1)
    result["b_peak_to_peak_t"] = span
    result["b_rms_t"] = np.sqrt(np.mean(signal**2, axis=1))
    result["b_centered_rms_t"] = np.sqrt(np.mean(centered**2, axis=1))
    result["b_crest_factor"] = result["b_peak_abs_t"] / result["b_rms_t"]
    result["slope_rms_rel"] = np.sqrt(np.mean(diff**2, axis=1)) / span
    result["slope_p90_rel"] = np.quantile(abs_diff, 0.90, axis=1)
    result["plateau_fraction_0p001"] = np.mean(abs_diff < 0.001, axis=1)
    result["curvature_rms_rel"] = np.sqrt(np.mean(second**2, axis=1)) / span
    for harmonic in range(2, 6):
        result[f"harmonic_{harmonic}_to_1"] = spectrum[:, harmonic] / fundamental
    result["harmonic_2_10_rss_to_1"] = np.sqrt(
        np.sum(spectrum[:, 2:11] ** 2, axis=1)
    ) / fundamental
    result["transmission_proxy_hz_t"] = (
        result["frequency_hz"] * result["b_peak_abs_t"]
    )
    numeric = result.select_dtypes(include="number").drop(
        columns=["sample_id", "loss_w_m3", "waveform", "waveform_code"], errors="ignore"
    )
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        bad = numeric.columns[~np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=0)].tolist()
        raise ValueError(f"Feature matrix contains nonfinite values in {bad}")
    return result


def describe(series: pd.Series) -> dict:
    return {
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def profile(table: pd.DataFrame) -> dict:
    stats = {
        "rows": len(table),
        "material_counts": {str(k): int(v) for k, v in table.material.value_counts().sort_index().items()},
        "temperature_counts": {str(k): int(v) for k, v in table.temperature_c.value_counts().sort_index().items()},
        "frequency_hz": describe(table.frequency_hz),
        "b_peak_abs_t": describe(table.b_peak_abs_t),
        "b_peak_to_peak_t": describe(table.b_peak_to_peak_t),
    }
    if table.waveform.notna().any():
        stats["waveform_counts"] = {str(k): int(v) for k, v in table.waveform.value_counts().items()}
    if table.loss_w_m3.notna().any():
        stats["loss_w_m3"] = describe(table.loss_w_m3)
    return stats


def coverage(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    fields = ["frequency_hz", "b_peak_abs_t", "b_peak_to_peak_t"]
    result = {}
    for field in fields:
        lower = train.groupby("material")[field].min()
        upper = train.groupby("material")[field].max()
        lo = test.material.map(lower)
        hi = test.material.map(upper)
        mask = (test[field] < lo) | (test[field] > hi)
        result[field] = {
            "outside_material_training_range_count": int(mask.sum()),
            "outside_material_training_range_sample_ids": test.loc[mask, "sample_id"].astype(int).tolist(),
        }
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    for key, filename in INPUT_NAMES.items():
        source = pd.read_csv(INPUT / filename, low_memory=False)
        if source.shape[1] != 1033 or len([c for c in source if c in TRACE_COLS]) != 1024:
            raise ValueError(f"Unexpected schema in {filename}")
        tables[key] = extract_features(source)
        tables[key].to_csv(OUTPUT / f"features_{key}.csv", index=False, encoding="utf-8-sig", float_format="%.17g")

    train = tables["train"]
    medians = (
        train.groupby(["material", "temperature_c", "waveform"], observed=True)["loss_w_m3"]
        .agg(["size", "median"])
        .reset_index()
        .rename(columns={"size": "n", "median": "median_loss_w_m3"})
    )
    medians.to_csv(OUTPUT / "loss_group_medians.csv", index=False, encoding="utf-8-sig")

    wave_features = [
        "b_crest_factor", "slope_rms_rel", "slope_p90_rel",
        "plateau_fraction_0p001", "curvature_rms_rel", "harmonic_2_10_rss_to_1",
    ]
    feature_summary = (
        train.groupby("waveform", observed=True)[wave_features]
        .agg(["median", "std"])
    )
    feature_summary.columns = [f"{field}_{stat}" for field, stat in feature_summary.columns]
    feature_summary.reset_index().to_csv(OUTPUT / "waveform_feature_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "profiles": {key: profile(value) for key, value in tables.items()},
        "test_coverage": {
            key: coverage(train, tables[key]) for key in ("test_waveform", "test_loss")
        },
        "loss_group_cells": len(medians),
        "minimum_group_size": int(medians.n.min()),
        "feature_columns": [c for c in train.columns if c not in [
            "source_file", "source_row", "sample_id", "material", "temperature_c",
            "frequency_hz", "loss_w_m3", "waveform", "waveform_code",
        ]],
    }
    (OUTPUT / "eda_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "feature_rows": {key: len(value) for key, value in tables.items()},
        "feature_count": len(report["feature_columns"]),
        "minimum_group_size": report["minimum_group_size"],
        "test_coverage": report["test_coverage"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
