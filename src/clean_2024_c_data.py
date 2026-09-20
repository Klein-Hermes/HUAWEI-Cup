"""Validate and standardize the 2024 CPMCM C problem data attachments.

Run from the repository root with: python src/clean_2024_c_data.py
The original attachments are always read only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "2024_C题"
OUT = ROOT / "data" / "processed" / "2024_C题"
WAVE_CODES = {"正弦波": 1, "三角波": 2, "梯形波": 3}
TRACE_COLS = [f"b_{i:04d}_t" for i in range(1024)]
BASE_COLS = [
    "source_file", "source_row", "sample_id", "material", "temperature_c",
    "frequency_hz", "loss_w_m3", "waveform", "waveform_code",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_and_standardize(path: Path, material: int | None = None) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(path, encoding="utf-8") if path.suffix == ".csv" else pd.read_excel(path, sheet_name=0)
    is_train = material is not None
    is_test2 = "附件二" in path.name
    expected_cols = 1028 if is_train or is_test2 else 1029
    if source.shape[1] != expected_cols:
        raise ValueError(f"{path.name}: expected {expected_cols} columns, got {source.shape[1]}")

    # Source row is the 1-based spreadsheet row, including the header row.
    source_row = np.arange(2, len(source) + 2, dtype=np.int32)
    trace = source.iloc[:, 4 if is_train or is_test2 else 5:].copy()
    trace.columns = TRACE_COLS
    trace = trace.apply(pd.to_numeric, errors="raise")
    fields = {
        "source_file": path.name,
        "source_row": source_row,
        "sample_id": pd.NA if is_train else source.iloc[:, 0],
        "material": material if is_train else source.iloc[:, 3].map({f"材料{i}": i for i in range(1, 5)}),
        "temperature_c": source.iloc[:, 0 if is_train else 1],
        "frequency_hz": source.iloc[:, 1 if is_train else 2],
        "loss_w_m3": source.iloc[:, 2] if is_train else pd.NA,
        "waveform": source.iloc[:, 3] if is_train else pd.NA if is_test2 else source.iloc[:, 4],
    }
    clean = pd.DataFrame(fields)
    clean["waveform_code"] = clean["waveform"].map(WAVE_CODES)
    for col in ["sample_id", "material", "temperature_c", "frequency_hz", "loss_w_m3"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    if clean[["material", "temperature_c", "frequency_hz"]].isna().any().any():
        raise ValueError(f"{path.name}: invalid required metadata")
    if is_train and (clean["loss_w_m3"].isna().any() or clean["waveform_code"].isna().any()):
        raise ValueError(f"{path.name}: invalid training target or waveform")
    if not is_train and not is_test2 and clean["waveform_code"].isna().any():
        raise ValueError(f"{path.name}: invalid waveform in test set 3")
    if not is_train and (clean["sample_id"].to_numpy() != np.arange(1, len(clean) + 1)).any():
        raise ValueError(f"{path.name}: sample IDs are not consecutive and ordered")
    if not clean["temperature_c"].isin([25, 50, 70, 90]).all():
        raise ValueError(f"{path.name}: unexpected temperature")
    if not clean["material"].isin([1, 2, 3, 4]).all():
        raise ValueError(f"{path.name}: unexpected material")
    if (clean["frequency_hz"] <= 0).any() or (is_train and (clean["loss_w_m3"] <= 0).any()):
        raise ValueError(f"{path.name}: nonpositive frequency or loss")
    if not np.isfinite(trace.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path.name}: nonfinite waveform sample")
    if (trace.max(axis=1) - trace.min(axis=1) <= 0).any():
        raise ValueError(f"{path.name}: constant waveform")

    # Exact duplicates are judged on all original fields; metadata repeats are valid measurements.
    duplicate_mask = source.duplicated(keep="first") if is_train else pd.Series(False, index=source.index)
    report = {
        "source": path.name,
        "sha256": sha256(path),
        "input_rows": len(source),
        "input_columns": source.shape[1],
        "missing_source_cells": int(source.isna().sum().sum()),
        "metadata_repeats": int(source.iloc[:, :4 if is_train else 5].duplicated().sum()),
        "exact_duplicate_rows_removed": source_row[duplicate_mask.to_numpy()].tolist(),
        "frequency_range_hz": [int(clean["frequency_hz"].min()), int(clean["frequency_hz"].max())],
        "waveform_counts": source.iloc[:, 3 if is_train else 4].value_counts().to_dict()
        if not is_test2 else {},
    }
    keep = ~duplicate_mask.to_numpy()
    clean = clean.loc[keep].reset_index(drop=True)
    trace = trace.loc[keep].reset_index(drop=True)
    clean = pd.concat([clean[BASE_COLS], trace], axis=1)
    report["output_rows"] = len(clean)
    return clean, report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = []
    reports = []
    for material in range(1, 5):
        path = RAW / f"appendix1_m{material}.csv"
        table, report = check_and_standardize(path, material)
        tables.append(table)
        reports.append(report)
    train = pd.concat(tables, ignore_index=True)
    tests = []
    for name, out_name, count in [
        ("附件二（测试集）.xlsx", "test_waveform.csv", 80),
        ("附件三（测试集）.xlsx", "test_loss.csv", 400),
    ]:
        table, report = check_and_standardize(RAW / name)
        if len(table) != count:
            raise ValueError(f"{name}: expected {count} samples, got {len(table)}")
        tests.append((table, OUT / out_name))
        reports.append(report)

    # All inputs are validated before any output is written.
    train_path = OUT / "train.csv"
    train.to_csv(train_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    for table, path in tests:
        table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")

    summary = {
        "rules": [
            "Keep original attachments unchanged.",
            "Rename metadata and 1024 waveform columns consistently.",
            "Encode material as 1..4 and waveform as 1=sinusoidal, 2=triangular, 3=trapezoidal.",
            "Remove only exact duplicate training rows; retain metadata repeats and unusual but valid values.",
            "Leave deliberately unavailable test targets blank; do not impute, clip, or scale measurements.",
        ],
        "source_reports": reports,
        "outputs": {
            p.name: {"rows": len(t), "columns": t.shape[1], "sha256": sha256(p)}
            for t, p in [(train, train_path), *tests]
        },
    }
    (OUT / "cleaning_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
