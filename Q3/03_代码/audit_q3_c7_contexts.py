#!/usr/bin/env python3
"""Audit C7 model context limits for the Q3 attention-cost scenarios.

Run from the repository root:
    python Q3/03_代码/audit_q3_c7_contexts.py

The script uses only the Python standard library and writes the model-level
audit, a machine-readable summary, and a reproducibility manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_RELATIVE = Path(
    "中文题目/F题/real_attachments/C_efficiency_evolution/"
    "model_architecture_metadata.csv"
)
DETAIL_RELATIVE = Path("Q3/02_数据审计/C7_context_by_model.csv")
SUMMARY_RELATIVE = Path("Q3/04_结果/C7_context_audit_summary.json")
MANIFEST_RELATIVE = Path("Q3/04_结果/复现清单.json")

ETA = 2e-4
TRAINING_COEFFICIENT = 6
CRITICAL_LENGTH = TRAINING_COEFFICIENT / ETA


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"model_name", "max_position_embeddings"}
        headers = set(reader.fieldnames or [])
        missing_headers = sorted(required - headers)
        if missing_headers:
            raise ValueError(f"Required C7 columns are missing: {missing_headers}")

        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for source_row, raw in enumerate(reader, start=2):
            name = (raw.get("model_name") or "").strip()
            raw_length = (raw.get("max_position_embeddings") or "").strip()
            if not name:
                errors.append(f"CSV row {source_row}: model_name is blank")
                continue
            if not raw_length:
                errors.append(f"CSV row {source_row}: max_position_embeddings is blank")
                continue
            try:
                length = int(raw_length)
            except ValueError:
                errors.append(
                    f"CSV row {source_row}: max_position_embeddings is not an integer: "
                    f"{raw_length!r}"
                )
                continue
            if length <= 0:
                errors.append(
                    f"CSV row {source_row}: max_position_embeddings must be positive, got {length}"
                )
                continue
            rows.append(
                {
                    "source_csv_row": source_row,
                    "model_name": name,
                    "max_position_embeddings": length,
                }
            )

        if errors:
            raise ValueError("C7 input validation failed:\n- " + "\n- ".join(errors))
        if not rows:
            raise ValueError("C7 input contains no model rows")

        names = [row["model_name"] for row in rows]
        duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicate_names:
            raise ValueError(f"Duplicate model_name values found: {duplicate_names}")
        return rows


def write_detail(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_csv_row",
        "model_name",
        "max_position_embeddings",
        "context_to_critical_ratio",
        "above_critical_length",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            length = row["max_position_embeddings"]
            writer.writerow(
                {
                    **row,
                    "context_to_critical_ratio": f"{length / CRITICAL_LENGTH:.10f}",
                    "above_critical_length": str(length > CRITICAL_LENGTH).lower(),
                }
            )


def build_summary(rows: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    lengths = [row["max_position_embeddings"] for row in rows]
    frequency = Counter(lengths)
    above = sum(length > CRITICAL_LENGTH for length in lengths)
    return {
        "artifact": "Q3 C7 context length audit",
        "scope": "Architecture max context length scenarios; not a claim about deployed task windows.",
        "input": {
            "path": relative_posix(input_path),
            "sha256": sha256_file(input_path),
            "required_columns": ["model_name", "max_position_embeddings"],
        },
        "validation": {
            "row_count": len(rows),
            "unique_model_name_count": len({row["model_name"] for row in rows}),
            "blank_model_name_count": 0,
            "blank_or_invalid_context_count": 0,
            "duplicate_model_name_count": 0,
            "all_context_lengths_positive_integers": True,
        },
        "critical_length": {
            "training_cost_coefficient": TRAINING_COEFFICIENT,
            "attention_coefficient_eta": ETA,
            "tokens": int(CRITICAL_LENGTH),
            "derivation": "C_train=6ND and C_attn=eta*N*D*L_ctx; equality gives L_crit=6/eta.",
            "attention_to_training_cost_ratio": "L_ctx / L_crit",
        },
        "context_length_distribution": [
            {"max_position_embeddings": length, "model_count": frequency[length]}
            for length in sorted(frequency)
        ],
        "threshold_comparison": {
            "above_critical_count": above,
            "at_or_below_critical_count": len(rows) - above,
            "above_critical_fraction": above / len(rows),
            "definition": "above means strictly greater than L_crit",
        },
        "interpretation_limit": (
            "max_position_embeddings is an architecture-supported upper bound; "
            "it may differ from the context window used by the task."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    input_path = REPO_ROOT / INPUT_RELATIVE
    detail_path = REPO_ROOT / DETAIL_RELATIVE
    summary_path = REPO_ROOT / SUMMARY_RELATIVE
    manifest_path = REPO_ROOT / MANIFEST_RELATIVE

    if not input_path.is_file():
        raise FileNotFoundError(f"C7 input not found: {input_path}")

    rows = load_rows(input_path)
    write_detail(rows, detail_path)

    summary = build_summary(rows, input_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "artifact": "Q3 C7 audit reproducibility manifest",
        "command": "python Q3/03_代码/audit_q3_c7_contexts.py",
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "script": {
            "path": relative_posix(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "input": {
            "path": relative_posix(input_path),
            "sha256": sha256_file(input_path),
        },
        "parameters": {
            "training_cost_coefficient": TRAINING_COEFFICIENT,
            "attention_coefficient_eta": ETA,
            "critical_context_length": int(CRITICAL_LENGTH),
            "critical_comparison": "strictly greater than 30,000 tokens",
        },
        "outputs": {
            relative_posix(detail_path): sha256_file(detail_path),
            relative_posix(summary_path): sha256_file(summary_path),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary["validation"], ensure_ascii=False))
    print(
        "C7 distribution: "
        + ", ".join(
            f"{entry['max_position_embeddings']}={entry['model_count']}"
            for entry in summary["context_length_distribution"]
        )
    )
    above_count = summary["threshold_comparison"]["above_critical_count"]
    print(
        f"Above {int(CRITICAL_LENGTH):,} tokens: {above_count}/{len(rows)} "
        f"({above_count / len(rows):.1%})"
    )
    print(f"Wrote {relative_posix(detail_path)}")
    print(f"Wrote {relative_posix(summary_path)}")
    print(f"Wrote {relative_posix(manifest_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
