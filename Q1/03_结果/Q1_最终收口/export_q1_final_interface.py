"""Export the owner-selected Q1 q_huber interface from frozen candidate files.

This script only extracts existing q_equal/q_huber sample scores and the
precomputed q_huber domain summaries. It does not score indicators, estimate
weights, refit Huber parameters, or rerun Q1.2/Q1.3.
Run from the project root on a fresh results/q1_final directory.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "q1_1" / "v1"
OUT = ROOT / "results" / "q1_final"
Q_VERSION = "Q1-q_huber-v1"
SELECTED = "q_huber"
SENSITIVITY = "q_equal"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path, compressed: bool = False) -> list[dict[str, str]]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require_source_hash(manifest: dict, name: str) -> Path:
    path = SOURCE / name
    expected = manifest["outputs_sha256"][name]
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Source hash mismatch for {path.relative_to(ROOT)}: {actual} != {expected}")
    return path


def main() -> None:
    manifest_path = SOURCE / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = require_source_hash(source_manifest, "sample_scores.csv.gz")
    domain_path = require_source_hash(source_manifest, "domain_summary.csv")
    ci_path = require_source_hash(source_manifest, "domain_confidence_intervals.csv")

    human_status_path = ROOT / "Q1" / "03_结果" / "Q1.1" / "human_spotcheck_final" / "received_20260924" / "decision_status.json"
    human_status = json.loads(human_status_path.read_text(encoding="utf-8"))
    if human_status.get("candidate_selected") is not None:
        raise ValueError("Human review now selects a candidate; re-audit the approved basis before export")

    samples = read_csv(sample_path, compressed=True)
    sample_fields = ["dataset", "source_domain", "id", "q_equal", "q_huber", "n_valid_indicators", "indicator_coverage", "weighted_coverage", "outlier_indicator_count", "unit_suspect_indicator_count", "unicode_cf_cc_flag", "unicode_broad_flag"]
    if not samples or any(any(field not in row for field in sample_fields) for row in samples):
        raise ValueError("Unexpected sample candidate schema")

    seen: set[tuple[str, str, str]] = set()
    interface_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    for row in samples:
        key = (row["dataset"], row["source_domain"], row["id"])
        if key in seen:
            raise ValueError(f"Duplicate sample key: {key}")
        seen.add(key)
        q_equal = float(row["q_equal"])
        q_huber = float(row["q_huber"])
        if not (0.0 <= q_equal <= 1.0 and 0.0 <= q_huber <= 1.0):
            raise ValueError(f"Candidate score outside [0,1] for {key}")
        interface_rows.append({
            "dataset": row["dataset"],
            "source_domain": row["source_domain"],
            "sample_id": row["id"],
            "q_operational": row[SELECTED],
            "q_version": Q_VERSION,
        })
        audit_rows.append({
            "dataset": row["dataset"],
            "source_domain": row["source_domain"],
            "sample_id": row["id"],
            "q_equal": row["q_equal"],
            "q_huber": row["q_huber"],
            "q_operational": row[SELECTED],
            "q_sensitivity": row[SENSITIVITY],
            "q_version": Q_VERSION,
            "q_sensitivity_candidate": SENSITIVITY,
            "n_valid_indicators": row["n_valid_indicators"],
            "indicator_coverage": row["indicator_coverage"],
            "weighted_coverage": row["weighted_coverage"],
            "outlier_indicator_count": row["outlier_indicator_count"],
            "unit_suspect_indicator_count": row["unit_suspect_indicator_count"],
            "unicode_cf_cc_flag": row["unicode_cf_cc_flag"],
            "unicode_broad_flag": row["unicode_broad_flag"],
        })

    expected_n = sum(source_manifest["execution_scope"]["rows_selected"].values())
    if len(samples) != expected_n:
        raise ValueError(f"Source row count mismatch: {len(samples)} != {expected_n}")

    domain_rows = read_csv(domain_path)
    ci_rows = read_csv(ci_path)
    domain_by_key = {
        (row["dataset"], row["source_domain"]): row
        for row in domain_rows
        if row["scenario"] == "all_a1" and row["candidate"] == SELECTED
    }
    ci_by_key = {
        (row["dataset"], row["source_domain"]): row
        for row in ci_rows
        if row["candidate"] == SELECTED
    }
    if not domain_by_key or set(domain_by_key) != set(ci_by_key):
        raise ValueError("q_huber domain summaries and confidence intervals do not have identical coverage")

    domain_interface: list[dict[str, str]] = []
    for key, row in sorted(domain_by_key.items()):
        ci = ci_by_key[key]
        if int(row["n_scored"]) != int(row["n_records"]) or float(row["scored_rate"]) != 1.0:
            raise ValueError(f"Incomplete domain score coverage for {key}")
        domain_interface.append({
            "dataset": row["dataset"],
            "source_domain": row["source_domain"],
            "q_operational": row["domain_score"],
            "q_version": Q_VERSION,
            "n_records": row["n_records"],
            "n_scored": row["n_scored"],
            "scored_rate": row["scored_rate"],
            "ci95_low": ci["ci_low"],
            "ci95_high": ci["ci_high"],
            "aggregation_method": "A1-calibrated domain Huber location over sample-level q_huber; A2/A3 use frozen A1 domain threshold",
            "domain_huber_delta_from_a1": row["domain_huber_delta_from_a1"],
        })

    outputs = {
        "q1_final_quality.csv": (
            interface_rows,
            ["dataset", "source_domain", "sample_id", "q_operational", "q_version"],
        ),
        "q1_candidate_audit.csv": (
            audit_rows,
            ["dataset", "source_domain", "sample_id", "q_equal", "q_huber", "q_operational", "q_sensitivity", "q_version", "q_sensitivity_candidate", "n_valid_indicators", "indicator_coverage", "weighted_coverage", "outlier_indicator_count", "unit_suspect_indicator_count", "unicode_cf_cc_flag", "unicode_broad_flag"],
        ),
        "q1_final_domain_quality.csv": (
            domain_interface,
            ["dataset", "source_domain", "q_operational", "q_version", "n_records", "n_scored", "scored_rate", "ci95_low", "ci95_high", "aggregation_method", "domain_huber_delta_from_a1"],
        ),
    }
    existing = [name for name in outputs if (OUT / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing final interface file(s): {existing}")
    for name, (rows, fields) in outputs.items():
        write_csv(OUT / name, rows, fields)

    # Verify the exported operational values exactly match the q_huber source strings.
    final_rows = read_csv(OUT / "q1_final_quality.csv")
    if len(final_rows) != len(samples):
        raise ValueError("Exported q1_final_quality.csv row count mismatch")
    final_by_key = {(r["dataset"], r["source_domain"], r["sample_id"]): r for r in final_rows}
    for row in samples:
        key = (row["dataset"], row["source_domain"], row["id"])
        if final_by_key[key]["q_operational"] != row[SELECTED]:
            raise ValueError(f"Exported q_huber differs from frozen source for {key}")

    print(json.dumps({
        "status": "PASS",
        "selection_basis": "project_operational_choice",
        "q_operational": SELECTED,
        "q_sensitivity": SENSITIVITY,
        "q_version": Q_VERSION,
        "sample_rows": len(interface_rows),
        "sample_key_unique": len(seen) == len(interface_rows),
        "domain_rows": len(domain_interface),
        "source_hashes_verified": {
            "sample_scores.csv.gz": sha256(sample_path),
            "domain_summary.csv": sha256(domain_path),
            "domain_confidence_intervals.csv": sha256(ci_path),
        },
        "outputs": {name: {"rows": len(rows), "sha256": sha256(OUT / name)} for name, (rows, _) in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
