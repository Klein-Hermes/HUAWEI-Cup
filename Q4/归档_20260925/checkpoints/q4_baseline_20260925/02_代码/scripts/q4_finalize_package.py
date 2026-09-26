"""Verify the Q4 provisional package and write a complete input/output hash manifest.

AI assistance: OpenAI Codex desktop 26.917.71314 (build 10954; prod; updater
checked 2026-09-25, up_to_date); developer OpenAI. Model selector name: ; model
version/publication date: . These fields are blank per user instruction.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
DATA = ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"
RESULTS = Q4 / "03_结果" / "results"
MANIFEST = RESULTS / "q4_package_manifest.json"
REVIEW_FILE = Q4 / "05_审计交付" / "q4_quality_review.md"
SKIP_DIR_NAMES = {"__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_stage_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    checked = 0
    for relative, expected in data.get("outputs_sha256", {}).items():
        target = ROOT / Path(relative.replace("/", "\\"))
        if not target.is_file():
            raise FileNotFoundError(f"Stage output listed but missing: {target}")
        actual = sha256(target)
        if actual != expected:
            raise ValueError(f"Stage output hash mismatch in {path.name}: {relative}")
        checked += 1
    return {"path": rel(path), "sha256": sha256(path), "verified_output_hashes": checked}


def main() -> None:
    stage_paths = [
        RESULTS / "q4_c8_manifest.json",
        RESULTS / "q4_auxiliary_audit_manifest.json",
        RESULTS / "q4_modeling_manifest.json",
        RESULTS / "q4_registry_coverage_manifest.json",
    ]
    stage_results = [verify_stage_manifest(path) for path in stage_paths]

    registry = read_csv(RESULTS / "q4_model_registry.csv")
    match_audit = read_csv(RESULTS / "q4_match_audit.csv")
    coverage = read_csv(RESULTS / "q4_coverage.csv")
    forecast = read_csv(RESULTS / "q4_forecast_feasibility.csv")
    qa = json.loads((RESULTS / "q4_figure_qa.json").read_text(encoding="utf-8"))

    registry_ids = [row.get("registry_id", "") for row in registry]
    if len(registry) != 1860 or len(set(registry_ids)) != len(registry_ids):
        raise ValueError("Registry must contain 1,860 unique C8 latest-directory IDs")
    if len(match_audit) != 1860:
        raise ValueError("Match audit row count must equal the 1,860 latest C8 directories")
    candidates = [row for row in match_audit if row.get("provisional_unique_name_candidate") == "True"]
    strict = [row for row in match_audit if row.get("strict_version_confirmed") == "True"]
    if len(candidates) != 1819 or strict:
        raise ValueError(f"Unexpected match cohort: candidates={len(candidates)}, strict={len(strict)}")
    def cov_count(level: str) -> int:
        for row in coverage:
            if row.get("level") == level:
                return int(float(row["n_models"]))
        raise KeyError(f"Coverage row missing: {level}")
    if cov_count("Complete six-family score index") != 789:
        raise ValueError("Expected 789 complete six-family score records")
    if cov_count("Complete seven-family score index") != 0:
        raise ValueError("Unexpected complete seven-family score count")
    if cov_count("Core6 + positive parameters + submission date") != 788:
        raise ValueError("Unexpected core6/parameter/date intersection")
    if len(forecast) != 2:
        raise ValueError("Forecast feasibility file must contain 12- and 24-month rows")
    for row in forecast:
        if row.get("status") != "not_estimable_from_supplied_history":
            raise ValueError("Annual forecast must remain suppressed with current history")
        for field in ("point_forecast", "prediction_interval_low", "prediction_interval_high"):
            if row.get(field, "").strip():
                raise ValueError(f"Unsupported annual numeric forecast found: {field}")

    if qa.get("manual_visual_review") != "PASS":
        raise ValueError("Figure visual review is not marked PASS")
    for figure in qa.get("figures", []):
        if figure.get("program_status") != "PASS" or figure.get("manual_visual_review") != "PASS":
            raise ValueError(f"Figure QA is incomplete: {figure.get('name')}")
        for key in ("png", "svg", "grayscale_png"):
            target = ROOT / Path(figure[key].replace("/", "\\"))
            if not target.is_file():
                raise FileNotFoundError(f"Figure listed in QA but missing: {target}")
            if target.suffix.lower() == ".png":
                with Image.open(target) as image:
                    dpi = image.info.get("dpi")
                    if not dpi or round(float(dpi[0])) < 300:
                        raise ValueError(f"PNG DPI below 300 or unavailable: {target}")

    csv_inputs = sorted(DATA.glob("*.csv"))
    file_registry = read_csv(RESULTS / "q4_c8_file_registry.csv")
    c8_files = sorted({ROOT / Path(row["source_file"].replace("/", "\\")) for row in file_registry if row.get("source_file")})
    raw_inputs = [path for path in [*csv_inputs, *c8_files] if path.is_file()]
    if len(c8_files) != 1958:
        raise ValueError(f"Expected 1,958 source JSON files in the C8 registry, found {len(c8_files)}")

    output_files = []
    for path in Q4.rglob("*"):
        if not path.is_file() or path == MANIFEST or path == REVIEW_FILE:
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts) or path.suffix.lower() == ".pyc":
            continue
        output_files.append(path)
    output_files = sorted(output_files)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "provisional_not_frozen",
        "generated_at_local": "",
        "ai_disclosure": "OpenAI Codex desktop 26.917.71314 (build 10954; prod; updater checked 2026-09-25, up_to_date); developer OpenAI; model selector name: ; model version/publication date: .",
        "scope": "Q4 workflow steps 0-10; source data hashes and all Q4 code, documentation, tables, and figures; independent review file is outside the hash set to avoid a circular reference.",
        "source_input_file_count": len(raw_inputs),
        "c8_json_source_file_count": len(c8_files),
        "stage_manifests_verified": stage_results,
        "inputs_sha256": {rel(path): sha256(path) for path in raw_inputs},
        "outputs_sha256": {rel(path): sha256(path) for path in output_files},
        "output_file_count": len(output_files),
        "verified_invariants": {
            "c8_latest_registry_rows": len(registry), "c8_match_audit_rows": len(match_audit),
            "unique_name_candidates": len(candidates), "strict_version_confirmed": len(strict),
            "complete_core6": cov_count("Complete six-family score index"),
            "complete_full7": cov_count("Complete seven-family score index"),
            "core6_positive_parameter_submission_date": cov_count("Core6 + positive parameters + submission date"),
            "annual_numeric_forecasts_issued": False,
            "all_figure_program_and_visual_qa_pass": True,
        },
        "independent_review_record": "Q4/05_审计交付/q4_quality_review.md; excluded from this manifest to avoid a circular hash.",
        "cohort_decision": {
            "status": "accepted_provisional",
            "decision": "Adopt the provisional cohort and metric choices documented in Q4/01_方案说明/q4_cohort_decision.md.",
            "recorded_by": "user",
            "recorded_on": "2026-09-25",
        },
        "pending_before_team_freeze": [
            "AI model selector name and model version/publication date are blank per user instruction; disclosure must be completed before final submission.",
        ],
    }
    manifest["generated_at_local"] = datetime.now().astimezone().isoformat(timespec="seconds")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"package_manifest={rel(MANIFEST)}")
    print(f"raw_inputs={len(raw_inputs)} c8_json={len(c8_files)} q4_files={len(output_files)} stage_output_hashes={sum(s['verified_output_hashes'] for s in stage_results)}")
    print(f"core6=789 full7=0 scale_time=788 annual_forecasts=0 visual_QA=PASS team_freeze=pending_ai_disclosure")
    print(f"independent_review_record={'present' if REVIEW_FILE.is_file() else 'pending'}")
    print(f"manifest_sha256={sha256(MANIFEST)}")


if __name__ == "__main__":
    main()
