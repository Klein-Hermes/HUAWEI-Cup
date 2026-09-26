"""Verify the Q4 package and write a complete input/output hash manifest.

AI assistance: OpenAI Codex desktop 26.917.71314 (build 10954; prod);
verified model ID gpt-6-luna (official name GPT-6 Luna), developer OpenAI,
official release date 2026-09-22. See the canonical Q4 AI disclosure record.
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
ATTACHMENTS = ROOT / "中文题目" / "F题" / "real_attachments"
RESULTS = Q4 / "03_结果" / "results"
MANIFEST = RESULTS / "q4_package_manifest.json"
REVIEW_FILE = Q4 / "05_审计交付" / "q4_quality_review.md"
AI_DISCLOSURE_PATH = Q4 / "05_审计交付" / "ai_disclosure.json"
FREEZE_APPROVAL_PATH = Q4 / "05_审计交付" / "q4_freeze_approval.json"
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
        RESULTS / "q4_loss_benchmark_feasibility_manifest.json",
        RESULTS / "q4_dynamic_decomposition_manifest.json",
    ]
    stage_results = [verify_stage_manifest(path) for path in stage_paths]
    gate_manifest = json.loads((RESULTS / "q4_loss_benchmark_feasibility_manifest.json").read_text(encoding="utf-8"))
    if gate_manifest.get("status") != "BLOCKED_NO_FIT" or gate_manifest.get("fit_executed") is not False:
        raise ValueError("Loss–Benchmark gate must remain BLOCKED_NO_FIT with fit_executed=false")
    dynamic_manifest = json.loads((RESULTS / "q4_dynamic_decomposition_manifest.json").read_text(encoding="utf-8"))
    if dynamic_manifest.get("smoke_run") is not False or dynamic_manifest.get("loss_bridge_fit_executed") is not False:
        raise ValueError("The package must reference the full descriptive run and no bridge fit")
    if dynamic_manifest.get("analysis_scope", {}).get("ar1_annual_forecast_authorized") is not False:
        raise ValueError("AR(1) descriptive fit cannot authorize annual forecasts")
    scale_support = read_csv(RESULTS / "q4_scale_tech_support_audit.csv")
    if len(scale_support) != 1:
        raise ValueError("Expected one endpoint common-support audit row")
    scale_row = scale_support[0]
    if scale_row.get("smoke_run") != "False" or scale_row.get("status") != "overlap":
        raise ValueError("Full common-support decomposition is missing or unsupported")
    if int(float(scale_row.get("design_rank", 0))) != int(float(scale_row.get("design_columns", -1))):
        raise ValueError("Common-support decomposition design matrix is rank deficient")
    if float(scale_row.get("component_sum_abs_error", "inf")) > 1e-10 or float(scale_row.get("observed_reconstruction_abs_error", "inf")) > 1e-8:
        raise ValueError("Common-support decomposition arithmetic closure failed")
    c3_years = read_csv(RESULTS / "q4_c3_year_summary.csv")
    if not any(row.get("Source") == "Open LLM Leaderboard" and row.get("Year") == "2025" and row.get("coverage_status") == "partial_year_coverage_only" for row in c3_years):
        raise ValueError("C3 2025 must remain labeled as partial-year coverage")

    registry = read_csv(RESULTS / "q4_model_registry.csv")
    match_audit = read_csv(RESULTS / "q4_match_audit.csv")
    coverage = read_csv(RESULTS / "q4_coverage.csv")
    forecast = read_csv(RESULTS / "q4_forecast_feasibility.csv")
    qa = json.loads((RESULTS / "q4_figure_qa.json").read_text(encoding="utf-8"))
    repro = json.loads((RESULTS / "复现清单.json").read_text(encoding="utf-8"))
    ai_disclosure = json.loads(AI_DISCLOSURE_PATH.read_text(encoding="utf-8"))
    freeze_approval = json.loads(FREEZE_APPROVAL_PATH.read_text(encoding="utf-8"))
    package_status = freeze_approval.get("package_status", "provisional_not_frozen")
    if package_status not in {"provisional_not_frozen", "frozen"}:
        raise ValueError(f"Unsupported Q4 package status: {package_status}")
    if package_status == "frozen" and freeze_approval.get("authorization_confirmed") is not True:
        raise ValueError("A frozen Q4 package requires explicit user authorization evidence")
    data_dictionary = ROOT / "中文题目" / "F题" / "数据说明(无隐藏字段版本）.pdf"
    hidden_dictionary = ROOT / "中文题目" / "F题" / "数据说明.pdf"
    expected_dictionary_hash = "b2c8e1b997f25ab7f5c83539931dfe2260262078fac85ed09f2e99b801f3866f"
    if not data_dictionary.is_file() or sha256(data_dictionary) != expected_dictionary_hash:
        raise ValueError("The Q4 field dictionary is not the verified no-hidden-fields PDF")
    if hidden_dictionary.exists():
        raise ValueError("Unexpected alternate hidden-field data dictionary is present")
    dictionary_record = repro.get("data_dictionary_check", {})
    if dictionary_record.get("version") != "no_hidden_fields" or dictionary_record.get("sha256") != expected_dictionary_hash:
        raise ValueError("Reproduction manifest does not pin the verified no-hidden-fields dictionary")
    disclosure_model = ai_disclosure.get("model_identity", {})
    disclosure_source = repro.get("ai_disclosure_source", {})
    if disclosure_source.get("file") != rel(AI_DISCLOSURE_PATH) or disclosure_source.get("sha256") != sha256(AI_DISCLOSURE_PATH):
        raise ValueError("Reproduction manifest does not pin the current canonical AI disclosure record")
    if repro.get("ai_disclosure_record") != ai_disclosure:
        raise ValueError("Reproduction manifest AI disclosure record is stale")
    if repro.get("freeze_authorization_record") != freeze_approval:
        raise ValueError("Reproduction manifest freeze authorization record is stale")
    if repro.get("freeze_authorization_source", {}).get("sha256") != sha256(FREEZE_APPROVAL_PATH):
        raise ValueError("Reproduction manifest does not pin the current freeze authorization")
    if disclosure_model.get("runtime_reported_family") != repro.get("key_decisions", {}).get("ai_runtime_model_family"):
        raise ValueError("AI runtime model-family disclosure does not match the canonical record")
    if (repro.get("key_decisions", {}).get("ai_model_selector_name") or "") != (disclosure_model.get("selector_display_name") or ""):
        raise ValueError("AI selector disclosure does not match the canonical record")
    if (repro.get("key_decisions", {}).get("ai_model_id") or "") != (disclosure_model.get("model_variant_or_id") or ""):
        raise ValueError("AI model ID disclosure does not match the canonical record")
    expected_additional_ai_ids = "; ".join(
        tool.get("model_id", "") for tool in ai_disclosure.get("additional_ai_tools", []) if tool.get("model_id")
    )
    if (repro.get("key_decisions", {}).get("additional_ai_model_ids") or "") != expected_additional_ai_ids:
        raise ValueError("Additional AI model IDs in the reproduction manifest do not match the canonical record")
    expected_model_version_date = "; ".join(
        value for value in (disclosure_model.get("model_version"), disclosure_model.get("publication_date")) if value
    )
    if (repro.get("key_decisions", {}).get("ai_model_version_or_publication_date") or "") != expected_model_version_date:
        raise ValueError("AI model version/date disclosure does not match the canonical record")
    figures = qa.get("figures", [])
    candidate_figures = [figure for figure in figures if figure.get("class") != "独立流程图"]
    flow_figures = [figure for figure in figures if figure.get("class") == "独立流程图"]
    if qa.get("candidate_figure_count") != 9 or len(candidate_figures) != 9 or len(flow_figures) != 1 or len(figures) != 10 or qa.get("candidate_class_counts") != {"raw": 3, "process": 3, "result": 3}:
        raise ValueError("Expected nine Q4 candidates (three per class) and one separate flowchart")
    if qa.get("file_audit", {}).get("status") != "PASS" or qa.get("file_audit", {}).get("checked_assets") != 30:
        raise ValueError("Figure file audit must cover all 30 color/SVG/grayscale assets")

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
    reference_inputs = [
        ROOT / "中文题目" / "F题" / "算力约束下提升大语言模型能力的资源配置建模.docx",
        data_dictionary,
        ATTACHMENTS / "source_manifest.json",
        ATTACHMENTS / "attachment_size_summary.csv",
    ]
    loss_gate_inputs = [
        ROOT / "中文题目" / "F题" / "real_attachments" / "B_scaling_laws" / "pythia_training_log_existing.csv",
        ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "b1_fitted_predictions.csv",
        ROOT / "Q2" / "05_审计交付" / "loss_protocol_matrix.csv",
    ]
    if any(not path.is_file() for path in loss_gate_inputs):
        missing = [rel(path) for path in loss_gate_inputs if not path.is_file()]
        raise FileNotFoundError("Loss gate provenance input missing: " + "; ".join(missing))
    repro_inputs = repro.get("inputs", {})
    for path in loss_gate_inputs:
        entry = repro_inputs.get(rel(path), {})
        if entry.get("sha256") != sha256(path):
            raise ValueError(f"Reproduction manifest does not pin the current Loss gate input: {rel(path)}")
    raw_inputs = [path for path in [*csv_inputs, *c8_files, *reference_inputs, *loss_gate_inputs] if path.is_file()]
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
        "status": package_status,
        "generated_at_local": "",
        "ai_disclosure": "OpenAI Codex desktop 26.917.71314 (build 10954; prod), model gpt-6-luna (GPT-6 Luna; 2026-09-22); related ChatGPT conversation user-identified as gpt-5.6-sol (GPT-5.6 Sol; 2026-07-09); developer OpenAI. See ai_disclosure_record for evidence provenance.",
        "ai_disclosure_record": ai_disclosure,
        "ai_disclosure_source": {"file": rel(AI_DISCLOSURE_PATH), "sha256": sha256(AI_DISCLOSURE_PATH)},
        "freeze_authorization_record": freeze_approval,
        "freeze_authorization_source": {"file": rel(FREEZE_APPROVAL_PATH), "sha256": sha256(FREEZE_APPROVAL_PATH)},
        "scope": "Q4 internal analytical and reproducibility package, workflow steps 0-10; source data hashes and all Q4 code, documentation, tables, and figures. This is not a full-paper or competition-submission approval. The independent review file is outside the hash set to avoid a circular reference.",
        "source_input_file_count": len(raw_inputs),
        "c8_json_source_file_count": len(c8_files),
        "field_dictionary": {
            "path": rel(data_dictionary), "version": "no_hidden_fields", "sha256": sha256(data_dictionary),
            "role": "visible field/schema reference; not statistical data or a modeling instruction source",
            "alternate_hidden_version_present": hidden_dictionary.exists(),
        },
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
            "no_hidden_field_dictionary_hash_confirmed": sha256(data_dictionary) == expected_dictionary_hash,
        },
        "independent_review_record": "Q4/05_审计交付/q4_quality_review.md; excluded from this manifest to avoid a circular hash.",
        "cohort_decision": {
            "status": "accepted_and_included_in_frozen_package" if package_status == "frozen" else "accepted_provisional",
            "decision": "Adopt the cohort and metric choices documented in Q4/01_方案说明/q4_cohort_decision.md.",
            "recorded_by": "user",
            "recorded_on": "2026-09-25",
        },
        "pending_before_competition_submission": ai_disclosure.get("completion_items", []),
    }
    manifest["generated_at_local"] = datetime.now().astimezone().isoformat(timespec="seconds")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"package_manifest={rel(MANIFEST)}")
    print(f"raw_inputs={len(raw_inputs)} c8_json={len(c8_files)} q4_files={len(output_files)} stage_output_hashes={sum(s['verified_output_hashes'] for s in stage_results)}")
    print(f"core6=789 full7=0 scale_time=788 annual_forecasts=0 visual_QA=PASS q4_package={package_status} other_ai_scope=pending_before_competition_submission")
    print(f"independent_review_record={'present' if REVIEW_FILE.is_file() else 'pending'}")
    print(f"manifest_sha256={sha256(MANIFEST)}")


if __name__ == "__main__":
    main()
