"""Write the Q4 raw-input, code, environment, and command provenance manifest.

AI assistance: OpenAI Codex desktop 26.917.71314 (build 10954; prod);
verified model ID gpt-6-luna (official name GPT-6 Luna), developer OpenAI,
official release date 2026-09-22. See Q4/05_审计交付/AI工具使用记录.md and
ai_disclosure.json for evidence scope and unresolved non-Codex scope.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
RESULTS = Q4 / "03_结果" / "results"
DATA = ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"
ATTACHMENTS = ROOT / "中文题目" / "F题" / "real_attachments"
OUTPUT = RESULTS / "复现清单.json"
AI_DISCLOSURE_PATH = Q4 / "05_审计交付" / "ai_disclosure.json"
FREEZE_APPROVAL_PATH = Q4 / "05_审计交付" / "q4_freeze_approval.json"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ai_disclosure = json.loads(AI_DISCLOSURE_PATH.read_text(encoding="utf-8"))
    freeze_approval = json.loads(FREEZE_APPROVAL_PATH.read_text(encoding="utf-8"))
    package_status = freeze_approval.get("package_status", "provisional_not_frozen")
    if package_status not in {"provisional_not_frozen", "frozen"}:
        raise RuntimeError(f"Unsupported Q4 package status: {package_status}")
    if package_status == "frozen" and freeze_approval.get("authorization_confirmed") is not True:
        raise RuntimeError("A frozen Q4 package requires explicit user authorization evidence")
    model_identity = ai_disclosure["model_identity"]
    if not model_identity.get("runtime_reported_family"):
        raise RuntimeError("AI disclosure record is missing the runtime-reported model family")

    registry_path = RESULTS / "q4_c8_file_registry.csv"
    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        registry = list(csv.DictReader(handle))
    c8_paths = sorted({(ROOT / Path(row["source_file"].replace("/", "\\"))).resolve() for row in registry if row.get("source_file")})
    if len(c8_paths) != 1958 or any(not path.is_file() for path in c8_paths):
        raise RuntimeError(f"C8 registry/input mismatch: expected 1,958 files, found {len(c8_paths)}")

    c_csv_paths = sorted(DATA.glob("*.csv"))
    problem_doc = ROOT / "中文题目" / "F题" / "算力约束下提升大语言模型能力的资源配置建模.docx"
    data_dictionary = ROOT / "中文题目" / "F题" / "数据说明(无隐藏字段版本）.pdf"
    source_manifest = ATTACHMENTS / "source_manifest.json"
    attachment_size_summary = ATTACHMENTS / "attachment_size_summary.csv"
    hidden_dictionary_alternate = ROOT / "中文题目" / "F题" / "数据说明.pdf"
    loss_gate_inputs = [
        ROOT / "中文题目" / "F题" / "real_attachments" / "B_scaling_laws" / "pythia_training_log_existing.csv",
        ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "b1_fitted_predictions.csv",
        ROOT / "Q2" / "05_审计交付" / "loss_protocol_matrix.csv",
    ]

    required = [*c_csv_paths, *c8_paths, problem_doc, data_dictionary, source_manifest, attachment_size_summary, *loss_gate_inputs]
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError("Required provenance input missing: " + "; ".join(missing[:10]))

    inputs: dict[str, dict[str, object]] = {}
    for path in c_csv_paths:
        inputs[rel(path)] = {"sha256": sha256(path), "role": "C1-C7 attached tabular source; source-specific use is recorded in stage manifests"}
    for path in c8_paths:
        inputs[rel(path)] = {"sha256": sha256(path), "role": "C8 per-model evaluation JSON; parser status is recorded in q4_c8_file_registry.csv"}
    inputs[rel(problem_doc)] = {"sha256": sha256(problem_doc), "role": "formal problem statement reference; not parsed as numerical observations"}
    inputs[rel(data_dictionary)] = {
        "sha256": sha256(data_dictionary),
        "role": "visible field dictionary/reference only; not parsed as numerical observations or modeling instructions",
        "version": "no_hidden_fields",
    }
    inputs[rel(source_manifest)] = {"sha256": sha256(source_manifest), "role": "attachment provenance inventory"}
    inputs[rel(attachment_size_summary)] = {"sha256": sha256(attachment_size_summary), "role": "attachment provenance inventory"}
    loss_gate_roles = [
        "Q2 Loss gate source; Pythia training-log evidence",
        "Q2 Loss gate target-support evidence; fitted B1 predictions",
        "Q2 Loss gate protocol evidence; measured evaluation-unit and protocol status",
    ]
    for path, role in zip(loss_gate_inputs, loss_gate_roles, strict=True):
        inputs[rel(path)] = {"sha256": sha256(path), "role": role}

    script_paths = sorted((Q4 / "02_代码" / "scripts").glob("*.py"))
    code_hashes = {rel(path): sha256(path) for path in script_paths}
    contracts = [
        Q4 / "01_方案说明" / "q4_cohort_decision.md",
        Q4 / "01_方案说明" / "q4_input_contract_v0.1.md",
        Q4 / "01_方案说明" / "q4_model_contract_v0.6.md",
        Q4 / "01_方案说明" / "q4_c8_schema_notes.md",
    ]
    contracts_hashes = {rel(path): sha256(path) for path in contracts if path.is_file()}

    packages = {}
    for name in ("numpy", "pandas", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    python = r"C:\Users\86147\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    commands = [
        {"step": 1, "purpose": "parse C8 and rebuild C1/C2 alignment and source candidates", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_c8_pipeline.py'"},
        {"step": 2, "purpose": "audit C3-C7 and monthly source coverage", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_auxiliary_source_audit.py'"},
        {"step": 3, "purpose": "run the Loss-Benchmark pre-fit feasibility gate; it must remain BLOCKED_NO_FIT", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_loss_bridge_feasibility.py'"},
        {"step": 4, "purpose": "force base task aggregation/frontier outputs before dynamic analysis", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_modeling_analysis.py' --bootstrap-only"},
        {"step": 5, "purpose": "run full dynamic decomposition and C3/C4 year summaries", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_dynamic_decomposition.py'"},
        {"step": 6, "purpose": "rerun modeling to integrate dynamic results in the report and full stage manifest", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_modeling_analysis.py'"},
        {"step": 7, "purpose": "build model registry, match audit, and coverage report after final modeling outputs", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_registry_coverage.py'"},
        {"step": 8, "purpose": "regenerate nine candidate figures, separate flowchart, and figure contracts", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_make_figures.py'"},
        {"step": 9, "purpose": "strictly audit all Q4 PNG/SVG figures", "command": "& $Python 'C:\\Users\\86147\\.codex\\skills\\math-modeling\\references\\roles\\编程手\\scripts\\figure_audit.py' 'Q4/04_图表' --questions q4 --min-dpi 300 --strict\n$figureAssets = Get-ChildItem 'Q4/04_图表' -Recurse -File | Where-Object { $_.Extension -in '.png', '.svg' } | ForEach-Object FullName\n& $Python 'C:\\Users\\86147\\.codex\\skills\\math-modeling\\tools\\figure\\scripts\\check_figure.py' $figureAssets --min-dpi 300 --strict"},
        {"step": 10, "purpose": "after a person inspects all color and grayscale previews, record visual QA", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_record_figure_visual_review.py'"},
        {"step": 11, "purpose": "write input/code/environment provenance and final ordered reproduction commands", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_write_repro_manifest.py'"},
        {"step": 12, "purpose": "verify invariants and write the package hash manifest using the recorded authorization state", "command": f"& '{python}' 'Q4/02_代码/scripts/q4_finalize_package.py'"},
    ]

    stage_names = [
        "q4_c8_manifest.json", "q4_auxiliary_audit_manifest.json", "q4_loss_benchmark_feasibility_manifest.json",
        "q4_dynamic_decomposition_manifest.json", "q4_modeling_manifest.json", "q4_registry_coverage_manifest.json",
    ]
    stage_hashes = {
        rel(RESULTS / name): sha256(RESULTS / name)
        for name in stage_names if (RESULTS / name).is_file()
    }
    dictionary_hash = sha256(data_dictionary)
    payload = {
        "schema_version": 1,
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": package_status,
        "purpose": "Q4 ordered reproduction record; raw inputs, code, environment, critical contracts, and stage manifests are pinned by SHA-256.",
        "raw_input_counts": {"C1_C7_top_level_csv": len(c_csv_paths), "C8_json": len(c8_paths), "total_including_docs_and_provenance": len(inputs)},
        "inputs": inputs,
        "data_dictionary_check": {
            "file": rel(data_dictionary), "sha256": dictionary_hash,
            "version": "no_hidden_fields",
            "use": "field names/schema reference only; not parsed by the statistical pipeline",
            "alternate_hidden_version_file": rel(hidden_dictionary_alternate),
            "alternate_hidden_version_present": hidden_dictionary_alternate.is_file(),
        },
        "formal_problem_statement": {"file": rel(problem_doc), "sha256": sha256(problem_doc)},
        "attachment_source_manifest": {"file": rel(source_manifest), "sha256": sha256(source_manifest)},
        "code_sha256": code_hashes,
        "model_contract_sha256": contracts_hashes,
        "stage_manifests_sha256": stage_hashes,
        "ai_disclosure_record": ai_disclosure,
        "ai_disclosure_source": {"file": rel(AI_DISCLOSURE_PATH), "sha256": sha256(AI_DISCLOSURE_PATH)},
        "freeze_authorization_record": freeze_approval,
        "freeze_authorization_source": {"file": rel(FREEZE_APPROVAL_PATH), "sha256": sha256(FREEZE_APPROVAL_PATH)},
        "environment": {
            "python_executable": python,
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "declared_requirements": ["numpy", "pandas", "Pillow"],
            "matplotlib_available": importlib.util.find_spec("matplotlib") is not None,
            "figure_rendering": "Pillow raster plus direct editable SVG primitives",
        },
        "key_decisions": {
            "q4_package_status": package_status,
            "cohort_and_metrics": "user-approved decision included in the frozen Q4 package; see q4_cohort_decision.md",
            "loss_benchmark": "pre-fit gate only; BLOCKED_NO_FIT; fit_executed=false",
            "annual_forecast": "no numeric 12/24 month forecast because rolling-origin support is absent",
            "ai_runtime_model_family": model_identity["runtime_reported_family"],
            "ai_model_selector_name": model_identity.get("selector_display_name") or "",
            "ai_model_id": model_identity.get("model_variant_or_id") or "",
            "additional_ai_model_ids": "; ".join(
                tool.get("model_id", "") for tool in ai_disclosure.get("additional_ai_tools", []) if tool.get("model_id")
            ),
            "ai_model_version_or_publication_date": "; ".join(
                value for value in (model_identity.get("model_version"), model_identity.get("publication_date")) if value
            ),
        },
        "ordered_commands": commands,
        "rebuild_notes": [
            "The first q4_modeling_analysis.py --bootstrap-only run deliberately writes base task/frontier outputs and exits with q4_modeling_bootstrap_manifest.json, regardless of any older dynamic files in results/.",
            "After q4_dynamic_decomposition.py runs, the second q4_modeling_analysis.py run integrates dynamic outputs and removes the temporary bootstrap manifest.",
            "Run q4_registry_coverage.py after the final modeling pass so its manifest hashes refer to final aggregate and monthly tables.",
            "Do not run the visual-review recorder until every color and grayscale preview has been opened and inspected.",
            "The Q4 analytical package was frozen on 2026-09-26 after explicit user authorization. The Q4-related ChatGPT output mapping is recorded at summary level. Whether any additional AI tool/model contributed is not confirmed and must be checked before final competition-paper submission; this package does not claim that none did.",
        ],
        "self_hash_note": "This file is intentionally not listed in its own inputs, code, or stage hashes; q4_package_manifest.json hashes this output.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"repro_manifest={rel(OUTPUT)} inputs={len(inputs)} C8_json={len(c8_paths)} code={len(code_hashes)}")
    print(f"no_hidden_data_dictionary_sha256={dictionary_hash} q4_package={package_status}")


if __name__ == "__main__":
    main()
