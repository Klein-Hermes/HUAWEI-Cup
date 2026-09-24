"""Audit whether B1-B5 contain traceable training-mixture p for Q2.

This is a provenance and identifiability-input audit only. It does not fit a
model, infer missing mixture shares, or read the hidden text layer of the
contest PDF.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ATTACHMENTS = ROOT / "中文题目" / "F题" / "real_attachments"
B_DIR = ATTACHMENTS / "B_scaling_laws"
BASELINE_OUT = ROOT / "results" / "q2_scaling_baseline" / "v1"
OUT_DIR = ROOT / "results" / "q2_p_feasibility_audit" / "v1"
Q2_OUT_DIR = ROOT / "Q2" / "03_结果" / "p配比可行性审计" / "v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def canonical_row_hash(row: dict[str, str]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pct_or_text(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


def detect_p_columns(headers: list[str]) -> list[str]:
    candidates = []
    for name in headers:
        lower = name.strip().lower()
        if lower in {"p", "p_vector", "mixture", "mixture_vector", "data_mix"}:
            candidates.append(name)
        elif lower.startswith("p_") or lower.startswith("mixture_share_"):
            candidates.append(name)
    return candidates


def collect_input_paths() -> list[Path]:
    paths = [
        ATTACHMENTS / "source_manifest.json",
        B_DIR / "pythia_training_log_existing.csv",
        B_DIR / "cerebras_training_log.csv",
        B_DIR / "scaling_baseline.csv",
        B_DIR / "published_scaling_data.csv",
        B_DIR / "pythia_checkpoint_index.csv",
        B_DIR / "open_model_family_metadata.csv",
        BASELINE_OUT / "gate0_b1_trajectory_mapping.csv",
        ATTACHMENTS / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv",
        ATTACHMENTS / "A_data_value" / "domain_mapping_guide.csv",
        ROOT / "题目分析报告.md",
        ROOT / "Q2" / "01_方案说明" / "题目分析报告.md",
        ROOT / "docs" / "F题_Q2公共实验框架.md",
        ROOT / "Q2" / "01_方案说明" / "F题_Q2公共实验框架.md",
    ]
    paths.extend(sorted((B_DIR / "training_trajectories").glob("*.csv")))
    return paths


def b4_b5_row_id(dataset: str, row_number: int, fingerprint: str) -> str:
    # The line number is only a locator inside the hashed source CSV; it is not
    # a cross-file join key. The row fingerprint binds the locator to content.
    return f"{dataset}:row_{row_number:04d}:{fingerprint[:12]}"


def build_audit() -> dict[str, Any]:
    source_manifest_path = ATTACHMENTS / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest_by_file = {
        str(entry.get("file", "")): entry
        for entry in source_manifest
        if isinstance(entry, dict)
    }

    b1_path = B_DIR / "pythia_training_log_existing.csv"
    b2_path = B_DIR / "cerebras_training_log.csv"
    b4_path = B_DIR / "scaling_baseline.csv"
    b5_path = B_DIR / "published_scaling_data.csv"
    b1_headers, b1_rows = read_csv(b1_path)
    b2_headers, b2_rows = read_csv(b2_path)
    b4_headers, b4_rows = read_csv(b4_path)
    b5_headers, b5_rows = read_csv(b5_path)

    gate_map_path = BASELINE_OUT / "gate0_b1_trajectory_mapping.csv"
    gate_headers, gate_rows = read_csv(gate_map_path)
    del gate_headers
    gate_by_n = {
        round(float(row["N_params_B"]), 9): row
        for row in gate_rows
    }

    rows: list[dict[str, Any]] = []
    b1_track_stats: list[dict[str, Any]] = []
    for n_value, gate in sorted(gate_by_n.items()):
        track_rows = [
            row for row in b1_rows
            if (number := as_float(row.get("N_params_B"))) is not None
            and abs(number - n_value) < 1e-8
        ]
        ds = [as_float(row.get("D_tokens_B")) for row in track_rows]
        steps = [row.get("steps", "") for row in track_rows]
        model_repo = gate["model_repo"]
        stats = {
            "dataset": "B1",
            "unit_type": "training_trajectory",
            "unit_id": model_repo,
            "model_family": "Pythia",
            "model_repo": model_repo,
            "N_params_B": pct_or_text(n_value),
            "record_count": len(track_rows),
            "distinct_D_count": len(set(value for value in ds if value is not None)),
            "D_min_B_tokens": pct_or_text(min((value for value in ds if value is not None), default=None)),
            "D_max_B_tokens": pct_or_text(max((value for value in ds if value is not None), default=None)),
            "loss_column": "val_loss",
            "source_role": "real Pythia checkpoint log; primary B1 fit source",
            "track_key_status": "model_repo mapped through Gate0/B12; run_id is row/checkpoint-level, not trajectory key",
            "p_recipe_group_candidate": "pythia_shared_training_data_order_policy",
            "p_recipe_status": "各规模共享数据顺序政策；数值化项目 17 域配比缺失或未联接",
            "p_D_status": "not reconstructed; local log has no per-source token counts or token-stream position map",
            "p_loss_join_status": "NO: no p columns or traceable p-to-Loss key in B1 table",
            "independent_recipe_count_contribution": "0 verified numeric p vectors; 8 sizes share one documented data-order policy",
            "eligible_as_real_p_variation": "NO under current local evidence",
            "audit_locator": "",
            "record_fingerprint_sha256": "",
            "limitations": "Pythia official data/code make a future reconstruction pathway plausible; exact project-17 mapping and row/checkpoint join remain absent.",
        }
        b1_track_stats.append(stats)
        rows.append(stats)

    b2_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in b2_rows:
        b2_groups[row.get("run_id", "")].append(row)
    for run_id, track_rows in sorted(b2_groups.items()):
        first = track_rows[0]
        ds = [as_float(row.get("D_tokens_B")) for row in track_rows]
        n_value = as_float(first.get("N_params_B"))
        rows.append({
            "dataset": "B2",
            "unit_type": "semi_synthetic_track",
            "unit_id": run_id,
            "model_family": "Cerebras-GPT",
            "model_repo": "",
            "N_params_B": pct_or_text(n_value),
            "record_count": len(track_rows),
            "distinct_D_count": len(set(value for value in ds if value is not None)),
            "D_min_B_tokens": pct_or_text(min((value for value in ds if value is not None), default=None)),
            "D_max_B_tokens": pct_or_text(max((value for value in ds if value is not None), default=None)),
            "loss_column": "val_loss",
            "source_role": "semi-synthetic Pythia-calibrated trajectory; not a real Cerebras run",
            "track_key_status": "run_id groups checkpoints by synthetic model track",
            "p_recipe_group_candidate": "unknown_not_counted",
            "p_recipe_status": "not present; source manifest identifies this table as semi-synthetic",
            "p_D_status": "not reconstructable from this synthetic loss table",
            "p_loss_join_status": "NO: no p columns or verified p-to-Loss key",
            "independent_recipe_count_contribution": "0 (excluded from real-data gate)",
            "eligible_as_real_p_variation": "NO: semi-synthetic",
            "audit_locator": "",
            "record_fingerprint_sha256": "",
            "limitations": "Structural missing gpu_days, step_time_ms, and grad_norm_avg are preserved; no field is imputed.",
        })

    trajectory_files = sorted((B_DIR / "training_trajectories").glob("*.csv"))
    for path in trajectory_files:
        headers, track_rows = read_csv(path)
        ds = [as_float(row.get("D_tokens_B")) for row in track_rows]
        n_value = as_float(track_rows[0].get("N_params_B")) if track_rows else None
        interpolation_flags = {row.get("interpolated", "") for row in track_rows}
        rows.append({
            "dataset": "B3",
            "unit_type": "interpolated_trajectory",
            "unit_id": path.stem,
            "model_family": "Pythia-derived",
            "model_repo": "",
            "N_params_B": pct_or_text(n_value),
            "record_count": len(track_rows),
            "distinct_D_count": len(set(value for value in ds if value is not None)),
            "D_min_B_tokens": pct_or_text(min((value for value in ds if value is not None), default=None)),
            "D_max_B_tokens": pct_or_text(max((value for value in ds if value is not None), default=None)),
            "loss_column": "val_loss",
            "source_role": "derived/interpolated Pythia trajectory; not independent real evidence",
            "track_key_status": "filename and trajectory file identify derived size path; all rows marked interpolated",
            "p_recipe_group_candidate": "inherits_pythia_shared_policy_unquantified",
            "p_recipe_status": "no p columns; derived from Pythia path and not an independent recipe",
            "p_D_status": "not reconstructed; interpolation creates no token-source composition observations",
            "p_loss_join_status": "NO: no p columns or verified p-to-Loss key",
            "independent_recipe_count_contribution": "0 (derived data excluded from real-data gate)",
            "eligible_as_real_p_variation": "NO: interpolated",
            "audit_locator": path.relative_to(ROOT).as_posix(),
            "record_fingerprint_sha256": sha256(path),
            "limitations": f"interpolated flag values={','.join(sorted(interpolation_flags))}; source file headers={','.join(headers)}",
        })

    def add_cross_section(dataset: str, table_path: Path, table_rows: list[dict[str, str]]) -> None:
        occurrences: Counter[str] = Counter()
        for row_number, raw_row in enumerate(table_rows, start=2):
            fingerprint = canonical_row_hash(raw_row)
            occurrences[fingerprint] += 1
            family = raw_row.get("family", "")
            source = raw_row.get("source", "")
            n_value = as_float(raw_row.get("N_params_B"))
            d_value = as_float(raw_row.get("D_tokens_B"))
            locator = b4_b5_row_id(dataset, row_number, fingerprint)
            rows.append({
                "dataset": dataset,
                "unit_type": "cross_sectional_model_point",
                "unit_id": locator,
                "model_family": family,
                "model_repo": "",
                "N_params_B": pct_or_text(n_value),
                "record_count": 1,
                "distinct_D_count": 1 if d_value is not None else 0,
                "D_min_B_tokens": pct_or_text(d_value),
                "D_max_B_tokens": pct_or_text(d_value),
                "loss_column": "val_loss",
                "source_role": "curated convergence/literature point; cross-sectional, not a training trajectory",
                "track_key_status": (
                    "family label only; no row-level model/run/checkpoint identifier"
                    if dataset == "B4"
                    else "paper citation string and family label only; no row-level run/checkpoint key"
                ),
                "p_recipe_group_candidate": "unresolved_source_specific_recipe",
                "p_recipe_status": (
                    "no source field or p vector in row; provenance must be resolved before assigning a recipe"
                    if dataset == "B4"
                    else f"source citation lead={source}; no 17-domain p vector or run-level link in row"
                ),
                "p_D_status": "not applicable to a single endpoint point unless its run-level exposure path is separately sourced",
                "p_loss_join_status": "NO: no p columns or verified p-to-Loss key",
                "independent_recipe_count_contribution": "0 verified p vectors; do not count family/source labels as vectors",
                "eligible_as_real_p_variation": "POTENTIAL ONLY after row-to-paper/model and taxonomy mapping is audited",
                "audit_locator": f"{table_path.relative_to(ROOT).as_posix()}#csv_line={row_number}; locator only, not a join key",
                "record_fingerprint_sha256": fingerprint,
                "limitations": f"source={source}; identical-row occurrence={occurrences[fingerprint]}",
            })

    add_cross_section("B4", b4_path, b4_rows)
    add_cross_section("B5", b5_path, b5_rows)

    p_columns_by_dataset = {
        "B1": detect_p_columns(b1_headers),
        "B2": detect_p_columns(b2_headers),
        "B3": [],
        "B4": detect_p_columns(b4_headers),
        "B5": detect_p_columns(b5_headers),
    }
    a_mix_headers, _ = read_csv(ATTACHMENTS / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv")
    reference_p_columns = [name for name in a_mix_headers if name.startswith("train_the_pile_")]
    mapping_headers, mapping_rows = read_csv(ATTACHMENTS / "A_data_value" / "domain_mapping_guide.csv")
    mapping_counts = Counter(row.get("mapping_type", "") for row in mapping_rows)

    b5_source_groups: dict[str, dict[str, Any]] = {}
    for raw_row in b5_rows:
        source = raw_row.get("source", "")
        group = b5_source_groups.setdefault(source, {"rows": 0, "families": set()})
        group["rows"] += 1
        group["families"].add(raw_row.get("family", ""))

    input_files = []
    for path in collect_input_paths():
        if path.exists() and path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            input_files.append({
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

    declared_size_checks = []
    for rel, entry in manifest_by_file.items():
        if rel.startswith("B_scaling_laws/"):
            path = ATTACHMENTS / rel
            if path.exists() and path.is_file() and "bytes" in entry:
                declared_size_checks.append({
                    "path": (Path("中文题目") / "F题" / "real_attachments" / rel).as_posix(),
                    "manifest_bytes": entry["bytes"],
                    "actual_bytes": path.stat().st_size,
                    "match": entry["bytes"] == path.stat().st_size,
                })

    structural_missing = {
        column: sum(1 for row in b2_rows if not row.get(column, ""))
        for column in ("gpu_days", "step_time_ms", "grad_norm_avg")
    }
    b1_unique_run_ids = len({row.get("run_id", "") for row in b1_rows})
    b1_n_distinct_d = len({row.get("D_tokens_B", "") for row in b1_rows})
    b2_n_distinct_d = len({row.get("D_tokens_B", "") for row in b2_rows})
    b4_family_count = len({row.get("family", "") for row in b4_rows})
    b5_family_count = len({row.get("family", "") for row in b5_rows})

    summary: dict[str, Any] = {
        "audit_date": date.today().isoformat(),
        "scope": "B1-B5 p provenance and data-identifiability pre-gate only; no model fitting",
        "overall_p_only_data_gate": "FAIL",
        "gate_interpretation": "Current local B1-B5 Loss records do not provide a verified, row-linked 17-domain p vector or enough verified between-recipe variation to begin formal p-only M1 fitting.",
        "provenance_reconstruction_path": "CONDITIONAL",
        "provenance_reconstruction_note": "Some primary publications and the Pythia repository expose source or mixture information, but it is not consistently row-linked to B4/B5 and is not normalized to the project 17-domain p contract. B1 official dataloader reconstruction would still require source-token counts, exact checkpoint alignment, and a validated taxonomy crosswalk.",
        "model_fitted": False,
        "collinearity_test": "NOT_TESTABLE: no B1-B5 table contains joined p components; no VIF/correlation/regression was run.",
        "p_recipe_between_track_result": "B1 has one documented shared data-order policy across eight sizes; it is not eight independent mixture vectors. Numeric shares in the project p taxonomy are absent.",
        "p_D_result": "Not reconstructed. The local logs do not contain per-source token counts by checkpoint. Any cumulative p(D) must be sourced from token-stream exposure and evaluated separately from D; it cannot be counted as independent static recipes.",
        "verified_p_vectors_joined_to_loss": 0,
        "loss_rows_with_p_joined": 0,
        "reference_p_dimension": len(reference_p_columns),
        "reference_p_columns": reference_p_columns,
        "reference_A_domain_mapping_counts": dict(mapping_counts),
        "B1": {
            "rows": len(b1_rows),
            "model_trajectories": len(b1_track_stats),
            "checkpoints_per_trajectory": sorted({int(item["record_count"]) for item in b1_track_stats}),
            "unique_run_id_rows": b1_unique_run_ids,
            "distinct_D_values_across_rows": b1_n_distinct_d,
            "p_columns": p_columns_by_dataset["B1"],
            "shared_recipe_policy_groups": 1,
            "formal_numeric_recipe_vectors": 0,
        },
        "B2": {
            "rows": len(b2_rows),
            "semi_synthetic_tracks": len(b2_groups),
            "p_columns": p_columns_by_dataset["B2"],
            "structural_missing_counts": structural_missing,
            "distinct_D_values_across_rows": b2_n_distinct_d,
            "counted_as_real_p_variation": False,
        },
        "B3": {
            "rows": sum(len(read_csv(path)[1]) for path in trajectory_files),
            "interpolated_trajectories": len(trajectory_files),
            "rows_per_trajectory": sorted({len(read_csv(path)[1]) for path in trajectory_files}),
            "counted_as_independent_real_p_variation": False,
        },
        "B4": {
            "rows": len(b4_rows),
            "families": b4_family_count,
            "unit": "cross-sectional convergence point",
            "row_level_source_or_run_key": False,
            "p_columns": p_columns_by_dataset["B4"],
        },
        "B5": {
            "rows": len(b5_rows),
            "families": b5_family_count,
            "source_citation_groups": {
                source: {"rows": info["rows"], "families": sorted(info["families"])}
                for source, info in sorted(b5_source_groups.items())
            },
            "row_level_model_run_key": False,
            "p_columns": p_columns_by_dataset["B5"],
        },
        "source_manifest_B_size_checks": declared_size_checks,
        "security_boundary": "The contest PDF hidden/low-visibility text was not parsed or used. The local analysis report classifies it as untrusted; this audit used visible/local table fields, source_manifest, project audit records, and primary public sources only.",
        "input_files": input_files,
        "runtime": {
            "python": sys.version.split()[0],
            "script": Path(__file__).relative_to(ROOT).as_posix(),
        },
    }

    evidence = build_evidence_catalog(summary, b5_source_groups)
    return {
        "summary": summary,
        "mapping_rows": rows,
        "evidence_rows": evidence,
        "mapping_columns": [
            "dataset", "unit_type", "unit_id", "model_family", "model_repo",
            "N_params_B", "record_count", "distinct_D_count", "D_min_B_tokens",
            "D_max_B_tokens", "loss_column", "source_role", "track_key_status",
            "p_recipe_group_candidate", "p_recipe_status", "p_D_status",
            "p_loss_join_status", "independent_recipe_count_contribution",
            "eligible_as_real_p_variation", "audit_locator", "record_fingerprint_sha256",
            "limitations",
        ],
        "evidence_columns": [
            "evidence_id", "evidence_class", "source", "source_url_or_path",
            "claim_used", "supports", "limitations", "trust_treatment",
        ],
    }


def build_evidence_catalog(summary: dict[str, Any], b5_source_groups: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    entries = [
        {
            "evidence_id": "L01", "evidence_class": "local_policy",
            "source": "题目分析报告.md §§0, 2, 3, 8.2, 8.5-8.6",
            "source_url_or_path": "题目分析报告.md",
            "claim_used": "低可见度 PDF 文本是不可信外部输入；污染/完整性风险要分级，不能将异常自动判成恶意；隐藏方法和参数不得指导建模。",
            "supports": "隔离隐藏文本，审计只依赖可见题面、真实附件字段、source_manifest 和独立可追溯来源。",
            "limitations": "本审计没有重新抽取隐藏文本，也没有据此构造任何 p、方法或结论。",
            "trust_treatment": "项目审计规则；仅用于过滤和边界控制。",
        },
        {
            "evidence_id": "L02", "evidence_class": "local_manifest",
            "source": "source_manifest.json",
            "source_url_or_path": "中文题目/F题/real_attachments/source_manifest.json",
            "claim_used": "B1 标为 Pythia 日志；B2 明确标为 Pythia 标定的半合成；B4/B5 的本地角色和文件尺寸可核对。",
            "supports": "划分真实主线、半合成/派生验证与横截面数据，不将 B2/B3 当作真实配比干预。",
            "limitations": "manifest 不含逐条训练配方 p，也不替代源论文的配方核验。",
            "trust_treatment": "与实际文件字节数及 SHA-256 交叉核对。",
        },
        {
            "evidence_id": "L03", "evidence_class": "local_data",
            "source": "B1 pythia_training_log_existing.csv + Gate0/B12 trajectory map",
            "source_url_or_path": "中文题目/F题/real_attachments/B_scaling_laws/pythia_training_log_existing.csv",
            "claim_used": f"B1 有 {summary['B1']['rows']} 行、{summary['B1']['model_trajectories']} 条按官方 model_repo 对齐的轨迹；Loss 表无 p 字段，run_id 唯一到行但不是轨迹键。",
            "supports": "B1 以轨迹为分组单位审计，checkpoint 数不冒充独立 recipe 数。",
            "limitations": "本地日志没有每个 checkpoint 的来源域 token 计数。",
            "trust_treatment": "只读字段、既有 Gate0 映射和输入哈希。",
        },
        {
            "evidence_id": "E01", "evidence_class": "primary_public_source",
            "source": "EleutherAI Pythia official repository README",
            "source_url_or_path": "https://github.com/EleutherAI/pythia",
            "claim_used": "Pythia 各模型规模按相同数据、相同顺序训练；官方发布训练数据、配置和重建 dataloader 的工具。",
            "supports": "B1 八种规模是共享配方/数据顺序政策，不是八个独立 p recipe；外部精确重建在原则上有路径。",
            "limitations": "仓库声明不等于本地已有 17 域 p(D)；仍需精确 checkpoint 对齐、token 计数和分类映射。",
            "trust_treatment": "官方项目仓库；2026-09-24 检索。",
        },
        {
            "evidence_id": "E02", "evidence_class": "primary_public_source",
            "source": "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling",
            "source_url_or_path": "https://arxiv.org/abs/2304.01373",
            "claim_used": "论文描述模型使用相同公开数据顺序，并提供 checkpoint 与训练数据重建工具。",
            "supports": "与官方仓库相互核对 B1 共同配方和可能的 p(D) 重建路径。",
            "limitations": "不能据此把训练规模/检查点数量当成配比独立变化。",
            "trust_treatment": "原始论文；2026-09-24 检索。",
        },
        {
            "evidence_id": "L04", "evidence_class": "local_data",
            "source": "B2 Cerebras training log + source_manifest.json",
            "source_url_or_path": "中文题目/F题/real_attachments/B_scaling_laws/cerebras_training_log.csv",
            "claim_used": f"B2 有 {summary['B2']['rows']} 行、{summary['B2']['semi_synthetic_tracks']} 条合成轨迹；gpu_days、step_time_ms、grad_norm_avg 全缺失，且无 p 字段。",
            "supports": "B2 只作半合成压力/迁移检查，不作为真实独立 recipe 计数。",
            "limitations": "缺失字段为结构性缺失，不补零。",
            "trust_treatment": "manifest 与实际缺失模式交叉核对。",
        },
        {
            "evidence_id": "E03", "evidence_class": "primary_public_source",
            "source": "Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster",
            "source_url_or_path": "https://arxiv.org/abs/2304.03208",
            "claim_used": "原始 Cerebras-GPT 论文说明模型在 EleutherAI Pile 上训练。",
            "supports": "仅作为 B2 数据来源背景，不能把本地半合成轨迹改称原始训练日志。",
            "limitations": "论文语料声明不提供本地 B2 每条 Loss 的 17 域 p 联接。",
            "trust_treatment": "原始论文；2026-09-24 检索。",
        },
        {
            "evidence_id": "L05", "evidence_class": "local_data",
            "source": "B3 training_trajectories/*.csv",
            "source_url_or_path": "中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/",
            "claim_used": "8 条 Pythia 派生路径各 500 点；interpolated 列均为 1；文件无 p 字段。",
            "supports": "B3 是插值产物，不产生独立配比变化或真实训练配方。",
            "limitations": "B3 点值的来源仍依赖 Pythia B1；不作为 p 多样性增量。",
            "trust_treatment": "逐文件读取列、行数、插值标志并计算文件哈希。",
        },
        {
            "evidence_id": "L06", "evidence_class": "local_data",
            "source": "B4 scaling_baseline.csv",
            "source_url_or_path": "中文题目/F题/real_attachments/B_scaling_laws/scaling_baseline.csv",
            "claim_used": f"B4 有 {summary['B4']['rows']} 个点、{summary['B4']['families']} 个 family；只有 family/N/D/Loss/converged 字段，无 p、逐行 source 或运行键。",
            "supports": "B4 只能作为横截面来源审计；不能按 family 直接构造训练轨迹或 p。",
            "limitations": "source_manifest 只说明是 curated snapshots；需要补齐每条记录的原始模型与论文定位。",
            "trust_treatment": "本地表字段 + manifest；记录哈希作为审计定位符，不把行号当 join key。",
        },
        {
            "evidence_id": "L07", "evidence_class": "local_data",
            "source": "B5 published_scaling_data.csv",
            "source_url_or_path": "中文题目/F题/real_attachments/B_scaling_laws/published_scaling_data.csv",
            "claim_used": f"B5 有 {summary['B5']['rows']} 个点、{summary['B5']['families']} 个 family、{len(summary['B5']['source_citation_groups'])} 个 source 字符串；无 p 或逐条运行键。",
            "supports": "来源字符串只是追溯线索，不等于已建立某个模型运行到 Loss 行的 recipe 联接。",
            "limitations": "不同论文披露粒度和数据分类不同；不得把同一作者/家族或其他版本比例直接转给某行。",
            "trust_treatment": "按原字符串分组并保留到逐行映射表。",
        },
        {
            "evidence_id": "E04", "evidence_class": "primary_public_source",
            "source": "Language Models are Few-Shot Learners (Brown et al., 2020)",
            "source_url_or_path": "https://arxiv.org/abs/2005.14165",
            "claim_used": "GPT-3 论文 Table 2.2 报告一个 5 类训练 mix 的抽样权重。",
            "supports": "说明个别公开论文能提供候选配方；该候选不能自动赋给 B5 的 Kaplan et al. 2020 行。",
            "limitations": "B5 的 source 字段是 Kaplan et al. 2020；Brown 的 GPT-3 配方不是该字段的同一引用，且五类 taxonomy 不等于项目 17 维 p。",
            "trust_treatment": "只作反例/来源区分；不将此配方写入 B5 映射。",
        },
        {
            "evidence_id": "E05", "evidence_class": "primary_public_source",
            "source": "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
            "source_url_or_path": "https://arxiv.org/abs/2302.13971",
            "claim_used": "原始 LLaMA 论文 Table 1 披露 LLaMA v1 的 7 类采样比例。",
            "supports": "B5 中 LLaMA v1 行存在候选配方重建线索。",
            "limitations": "B5 同一 source 字符串还包括 LLaMA-2 行；不能把 LLaMA v1 配方自动复制给 LLaMA-2，且该 7 类表尚未映射为项目 17 类、也未与 B5 行逐条核对。",
            "trust_treatment": "候选来源；暂不作为已联接的正式 p。",
        },
        {
            "evidence_id": "E06", "evidence_class": "primary_public_source",
            "source": "OPT: Open Pre-trained Transformer Language Models (Zhang et al., 2022)",
            "source_url_or_path": "https://arxiv.org/abs/2205.01068",
            "claim_used": "论文数据卡列出 OPT 训练语料的多个组成来源，但并未在本审计所需的项目 17 域合同下给出可直接使用的权重向量。",
            "supports": "B5 OPT 的 source 字段可作为查证入口，不足以生成正式 p。",
            "limitations": "不能把各语料容量、验证样本抽样比例或 token 总量当作训练抽样权重。",
            "trust_treatment": "原始论文；不做比例推断。",
        },
        {
            "evidence_id": "E07", "evidence_class": "primary_public_source",
            "source": "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model",
            "source_url_or_path": "https://arxiv.org/abs/2211.05100",
            "claim_used": "BLOOM 使用 ROOTS 的大量多语言来源。",
            "supports": "B5 的 BLOOM source 不能直接等同到项目的 17 个 Pile/RegMix 域。",
            "limitations": "需逐来源构造可审计 crosswalk 和训练 token 权重；语言/语料清单不等于 p。",
            "trust_treatment": "原始论文；不把不同 taxonomy 强制并成 17 维。",
        },
        {
            "evidence_id": "E08", "evidence_class": "primary_public_source",
            "source": "PaLM: Scaling Language Modeling with Pathways",
            "source_url_or_path": "https://arxiv.org/abs/2204.02311",
            "claim_used": "PaLM 是独立训练来源；本审计没有发现本地 B5 到项目 17 维配比的逐条连接证据。",
            "supports": "将 B5 PaLM 条目保留为文献候选，而不把源列表/语言统计当成域配比。",
            "limitations": "若后续声称有具体权重，须能在原论文/官方配置中定位并证明分母定义。",
            "trust_treatment": "原始论文；当前仅记 provenance lead。",
        },
        {
            "evidence_id": "E09", "evidence_class": "primary_public_source",
            "source": "Scaling Laws for Neural Language Models (Kaplan et al., 2020)",
            "source_url_or_path": "https://arxiv.org/abs/2001.08361",
            "claim_used": "B5 原始 source 字符串为 Kaplan et al. 2020，覆盖 GPT-2/GPT-3 family 标签；需要按该 source 实际记录定位训练运行。",
            "supports": "明确禁止把 Brown et al. 的 GPT-3 训练 mix 仅因 family 标签相似就转移到 Kaplan 来源行。",
            "limitations": "B5 CSV 没有逐行原始模型 ID/训练运行键，当前 citation 字符串不足以确定各行的精确数据配方。",
            "trust_treatment": "原始论文；只用于来源辨析，不臆造或转填比例。",
        },
        {
            "evidence_id": "E10", "evidence_class": "primary_public_source",
            "source": "Training Compute-Optimal Large Language Models (Hoffmann et al., 2022) and original Gopher paper",
            "source_url_or_path": "https://arxiv.org/abs/2203.15556 ; https://arxiv.org/abs/2112.11446",
            "claim_used": "B5 把 Chinchilla 与 Gopher 点归在 Hoffmann et al. 2022 source 字符串下；Chinchilla 与 Gopher 是不同训练运行/模型来源。",
            "supports": "每个模型家族须回到自己的训练报告确认 recipe，不能把论文比较中引用的模型当成该论文作者训练的配方。",
            "limitations": "B5 缺逐行 run/version key，当前不能判定哪些 Gopher 点可与原始 Gopher 报告精确连接。",
            "trust_treatment": "原始论文交叉核对；当前保留为待解析来源线索。",
        },
        {
            "evidence_id": "L08", "evidence_class": "local_reference_schema",
            "source": "A train_mixture_1m.csv + domain_mapping_guide.csv",
            "source_url_or_path": "中文题目/F题/real_attachments/A_data_value/",
            "claim_used": f"Q1 配比参考合同为 {summary['reference_p_dimension']} 个 train_the_pile_* 分量；映射表描述 mixture_domain 到 quality_domain 的映射，不提供 B1-B5 的 Loss 级训练配方。",
            "supports": "固定 17 分量参照 schema，避免把 Q1 行或推断域映射拼接到 B 组。",
            "limitations": "A 表配比不能按参数规模、来源名、行序或 family 直接联接到 B 组 Loss。",
            "trust_treatment": "仅 schema/reference；不作为 B 组观测。",
        },
    ]
    return entries


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(summary: dict[str, Any], mapping_rows: list[dict[str, Any]], evidence_rows: list[dict[str, str]]) -> str:
    b5_sources = summary["B5"]["source_citation_groups"]
    source_lines = [
        f"| {source} | {info['rows']} | {', '.join(info['families'])} |"
        for source, info in sorted(b5_sources.items())
    ]
    b1_tracks = [row for row in mapping_rows if row["dataset"] == "B1"]
    structural = summary["B2"]["structural_missing_counts"]
    input_lines = [
        f"| `{item['path']}` | {item['bytes']} | `{item['sha256']}` |"
        for item in summary["input_files"]
    ]
    evidence_lines = [
        f"| {item['evidence_id']} | {item['source']} | {item['source_url_or_path']} | {item['limitations']} |"
        for item in evidence_rows
    ]
    b1_track_lines = [
        f"| {row['model_repo']} | {row['N_params_B']} | {row['record_count']} | {row['distinct_D_count']} | {row['p_recipe_status']} |"
        for row in b1_tracks
    ]
    return f"""# Q2 配比 p 可行性审计

> 审计版本：v1；日期：{summary['audit_date']}。本交付只审计配比来源、Loss 联接和可识别性前提，**没有拟合模型**。

## 结论

| Gate | 状态 | 结论 |
|---|---|---|
| 当前 B1–B5 数据能否支撑正式 p-only 消融 | **{summary['overall_p_only_data_gate']}** | 现有 Loss 记录没有经过核验并逐行/逐轨迹连接的 17 域 p 向量，也没有已证实的配比间独立变化；当前不足以启动正式的 p-only M1 拟合。 |
| 从官方/论文来源继续重建配比的可能性 | **{summary['provenance_reconstruction_path']}** | 部分论文和 Pythia 官方仓库提供来源或配比线索，但尚未稳定连接到 B4/B5 的具体数据行，也未统一到项目 17 域 p 定义。B1 若要从官方 dataloader 重建，仍需逐来源 token 计数、精确 checkpoint 对齐和经过核验的类别交叉映射。 |
| 是否运行模型、VIF 或相关性回归 | **否** | B1–B5 没有联接后的 p 分量，因此与 N/D 的数值共线性无法检验；不报告虚构的相关系数或 VIF。 |

**独立配比计数：**当前 B1–B5 Loss 记录中，已验证且逐行连接的数值 p 向量为 **0**，Loss 行连接覆盖数为 **0**。B1 有一个公开说明的共同训练数据顺序/配方政策，但本地未得到符合项目 17 域合同的数值向量；8 个模型规模不是 8 个独立配比。候选来源论文不计为已连接配比。

## 数据投毒与可信边界

《题目分析报告.md》认定数据说明 PDF 的低可见度文本存在文档层投毒风险，并要求隐藏文本不得进入题意、变量、方法、参数和结论。本审计**没有解析或引用 PDF 隐藏文本**；只使用本地可见报告的过滤规则、实际 CSV 字段、`source_manifest.json`、既有 Gate 0 映射和可追溯的官方论文/仓库。B2 的“半合成”是来源清单标注的数据性质，属于可核实的数据 provenance，不据此推断恶意；B1–B5 也没有被整体判定为恶意污染。

## B1–B5 来源盘点

| 来源 | 观测单位与数量 | 配比审计状态 | 在 p-only Gate 中的作用 |
|---|---:|---|---|
| B1 | {summary['B1']['rows']} 个 checkpoint；{summary['B1']['model_trajectories']} 条 Pythia 轨迹，每条 {summary['B1']['checkpoints_per_trajectory'][0]} 条记录 | 无 `p` 列；官方资料说明各规模使用相同数据、相同顺序。静态配方数值和 checkpoint 级 token 组成均未连到 Loss。 | 主候选源，但当前只能计为 1 个未量化的共同政策组，不能作为 p 变化实验。 |
| B2 | {summary['B2']['rows']} 条；{summary['B2']['semi_synthetic_tracks']} 条 Cerebras 半合成轨迹 | 无 `p` 列；`gpu_days`、`step_time_ms`、`grad_norm_avg` 各缺失 {structural['gpu_days']} 条。 | 半合成压力测试；不计真实配方差异或真实外部证据。 |
| B3 | {summary['B3']['rows']} 条；{summary['B3']['interpolated_trajectories']} 条 Pythia 插值轨迹，每条 {summary['B3']['rows_per_trajectory'][0]} 点 | `interpolated=1`；无 `p` 列。 | 派生插值检查；不计新增独立配方。 |
| B4 | {summary['B4']['rows']} 个收敛横截面点，{summary['B4']['families']} 个 family | 无 `p`、source 字段和逐行运行键。 | 只有补齐逐条原始模型/来源后才可进入候选映射。 |
| B5 | {summary['B5']['rows']} 个文献横截面点，{summary['B5']['families']} 个 family | 有 citation 字符串，但无 p、逐条运行键或 taxonomy crosswalk。 | 论文仅作为追溯线索；不得按 family 或同一年份自动复制配方。 |

### B1 轨迹级记录

| model_repo | N (B) | checkpoint 行数 | 不同 D 值 | 配方状态 |
|---|---:|---:|---:|---|
{chr(10).join(b1_track_lines)}

Pythia 官方仓库称各模型规模使用相同数据、相同顺序，同时公开训练数据和重建 dataloader 的工具。[官方仓库](https://github.com/EleutherAI/pythia) [论文](https://arxiv.org/abs/2304.01373)。因此，B1 的 8 条规模轨迹共享一个数据顺序政策，不形成 8 个独立静态配比。未来从数据流重建的累计组成 $p(D)$ 可能随进度改变，但目前本地文件没有逐来源 token 计数；且该路径是同一训练顺序下的暴露轨迹，不能当成跨轨迹独立 recipe 变化。

## 可追溯来源审计

### B4/B5 的来源线索

| B5 `source` 字符串 | 点数 | family | 处理结论 |
|---|---:|---|---|
{chr(10).join(source_lines)}

- B4 文件只有 `family, N_params_B, D_tokens_B, val_loss, is_converged`。它没有逐行引用，因此现在无法判断某一 Loss 点对应哪篇论文、哪个具体模型版本和训练配方。B4 中还有 Pythia、Cerebras-GPT 等与其他来源可能重叠的 family；在无 row-level identity 时，不把它们计作独立验证或独立配比。
- B5 的 source 字符串能帮助定位论文，但**citation 不等于 recipe join**。例如 Brown et al. 的 GPT-3 论文确实公布过训练 mix 权重，但 B5 标记为 `Kaplan et al. 2020`；不能仅凭 `GPT-3` family 把 Brown 论文的五类权重塞给 Kaplan 来源记录。原始 LLaMA 论文披露了 LLaMA v1 的配比，但 B5 同一 source 字符串下同时有 LLaMA 与 LLaMA-2；不能自动把 v1 配方复制给 v2。OPT、BLOOM、PaLM 等来源的语料清单/来源描述也不自动等于统一的 17 域权重向量。
- B5 将 Chinchilla 和 Gopher 点都记在 `Hoffmann et al. 2022` 下；Gopher 的原始训练报告另有其文献来源。没有逐行版本/运行键前，不把 Chinchilla 文献或比较表中引用的 Gopher 记录当成相同训练配方。
- 因此，文献级可恢复性为 **CONDITIONAL**：个别模型/版本存在配比线索，但必须完成“source 字符串 → 具体论文版本 → 模型点 → 训练 recipe → 项目 17 域”的逐行证据链，处理缺失域和不完整覆盖，并确认分母是训练采样 token 而非语料大小、验证抽样比例或参数规模。

## `p_recipe`、`p(D)` 与共线性

1. **`p_recipe`** 是训练配置中的固定采样权重。只按独立配置/配方计数，不能把每个 checkpoint、每种 `N` 或重复文献点都计为新配方。当前可确认 B1 只有一组共同数据顺序政策，尚无数值 17 维向量；B2 合成轨迹不进真实数据计数；B4/B5 的行级配方尚未连接。
2. **`p(D)`** 是截至累计 token 进度 `D` 的已观测数据组成。重建它需要 checkpoint 对应的 token stream/source token counts。不得把 `p(D)` 误标成静态配方；也不得把沿同一数据顺序得到的许多检查点当作许多独立 mixture。
3. **共线性检查状态为不可检验。** 因为没有行级 p 矩阵，本次不计算 p 与 `D/N` 的相关、VIF、回归残差或有效秩。下一轮必须先在已核验的配方集上将组成向量转换为可审计的对比坐标，再检查独立配方间 p 的变化能否与 `log N`、`log D`、模型族和数据来源区分；对 `p(D)` 还要单独检查它由 `D` 决定的结构关系。
4. 项目 A 侧的 `train_mixture_1m.csv` 有 {summary['reference_p_dimension']} 个 `train_the_pile_*` 分量。`domain_mapping_guide.csv` 是 A 侧 mixture domain 到 quality domain 的映射，不是 B 训练轨迹的 p 观测。不得把 A4–A15 的配比按行序、模型大小或 family 拼到 B Loss 行上，也不得将部分类别无说明归一化为完整 p。

## PASS / CONDITIONAL / FAIL 判据

| 状态 | 最低条件 | 当前判定 |
|---|---|---|
| PASS | 对真实 B Loss 记录有逐条/逐轨迹可追溯 p；配置来源和分母清楚；存在足够独立 recipe 对比；控制 N、D、模型族/来源后仍有可用变异。 | **未满足** |
| CONDITIONAL | 某些源可重建 p，但需补充原始配置、模型版本、行级连接、类目交叉表或有限配方支持；只能继续 provenance reconstruction，不能宣称 p-only 可识别。 | **适用于继续追溯工作** |
| FAIL | 当前数据不满足上述 M1 输入条件，例如 p 无法追溯、有效配方无变化或与 N/D/来源完全混杂。 | **当前 p-only 数据 Gate 最终状态** |

**对 Q2 的影响：**本报告不是否定 Q2 总题或经典 B1 基线；它只表示现有 B1–B5 不能支持正式 p-only M1 的可识别性主张。若未来完成可验证的 17 域 p 与 Loss 行连接，并证明存在不被 N/D/族别吸收的独立变化，可重新开启 Gate。Q1.3 的 A 组 p-only 模型不能代替 B 组真实训练配方。

## 输入与复现

- 唯一复现命令：`python src/f_q2_p_feasibility_audit.py`
- Python：`{summary['runtime']['python']}`；仅使用 Python 标准库。
- 本报告对应的输入及 SHA-256：

| 输入 | 字节数 | SHA-256 |
|---|---:|---|
{chr(10).join(input_lines)}

详见同目录 `q2_p_trajectory_mapping.csv`、`q2_p_source_evidence_catalog.csv`、`q2_p_audit_summary.json` 与 `q2_p_audit_manifest.json`。映射 CSV 对 B4/B5 使用“源文件行号+记录哈希”作为审计定位符，**不是跨文件拼接键**。

## 来源目录

| ID | 来源 | URL/路径 | 限制 |
|---|---|---|---|
{chr(10).join(evidence_lines)}
"""


def write_outputs(audit: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / "q2_p_trajectory_mapping.csv"
    evidence_path = out_dir / "q2_p_source_evidence_catalog.csv"
    summary_path = out_dir / "q2_p_audit_summary.json"
    report_path = out_dir / "q2_p_feasibility_audit.md"
    manifest_path = out_dir / "q2_p_audit_manifest.json"

    write_csv(mapping_path, audit["mapping_columns"], audit["mapping_rows"])
    write_csv(evidence_path, audit["evidence_columns"], audit["evidence_rows"])
    summary_path.write_text(
        json.dumps(audit["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        report_markdown(audit["summary"], audit["mapping_rows"], audit["evidence_rows"]),
        encoding="utf-8",
    )
    outputs = [mapping_path, evidence_path, summary_path, report_path]
    manifest = {
        "audit_version": "v1",
        "audit_date": audit["summary"]["audit_date"],
        "command": "python src/f_q2_p_feasibility_audit.py",
        "python_version": sys.version.split()[0],
        "script_path": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_files": audit["summary"]["input_files"],
        "output_files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
        "model_fitted": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [*outputs, manifest_path]


def mirror_deliverables(output_paths: list[Path]) -> list[Path]:
    Q2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mirrored = []
    for source in output_paths:
        destination = Q2_OUT_DIR / source.name
        shutil.copy2(source, destination)
        mirrored.append(destination)
    q2_script = ROOT / "Q2" / "02_代码" / "f_q2_p_feasibility_audit.py"
    q2_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), q2_script)
    mirrored.append(q2_script)
    docs_report = ROOT / "docs" / "Q2_p配比可行性审计.md"
    docs_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_DIR / "q2_p_feasibility_audit.md", docs_report)
    mirrored.append(docs_report)
    return mirrored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-mirror-q2", action="store_true", help="do not copy audit deliverables into Q2/")
    args = parser.parse_args()

    audit = build_audit()
    outputs = write_outputs(audit, args.output_dir)
    mirrored = [] if args.no_mirror_q2 else mirror_deliverables(outputs)
    print(json.dumps({
        "status": audit["summary"]["overall_p_only_data_gate"],
        "model_fitted": False,
        "B1_tracks": audit["summary"]["B1"]["model_trajectories"],
        "B1_B5_loss_rows_with_p": audit["summary"]["loss_rows_with_p_joined"],
        "outputs": [path.relative_to(ROOT).as_posix() for path in outputs],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
