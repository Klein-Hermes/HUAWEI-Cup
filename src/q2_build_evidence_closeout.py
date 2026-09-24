#!/usr/bin/env python3
"""Build the Q2 A/B evidence matrix and current-data closeout artifacts.

The generator only reads already audited Q1/Q2 outputs. It does not search for
new sources, inspect hidden PDF text, refit any model, or alter original data.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_CANONICAL = ROOT / "results" / "q2_closeout" / "v1"
OUT_Q2 = ROOT / "Q2" / "03_结果" / "综合收口" / "v1"
OUT_DOCS = ROOT / "docs"
Q2_PACKAGE = ROOT / "Q2"

FILES = {
    "q1_freeze": ROOT / "results" / "q1_final" / "q1_final_freeze.json",
    "q1_freeze_decision": ROOT / "results" / "q1_final" / "q1_final_decision.md",
    "q1_p_only_metrics": ROOT / "results" / "q1_3" / "v1" / "holdout_metrics.csv",
    "q1_estimate_metrics": ROOT / "results" / "q1_3" / "v1" / "estimate_consistency_metrics.csv",
    "q1_pq_comparison": ROOT / "results" / "q1_3" / "pq_extension_v1" / "paired_comparison.csv",
    "q1_pq_identifiability": ROOT / "results" / "q1_3" / "pq_extension_v1" / "identifiability_audit.csv",
    "m0_parameters": ROOT / "results" / "q2_scaling_baseline" / "v1" / "fit_parameters.json",
    "m0_effect_summary": ROOT / "results" / "q2_scaling_baseline" / "v1" / "q2_m0_marginal_effects_by_size.csv",
    "m0_effect_report": ROOT / "results" / "q2_scaling_baseline" / "v1" / "q2_m0_marginal_effects_report.md",
    "validation_by_source": ROOT / "results" / "q2_scaling_baseline" / "v1" / "validation_metrics_by_source.csv",
    "validation_strata": ROOT / "results" / "q2_scaling_baseline" / "v1" / "validation_metrics_by_stratum.csv",
    "validation_comparability": ROOT / "results" / "q2_scaling_baseline" / "v1" / "validation_comparability.json",
    "b1_predictions": ROOT / "results" / "q2_scaling_baseline" / "v1" / "b1_fitted_predictions.csv",
    "b1_mapping": Q2_PACKAGE / "03_结果" / "经典ScalingLaw基线" / "v1" / "gate0_b1_trajectory_mapping.csv",
    "b2_report": Q2_PACKAGE / "03_结果" / "B2半合成压力测试" / "v1" / "b2_cerebras_stress_report.md",
    "b3_report": Q2_PACKAGE / "03_结果" / "经典ScalingLaw基线" / "v1" / "b3_validation_closeout.md",
    "b4b5_audit": Q2_PACKAGE / "03_结果" / "经典ScalingLaw基线" / "v1" / "b4_b5_loss_comparability_audit.md",
    "p_gate_json": Q2_PACKAGE / "03_结果" / "p配比可行性审计" / "v1" / "q2_p_audit_summary.json",
    "p_gate_report": Q2_PACKAGE / "03_结果" / "p配比可行性审计" / "v1" / "q2_p_feasibility_audit.md",
    "topic_analysis": Q2_PACKAGE / "01_方案说明" / "题目分析报告.md",
    "route_review": Q2_PACKAGE / "01_方案说明" / "Q2_主模型路线复审_2026-09-24.md",
}

ARTIFACT_NAMES = [
    "q2_evidence_matrix.csv",
    "q2_identifiability_gates.csv",
    "q2_model_status.csv",
    "q2_q_incremental_effect_gate.md",
    "q2_stage_conclusion.md",
]
MANIFEST_NAME = "q2_closeout_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 5) -> str:
    return f"{float(value):.{digits}g}"


def ensure_inputs() -> dict[str, str]:
    missing = [rel(path) for path in FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audited closeout inputs: " + ", ".join(missing))
    return {key: sha256(path) for key, path in FILES.items()}


def load_and_check() -> dict:
    hashes = ensure_inputs()
    q_freeze = read_json(FILES["q1_freeze"])
    p_gate = read_json(FILES["p_gate_json"])
    m0 = read_json(FILES["m0_parameters"])
    comparability = read_json(FILES["validation_comparability"])
    p_only = pd.read_csv(FILES["q1_p_only_metrics"])
    estimate = pd.read_csv(FILES["q1_estimate_metrics"])
    pq = pd.read_csv(FILES["q1_pq_comparison"])
    pq_ident = pd.read_csv(FILES["q1_pq_identifiability"])
    val = pd.read_csv(FILES["validation_by_source"])
    strata = pd.read_csv(FILES["validation_strata"])
    b1 = pd.read_csv(FILES["b1_predictions"])
    mapping = pd.read_csv(FILES["b1_mapping"])
    effects = pd.read_csv(FILES["m0_effect_summary"])

    if q_freeze.get("status") != "FROZEN_WITH_LIMITATIONS" or q_freeze.get("operational_q") != "q_huber" or q_freeze.get("sensitivity_q") != "q_equal":
        raise ValueError("Current Q1 final freeze does not match the expected q_huber/q_equal interface")
    if p_gate.get("overall_p_only_data_gate") != "FAIL":
        raise ValueError("Current B-side p gate is not FAIL; review before applying this closeout")
    if int(p_gate.get("verified_p_vectors_joined_to_loss", -1)) != 0 or int(p_gate.get("loss_rows_with_p_joined", -1)) != 0:
        raise ValueError("p gate counts changed; rebuild the current-data closeout from the new evidence")
    if len(b1) != 1176 or len(mapping) != 8 or mapping["model_repo"].nunique() != 8:
        raise ValueError("B1 trajectory counts differ from the audited 1,176-row/eight-track contract")
    if len(effects) != 8 or not (effects["n_checkpoints"] == 147).all():
        raise ValueError("M0 effect summary is missing one or more of the eight 147-checkpoint tracks")
    if set(val["dataset"].astype(str)) != {"B2", "B3", "B4", "B5"}:
        raise ValueError("Validation source table no longer contains exactly B2-B5")
    if val.set_index("dataset")["n"].to_dict() != {"B2": 1029, "B3": 4000, "B4": 57, "B5": 44}:
        raise ValueError("B2-B5 source counts changed; refresh the closeout evidence")
    if comparability.get("pooled_validation_metric") is not None:
        raise ValueError("Unexpected pooled B2-B5 metric; inspect the validation contract")
    if not ((pq_ident["qcovered_is_deterministic_function_of_p"] == True).all() and (pq_ident["independent_quality_effect_identified"] == False).all()):
        raise ValueError("Q1.3 p/Q identifiability evidence changed; review before closing the Q gate")

    p_only_main = p_only[(p_only["model"] == "simplex_ridge") & (p_only["variant"] == "closed")]
    p_only_macro = p_only_main.groupby("split", as_index=False).agg(
        role=("role", "first"),
        n=("n", "first"),
        n_targets=("target", "nunique"),
        macro_mae=("mae", "mean"),
        macro_rmse=("rmse", "mean"),
        macro_r2=("r2", "mean"),
        macro_spearman=("spearman_rho", "mean"),
    )
    estimate_main = estimate[
        (estimate["model"] == "simplex_ridge")
        & (estimate["variant"] == "closed")
        & (estimate["role"] == "estimate_consistency_only")
    ]
    estimate_macro = estimate_main.groupby("split", as_index=False).agg(
        role=("role", "first"),
        n=("n", "first"),
        n_targets=("target", "nunique"),
        macro_mae=("mae", "mean"),
        macro_rmse=("rmse", "mean"),
        macro_r2=("r2", "mean"),
        macro_spearman=("spearman_rho", "mean"),
    )
    if set(estimate_macro["split"]) != {"estimate_10b", "estimate_70b"} or not (estimate_macro["n_targets"] == 13).all():
        raise ValueError("Q1.3 estimate-consistency metrics changed; inspect the 10B/70B evidence before closeout")
    pq_main = pq[(pq["mapping_scenario"] == "direct_only") & (pq["q_scenario"] == "q_huber_candidate")]
    pq_oof = pq_main[pq_main["split"] == "train_1m_nested_cv"].set_index("model")
    if not {"p_only_quadratic", "p_plus_Qcovered"}.issubset(pq_oof.index):
        raise ValueError("Missing the primary A-side Q1.3 p-only versus p+Q OOF comparison")
    return {
        "hashes": hashes,
        "q_freeze": q_freeze,
        "p_gate": p_gate,
        "m0": m0,
        "comparability": comparability,
        "p_only_macro": p_only_macro,
        "estimate_macro": estimate_macro,
        "pq_main": pq_main,
        "pq_oof": pq_oof,
        "pq_ident": pq_ident,
        "validation": val.set_index("dataset"),
        "strata": strata,
        "b1": b1,
        "mapping": mapping,
        "effects": effects,
    }


def evidence_rows(data: dict) -> pd.DataFrame:
    p_only = data["p_only_macro"].set_index("split")
    estimates = data["estimate_macro"].set_index("split")
    p1m = p_only.loc["test_1m"]
    p60 = p_only.loc["test_60m"]
    p1b = p_only.loc["test_1b"]
    est10 = estimates.loc["estimate_10b"]
    est70 = estimates.loc["estimate_70b"]
    pq_oof = data["pq_oof"]
    p2 = pq_oof.loc["p_only_quadratic"]
    pq = pq_oof.loc["p_plus_Qcovered"]
    qid = data["pq_ident"]
    val = data["validation"]
    m0 = data["m0"]
    p_gate = data["p_gate"]
    q_freeze = data["q_freeze"]

    rows = [
        {
            "evidence_id": "Q1_FINAL_Q",
            "module": "Q interface",
            "dataset_or_artifact": "Q1-q_huber-v1",
            "role": "Frozen operational quality input",
            "sample_or_units": f"{q_freeze.get('sample_rows')} sample rows; {q_freeze.get('domain_rows')} domain rows",
            "status": "FROZEN_WITH_LIMITATIONS",
            "evidence": "Operational Q is q_huber; q_equal is the sensitivity score. The small translation-assisted review did not distinguish them; selection was an owner operational choice.",
            "supports": "Use a versioned, consistent Q1 interface in later analysis.",
            "does_not_support": "Claiming q_huber is a human-review winner or that Q has an independent B-side Loss effect.",
            "source_paths": "results/q1_final/q1_final_freeze.json; results/q1_final/q1_final_decision.md",
        },
        {
            "evidence_id": "A_P_ONLY",
            "module": "A-side p-only",
            "dataset_or_artifact": "A4-A5 with held-out A6-A11",
            "role": "Source-specific recipe-to-Loss experiment",
            "sample_or_units": f"1M test n={int(p1m['n'])}; 60M test n={int(p60['n'])}; 1B test n={int(p1b['n'])}; 13 targets each",
            "status": "COMPLETE_WITH_LIMITS",
            "evidence": f"Simplex ridge closed-mixture macro R2: 1M={float(p1m['macro_r2']):.6f}, 60M={float(p60['macro_r2']):.6f}, 1B={float(p1b['macro_r2']):.6f}.",
            "supports": "A-side same-scale p/Loss association under the specific A4/A5 design; cross-scale failure is evidence against transferring the fit unchanged.",
            "does_not_support": "B1/B-side p effects, a universal cross-scale recipe effect, or causal claims.",
            "source_paths": "results/q1_3/v1/holdout_metrics.csv; results/q1_3/v1/q1_3_run_report.md",
        },
        {
            "evidence_id": "A_PQ",
            "module": "A-side p+Q sensitivity",
            "dataset_or_artifact": "A4-A5 nested OOF; direct-only q_huber candidate",
            "role": "Derived-feature predictive comparison and identifiability audit",
            "sample_or_units": f"OOF recipes={int(p2['n_recipes'])}; common support only",
            "status": "COMPLETE_WITH_LIMITS",
            "evidence": f"Nested OOF normalized RMSE: quadratic p-only={float(p2['nested_macro_normalized_rmse']):.6f}; p+Qcovered={float(pq['nested_macro_normalized_rmse']):.6f}; delta={float(pq['nested_macro_normalized_rmse']-p2['nested_macro_normalized_rmse']):+.6f}. Qcovered is deterministic from p in all audited scenarios.",
            "supports": "The tested Qcovered feature did not improve A4/A5 nested OOF prediction over quadratic p-only under the primary direct-only mapping.",
            "does_not_support": "Independent Q effect; transfer to B Loss; interpreting predictive differences as significance or causality.",
            "source_paths": "results/q1_3/pq_extension_v1/paired_comparison.csv; results/q1_3/pq_extension_v1/identifiability_audit.csv",
        },
        {
            "evidence_id": "A_ESTIMATES",
            "module": "A12-A15 scale estimates",
            "dataset_or_artifact": "10B/70B estimate tables",
            "role": "Estimate-table consistency only",
            "sample_or_units": f"10B: n={int(est10['n'])} per target; 70B: n={int(est70['n'])} per target; {int(est10['n_targets'])} targets",
            "status": "NOT_VALIDATION",
            "evidence": f"Macro RMSE: 10B={float(est10['macro_rmse']):.4f}, 70B={float(est70['macro_rmse']):.4f}; macro R2: 10B={float(est10['macro_r2']):.3f}, 70B={float(est70['macro_r2']):.3f}. These rows are estimate-table consistency checks only.",
            "supports": "Describing agreement or disagreement with those supplied estimates.",
            "does_not_support": "Held-out test accuracy or external validation.",
            "source_paths": "results/q1_3/v1/estimate_consistency_metrics.csv; results/q1_3/v1/q1_3_run_report.md",
        },
        {
            "evidence_id": "B1_M0",
            "module": "B1 classical scaling baseline",
            "dataset_or_artifact": "B1 Pythia training trajectories",
            "role": "Primary M0 fit",
            "sample_or_units": "1,176 checkpoints; 8 model_repo tracks; 147 checkpoints per track; 1,000 saved cluster-bootstrap fits",
            "status": "COMPLETE",
            "evidence": f"Frozen M0 fit R2={float(m0['metrics']['r2']):.9f}; N/D local derivatives and elasticities now have checkpoint and track summaries.",
            "supports": "The classic N,D reference law and local model-implied N,D effects over B1 support.",
            "does_not_support": "Independent p/Q effects, causal claims, or external validity from repeated checkpoints alone.",
            "source_paths": "results/q2_scaling_baseline/v1/fit_parameters.json; results/q2_scaling_baseline/v1/q2_m0_marginal_effects_report.md",
        },
    ]
    role_text = {
        "B2": "Semi-synthetic pressure test; not an independent real training validation.",
        "B3": "Interpolation from eight Pythia trajectories; not 4,000 independent experiments or external validation.",
        "B4": "Cross-family snapshots; Loss protocol and row provenance insufficient for absolute comparison to B1.",
        "B5": "Multi-paper literature points; common evaluation protocol and row-level provenance are not established.",
    }
    for dataset in ["B2", "B3", "B4", "B5"]:
        row = val.loc[dataset]
        status = "LIMITED_DIAGNOSTIC" if dataset in {"B2", "B3"} else "NOT_COMPARABLE_TO_B1"
        rows.append({
            "evidence_id": dataset,
            "module": "B-side validation",
            "dataset_or_artifact": dataset,
            "role": role_text[dataset],
            "sample_or_units": f"n={int(row['n'])}",
            "status": status,
            "evidence": f"RMSE={float(row['rmse']):.6g}; MAE={float(row['mae']):.6g}; bias={float(row['bias_pred_minus_obs']):.6g}; R2={float(row['r2']):.6g}; Spearman={float(row['spearman']):.6g}.",
            "supports": {
                "B2": "Stress-test behavior for the constructed semi-synthetic scenario.",
                "B3": "Same-source interpolation shape check only.",
                "B4": "Descriptive error diagnostics; no absolute Loss-scale validation.",
                "B5": "Descriptive source-separated diagnostics; no pooled or absolute validation.",
            }[dataset],
            "does_not_support": {
                "B2": "Independent real Cerebras validation.",
                "B3": "Cross-source generalization or independent training-run validation.",
                "B4": "Direct absolute Loss comparison to B1 or a verified source-internal trend at present.",
                "B5": "Direct absolute Loss comparison to B1; source-internal trends remain candidates until row citations are verified.",
            }[dataset],
            "source_paths": {
                "B2": "results/q2_scaling_baseline/v1/validation_metrics_by_source.csv; Q2/03_结果/B2半合成压力测试/v1/b2_cerebras_stress_report.md",
                "B3": "results/q2_scaling_baseline/v1/validation_metrics_by_source.csv; Q2/03_结果/经典ScalingLaw基线/v1/b3_validation_closeout.md",
                "B4": "results/q2_scaling_baseline/v1/validation_metrics_by_source.csv; Q2/03_结果/经典ScalingLaw基线/v1/b4_b5_loss_comparability_audit.md",
                "B5": "results/q2_scaling_baseline/v1/validation_metrics_by_source.csv; Q2/03_结果/经典ScalingLaw基线/v1/b4_b5_loss_comparability_audit.md",
            }[dataset],
        })
    rows.extend([
        {
            "evidence_id": "P_GATE",
            "module": "p identifiability gate",
            "dataset_or_artifact": "B1-B5 row/trajectory p linkage",
            "role": "M1 data-readiness gate",
            "sample_or_units": f"verified p vectors joined to Loss={int(p_gate['verified_p_vectors_joined_to_loss'])}; Loss rows joined={int(p_gate['loss_rows_with_p_joined'])}",
            "status": "FAIL_CURRENT_DATA",
            "evidence": "No verified numeric 17-domain p vector is joined to B1-B5 Loss; B1 has one shared data-order policy, not eight numeric mixture vectors; p(D) is not reconstructed.",
            "supports": "Concluding that current B data cannot identify an independent p increment.",
            "does_not_support": "Concluding that p has no effect.",
            "source_paths": "Q2/03_结果/p配比可行性审计/v1/q2_p_audit_summary.json; Q2/03_结果/p配比可行性审计/v1/q2_p_feasibility_audit.md",
        },
        {
            "evidence_id": "Q_GATE",
            "module": "Q incremental-effect gate",
            "dataset_or_artifact": "Q1-q_huber-v1 plus B1-B5",
            "role": "M2 independent-Q data-readiness gate",
            "sample_or_units": "Q1 interface frozen; no validated B-side independent Q variation joined with p, N, D and Loss",
            "status": "FAIL_CURRENT_EVIDENCE",
            "evidence": "Q1 provides an operational Q interface, but the B-side joint design is absent; Q1.3 Qcovered(p) is deterministic from p and its audit marks independent quality effect as not identified.",
            "supports": "Concluding that current evidence cannot identify Q beyond p for B-side Loss.",
            "does_not_support": "Concluding that Q has no effect or that Q1 failed.",
            "source_paths": "results/q1_final/q1_final_freeze.json; results/q1_3/pq_extension_v1/identifiability_audit.csv; Q2/03_结果/p配比可行性审计/v1/q2_p_audit_summary.json",
        },
    ])
    return pd.DataFrame(rows)


def gate_table(data: dict) -> pd.DataFrame:
    q_freeze = data["q_freeze"]
    p_gate = data["p_gate"]
    q_rows = data["pq_ident"]
    return pd.DataFrame([
        {
            "gate": "p-effect identifiability",
            "status": "FAIL_CURRENT_DATA",
            "criterion_summary": "Verified B-side 17-domain recipe or p(D), linked to Loss/checkpoints, with independent variation beyond N, D and family.",
            "observed_evidence": f"verified p vectors={p_gate['verified_p_vectors_joined_to_loss']}; Loss rows with p={p_gate['loss_rows_with_p_joined']}; numeric recipe vectors in B1={p_gate['B1']['formal_numeric_recipe_vectors']}.",
            "decision": "M1 NOT ESTIMATED; this is an identification limit, not evidence that p has no effect.",
            "reopen_condition": "New traceable B-side source/token counts and checkpoint mapping that produce multiple non-confounded numeric recipes or p(D) trajectories.",
        },
        {
            "gate": "Q incremental-effect identifiability",
            "status": "FAIL_CURRENT_EVIDENCE",
            "criterion_summary": "A versioned Q value must link to B runs/checkpoints and vary independently of p after accounting for N and D.",
            "observed_evidence": f"Q1 operational={q_freeze['operational_q']} ({q_freeze['q_version']}); A-side Qcovered deterministic from p in {int(q_rows['qcovered_is_deterministic_function_of_p'].sum())}/{len(q_rows)} audit scenarios; independent quality effect identified in {int(q_rows['independent_quality_effect_identified'].sum())}/{len(q_rows)} scenarios; B-side p-Loss joins=0.",
            "decision": "M2 NOT ESTIMATED; this is an identification limit, not evidence that Q has no effect and not a failure of Q1.",
            "reopen_condition": "B-side Q-to-run/checkpoint links plus an identifiable design with Q variation not determined by p and not confounded with N/D/family; first resolve the p gate needed for the M2 nested comparison.",
        },
    ])


def status_table() -> pd.DataFrame:
    return pd.DataFrame([
        ["Q1 formal Q interface", "FROZEN_WITH_LIMITATIONS", "q_huber operational; q_equal sensitivity; review did not select a statistical winner"],
        ["M0 classical Loss=f(N,D)", "COMPLETE", "B1 fit frozen; 1,176 checkpoints across 8 tracks"],
        ["M0 Bootstrap", "COMPLETE_WITH_LIMITS", "1,000 complete-track resamples; only 8 independent trajectory clusters"],
        ["M0 N/D marginal effects and elasticities", "COMPLETE", "Checkpoint output and 8-track summary generated"],
        ["B2/B3 validation checks", "COMPLETE_WITH_LIMITS", "B2 semi-synthetic; B3 same-source interpolation"],
        ["B4/B5 cross-source validation", "AUDIT_COMPLETE_VALIDATION_NOT_ESTABLISHED", "All source groups remain NOT_COMPARABLE to B1"],
        ["A-side p-only evidence", "COMPLETE_WITH_LIMITS", "Same-scale signal; cross-scale generalization fails"],
        ["A-side p+Q sensitivity", "COMPLETE_WITH_LIMITS", "Qcovered derived from p; independent Q effect not identified"],
        ["B-side p identifiability", "FAIL_CURRENT_DATA", "0 verified p vectors joined to Loss"],
        ["M1 (N,D,p)", "NOT_ESTIMATED", "Do not fit before p gate passes"],
        ["B-side Q incremental-effect identifiability", "FAIL_CURRENT_EVIDENCE", "No linked, independent B-side Q variation"],
        ["M2 (N,D,p,Q)", "NOT_ESTIMATED", "Do not fit before p and independent-Q gates pass"],
        ["p×Q interaction extension", "NOT_ESTIMATED", "Optional only if evidence supports it; not fit under current gates"],
        ["Q2 current-data package", "FROZEN_WITH_LIMITATIONS", "M0 and evidence audit closed; full p/Q increment remains unidentified with current B data"],
    ], columns=["module", "status", "basis_and_boundary"])


def build_q_gate_report(data: dict, gates: pd.DataFrame) -> str:
    gate = gates.iloc[1]
    q = data["q_freeze"]
    ident = data["pq_ident"]
    return rf"""# Q2：Q 独立增量效应识别门禁

## 判定

**当前证据状态：{gate['status']}；M2 不进入正式估计。** 这表示现有 B 侧数据不能区分 Q 在 p 之外的增量作用，不表示 Q 对 Loss 没有效果，也不表示 Q1 失败。

## 条件核对

| 条件 | 当前证据 | 判定 |
|---|---|---|
| Q1 接口有版本且冻结 | 操作主 Q 为 `{q['operational_q']}`（`{q['q_version']}`）；敏感性 Q 为 `{q['sensitivity_q']}`。冻结状态为 `{q['status']}`。 | 满足接口冻结；该条件本身不等于 B 侧可估计 |
| Q 可连接到 B 侧训练运行/检查点 | 当前 B1–B5 交付物没有经核验、逐行连接到 Loss 的独立 Q 观测；B 侧 `p`—Loss 联接数为 0。 | 不满足当前可估计要求 |
| Q 在控制 p、N、D 后有独立变化 | A 侧 `Q_covered(p)` 在 {len(ident)} 个映射/候选审计情景中均标记为 p 的确定性函数；独立质量效应均未识别。B 侧没有已连接 p 的设计可执行条件独立性检验。 | 未通过；不能分解独立效应 |
| 有足够独立变异支撑 M2 | 现有 B 侧联合设计没有可验证的 `(N,D,p,Q,Loss)` 行级支持。 | 未通过 |

## 结论与模型边界

题目分析报告规定在同一广义标度律内作嵌套比较：M0 令 `h=0`，M1 令 `h=g_p(p)`，M2 令 `h=g_p(p)+g_Q(Q)`。M2 的问题是“在 p 之外加入 Q 是否有增量”，因此必须先有 M1 所需的 p 数据链和可比较样本，再有独立于 p 的 Q 变化。`Q_covered(p)` 不能充当独立 Q 变化。

因此当前状态应记为：

- Q1 接口已按操作决定冻结，`q_huber` 为主分，`q_equal` 为敏感性分；小样本评审没有区分候选，也没有证明唯一胜者。
- B 侧 p Gate 为 FAIL，当前没有 p 向量与 Loss 行的可信连接；M1 不估计。
- 在 M1 数据基础及独立 Q 变化缺失时，M2 不估计；不报告 Q 边际效应、Q 弹性或数值质量—规模替代率。
- 以上不否定 Q 的潜在作用，只限定当前数据支持的结论范围。

## 重开门禁的最低证据

只有出现新的可追溯 B 侧记录，并同时满足以下条件，才重开：

1. `p_recipe` 与累计组成 `p(D)` 定义分开，能定位到真实训练轨迹/检查点和 Loss；
2. 有多个可比且不被 N、D、模型族完全吸收的独立配比；
3. 版本化 Q 能对应到相同运行/检查点，并在给定 p、N、D 后仍有可识别的独立变化；
4. Loss 与评测口径足以支持共同样本上的嵌套比较。

此前的一次性来源补查已经完成；在没有新数据、日志或正式来源材料时，不重复搜索或拟合。

## 证据文件

- p 门禁：`Q2/03_结果/p配比可行性审计/v1/q2_p_audit_summary.json`
- Q1 冻结：`results/q1_final/q1_final_freeze.json`
- A 侧 Q 可识别性：`results/q1_3/pq_extension_v1/identifiability_audit.csv`
- Q2 模型层级：`Q2/01_方案说明/题目分析报告.md`
"""


def build_stage_conclusion(data: dict, gates: pd.DataFrame, statuses: pd.DataFrame) -> str:
    p_only = data["p_only_macro"].set_index("split")
    oof = data["pq_oof"]
    p2 = oof.loc["p_only_quadratic"]
    pq = oof.loc["p_plus_Qcovered"]
    val = data["validation"]
    m0 = data["m0"]
    effects = data["effects"]
    gates_md = gates.rename(columns={
        "gate": "门禁",
        "status": "状态",
        "criterion_summary": "判定条件",
        "observed_evidence": "当前证据",
        "decision": "结论",
        "reopen_condition": "重开条件",
    }).to_markdown(index=False)
    status_md = statuses.rename(columns={
        "module": "模块",
        "status": "状态",
        "basis_and_boundary": "依据与边界",
    }).to_markdown(index=False)
    lo_epsn = float(effects["elasticity_N_median"].min())
    hi_epsn = float(effects["elasticity_N_median"].max())
    lo_epsd = float(effects["elasticity_D_median"].min())
    hi_epsd = float(effects["elasticity_D_median"].max())
    b2, b3, b4, b5 = (val.loc[k] for k in ["B2", "B3", "B4", "B5"])
    return rf"""# Q2 阶段性综合结论与当前数据收口

## 当前结论

截至本收口版本，Q2 的经典参照支路、M0 的 N/D 局部效应、B1–B5 证据审计及 p/Q 可识别性判断已经整理完成。现有数据**不能支持 B 侧 M1 或 M2 的正式系数估计**。因此当前数据包状态记为 `FROZEN_WITH_LIMITATIONS`：这表示基于现有证据的分析版本已收口，不表示题面要求的独立 p/Q 效应已经估计，也不把“不可识别”写成“没有作用”。

## 与题目模型层级的对应

题目分析报告的广义标度律为

\[L=E+A N^{{-\alpha}}+B D_{{\mathrm{{eff}}}}^{{-\beta}},\qquad D_{{\mathrm{{eff}}}}=D\exp\{{h(Q,\boldsymbol{{p}})\}}.\]

M0、M1、M2 是同一模型族内的嵌套限制：M0 令 `h=0`；M1 令 `h=g_p(p)`；M2 令 `h=g_p(p)+g_Q(Q)`。它们不是三套互不相关的模型。交互项只在数据支持时作为受限的额外扩展；当前没有进入该扩展的识别依据。

## 现有结果能回答什么

### 1. B1 经典 M0 与 N/D 局部效应

B1 有 1,176 个检查点，按 B12 映射为 8 条完整轨迹、每条 147 点；点数不等于独立轨迹数。冻结 M0 在 B1 上的拟合 `R²={float(m0['metrics']['r2']):.9f}`。按模型轨迹中位数汇总，N 弹性范围约为 [{lo_epsn:.5f}, {hi_epsn:.5f}]，D 弹性范围约为 [{lo_epsd:.5f}, {hi_epsd:.5f}]；完整逐检查点效应和 Bootstrap 区间见 `q2_m0_marginal_effects_by_checkpoint.csv` 与 `q2_m0_marginal_effects_by_size.csv`。

这些是条件于冻结 M0 的局部模型效应，单位与支持范围以 M0 报告为准。区间复用 1,000 次完整轨迹 cluster Bootstrap，但独立 cluster 只有 8 条，因此只作稳定性提示。

### 2. A 侧 p 与 p+Q 证据

Q1.3 的 A4/A5 p-only 模型在同尺度 1M 留出集上的 13 目标宏平均 `R²={float(p_only.loc['test_1m']['macro_r2']):.4f}`；跨尺度 60M 和 1B 分别为 `{float(p_only.loc['test_60m']['macro_r2']):.4f}` 与 `{float(p_only.loc['test_1b']['macro_r2']):.4f}`。因此可报告 A 来源内的配比—Loss 关系，但不能将其作为跨尺度通用规律或直接替代 B 侧证据。

在 A4/A5 的 direct-only 嵌套 OOF 中，二次 p-only 的标准化 RMSE 为 `{float(p2['nested_macro_normalized_rmse']):.6f}`，加入 `Q_covered(p)` 后为 `{float(pq['nested_macro_normalized_rmse']):.6f}`，变化为 `{float(pq['nested_macro_normalized_rmse']-p2['nested_macro_normalized_rmse']):+.6f}`。审计表明 `Q_covered(p)` 是 p 的确定性函数；该比较不识别独立 Q 效应。Q1 的 `q_huber` 冻结是操作性决定，有限人工核查没有证明其为唯一优胜候选。

### 3. B2–B5 验证边界

- B2：n={int(b2['n'])}，RMSE={float(b2['rmse']):.4f}，R²={float(b2['r2']):.4f}；它是半合成压力测试，不是真实 Cerebras 外部验证。
- B3：n={int(b3['n'])}，RMSE={float(b3['rmse']):.6f}，R²={float(b3['r2']):.6f}；4,000 点来自 8 条 Pythia 插值轨迹，只支持同来源插值检查。
- B4：n={int(b4['n'])}；B5：n={int(b5['n'])}。两者相对 B1 均为 `NOT_COMPARABLE`；其已有误差只能分源作描述，不构成同量纲外部验证。

## 按《题目分析报告》§4.2 的四项答复

以下四项是对题面要求的收口组织，不代表题目正式拆分为 Q2.1–Q2.4。验证边界纳入第 4 项及各来源结果说明。完整公式、来源内结果和逐项证据缺口见 `Q2/03_结果/完整回答补充/v1/q2_full_answer_evidence_supplement.md`。

### 1. 广义标度律：经典 N、D 基线及 p/Q 纳入条件

当前可报告的经验基线是 B1 上冻结的 M0：`L=E+A N^(-alpha)+B D^(-beta)`。它描述 B1 支持域内的 N/D 关系；B1 训练内拟合不等于独立验证。B 侧可核验的 p—Loss 行级连接为 0，且没有 p 之外独立变化的 Q 证据，故 M1/M2 保持 `NOT_ESTIMATED`；当前结果不能写成广义联合标度律已被估计或验证。

### 2. 边际效应与无量纲弹性

B1 的 M0 已给出逐检查点 `dL/dN`、`dL/dD` 与对应弹性，弹性以该点的 M0 预测 Loss 为分母。A4/A5 的 p 关联限于来源内；B6/B7 的 Q 响应是半合成条件结果。它们不构成同一批真实 B 侧运行上 p、Q 的联合边际效应或弹性。

### 3. 领域配比替代与互补

配比 `p` 位于单纯形，域间转移必须保持总和为 1。A4/A5 可报告 1M 同来源的配比关联和替代敏感性；线性主模型的零二阶项是模型限制，不是“没有互补”的证据。二次探索的候选区间经条件 max-|t| 校正后没有稳健的负向互补信号，少数正向区间也受同样本候选筛选限制。该证据不能迁移为 B 侧或跨尺度普遍规律。

### 4. 质量—规模等 Loss 替代条件与分来源验证

可以报告在明确模型假设下的解析等 Loss 条件；例如 `dN/dQ|L=-L_Q/L_N`，其数值依赖可识别的 Q 系数。B6/B7 给出的是半合成曲面上的代数示例，不是实测 Q 效应或现实资源替代率。验证上，B1 用于 M0 拟合；B2 只作半合成压力测试，B3 只作 Pythia 同来源插值检查，B4/B5 对 B1 均为 `NOT_COMPARABLE`。B6–B8 是质量响应补充，其中 B8 与 B6/B7 方向冲突；B9/B10 是超出 B1 支持域的大尺度情景外推，不是独立验证。真实经验 Q—N 替代率及联合 p/Q 模型的外部验证均未完成。

## p/Q 门禁与当前状态

{gates_md}

当前模块状态：

{status_md}

## 可写结论

> 现有证据支持以 B1 建立经典 N、D 参照关系，并报告其局部边际效应和弹性；A4/A5 提供来源内配比实验结果，但跨尺度预测未通过。B1–B5 当前没有可核验并连接到 Loss 的数值 p 向量，且 B4/B5 Loss 与 B1 的绝对可比性未建立，因此无法识别 B 侧 p 的独立增量效应。Q1 已冻结操作性质量接口，但当前 B 侧没有可识别的 Q—p—N—D—Loss 联合变化，且 `Q_covered(p)` 为 p 的派生特征，因此不能识别 Q 在 p 之外的独立增量效应。M1/M2 记为 `NOT_ESTIMATED`，不是“无效”或“拟合失败”。

本结论是当前数据条件下的收口，不是对 p 或 Q 无效的证明。只有新出现可追溯、行级匹配且有独立变化的 B 侧数据时，才重开门禁，并按 M0→M1→M2 的嵌套顺序估计。

## 数据完整性与复现边界

本收口复用已生成的结构化结果、哈希清单和可见审计报告；不重跑模型、不重复一次性来源搜索、不修改原始附件。题目分析报告所记录的低可见度/隐藏文本按不可信内容处理，不作为题意、数据、变量或方法证据。证据行及其来源见 `q2_evidence_matrix.csv`；复现输入与输出哈希见 `q2_closeout_manifest.json`。
"""


def write_artifacts() -> dict:
    data = load_and_check()
    evidence = evidence_rows(data)
    gates = gate_table(data)
    statuses = status_table()
    gate_md = build_q_gate_report(data, gates)
    conclusion_md = build_stage_conclusion(data, gates, statuses)
    csv_settings = {"index": False, "encoding": "utf-8-sig", "lineterminator": "\n"}
    artifacts = {
        "q2_evidence_matrix.csv": evidence.to_csv(**csv_settings).encode("utf-8-sig"),
        "q2_identifiability_gates.csv": gates.to_csv(**csv_settings).encode("utf-8-sig"),
        "q2_model_status.csv": statuses.to_csv(**csv_settings).encode("utf-8-sig"),
        "q2_q_incremental_effect_gate.md": gate_md.encode("utf-8"),
        "q2_stage_conclusion.md": conclusion_md.encode("utf-8"),
    }
    destinations = [OUT_CANONICAL, OUT_Q2, OUT_DOCS]
    output_hashes = {}
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            path = destination / name
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(content)
            temp.replace(path)
            output_hashes[rel(path)] = sha256(path)

    script_copy = Q2_PACKAGE / "02_代码" / Path(__file__).name
    script_copy.parent.mkdir(parents=True, exist_ok=True)
    script_copy.write_bytes(Path(__file__).read_bytes())
    output_hashes[rel(script_copy)] = sha256(script_copy)

    manifest = {
        "task": "Q2 A/B evidence matrix, independent-Q gate, and current-data closeout",
        "status": "FROZEN_WITH_LIMITATIONS",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "D:/Anaconda/python.exe src/q2_build_evidence_closeout.py",
        "script": rel(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "executable": sys.executable},
        "input_sha256": {rel(FILES[key]): digest for key, digest in data["hashes"].items()},
        "verification": {
            "B1_checkpoints": int(len(data["b1"])),
            "B1_independent_tracks": int(data["mapping"]["model_repo"].nunique()),
            "B1_checkpoints_per_track": sorted(data["mapping"]["B1_rows"].astype(int).unique().tolist()),
            "B2_B5_rows": {str(k): int(v) for k, v in data["validation"]["n"].items()},
            "verified_p_vectors_joined_to_loss": int(data["p_gate"]["verified_p_vectors_joined_to_loss"]),
            "qcovered_deterministic_scenarios": int(data["pq_ident"]["qcovered_is_deterministic_function_of_p"].sum()),
            "independent_Q_scenarios_identified": int(data["pq_ident"]["independent_quality_effect_identified"].sum()),
            "pooled_validation_metric": data["comparability"].get("pooled_validation_metric"),
        },
        "output_sha256_excluding_this_manifest": output_hashes,
        "manifest_note": "This manifest is mirrored to results/q2_closeout/v1, Q2/03_结果/综合收口/v1 and docs; self-hash is excluded to avoid recursion.",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for destination in destinations:
        (destination / MANIFEST_NAME).write_bytes(manifest_bytes)
    return {"artifacts": list(artifacts), "evidence_rows": len(evidence), "gate_rows": len(gates), "status_rows": len(statuses), "manifest": manifest}


def main() -> int:
    result = write_artifacts()
    print(f"Wrote {len(result['artifacts'])} closeout artifacts and the manifest to results/, Q2/, and docs/.")
    print(f"Evidence rows: {result['evidence_rows']}; gates: {result['gate_rows']}; status rows: {result['status_rows']}.")
    print("Gate decisions: p=FAIL_CURRENT_DATA; Q=FAIL_CURRENT_EVIDENCE; M1/M2=NOT_ESTIMATED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
