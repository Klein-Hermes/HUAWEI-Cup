#!/usr/bin/env python3
"""Analyze the frozen Q1.1 A1 double-blind review sample.

This script consumes the merged output of validate_q1_1_reviews.ps1. It does
not refit indicator weights or alter any Q1.1 score. It implements the
pre-registered quadratic weighted Cohen kappa, design-weighted Spearman
correlations, and paired stratified bootstrap. The final candidate remains
pending until the separate sensitivity-reversal gate is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "results" / "q1_1" / "review_intake" / "v1"
V1 = ROOT / "results" / "q1_1" / "v1"
RATINGS_DEFAULT = INTAKE / "validated_ratings.csv"
SCORES_DEFAULT = V1 / "sample_scores.csv.gz"
MAPPING_DEFAULT = V1 / "review_packet" / "restricted_id_mapping.csv"
DOMAIN_SUMMARY_DEFAULT = V1 / "domain_summary.csv"
OUTPUT_DEFAULT = ROOT / "results" / "q1_1" / "human_validation" / "v1"

CRITERIA = ("education", "readability", "coherence", "information", "overall")
CANDIDATES = ("q_equal", "q_huber")
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260924


def read_rows(path: Path, *, compressed: bool = False) -> List[dict]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rating(value: str, where: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: rating is not an integer: {value!r}") from exc
    if str(number) != str(value).strip() or not 1 <= number <= 5:
        raise ValueError(f"{where}: rating must be an integer from 1 to 5: {value!r}")
    return number


def quadratic_weighted_kappa(left: np.ndarray, right: np.ndarray) -> float:
    """Quadratic weighted Cohen kappa, with categories 1 through 5."""
    observed = np.zeros((5, 5), dtype=float)
    for a, b in zip(left, right):
        observed[int(a) - 1, int(b) - 1] += 1.0
    total = float(observed.sum())
    if total <= 0:
        return math.nan
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    categories = np.arange(1, 6, dtype=float)
    disagreement = ((categories[:, None] - categories[None, :]) / 4.0) ** 2
    expected_disagreement = float(np.sum(disagreement * expected))
    if expected_disagreement <= 0:
        return math.nan
    observed_disagreement = float(np.sum(disagreement * observed))
    return 1.0 - observed_disagreement / expected_disagreement


def weighted_midrank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted midranks using half the total design weight of tied values."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=float)
    cumulative = 0.0
    start = 0
    while start < ordered.size:
        end = start + 1
        while end < ordered.size and ordered[end] == ordered[start]:
            end += 1
        group_weight = float(np.sum(weights[order[start:end]]))
        ranks[order[start:end]] = cumulative + 0.5 * group_weight
        cumulative += group_weight
        start = end
    return ranks


def weighted_spearman(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    if x.size < 2 or y.size != x.size or weights.size != x.size:
        return math.nan
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(weights).all()):
        return math.nan
    if np.any(weights <= 0):
        return math.nan
    rank_x = weighted_midrank(x, weights)
    rank_y = weighted_midrank(y, weights)
    total_weight = float(np.sum(weights))
    mean_x = float(np.sum(weights * rank_x) / total_weight)
    mean_y = float(np.sum(weights * rank_y) / total_weight)
    centered_x = rank_x - mean_x
    centered_y = rank_y - mean_y
    sum_x = float(np.sum(weights * centered_x**2))
    sum_y = float(np.sum(weights * centered_y**2))
    denominator = math.sqrt(sum_x * sum_y)
    if denominator <= 0:
        return math.nan
    return float(np.sum(weights * centered_x * centered_y) / denominator)


def percentile_interval(values: Sequence[float]) -> Tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return math.nan, math.nan
    low, high = np.quantile(finite, [0.025, 0.975], method="linear")
    return float(low), float(high)


def load_analysis_rows(ratings_path: Path, scores_path: Path, mapping_path: Path,
                       domain_summary_path: Path) -> Tuple[List[dict], dict]:
    ratings = read_rows(ratings_path)
    mapping = read_rows(mapping_path)
    if not ratings or not mapping:
        raise ValueError("Ratings or restricted mapping is empty")

    rating_ids = [row.get("blind_id", "") for row in ratings]
    mapping_ids = [row.get("blind_id", "") for row in mapping]
    if len(set(rating_ids)) != len(rating_ids) or len(set(mapping_ids)) != len(mapping_ids):
        raise ValueError("Duplicate blind_id found in ratings or restricted mapping")
    if set(rating_ids) != set(mapping_ids):
        raise ValueError("Ratings and restricted mapping do not contain the same blind_id set")

    for row in ratings:
        for rater in ("rater_1", "rater_2"):
            for criterion in CRITERIA:
                column = f"{rater}_{criterion}"
                parse_rating(row.get(column, ""), f"{row['blind_id']}/{column}")

    wanted = {(row["source_domain"], row["raw_id"]) for row in mapping}
    score_by_key: Dict[Tuple[str, str], dict] = {}
    for row in read_rows(scores_path, compressed=True):
        if row.get("dataset", "").lower() != "a1":
            continue
        key = (row.get("source_domain", ""), row.get("id", ""))
        if key not in wanted:
            continue
        if key in score_by_key:
            raise ValueError(f"Duplicate Q1.1 sample score row: {key}")
        score_by_key[key] = row
    if set(score_by_key) != wanted:
        missing = len(wanted - set(score_by_key))
        raise ValueError(f"Could not join {missing} blind-review IDs to Q1.1 scores")

    mapping_by_id = {row["blind_id"]: row for row in mapping}
    ratings_by_id = {row["blind_id"]: row for row in ratings}
    joined: List[dict] = []
    strata_counts: Dict[Tuple[str, str, str], int] = {}
    strata_expected: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    for blind_id in rating_ids:
        m = mapping_by_id[blind_id]
        r = ratings_by_id[blind_id]
        score = score_by_key[(m["source_domain"], m["raw_id"])]
        stratum = (m["source_domain"], m["q_equal_tertile"], m["broad_unicode_flag"])
        n_h, N_h = int(m["n_h"]), int(m["N_h"])
        if n_h <= 0 or N_h < n_h:
            raise ValueError(f"Invalid sampling stratum counts for {blind_id}: N_h={N_h}, n_h={n_h}")
        expected_counts = (N_h, n_h)
        if stratum in strata_expected and strata_expected[stratum] != expected_counts:
            raise ValueError(f"Inconsistent N_h/n_h within stratum {stratum}")
        strata_expected[stratum] = expected_counts
        strata_counts[stratum] = strata_counts.get(stratum, 0) + 1
        joined_row = {
            "blind_id": blind_id,
            "source_domain": m["source_domain"],
            "stratum": stratum,
            "design_weight": float(m["design_weight"]),
            "q_equal": float(score["q_equal"]),
            "q_huber": float(score["q_huber"]),
        }
        for criterion in CRITERIA:
            joined_row[criterion] = (
                parse_rating(r[f"rater_1_{criterion}"], f"{blind_id}/rater_1_{criterion}")
                + parse_rating(r[f"rater_2_{criterion}"], f"{blind_id}/rater_2_{criterion}")
            ) / 2.0
        joined.append(joined_row)

    for stratum, actual_n in strata_counts.items():
        expected_n = strata_expected[stratum][1]
        if actual_n != expected_n:
            raise ValueError(f"Sample count differs from frozen n_h for {stratum}: {actual_n} != {expected_n}")

    expected_domains = {
        row["source_domain"] for row in read_rows(domain_summary_path)
        if row.get("dataset", "").lower() == "a1"
    }
    observed_domains = {row["source_domain"] for row in joined}
    if observed_domains != expected_domains or len(expected_domains) != 7:
        raise ValueError("Blind review does not cover exactly the frozen seven A1 source domains")

    context = {
        "n": len(joined),
        "domains": sorted(expected_domains),
        "strata": len(strata_counts),
        "strata_counts": {"|".join(key): value for key, value in sorted(strata_counts.items())},
    }
    return joined, context


def point_correlations(rows: Sequence[dict]) -> Dict[Tuple[str, str, str], float]:
    domains = sorted({row["source_domain"] for row in rows})
    results: Dict[Tuple[str, str, str], float] = {}
    for criterion in CRITERIA:
        for candidate in CANDIDATES:
            domain_values = []
            for domain in domains:
                subset = [row for row in rows if row["source_domain"] == domain]
                rho = weighted_spearman(
                    np.asarray([row[candidate] for row in subset], dtype=float),
                    np.asarray([row[criterion] for row in subset], dtype=float),
                    np.asarray([row["design_weight"] for row in subset], dtype=float),
                )
                results[(criterion, candidate, domain)] = rho
                domain_values.append(rho)
            results[(criterion, candidate, "A1_macro")] = (
                float(np.mean(domain_values)) if all(np.isfinite(domain_values)) else math.nan
            )
    return results


def bootstrap_correlations(rows: Sequence[dict], repetitions: int, seed: int
                           ) -> Dict[Tuple[str, str, str], List[float]]:
    rng = np.random.default_rng(seed)
    domains = sorted({row["source_domain"] for row in rows})
    domain_indices: Dict[str, np.ndarray] = {
        domain: np.asarray([i for i, row in enumerate(rows) if row["source_domain"] == domain], dtype=int)
        for domain in domains
    }
    stratum_indices: Dict[Tuple[str, str, str], np.ndarray] = {}
    for i, row in enumerate(rows):
        stratum_indices.setdefault(row["stratum"], []).append(i)
    stratum_indices = {key: np.asarray(value, dtype=int) for key, value in stratum_indices.items()}
    results: Dict[Tuple[str, str, str], List[float]] = {}
    output_domains = domains + ["A1_macro"]
    for criterion in CRITERIA:
        for candidate in CANDIDATES:
            for domain in output_domains:
                results[(criterion, candidate, domain)] = []
    for criterion in CRITERIA:
        results[(criterion, "delta_huber_minus_equal", "A1_macro")] = []
        for domain in domains:
            results[(criterion, "delta_huber_minus_equal", domain)] = []

    for _ in range(repetitions):
        sampled_by_domain: Dict[str, List[np.ndarray]] = {domain: [] for domain in domains}
        for stratum, indices in stratum_indices.items():
            draw = rng.choice(indices, size=indices.size, replace=True)
            sampled_by_domain[stratum[0]].append(draw)
        draw_rhos: Dict[Tuple[str, str], Dict[str, float]] = {}
        for criterion in CRITERIA:
            for domain in domains:
                indices = np.concatenate(sampled_by_domain[domain])
                weights = np.asarray([rows[i]["design_weight"] for i in indices], dtype=float)
                human = np.asarray([rows[i][criterion] for i in indices], dtype=float)
                for candidate in CANDIDATES:
                    model = np.asarray([rows[i][candidate] for i in indices], dtype=float)
                    rho = weighted_spearman(model, human, weights)
                    results[(criterion, candidate, domain)].append(rho)
                    draw_rhos.setdefault((criterion, domain), {})[candidate] = rho
            for candidate in CANDIDATES:
                domain_draws = [draw_rhos[(criterion, domain)][candidate] for domain in domains]
                macro = float(np.mean(domain_draws)) if all(np.isfinite(domain_draws)) else math.nan
                results[(criterion, candidate, "A1_macro")].append(macro)
            for domain in domains:
                pair = draw_rhos[(criterion, domain)]
                delta = pair["q_huber"] - pair["q_equal"] if all(
                    np.isfinite(pair[candidate]) for candidate in CANDIDATES
                ) else math.nan
                results[(criterion, "delta_huber_minus_equal", domain)].append(delta)
            macro_equal = results[(criterion, "q_equal", "A1_macro")][-1]
            macro_huber = results[(criterion, "q_huber", "A1_macro")][-1]
            macro_delta = macro_huber - macro_equal if np.isfinite(macro_equal) and np.isfinite(macro_huber) else math.nan
            results[(criterion, "delta_huber_minus_equal", "A1_macro")].append(macro_delta)
    return results


def make_correlation_rows(point: Mapping[Tuple[str, str, str], float],
                          draws: Mapping[Tuple[str, str, str], Sequence[float]],
                          rows: Sequence[dict], repetitions: int) -> List[dict]:
    output: List[dict] = []
    for (criterion, candidate, domain), estimate in point.items():
        values = draws[(criterion, candidate, domain)]
        low, high = percentile_interval(values)
        valid = sum(np.isfinite(values))
        n = sum(row["source_domain"] == domain for row in rows) if domain != "A1_macro" else len(rows)
        output.append({
            "criterion": criterion,
            "candidate": candidate,
            "domain": domain,
            "n": n,
            "design_weighted_spearman": estimate,
            "ci_low": low,
            "ci_high": high,
            "ci_half_width": (high - low) / 2 if np.isfinite(low) and np.isfinite(high) else math.nan,
            "bootstrap_repetitions": repetitions,
            "valid_repetitions": valid,
            "valid_repetition_rate": valid / repetitions,
        })
    return output


def make_difference_rows(point: Mapping[Tuple[str, str, str], float],
                         draws: Mapping[Tuple[str, str, str], Sequence[float]],
                         repetitions: int, domains: Sequence[str]) -> List[dict]:
    output: List[dict] = []
    for criterion in CRITERIA:
        for domain in list(domains) + ["A1_macro"]:
            values = draws[(criterion, "delta_huber_minus_equal", domain)]
            low, high = percentile_interval(values)
            valid = sum(np.isfinite(values))
            equal_point = point[(criterion, "q_equal", domain)]
            huber_point = point[(criterion, "q_huber", domain)]
            estimate = huber_point - equal_point if np.isfinite(equal_point) and np.isfinite(huber_point) else math.nan
            output.append({
                "criterion": criterion,
                "domain": domain,
                "difference_huber_minus_equal": estimate,
                "ci_low": low,
                "ci_high": high,
                "ci_half_width": (high - low) / 2 if np.isfinite(low) and np.isfinite(high) else math.nan,
                "bootstrap_repetitions": repetitions,
                "valid_repetitions": valid,
                "valid_repetition_rate": valid / repetitions,
            })
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=RATINGS_DEFAULT,
                        help="Merged, validated rating CSV from validate_q1_1_reviews.ps1")
    parser.add_argument("--scores", type=Path, default=SCORES_DEFAULT)
    parser.add_argument("--mapping", type=Path, default=MAPPING_DEFAULT,
                        help="Restricted mapping; keep this file internal")
    parser.add_argument("--domain-summary", type=Path, default=DOMAIN_SUMMARY_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--force", action="store_true", help="Allow replacing prior analysis outputs")
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    output_paths = [args.output_dir / name for name in
                    ("reviewer_agreement.csv", "review_correlations.csv",
                     "review_correlation_differences.csv", "decision_status.json", "manifest.json")]
    if not args.force and any(path.exists() for path in output_paths):
        raise FileExistsError("Analysis output already exists; choose another --output-dir or pass --force")

    joined, context = load_analysis_rows(args.ratings, args.scores, args.mapping, args.domain_summary)
    reliability_rows = []
    rating_rows = read_rows(args.ratings)
    for criterion in CRITERIA:
        left = np.asarray([parse_rating(row[f"rater_1_{criterion}"], criterion) for row in rating_rows], dtype=int)
        right = np.asarray([parse_rating(row[f"rater_2_{criterion}"], criterion) for row in rating_rows], dtype=int)
        reliability_rows.append({
            "criterion": criterion,
            "quadratic_weighted_cohen_kappa": quadratic_weighted_kappa(left, right),
            "n_double_rated": int(left.size),
            "kappa_weights": "1 - (a-b)^2/16",
        })

    point = point_correlations(joined)
    draws = bootstrap_correlations(joined, args.bootstrap, args.seed)
    correlation_rows = make_correlation_rows(point, draws, joined, args.bootstrap)
    difference_rows = make_difference_rows(point, draws, args.bootstrap, context["domains"])

    primary = {(row["candidate"], row["domain"]): row for row in correlation_rows
               if row["criterion"] == "overall" and row["domain"] == "A1_macro"}
    difference = next(row for row in difference_rows
                      if row["criterion"] == "overall" and row["domain"] == "A1_macro")
    precision_ok = all(
        row["valid_repetition_rate"] >= 0.95 and row["ci_half_width"] <= 0.15
        for row in [primary[("q_equal", "A1_macro")], primary[("q_huber", "A1_macro")], difference]
    )
    equal_lower = primary[("q_equal", "A1_macro")]["ci_low"]
    huber_lower = primary[("q_huber", "A1_macro")]["ci_low"]
    delta_lower = difference["ci_low"]
    if not precision_ok:
        preliminary_status = "precision_insufficient_no_candidate_passes"
    elif huber_lower > 0 and delta_lower > 0:
        preliminary_status = "huber_baseline_conditions_met_sensitivity_gate_pending"
    elif equal_lower > 0:
        preliminary_status = "equal_baseline_conditions_met_sensitivity_gate_pending"
    else:
        preliminary_status = "neither_candidate_validated_on_baseline"

    decision = {
        "primary_outcome": "mean of two raters' overall-quality ratings",
        "baseline_status": preliminary_status,
        "candidate_selected": None,
        "sensitivity_reversal_gate": "pending",
        "selection_note": (
            "This script computes the frozen A1 baseline agreement/correlation analysis. "
            "It does not claim a final Q until the pre-registered sensitivity-reversal gate "
            "for leave-one-indicator, weight/Huber, and contamination/unit scenarios is reviewed."
        ),
        "precision_gate": {
            "max_ci_half_width": 0.15,
            "min_valid_bootstrap_rate": 0.95,
            "passed_for_baseline": bool(precision_ok),
        },
        "bootstrap_repetitions": args.bootstrap,
        "seed": args.seed,
        "sample": context,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "reviewer_agreement.csv", reliability_rows)
    write_csv(args.output_dir / "review_correlations.csv", correlation_rows)
    write_csv(args.output_dir / "review_correlation_differences.csv", difference_rows)
    (args.output_dir / "decision_status.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    input_paths = [args.ratings, args.scores, args.mapping, args.domain_summary, Path(__file__)]
    output_names = [path.name for path in output_paths[:-1]]
    manifest = {
        "artifact": "Q1.1 A1 blind-review analysis",
        "status": "computed_baseline_sensitivity_gate_pending",
        "bootstrap_repetitions": args.bootstrap,
        "seed": args.seed,
        "inputs_sha256": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name: sha256(path)
                           for path in input_paths},
        "outputs_sha256": {
            name: sha256(args.output_dir / name) for name in output_names
        },
        "sensitivity_reversal_gate": "not computed by this script",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Baseline review analysis saved to {args.output_dir}")
    print(f"Status: {preliminary_status}; final selection remains pending sensitivity gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
