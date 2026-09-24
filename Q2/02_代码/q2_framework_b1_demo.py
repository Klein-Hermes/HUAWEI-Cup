#!/usr/bin/env python3
"""Run the existing classical Q2 model through the shared CV/output contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from f_q2_scaling_baseline import fit_multistart
from q2_experiment_framework import load_q2_data, make_cv_splits, run_experiment, save_cv_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "中文题目" / "F题" / "real_attachments" / "B_scaling_laws" / "pythia_training_log_existing.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "q2_public_framework"
SEED = 20260924


def fit_m0(x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    fit_frame = pd.DataFrame({
        "N_params_B": x_train["N"].to_numpy(dtype=float),
        "D_tokens_B": x_train["D"].to_numpy(dtype=float),
        "val_loss": np.asarray(y_train, dtype=float),
    })
    params, _, _ = fit_multistart(fit_frame, starts=8)
    e, a, alpha, b, beta = params
    n = x_test["N"].to_numpy(dtype=float)
    d = x_test["D"].to_numpy(dtype=float)
    return e + a * np.power(n, -alpha) + b * np.power(d, -beta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive")

    dataset = load_q2_data([{
        "path": DATA_PATH,
        "source": "B1",
        "columns": {"N": "N_params_B", "D": "D_tokens_B", "loss": "val_loss", "group": "N_params_B"},
        "q_status": "absent",
    }])
    split_manifest = make_cv_splits(dataset, n_splits=args.splits, seed=SEED, group_col="group")
    splits_path = OUTPUT_ROOT / "q2_cv_splits.json"
    save_cv_splits(split_manifest, splits_path)

    run_experiment(
        dataset,
        experiment="B1",
        model="M0",
        fit_predict=fit_m0,
        splits=split_manifest,
        output_dir=OUTPUT_ROOT / "B1",
        seed=SEED,
        bootstrap_resamples=args.bootstrap_resamples,
        config={
            "purpose": "P1 framework integration run; not a replacement for the frozen full-data B1 reference fit",
            "fit": "existing additive classical scaling law, re-fitted within each grouped CV training fold",
            "group_contract": "N_params_B identifies the eight complete trajectories; Gate 0 maps them to B12 model_repo clusters",
            "q_or_p_used": False,
        },
    )

    result = json.loads((OUTPUT_ROOT / "B1" / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps({"metrics": result["overall_oof"], "bootstrap": result["bootstrap"], "outputs": OUTPUT_ROOT.as_posix()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
