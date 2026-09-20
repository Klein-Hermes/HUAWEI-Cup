"""Question 1: classify excitation waveforms from shape features.

Pilot: D:\Anaconda\envs\huawei-cup\python.exe src\solve_q1_waveform.py --pilot
Full:  D:\Anaconda\envs\huawei-cup\python.exe src\solve_q1_waveform.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "results" / "2024_C题_eda"
OUT = ROOT / "results" / "2024_C题_q1"
SEED = 42
LABELS = [1, 2, 3]
FEATURES = [
    "b_crest_factor", "slope_rms_rel", "slope_p90_rel",
    "plateau_fraction_0p001", "curvature_rms_rel",
    "harmonic_2_to_1", "harmonic_3_to_1", "harmonic_4_to_1",
    "harmonic_5_to_1", "harmonic_2_10_rss_to_1",
]
SPECIAL_IDS = [1, 5, 15, 25, 35, 45, 55, 65, 75, 80]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(EDA / "features_train.csv")
    test = pd.read_csv(EDA / "features_test_waveform.csv")
    if len(train) != 12399 or len(test) != 80:
        raise ValueError("Unexpected row count")
    if train[FEATURES].isna().any().any() or test[FEATURES].isna().any().any():
        raise ValueError("Missing classification features")
    if not np.isfinite(train[FEATURES].to_numpy()).all() or not np.isfinite(test[FEATURES].to_numpy()).all():
        raise ValueError("Nonfinite classification features")
    if set(train.waveform_code.unique()) != set(LABELS):
        raise ValueError("Unexpected class labels")
    if test.sample_id.tolist() != list(range(1, 81)):
        raise ValueError("Test IDs are not in the expected order")
    processed = ROOT / "data" / "processed" / "2024_C题"
    for features, filename in [(train, "train.csv"), (test, "test_waveform.csv")]:
        source = pd.read_csv(processed / filename, usecols=["source_file", "source_row", "sample_id"])
        if not features[["source_file", "source_row", "sample_id"]].equals(source):
            raise ValueError(f"Feature row identity does not match {filename}")
    return train, test


def pilot(train: pd.DataFrame) -> None:
    # A real, bounded vertical slice before full cross-validation.
    selected = (
        train.groupby("waveform_code", group_keys=False)
        .sample(n=100, random_state=SEED)
    )
    x_train, x_hold, y_train, y_hold = train_test_split(
        selected[FEATURES], selected.waveform_code.astype(int),
        test_size=0.20, stratify=selected.waveform_code, random_state=SEED,
    )
    clf = model()
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_hold)
    print(json.dumps({
        "mode": "pilot", "n_train": len(x_train), "n_holdout": len(x_hold),
        "accuracy": accuracy_score(y_hold, prediction),
        "macro_f1": f1_score(y_hold, prediction, average="macro"),
        "confusion_matrix_labels_1_2_3": confusion_matrix(y_hold, prediction, labels=LABELS).tolist(),
    }, ensure_ascii=False, indent=2))


def full(train: pd.DataFrame, test: pd.DataFrame) -> None:
    x = train[FEATURES]
    y = train.waveform_code.astype(int).to_numpy()
    material = train.material.to_numpy()
    groups = pd.factorize(pd.MultiIndex.from_frame(x), sort=False)[0]
    if pd.DataFrame({"group": groups, "label": y}).groupby("group").label.nunique().max() != 1:
        raise ValueError("Identical classification inputs have conflicting labels")
    fold_predictions = np.zeros(len(train), dtype=np.int8)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_idx, hold_idx in cv.split(x, y, groups):
        if np.intersect1d(groups[fit_idx], groups[hold_idx]).size:
            raise ValueError("Identical feature groups cross a validation split")
        clf = model()
        clf.fit(x.iloc[fit_idx], y[fit_idx])
        fold_predictions[hold_idx] = clf.predict(x.iloc[hold_idx])
    if not np.isin(fold_predictions, LABELS).all():
        raise ValueError("Incomplete cross-validation predictions")
    random_cv = {
        "accuracy": accuracy_score(y, fold_predictions),
        "macro_f1": f1_score(y, fold_predictions, average="macro"),
        "confusion_matrix_labels_1_2_3": confusion_matrix(y, fold_predictions, labels=LABELS).tolist(),
    }

    material_results = []
    material_predictions = np.zeros(len(train), dtype=np.int8)
    for held_material in sorted(np.unique(material)):
        fit_idx = np.where(material != held_material)[0]
        hold_idx = np.where(material == held_material)[0]
        clf = model()
        clf.fit(x.iloc[fit_idx], y[fit_idx])
        prediction = clf.predict(x.iloc[hold_idx])
        material_predictions[hold_idx] = prediction
        material_results.append({
            "held_out_material": int(held_material),
            "n": len(hold_idx),
            "accuracy": accuracy_score(y[hold_idx], prediction),
            "macro_f1": f1_score(y[hold_idx], prediction, average="macro"),
            "confusion_matrix_labels_1_2_3": confusion_matrix(y[hold_idx], prediction, labels=LABELS).tolist(),
        })

    clf = model()
    clf.fit(x, y)
    prediction = clf.predict(test[FEATURES]).astype(int)
    probability = clf.predict_proba(test[FEATURES])
    max_probability = probability.max(axis=1)
    if clf.classes_.tolist() != LABELS:
        raise ValueError("Probability columns do not match class codes")
    results = pd.DataFrame({
        "sample_id": test.sample_id.astype(int),
        "material": test.material.astype(int),
        "waveform_code": prediction,
        "waveform": pd.Series(prediction).map({1: "正弦波", 2: "三角波", 3: "梯形波"}),
        "probability_sine": probability[:, 0],
        "probability_triangle": probability[:, 1],
        "probability_trapezoid": probability[:, 2],
        "max_probability": max_probability,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "附件二_波形分类结果.csv", index=False, encoding="utf-8-sig", float_format="%.9g")
    pd.DataFrame({"source_file": train.source_file, "source_row": train.source_row,
                  "actual_code": y, "cv_predicted_code": fold_predictions,
                  "held_material_predicted_code": material_predictions}).to_csv(
        OUT / "训练集验证逐行结果.csv", index=False, encoding="utf-8-sig"
    )
    importance = pd.DataFrame({"feature": FEATURES, "importance": clf.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(
        OUT / "特征重要性.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "seed": SEED,
        "model": "RandomForestClassifier",
        "parameters": {"n_estimators": 300, "min_samples_leaf": 2,
                       "max_features": "sqrt", "class_weight": "balanced_subsample"},
        "features": FEATURES,
        "input_sha256": {p.name: digest(p) for p in [
            EDA / "features_train.csv", EDA / "features_test_waveform.csv"]},
        "stratified_group_5fold": random_cv,
        "leave_one_material_out": material_results,
        "test_class_counts": {str(k): int(v) for k, v in results.waveform_code.value_counts().sort_index().items()},
        "special_samples": results.loc[results.sample_id.isin(SPECIAL_IDS),
                                      ["sample_id", "waveform_code", "waveform", "max_probability"]].to_dict(orient="records"),
        "low_confidence_below_0p8": int((max_probability < 0.8).sum()),
    }
    (OUT / "问题1_验证摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "stratified_group_5fold": random_cv,
        "leave_one_material_out": material_results,
        "test_class_counts": summary["test_class_counts"],
        "low_confidence_below_0p8": summary["low_confidence_below_0p8"],
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    train, test = load()
    if args.pilot:
        pilot(train)
    else:
        full(train, test)


if __name__ == "__main__":
    main()
