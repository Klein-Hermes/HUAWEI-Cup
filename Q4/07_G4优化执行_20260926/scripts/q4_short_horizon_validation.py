"""Small-sample exact-calendar short-horizon diagnostics for Q4 G4.

This checks persistence and linear-time baselines at horizons that the supplied
monthly history can actually support. It does not validate 12/24-month skill.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "Q4" / "07_G4优化执行_20260926" / "results" / "forecast" / "q4_frontier_monthly_rebuilt.csv"
DEFAULT_OUTPUT = ROOT / "Q4" / "07_G4优化执行_20260926" / "results" / "forecast"
MIN_MONTH_N = 10
MIN_TRAIN_MONTHS = 3
HORIZONS = (1, 2, 3, 4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"month", "n", "frontier_p90"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = raw.loc[:, ["month", "n", "frontier_p90"]].copy()
    df["period"] = pd.PeriodIndex(df["month"].astype(str), freq="M")
    df["n"] = pd.to_numeric(df["n"], errors="raise").astype(int)
    df["frontier_p90"] = pd.to_numeric(df["frontier_p90"], errors="raise")
    if df["period"].duplicated().any():
        raise ValueError("Duplicate month rows")
    if not df["period"].is_monotonic_increasing:
        df = df.sort_values("period").reset_index(drop=True)
    if not np.isfinite(df["frontier_p90"].to_numpy()).all():
        raise ValueError("Non-finite frontier score")
    if not df["frontier_p90"].between(0, 100).all():
        raise ValueError("Frontier P90 must remain on the 0-100 score scale")
    if (df["n"] < 0).any():
        raise ValueError("Negative monthly sample size")
    return df


def score_fold(train: pd.DataFrame, target: pd.Series, horizon: int) -> list[dict]:
    origin = target["period"] - horizon
    train = train.loc[train["period"] <= origin].sort_values("period")
    if len(train) < MIN_TRAIN_MONTHS:
        return []

    y = train["frontier_p90"].to_numpy(dtype=float)
    x = train["period"].map(lambda p: p.ordinal).to_numpy(dtype=float)
    target_x = float(target["period"].ordinal)
    design = np.column_stack((np.ones(len(x)), x))
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2:
        raise ValueError("Linear trend design is rank deficient")
    predictions = {
        "last_value": float(y[-1]),
        "linear_time_trend": float(coef[0] + coef[1] * target_x),
    }
    rows = []
    for method, prediction in predictions.items():
        rows.append({
            "horizon_months": int(horizon),
            "forecast_origin": str(origin),
            "latest_training_month": str(train.iloc[-1]["period"]),
            "target_month": str(target["period"]),
            "training_months": int(len(train)),
            "training_last_observation_lag_months": int(origin.ordinal - train.iloc[-1]["period"].ordinal),
            "method": method,
            "observed_frontier_p90": float(target["frontier_p90"]),
            "predicted_frontier_p90": prediction,
            "absolute_error": abs(prediction - float(target["frontier_p90"])),
        })
    return rows


def evaluate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = validate_monthly(raw)
    eligible = data.loc[data["n"] >= MIN_MONTH_N].copy().reset_index(drop=True)
    if len(eligible) < MIN_TRAIN_MONTHS + 1:
        raise ValueError("Too few eligible months for a rolling diagnostic")

    folds: list[dict] = []
    for horizon in HORIZONS:
        for _, target in eligible.iterrows():
            folds.extend(score_fold(eligible, target, horizon))
    details = pd.DataFrame(folds)
    if details.empty:
        raise ValueError("No exact-calendar target folds were available")

    summaries = []
    for (horizon, method), group in details.groupby(["horizon_months", "method"], sort=True):
        summaries.append({
            "horizon_months": int(horizon),
            "method": method,
            "folds": int(len(group)),
            "mae": float(group["absolute_error"].mean()),
            "rmse": float(np.sqrt(np.mean(np.square(
                group["predicted_frontier_p90"] - group["observed_frontier_p90"]
            )))),
            "first_target_month": group["target_month"].min(),
            "last_target_month": group["target_month"].max(),
        })
    summary = pd.DataFrame(summaries)
    return data, details, summary


def run_smoke(input_path: Path) -> int:
    raw = pd.read_csv(input_path, encoding="utf-8-sig")
    validated = validate_monthly(raw)
    eligible_idx = validated.index[validated["n"] >= MIN_MONTH_N]
    if len(eligible_idx) < 6:
        raise ValueError("P1 real-input slice requires at least six eligible months")
    sample = validated.iloc[: int(eligible_idx[5]) + 1].copy()
    _, folds, summary = evaluate(sample)
    if folds.empty or summary.empty:
        raise AssertionError("No real-input folds were generated")
    for row in folds.itertuples(index=False):
        if (pd.Period(row.target_month, freq="M").ordinal
                - pd.Period(row.forecast_origin, freq="M").ordinal) != row.horizon_months:
            raise AssertionError("A real-input fold does not match its exact calendar horizon")
        if pd.Period(row.latest_training_month, freq="M") > pd.Period(row.forecast_origin, freq="M"):
            raise AssertionError("A real-input fold uses information after its forecast origin")
        if row.training_months < MIN_TRAIN_MONTHS:
            raise AssertionError("A real-input fold has too few training months")
    print(json.dumps({"status": "PASS", "input": str(input_path.resolve()),
                      "real_month_rows": int(len(sample)), "eligible_months": 6,
                      "fold_rows": int(len(folds)), "horizons": sorted(folds["horizon_months"].unique().tolist())},
                     ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="run a small calendar-horizon example only")
    args = parser.parse_args()
    if args.smoke:
        return run_smoke(args.input.resolve())

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path, encoding="utf-8-sig")
    source, folds, summary = evaluate(raw)
    folds_path = output_dir / "q4_short_horizon_validation_folds.csv"
    summary_path = output_dir / "q4_short_horizon_validation_summary.csv"
    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    reproduction_command = f'& "{Path(sys.executable).resolve()}" "{Path(__file__).resolve()}"'
    manifest = {
        "status": "short_horizon_diagnostic_only",
        "horizons_months": list(HORIZONS),
        "min_monthly_model_count": MIN_MONTH_N,
        "min_training_months": MIN_TRAIN_MONTHS,
        "origin_rule": "exact target calendar month minus h; train only on eligible months not later than origin",
        "methods": ["last_value", "linear_time_trend"],
        "annual_skill_status": "NOT_VALIDATED",
        "data_period_all_rows": [str(source["period"].min()), str(source["period"].max())],
        "data_period_eligible_rows": [str(source.loc[source["n"].ge(MIN_MONTH_N), "period"].min()),
                                      str(source.loc[source["n"].ge(MIN_MONTH_N), "period"].max())],
        "raw_month_rows": int(len(source)),
        "eligible_month_rows": int(source["n"].ge(MIN_MONTH_N).sum()),
        "calendar_gaps_among_eligible_months": int(sum(
            max(0, int(b.ordinal - a.ordinal) - 1)
            for a, b in zip(
                source.loc[source["n"].ge(MIN_MONTH_N), "period"].iloc[:-1],
                source.loc[source["n"].ge(MIN_MONTH_N), "period"].iloc[1:],
            )
        )),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "runtime": env,
        "unique_reproduction_command_from_project_root": reproduction_command,
        "folds_file": folds_path.name,
        "folds_sha256": sha256(folds_path),
        "summary_file": summary_path.name,
        "summary_sha256": sha256(summary_path),
        "note": "Small exact-calendar backtests are diagnostic only and do not validate 12/24-month predictive skill or interval coverage.",
    }
    manifest_path = output_dir / "q4_short_horizon_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# G4 短期限预测能力诊断",
        "",
        "**结论标签：** 仅为 1–4 个月短期限滚动诊断；不验证 12/24 个月预测技能。",
        "",
        f"- 原始月数：{len(source)}（{manifest['data_period_all_rows'][0]} 至 {manifest['data_period_all_rows'][1]}）；",
        f"- 每月模型数至少 {MIN_MONTH_N} 的有效月份：{manifest['eligible_month_rows']}；",
        f"- 有效月份中跳过的日历月：{manifest['calendar_gaps_among_eligible_months']}；",
        f"- 训练最少 {MIN_TRAIN_MONTHS} 个有效月份，按精确日历月回溯预测；缺月不会压缩成相邻月份。",
        "",
        "## 分期限回测",
        "",
        "| 期限（月） | 方法 | 有效折数 | MAE（Average P90 点） | RMSE（点） | 目标月范围 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for r in summary.sort_values(["horizon_months", "method"]).itertuples(index=False):
        lines.append(f"| {r.horizon_months} | {r.method} | {r.folds} | {r.mae:.3f} | {r.rmse:.3f} | {r.first_target_month}–{r.last_target_month} |")
    lines += [
        "",
        "有效折数随期限增长迅速减少；这些小样本结果只能比较短期基线，不足以证明年度预测能力或校准任何区间。主情景模型的一步 MAE 和持平基线仍以 `q4_forecast_one_step_validation.csv` 为准。",
        "",
        "## 可复现入口",
        "",
        "`" + reproduction_command.replace("`", "``") + "`",
        "",
        f"输入哈希：`{manifest['input_sha256']}`；脚本哈希：`{manifest['script_sha256']}`。",
        "",
        "逐折明细、汇总和环境信息分别见 `q4_short_horizon_validation_folds.csv`、`q4_short_horizon_validation_summary.csv`、`q4_short_horizon_validation_manifest.json`。",
    ]
    report_path = output_dir / "q4_short_horizon_validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["report_file"] = report_path.name
    manifest["report_sha256"] = sha256(report_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"wrote {folds_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
