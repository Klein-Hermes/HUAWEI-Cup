"""Direct C1 Average frontier model with C4 compute-slowdown scenarios.

This is a supplemental, conditional scenario analysis. It does not modify the
frozen Q4 v0.6 outputs or claim annual forecast skill from the short history.
"""
from __future__ import annotations

import hashlib
import argparse
from collections import Counter
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
REV = ROOT / "Q4" / "06_修订推进_20260926"
RESULTS = REV / "results" / "forecast"
BASE_RESULTS = ROOT / "Q4" / "03_结果" / "results"
SEED = 20260926
N_SIM = 12000
SCALING_EXPONENTS = (0.5, 0.73)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_inputs():
    files = {
        "task_aggregate": BASE_RESULTS / "q4_task_aggregate.csv",
        "frontier_monthly": BASE_RESULTS / "q4_frontier_monthly.csv",
        "type_crosswalk": BASE_RESULTS / "q4_type_group_crosswalk.csv",
        "c4_year_summary": BASE_RESULTS / "q4_c4_year_summary.csv",
        "one_month_backtest": BASE_RESULTS / "q4_forecast_backtest_1month.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    return {key: pd.read_csv(path) for key, path in files.items()}, files


def prepare_model_rows(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d = inputs["task_aggregate"].copy()
    crosswalk = inputs["type_crosswalk"]
    type_map = dict(zip(crosswalk["C1_Type_raw"].astype(str),
                        crosswalk["normalized_regression_group"].astype(str)))
    d["submission_date"] = pd.to_datetime(d["submission_date"], errors="coerce")
    d["score"] = pd.to_numeric(d["C1_Average_0_100"], errors="coerce")
    d["params_b"] = pd.to_numeric(d["parameter_count_B"], errors="coerce")
    d["month"] = d["submission_date"].dt.to_period("M").astype(str)
    d["month_index"] = ((d["submission_date"].dt.year - 2024) * 12
                         + d["submission_date"].dt.month - 6)
    d["type_group"] = d["C1_Type_raw"].astype(str).map(type_map).fillna("other_rare")
    valid = (
        d["open_weights_status"].eq("Yes")
        & d["submission_date"].notna()
        & d["score"].between(0, 100)
        & d["params_b"].gt(0)
        & d["type_group"].ne("not_in_scale_universe")
    )
    return d.loc[valid].copy().sort_values(["month", "Model"]).reset_index(drop=True)


def make_design(d: pd.DataFrame, type_levels: list[str], reference: str) -> tuple[np.ndarray, list[str]]:
    cols = ["intercept", "log10_params_b", "month_index"]
    arrays = [np.ones(len(d)), np.log10(d["params_b"].to_numpy(float)),
              d["month_index"].to_numpy(float)]
    for level in type_levels:
        if level == reference:
            continue
        cols.append("type=" + level)
        arrays.append(d["type_group"].eq(level).to_numpy(float))
    return np.column_stack(arrays), cols


def fit_hc3(d: pd.DataFrame, type_levels: list[str], reference: str):
    X, cols = make_design(d, type_levels, reference)
    y = d["score"].to_numpy(float)
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1] or len(y) <= X.shape[1]:
        raise ValueError(f"Rank/df failure: rank={rank}, columns={X.shape[1]}, n={len(y)}")
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    adj = resid / np.maximum(1.0 - leverage, 1e-9)
    meat = X.T @ (X * (adj * adj)[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    return {"beta": beta, "cov": cov, "columns": cols, "rank": rank,
            "n": len(y), "resid": resid, "r2": 1.0 - float(resid @ resid) / float(((y-y.mean())**2).sum())}


def c4_compute_scenarios(c4: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    d = c4[c4["publication_year"].isin([2022, 2023, 2024])].copy()
    d["compute_p90"] = pd.to_numeric(d["training_compute_p90_FLOP"], errors="coerce")
    d["year"] = pd.to_numeric(d["publication_year"], errors="coerce")
    d = d.dropna(subset=["compute_p90", "year"])
    if set(d["year"].astype(int)) != {2022, 2023, 2024}:
        raise ValueError("C4 2022-2024 annual P90 compute series is incomplete")
    coef = np.polyfit(d["year"].to_numpy(float) - 2022,
                      np.log(d["compute_p90"].to_numpy(float)), 1)
    log_growth = float(coef[0])
    annual_g = float(math.exp(log_growth))
    specs = [
        ("历史速度延续", "historical_rate", log_growth),
        ("中度放缓", "half_log_growth", 0.5 * log_growth),
        ("强放缓", "flat_compute", 0.0),
    ]
    rows = []
    for label, key, g in specs:
        growth = math.exp(g)
        rows.append({
            "scenario": key, "scenario_label_zh": label,
            "c4_annual_compute_growth_multiplier": growth,
            "compute_multiplier_12m": growth,
            "compute_multiplier_24m": growth ** 2,
            "parameter_multiplier_12m_assuming_N_proportional_C_half": growth ** 0.5,
            "parameter_multiplier_24m_assuming_N_proportional_C_half": growth,
            "scenario_basis": "log-linear trend of C4 annual training-compute P90, 2022-2024; scenario only, not C1 matched compute",
        })
    out = pd.DataFrame(rows)
    # Include C4 coverage/confidence counts for interpretation.
    c4_counts = d.set_index("year")
    for year in (2022, 2023, 2024):
        source = c4_counts.loc[year]
        out[f"c4_{year}_scalar_n"] = int(source["training_compute_scalar_n"])
        out[f"c4_{year}_confident_scalar_n"] = int(source["training_compute_confident_scalar_n"])
        out[f"c4_{year}_compute_p90_FLOP"] = float(source["compute_p90"])
    return out, annual_g


def monthly_frontier_from_rows(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in d.groupby("month", sort=True):
        rows.append({"month": month, "n": len(group),
                     "frontier_p90": float(group["score"].quantile(.9, interpolation="linear"))})
    return pd.DataFrame(rows)


def one_step_candidates(d: pd.DataFrame, frontier: pd.DataFrame, annual_compute_g: float):
    rows = []
    # Match the frozen Q4 short-horizon benchmark: only month bins with >=10
    # open-weight models are reliable; use the last five one-step targets.
    reliable = frontier[frontier["n"].ge(10)].reset_index(drop=True)
    if len(reliable) < 6:
        raise ValueError("Need at least six reliable monthly bins (n>=10) for one-step comparison")
    months = reliable["month"].tolist()
    values = reliable["frontier_p90"].to_numpy(float)
    month_index = np.array([pd.Period(month, freq="M").ordinal for month in months], dtype=float)
    growth_1m = annual_compute_g ** (1.0 / 12.0)
    delta_log_n_1m = 0.5 * math.log10(growth_1m)
    for target_i in range(max(3, len(months) - 5), len(months)):
        origin_i = target_i - 1
        origin_month, target_month = months[origin_i], months[target_i]
        train_frontier = reliable.iloc[:origin_i + 1].copy()
        y = train_frontier["frontier_p90"].to_numpy(float)
        origin_frontier = float(y[-1])
        observed = float(values[target_i])
        pred_map = {"last_value": origin_frontier}
        x_time = month_index[:origin_i + 1]
        slope, intercept = np.polyfit(x_time, y, 1)
        pred_map["linear_time_trend"] = float(intercept + slope * month_index[target_i])
        if len(y) >= 8:
            xa, ya = y[:-1], y[1:]
            ar_x = np.column_stack([np.ones(len(xa)), xa])
            if np.linalg.matrix_rank(ar_x) == 2:
                ar_beta = np.linalg.lstsq(ar_x, ya, rcond=None)[0]
                pred_map["AR1"] = float(ar_beta[0] + ar_beta[1] * y[-1])
        train_rows = d[d["month"].le(origin_month)].copy()
        try:
            # Determine categories within each training fold. Including categories
            # that only appear in future folds creates a rank-deficient design.
            fold_types = sorted(train_rows["type_group"].unique().tolist())
            reference = "chat_posttrained" if "chat_posttrained" in fold_types else fold_types[0]
            fit = fit_hc3(train_rows, fold_types, reference)
            b_n = fit["beta"][fit["columns"].index("log10_params_b")]
            b_t = fit["beta"][fit["columns"].index("month_index")]
            pred_map["size_time_C4_ref"] = float(origin_frontier + b_t + b_n * delta_log_n_1m)
        except (ValueError, np.linalg.LinAlgError):
            pass
        for method, pred in pred_map.items():
            rows.append({"origin_month": origin_month, "target_month": target_month,
                         "method": method, "predicted_p90": pred,
                         "observed_p90": observed, "error_pred_minus_observed": pred-observed,
                         "absolute_error": abs(pred-observed),
                         "train_frontier_months": len(y), "train_model_rows": len(train_rows),
                         "target_frontier_models_n": int(reliable.iloc[target_i]["n"])})
    # AR(1) has only two supported origins in the historical diagnostic and uses
    # all observed month bins; keep its shorter 8/9-bin window explicit.
    all_months = frontier["month"].tolist()
    all_values = frontier["frontier_p90"].to_numpy(float)
    for target_i in range(max(8, len(all_months) - 2), len(all_months)):
        y = all_values[:target_i]
        xa, ya = y[:-1], y[1:]
        ar_x = np.column_stack([np.ones(len(xa)), xa])
        if np.linalg.matrix_rank(ar_x) < 2:
            continue
        ar_beta = np.linalg.lstsq(ar_x, ya, rcond=None)[0]
        pred = float(ar_beta[0] + ar_beta[1] * y[-1])
        observed = float(all_values[target_i])
        rows.append({
            "origin_month": all_months[target_i - 1], "target_month": all_months[target_i],
            "method": "AR1", "predicted_p90": pred, "observed_p90": observed,
            "error_pred_minus_observed": pred - observed,
            "absolute_error": abs(pred - observed), "train_frontier_months": len(y),
            "train_model_rows": int(d["month"].le(all_months[target_i - 1]).sum()),
            "target_frontier_models_n": int(frontier.iloc[target_i]["n"]),
        })
    detail = pd.DataFrame(rows)
    summary = (detail.groupby("method", as_index=False)
               .agg(origins=("absolute_error", "size"), MAE=("absolute_error", "mean"),
                    median_absolute_error=("absolute_error", "median")))
    return detail, summary


def cluster_bootstrap_coefficients(d: pd.DataFrame, rng: np.random.Generator,
                                   draws: int = 1500) -> tuple[np.ndarray, int, dict[str, int]]:
    months = d["month"].drop_duplicates().to_numpy()
    result = []
    failures: Counter[str] = Counter()
    for _ in range(draws):
        picked = rng.choice(months, size=len(months), replace=True)
        pieces = [d[d["month"].eq(m)] for m in picked]
        boot = pd.concat(pieces, ignore_index=True)
        try:
            # A cluster bootstrap may omit rare categories. Build the design
            # from categories actually represented in each resample so absent
            # levels do not create all-zero columns and artificial rank failure.
            boot_types = sorted(boot["type_group"].unique().tolist())
            boot_reference = "chat_posttrained" if "chat_posttrained" in boot_types else boot_types[0]
            fit = fit_hc3(boot, boot_types, boot_reference)
            b_n = float(fit["beta"][fit["columns"].index("log10_params_b")])
            b_t = float(fit["beta"][fit["columns"].index("month_index")])
            result.append([b_n, b_t])
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures[f"{type(exc).__name__}: {str(exc)}"] += 1
            continue
    if len(result) < 200:
        raise ValueError(f"Too few successful month-cluster bootstrap replicates: {len(result)}")
    return np.asarray(result), len(result), dict(failures)


def cluster_bootstrap_frontier(march: pd.DataFrame, rng: np.random.Generator,
                               draws: int = 5000) -> tuple[np.ndarray, int]:
    d = march.copy()
    d["owner_cluster"] = d["Model"].astype(str).str.split("/").str[0]
    owners = d["owner_cluster"].drop_duplicates().to_numpy()
    if len(owners) < 3:
        raise ValueError("Need at least three owner clusters for frontier bootstrap")
    out = np.empty(draws, dtype=float)
    groups = {owner: d[d["owner_cluster"].eq(owner)] for owner in owners}
    for i in range(draws):
        sampled = rng.choice(owners, size=len(owners), replace=True)
        sample = pd.concat([groups[owner] for owner in sampled], ignore_index=True)
        out[i] = float(sample["score"].quantile(.9, interpolation="linear"))
    return out, len(owners)


def write_report(path: Path, diag: dict, scenarios: pd.DataFrame, forecast: pd.DataFrame,
                 cand_summary: pd.DataFrame) -> None:
    primary = forecast[forecast["is_primary_alpha_half"]].copy()
    sensitivity = forecast[~forecast["is_primary_alpha_half"]].copy()
    def table(frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 2) -> str:
        headers = [label for _, label in columns]
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for _, row in frame.iterrows():
            vals = []
            for key, _ in columns:
                value = row[key]
                if isinstance(value, (float, np.floating)):
                    vals.append(f"{value:.{digits}f}")
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    scenario_lines = []
    for _, row in scenarios.iterrows():
        scenario_lines.append(
            f"- **{row['scenario_label_zh']}**：C4 年算力乘数 {row['c4_annual_compute_growth_multiplier']:.3f}；"
            f"2022/2023/2024 C4 算力 P90 分别为 "
            f"{row['c4_2022_compute_p90_FLOP']:.3g}/"
            f"{row['c4_2023_compute_p90_FLOP']:.3g}/"
            f"{row['c4_2024_compute_p90_FLOP']:.3g} FLOP。"
            f"三年标量/Confident 数为 {int(row['c4_2022_scalar_n'])}/{int(row['c4_2022_confident_scalar_n'])}、"
            f"{int(row['c4_2023_scalar_n'])}/{int(row['c4_2023_confident_scalar_n'])}、"
            f"{int(row['c4_2024_scalar_n'])}/{int(row['c4_2024_confident_scalar_n'])}。"
        )
    c4_growth = float(scenarios.iloc[0]["c4_annual_compute_growth_multiplier"])
    point_cols = [
        ("horizon_months", "期限(月)"), ("target_month", "目标月"),
        ("scenario_label_zh", "C4 情景"), ("compute_multiplier", "累计算力乘数"),
        ("parameter_multiplier_under_alpha", "参数量乘数(α=.5)"),
        ("frontier_point_C1_Average_P90", "C1 Average P90 点值"),
        ("conditional_simulation_q025", "模拟分位数 2.5%"),
        ("conditional_simulation_q975", "模拟分位数 97.5%"),
    ]
    sens_cols = [
        ("horizon_months", "期限(月)"), ("target_month", "目标月"),
        ("scenario_label_zh", "C4 情景"), ("parameter_multiplier_under_alpha", "参数量乘数(α=.73)"),
        ("frontier_point_C1_Average_P90", "C1 Average P90 点值"),
        ("conditional_simulation_q025", "模拟分位数 2.5%"),
        ("conditional_simulation_q975", "模拟分位数 97.5%"),
    ]
    method_cols = [("method", "候选方法"), ("origins", "起点数"),
                   ("MAE", "一步 MAE"), ("median_absolute_error", "中位绝对误差")]
    report = f"""# Q4 Task 3：前沿预测模型与 C4 放缓情景

**交付状态：** `conditional_scenario_only`。年度预测技能 `NOT_VALIDATED`；模拟分位数带覆盖率 `UNKNOWN`。这是为了呈现题目明确要求的 12/24 月数值情景，不是经年度回测验证的预测。

## 目标与数据

- 预测原点：{diag['forecast_origin']}；目标月：{', '.join(diag['target_months'])}。
- 目标：每月新提交、开放权重模型的 C1 官方 `Average` 分数 P90，范围 0–100；不等于 Q4 六族相对百分位综合指数，也不经 Loss 桥换算。
- 201 条模型行、{diag['n_month_bins']} 个有记录月份；最新月前沿为 {diag['latest_month_frontier_p90']:.4f}，当月 18 个模型、{diag['latest_month_owner_clusters']} 个发布者簇。
- 月度 P90 从冻结包原始聚合数据重建，最大绝对差 {diag['monthly_frontier_rebuild_max_abs_difference']:.3g}。

## 候选模型与一步回测

规模—时间模型直接拟合 C1 原始 `Average`：

`score = α + β_N log10(parameter_count_B) + β_t month_index + Type effects + error`

本次估计 β_N={diag['beta_log10_parameter_count']:.3f}、β_t={diag['beta_month']:.3f}，R²={diag['model_r2']:.3f}。一步回测结果如下。持平与线性趋势共用最后 5 个可靠目标月；AR(1) 只有 2 个可用起点。

{table(cand_summary.sort_values('MAE'), method_cols, 3)}

规模—时间模型的一步 MAE 为 {float(cand_summary.loc[cand_summary['method'].eq('size_time_C4_ref'), 'MAE'].iloc[0]):.3f}，高于持平基线的 {float(cand_summary.loc[cand_summary['method'].eq('last_value'), 'MAE'].iloc[0]):.3f}。因此它仅用作题目要求的 C4 条件情景生成器，不宣称优于持平预测。线性趋势和 AR(1) 作为候选基准报告，不作为三情景主输出。

## C4 三种算力路径

三种路径均以 C4 年训练算力 P90 在 2022–2024 的三点对数线性趋势（年乘数 {c4_growth:.3f}）为参照。C4 只锚定算力路径，C4 与 C1 没有个体精确名称匹配，因此不估计个体能力—算力关系。

{chr(10).join(scenario_lines)}

主映射为 `N ∝ C^0.5`；额外给出 `N ∝ C^0.73` 结构敏感性。指数是文献启发的外加假设，不是本数据拟合所得。见方法方案的参考来源。

## 条件数值结果：α=0.5

{table(primary.sort_values(['horizon_months', 'scenario']), point_cols, 2)}

## 结构敏感性：α=0.73

{table(sensitivity.sort_values(['horizon_months', 'scenario']), sens_cols, 2)}

所有本次点值和模拟带端点均处于 0–100。此处只报告原始 P90 尺度，不做静默截断。

## 模拟带与限制

表内 2.5%–97.5% 数字是**固定 C4 情景下模拟分布的经验分位数**，不是经校准的 95% 预测区间。传播了月份簇系数 bootstrap（成功 {diag['month_cluster_bootstrap_successes']}/{diag['month_cluster_bootstrap_requested']} 次，失败 {diag['month_cluster_bootstrap_failures']} 次；分位数以可估计抽样为条件）、2025-03 发布者簇前沿 bootstrap（{diag['latest_month_owner_clusters']} 个簇），以及最近 5 个一步持平误差中心化后的独立逐月重抽样。系数和起点前沿分别重抽样，未传播二者的协方差。它依赖近期一步误差可代表未来、逐月独立同分布的假设；没有 12/24 月滚动起点，覆盖率未知，也没有传播 C4 增速和 α 选择的不确定性。C4 年度 P90 只有三个年份且混合置信等级。

Open LLM Leaderboard v1 于 2024-06 被替换，v2 于 2025-03 退役；不同版本不可拼成同口径长时间序列。因此本报告的目标遵循题目规定和已有 C1 定义，不声称退役排行榜之后仍以同一协议持续产出观测。

**完整追溯：** 候选回测、C4 情景、12/24 月结果和输入/输出 SHA-256 见本目录其他 CSV/JSON 文件。历史冻结包与 v0.7 的“年度技能不足”判定保留；v0.8 只授权条件情景数值展示。
"""
    path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run reduced Monte Carlo into results/smoke/forecast_model")
    args = parser.parse_args()
    output_dir = (REV / "results" / "smoke" / "forecast_model") if args.smoke else RESULTS
    n_sim = 800 if args.smoke else N_SIM
    coeff_boot_n = 400 if args.smoke else 1500
    anchor_boot_n = 1000 if args.smoke else 5000
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, input_paths = read_inputs()
    d = prepare_model_rows(inputs)
    observed_frontier = monthly_frontier_from_rows(d)
    attached_frontier = inputs["frontier_monthly"].rename(columns={
        "submission_month": "month",
        "monthly_entry_p90_frontier_C1_average_0_100": "attached_frontier_p90",
    })
    check = observed_frontier.merge(attached_frontier[["month", "attached_frontier_p90"]],
                                    on="month", how="outer", validate="one_to_one")
    if check["attached_frontier_p90"].isna().any() or check["frontier_p90"].isna().any():
        raise ValueError("Rebuilt monthly series does not match frozen frontier month coverage")
    max_diff = float((check["frontier_p90"] - check["attached_frontier_p90"]).abs().max())
    if max_diff > 1e-8:
        raise ValueError(f"Rebuilt monthly frontier does not reproduce baseline, max diff={max_diff}")

    scenarios, annual_g = c4_compute_scenarios(inputs["c4_year_summary"])
    frontier = observed_frontier.sort_values("month").reset_index(drop=True)
    latest_month = str(frontier.iloc[-1]["month"])
    h0 = pd.Period(latest_month, freq="M")
    origin_rows = d[d["month"].eq(latest_month)].copy()
    types = sorted(d["type_group"].unique().tolist())
    reference = "chat_posttrained" if "chat_posttrained" in types else types[0]
    fit = fit_hc3(d, types, reference)
    coeff_draws, coeff_success, coeff_failures = cluster_bootstrap_coefficients(
        d, np.random.default_rng(SEED), draws=coeff_boot_n)
    anchor_draws, n_owner_clusters = cluster_bootstrap_frontier(
        origin_rows, np.random.default_rng(SEED + 1), draws=anchor_boot_n)
    backtest_one = inputs["one_month_backtest"]
    recent = backtest_one[backtest_one["method"].eq("last_value")].tail(5)
    innovations = -pd.to_numeric(recent["error_pred_minus_observed"], errors="coerce").dropna().to_numpy(float)
    if len(innovations) < 3:
        raise ValueError("Fewer than three recent one-month baseline errors for conditional interval")
    innovations = innovations - innovations.mean()
    rng = np.random.default_rng(SEED + 2)

    b_n = float(fit["beta"][fit["columns"].index("log10_params_b")])
    b_t = float(fit["beta"][fit["columns"].index("month_index")])
    b_n_draws = coeff_draws[:, 0]
    b_t_draws = coeff_draws[:, 1]
    forecast_rows = []
    for scenario in scenarios.to_dict("records"):
        annual_growth = float(scenario["c4_annual_compute_growth_multiplier"])
        for horizon in (12, 24):
            years = horizon / 12.0
            compute_mult = annual_growth ** years
            idx = rng.integers(0, len(coeff_draws), size=n_sim)
            anchor = rng.choice(anchor_draws, size=n_sim, replace=True)
            shock_idx = rng.integers(0, len(innovations), size=(n_sim, horizon))
            shock = innovations[shock_idx].sum(axis=1)
            for alpha in SCALING_EXPONENTS:
                delta_log_n = alpha * math.log10(compute_mult)
                point = float(frontier.iloc[-1]["frontier_p90"] + b_t * horizon + b_n * delta_log_n)
                sim = (anchor + b_t_draws[idx] * horizon + b_n_draws[idx] * delta_log_n + shock)
                q_low, q_high = np.quantile(sim, [.025, .975])
                forecast_rows.append({
                    "forecast_origin": latest_month,
                    "horizon_months": horizon,
                    "target_month": str(h0 + horizon),
                    "scenario": scenario["scenario"],
                    "scenario_label_zh": scenario["scenario_label_zh"],
                    "scaling_exponent_alpha_N_proportional_C_alpha": alpha,
                    "is_primary_alpha_half": alpha == 0.5,
                    "c4_annual_compute_growth_multiplier": annual_growth,
                    "compute_multiplier": compute_mult,
                    "parameter_multiplier_under_alpha": compute_mult ** alpha,
                    "frontier_point_C1_Average_P90": point,
                    "conditional_simulation_q025": float(q_low),
                    "conditional_simulation_q975": float(q_high),
                    "point_outside_score_domain_0_100": not (0 <= point <= 100),
                    "simulation_band_outside_score_domain_0_100": not (q_low >= 0 and q_high <= 100),
                    "simulation_band_method": "month-cluster coefficient bootstrap + owner-cluster origin frontier bootstrap + iid resampling of centered five 1-month last-value errors",
                    "simulation_band_coverage_calibrated": False,
                    "annual_horizon_rolling_origin_count": 0,
                    "conditioned_on_type_mix_residual_distribution_and_protocol_stability": True,
                    "c4_is_individual_c1_compute_match": False,
                })
    forecast = pd.DataFrame(forecast_rows)

    point_diag = {
        "n_open_weight_model_rows": int(len(d)),
        "n_month_bins": int(len(frontier)),
        "forecast_origin": latest_month,
        "target_months": [str(h0 + 12), str(h0 + 24)],
        "latest_month_cohort_n": int(len(origin_rows)),
        "latest_month_frontier_p90": float(frontier.iloc[-1]["frontier_p90"]),
        "latest_month_owner_clusters": int(n_owner_clusters),
        "monthly_frontier_rebuild_max_abs_difference": max_diff,
        "model_n": int(fit["n"]), "model_rank": int(fit["rank"]),
        "model_columns": fit["columns"],
        "model_r2": float(fit["r2"]),
        "beta_log10_parameter_count": b_n,
        "beta_month": b_t,
        "reference_type_group": reference,
        "type_groups": types,
        "c4_reference_annual_compute_growth_multiplier": annual_g,
        "one_month_recent_last_value_errors_n": int(len(innovations)),
        "one_month_centered_innovation_sd": float(np.std(innovations, ddof=1)),
        "month_cluster_bootstrap_successes": int(coeff_success),
        "month_cluster_bootstrap_requested": int(coeff_boot_n),
        "month_cluster_bootstrap_failures": int(coeff_boot_n - coeff_success),
        "month_cluster_bootstrap_failure_reasons": coeff_failures,
        "annual_horizon_rolling_origins": 0,
        "annual_simulation_band_coverage_calibrated": False,
        "scaling_exponents": list(SCALING_EXPONENTS),
        "warning": "Conditional scenario simulation quantiles are not calibrated coverage intervals; they depend on stationary iid recent one-step errors and fixed C4/scaling scenarios. The historical source is retired and provides no annual rolling-origin validation.",
    }

    cand_detail, cand_summary = one_step_candidates(d, frontier, annual_g)
    observed_frontier.to_csv(output_dir / "q4_frontier_monthly_rebuilt.csv", index=False)
    scenarios.to_csv(output_dir / "q4_c4_compute_scenarios.csv", index=False)
    forecast.to_csv(output_dir / "q4_frontier_scenario_forecasts.csv", index=False)
    cand_detail.to_csv(output_dir / "q4_frontier_candidate_backtest.csv", index=False)
    cand_summary.to_csv(output_dir / "q4_frontier_candidate_summary.csv", index=False)
    write_report(output_dir / "q4_frontier_scenario_report.md", point_diag,
                 scenarios, forecast, cand_summary)
    (output_dir / "q4_frontier_model_diagnostics.json").write_text(
        json.dumps(point_diag, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "seed": SEED,
        "simulations_per_scenario_horizon": n_sim,
        "smoke_run": bool(args.smoke),
        "inputs": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                   for path in input_paths.values()},
        "script_sha256": sha256(Path(__file__).resolve()),
        "contract_and_method_sha256": {
            path.name: sha256(path) for path in [
                REV / "q4_task3_forecast_amendment_v0.8.md",
                REV / "q4_frontier_forecast_method_v0.1.md",
            ]
        },
        "outputs": {},
        "scope": "supplemental conditional forecast; frozen Q4 v0.6 not modified",
        "forecast_point_equation": "F0 + beta_month*h + beta_log10_params*alpha*log10(C_multiplier_h); alpha=0.5 primary and 0.73 sensitivity",
        "annual_validation_status": "NOT_VALIDATED; zero annual rolling-origin folds in supplied same-protocol data",
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "q4_frontier_forecast_manifest.json":
            manifest["outputs"][path.name] = sha256(path)
    (output_dir / "q4_frontier_forecast_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rows={len(d)} months={len(frontier)} origin={latest_month} beta_N={b_n:.6f} beta_t={b_t:.6f} C4_G={annual_g:.6f}")
    print(cand_summary.to_string(index=False))
    print(forecast[["horizon_months", "scenario_label_zh", "scaling_exponent_alpha_N_proportional_C_alpha",
                   "frontier_point_C1_Average_P90", "conditional_simulation_q025",
                   "conditional_simulation_q975"]].to_string(index=False))
    print(f"annual_origins=0 simulation_band_coverage_calibrated=false outputs={len(manifest['outputs'])}")


if __name__ == "__main__":
    main()
