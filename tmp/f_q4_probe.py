"""F 题 Q4 数据可达性体检（临时探针，输出到 stdout）。

检查每个 Q4 要求所需字段的覆盖率与可用样本量。
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"中文题目\F题\real_attachments\C_efficiency_evolution"
pd.set_option("display.width", 220)

lb = pd.read_csv(f"{BASE}/leaderboard_enhanced.csv")
lb["subdate"] = pd.to_datetime(lb["Submission Date"], errors="coerce")
print("== C2 leaderboard_enhanced ==", lb.shape)
print("字段覆盖率:")
for c in ["#Params (B)", "Submission Date", "Hub License", "Average ⬆️", "IFEval", "BBH",
          "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO", "Epoch_AI_Publication_Date",
          "Epoch_AI_Organization", "Epoch_AI_Open_Weights"]:
    print(f"   {c:26s} {lb[c].notna().mean():.3f}")
print("submission 日期范围:", lb["subdate"].min(), "->", lb["subdate"].max())

open_like = lb[lb["Type"].isin(["🟢 pretrained", "🟩 continuously pretrained"])].copy()
print("\n开源基础模型子集(pretrained + continuously pretrained):", open_like.shape)
print("   Params 非空:", open_like["#Params (B)"].notna().sum(),
      "| 日期非空:", open_like["subdate"].notna().sum(),
      "| Average 非空:", open_like["Average ⬆️"].notna().sum())
print("   按年:", open_like["subdate"].dt.year.value_counts().sort_index().to_dict())
print("   参数范围(B): %.3f -> %.1f" % (open_like["#Params (B)"].min(), open_like["#Params (B)"].max()))
print("   Epoch 匹配数:", int(open_like["Epoch_AI_Publication_Date"].notna().sum()))

print("\n== C3 timeseries ==")
ts = pd.read_csv(f"{BASE}/leaderboard_extended_timeseries.csv")
print(ts.shape, "| Year:", ts["Year"].min(), "->", ts["Year"].max())
print("Source 分布:", ts["Source"].value_counts().to_dict())
print("按年模型数:", ts["Year"].value_counts().sort_index().to_dict())

print("\n== C4 epoch ==")
ep = pd.read_csv(f"{BASE}/epoch_all_ai_models.csv")
for c in ["Parameters", "Training compute (FLOP)", "Training dataset size (total)",
          "Open model weights?", "Publication date", "Domain", "Task"]:
    print(f"   {c:34s} {ep[c].notna().mean():.3f}")
llm = ep[ep["Domain"].astype(str).str.contains("Language", case=False, na=False)]
print("   Domain 含 Language:", len(llm))
llm2 = llm[llm["Training compute (FLOP)"].notna() & llm["Publication date"].notna()]
print("   LLM 且算力+日期齐备:", len(llm2))
print("   开源权重字段取值:", llm["Open model weights?"].value_counts(dropna=False).to_dict())

print("\n== C6 桥接（主用）==")
br = pd.read_csv(f"{BASE}/loss_benchmark_bridge_expanded.csv")
print(br.shape, "| 可比性:", br["Loss_Comparability"].value_counts().to_dict())
sub = br.dropna(subset=["Val_Loss", "LB_Average"])
if len(sub) > 2:
    r = np.corrcoef(sub["Val_Loss"], sub["LB_Average"])[0, 1]
    print(f"   Val_Loss 与 LB_Average 相关系数 = {r:.4f} (n={len(sub)})")
    print("   Loss: %.3f -> %.3f | Average: %.2f -> %.2f"
          % (sub["Val_Loss"].min(), sub["Val_Loss"].max(),
             sub["LB_Average"].min(), sub["LB_Average"].max()))

print("\n== C1 leaderboard_cleaned（主用，纯 LB 六维）==")
c1 = pd.read_csv(f"{BASE}/leaderboard_cleaned.csv")
print(c1.shape, "| 与 C2 同源列差异:", set(lb.columns) - set(c1.columns))
print("   Type 分布:", c1["Type"].value_counts().to_dict())
