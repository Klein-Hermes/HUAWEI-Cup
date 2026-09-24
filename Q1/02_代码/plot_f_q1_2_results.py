#!/usr/bin/env python3
"""Reproducible Q1.2 plots from frozen norm_* inputs and result tables.

This plotting code never reads content text. Unicode flags are used only in
the explicit A1 calibration sensitivity panels, not as malicious-text labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--skill-root", type=Path, default=Path(r"C:\Users\86147\.codex\skills\math-modeling"))
parser.add_argument("--matplotlib-path", type=Path, default=Path("tmp/q1_2_figure_deps"))
parser.add_argument("--output", type=Path, default=Path("figures/q1_2_model_final"))
args = parser.parse_args()
ROOT = args.project_root.resolve()
SKILL = args.skill_root.resolve()
MPL = args.matplotlib_path if args.matplotlib_path.is_absolute() else ROOT / args.matplotlib_path
OUT = args.output if args.output.is_absolute() else ROOT / args.output
FIG = OUT
FIG.mkdir(parents=True, exist_ok=True)
GRAY = FIG / "grayscale_previews"
GRAY.mkdir(parents=True, exist_ok=True)
MPLCONF = ROOT / "tmp" / "q1_2_mplconfig"
MPLCONF.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONF))
sys.path[:0] = [str(MPL), str(SKILL / "tools" / "figure" / "scripts")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image
from export_figure import export_figure
from setup_style import setup_style
from visual_qa import audit_layout

style_info = setup_style(journal="general", lang="zh", use_sciplots=False, constrained_layout=False)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
COL = {"blue":"#0072B2", "orange":"#D55E00", "green":"#009E73",
       "purple":"#CC79A7", "sky":"#56B4E9", "amber":"#E69F00",
       "gray":"#68717B", "dark":"#20252B", "light":"#E8EDF2"}
PRE = ROOT / "results/q1_common_preprocess/v1"
RES = ROOT / "results/q1_2_conflict_model"
MAIN = RES / "v1"
STRICT = RES / "contamination_sensitivity_official_strict"
BROAD = RES / "contamination_sensitivity_official_broad"

def csv(path: Path, **kwargs):
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)

def save(fig, stem, size, caption):
    fig.text(.01, .004, caption, ha="left", va="bottom", fontsize=7.5, color="#444444", wrap=True)
    fig.set_size_inches(*size)
    if stem == "raw_q1_2_source_counts":
        fig.subplots_adjust(left=.12,right=.99,bottom=.30,top=.88)
    elif stem == "raw_q1_2_indicator_distributions":
        fig.subplots_adjust(left=.27,right=.98,bottom=.16,top=.96)
    elif stem == "raw_q1_2_endpoint_saturation":
        fig.subplots_adjust(left=.12,right=.91,bottom=.37,top=.91)
    else:
        fig.subplots_adjust(left=.15,right=.97,bottom=.20,top=.89)
    issues = audit_layout(fig)
    qa.append({"figure": stem, "issues": [{"severity":s,"message":m} for s,m in issues]})
    for severity, message in issues:
        print(f"VISUAL_QA {severity} {stem}: {message}")
    export_figure(fig, str(FIG / stem), formats=["svg", "png"],
                  size_inches=size, dpi=301, grayscale_preview=False)
    with Image.open(FIG / f"{stem}.png") as image:
        image.convert("L").save(GRAY / f"{stem}_grayscale.png", dpi=(301,301))
    plt.close(fig)

qa = []
indicators = csv(PRE / "quality_indicator_dictionary.csv")["indicator"].astype(str).tolist()
short_map = {
    "fineweb_edu":"FineWeb", "fluency_en":"Fluency", "modernbert_cleanliness":"MB-clean",
    "modernbert_readability":"MB-read", "modernbert_reasoning":"MB-reason", "modernbert_professionalism":"MB-pro",
    "dsir_books":"Books", "dsir_wiki":"Wiki", "dsir_math":"Math", "qurater":"Qurater", "ad_en":"AD-en",
    "rps_doc_word_count":"Words", "rps_doc_num_sentences":"Sent.", "rps_doc_unigram_entropy":"Entropy",
    "rps_doc_frac_unique_words":"Unique", "rps_doc_frac_no_alph_words":"Non-alpha",
    "rps_doc_frac_chars_top_2gram":"2-gram", "rps_doc_frac_chars_top_3gram":"3-gram",
    "rps_lines_uppercase_letter_fraction":"Uppercase", "rps_lines_ending_with_terminal_punctution_mark":"Terminal",
    "rps_lines_numerical_chars_fraction":"Numeric", "rps_doc_mean_word_length":"Mean-len",
}
short = [short_map.get(x, x) for x in indicators]
a1 = csv(PRE / "a1_preprocessed.csv.gz", compression="gzip")
scores = csv(MAIN / "sample_scores.csv.gz", compression="gzip")
summary = csv(MAIN / "domain_summary.csv")
thresholds = csv(MAIN / "thresholds.csv")
domains = ["arxiv","book","c4","commoncrawl","github","stackexchange","wikipedia"]

# Raw data: sample composition, indicator distributions, endpoint saturation.
cnt = scores.groupby(["dataset","source_domain"], observed=True).size().reset_index(name="n")
fig, ax = plt.subplots(figsize=(9.5,4.8))
colors = [COL["blue"] if d=="a1" else COL["green"] if d=="a2" else COL["orange"] for d in cnt.dataset]
bars = ax.bar(np.arange(len(cnt)), cnt.n, color=colors)
ax.set_xticks(np.arange(len(cnt)), cnt.dataset.str.upper()+" / "+cnt.source_domain, rotation=50, ha="right")
ax.set_yscale("log"); ax.set_ylabel("样本数（对数轴）"); ax.set_title("A1–A3 数据集与来源域样本量")
ax.grid(axis="y", color="#D9DEE5", linewidth=.7); ax.set_axisbelow(True)
for b,n in zip(bars,cnt.n): ax.text(b.get_x()+b.get_width()/2,b.get_height()+cnt.n.max()*.012,f"{n:,}",ha="center",fontsize=7)
save(fig,"raw_q1_2_source_counts",(9.5,4.8),"冻结预处理中的实际样本数；纵轴为对数刻度以同时显示小域与大型扩展集；A1 有七个来源域，A2/A3 分别为 arxiv/github 扩展集。")

metric_cols = ["norm_"+x for x in indicators]
vals = [a1[c].dropna().to_numpy(float) for c in metric_cols]
fig, ax = plt.subplots(figsize=(8.6,8.2))
bp=ax.boxplot(vals,orientation="horizontal",showfliers=False,patch_artist=True,widths=.66,
              medianprops={"color":COL["dark"],"linewidth":1.1})
for i,b in enumerate(bp["boxes"]): b.set_facecolor(COL["sky"] if i<11 else COL["amber"]); b.set_alpha(.72)
ax.set_yticks(np.arange(1,len(short)+1),short); ax.invert_yaxis(); ax.set_xlim(0,1)
ax.set_xlabel("冻结 norm 指标（统向、0–1）",labelpad=10); ax.set_title("A1 的 22 个质量指标分布")
ax.grid(axis="x",color="#D9DEE5",linewidth=.7); ax.set_axisbelow(True)
save(fig,"raw_q1_2_indicator_distributions",(8.6,8.2),"箱体为四分位距，中线为中位数，隐藏离群点仅为可读性；这些是公共预处理变换后的信号，不是原始单位。")

sat=[]
for d,g in a1.groupby("source_domain",observed=True,sort=False):
    x=g[metric_cols]; sat.append(((x.eq(0)|x.eq(1)).sum()/x.notna().sum()).fillna(0).to_numpy(float))
fig,ax=plt.subplots(figsize=(10.0,5.4))
sat_values=np.vstack(sat)*100
cmap=plt.get_cmap("cividis")
norm=mpl_colors.Normalize(vmin=0,vmax=max(5,float(np.nanpercentile(sat_values,98))))
for row in range(len(domains)):
    for col in range(len(short)):
        ax.add_patch(Rectangle((col,row),1,1,facecolor=cmap(norm(sat_values[row,col])),
                               edgecolor="white",linewidth=.25))
ax.set_xlim(0,len(short)); ax.set_ylim(len(domains),0)
ax.set_xticks(np.arange(len(short))+.5,short,rotation=90,ha="right"); ax.set_yticks(np.arange(len(domains))+.5,domains)
ax.tick_params(axis="x",labelsize=7)
ax.set_xlabel("指标"); ax.set_ylabel("来源域"); ax.set_title("A1 各来源域 norm 指标端点占比")
# Draw a vector color scale from rectangles; Matplotlib's default colorbar may
# embed a raster gradient in SVG, which defeats the editable-vector contract.
cax=fig.add_axes([.935,.39,.022,.42])
for i in range(80):
    y0=i/80
    cax.add_patch(Rectangle((0,y0),1,1/80,facecolor=cmap(norm(norm.vmin+y0*(norm.vmax-norm.vmin))),edgecolor="none"))
cax.set_xlim(0,1); cax.set_ylim(0,1); cax.set_xticks([])
ticks=np.linspace(0,1,6); cax.set_yticks(ticks,[f"{v:.0f}" for v in ticks*norm.vmax])
cax.yaxis.tick_right(); cax.tick_params(axis="y",labelsize=7,length=2)
cax.set_title("端点占比 (%)",fontsize=7,pad=4)
save(fig,"raw_q1_2_endpoint_saturation",(10.0,5.4),"每格为域内有效指标值等于 0 或 1 的样本比例；是预处理饱和诊断，不代表数据错误或污染。")

# Process: conflict distribution, pair diagnostics, marker-exclusion recalibration.
a1s=scores[scores.dataset.eq("a1")]
groups=[a1s.loc[a1s.source_domain.eq(d),"conflict_intensity"].dropna().to_numpy() for d in domains]
fig,ax=plt.subplots(figsize=(7.8,4.5))
vp=ax.violinplot(groups,positions=np.arange(1,8),showmedians=True,showextrema=False,widths=.82)
for body in vp["bodies"]: body.set_facecolor(COL["blue"]); body.set_edgecolor(COL["blue"]); body.set_alpha(.4)
vp["cmedians"].set_color(COL["dark"])
t=thresholds[thresholds.threshold_rule.eq("a1_source_domain_q95_linear")].set_index("source_domain").reindex(domains)
for i,d in enumerate(domains,1): ax.scatter(i,t.loc[d,"threshold"],marker="D",s=22,color=COL["orange"],label="A1 域内 Q95" if i==1 else None,zorder=3)
ax.set_xticks(np.arange(1,8),domains,rotation=25,ha="right"); ax.set_ylabel("样本连续冲突度 D_s")
ax.set_title("A1 来源域内冲突度与冻结阈值"); ax.grid(axis="y",color="#D9DEE5",linewidth=.7); ax.legend(frameon=False)
save(fig,"process_q1_2_conflict_distribution",(7.8,4.5),"小提琴宽度为密度，中线为中位数，橙菱形为 A1 域内经验 Q95；高冲突定义为 D_s 严格大于阈值。")

p=csv(MAIN/"pair_diagnostics.csv"); p=p[p.dataset.eq("a1")]
pm=p.groupby(["indicator_a","indicator_b"],as_index=False).agg(gap=("mean_abs_gap","mean"),rho=("spearman_rho","mean"),nd=("source_domain","nunique"))
if len(pm)!=231 or not pm.nd.eq(7).all(): raise ValueError("A1 宏平均诊断需含 231 对、每对七域")
top=pm.nlargest(10,"gap").index
fig,ax=plt.subplots(figsize=(7.1,5.1))
ax.scatter(pm.gap,pm.rho,s=23,color=COL["sky"],alpha=.72,edgecolor="white",linewidth=.35,label="其余指标对")
ax.scatter(pm.loc[top,"gap"],pm.loc[top,"rho"],s=32,color=COL["orange"],edgecolor="white",linewidth=.5,label="分差 Top 10")
ax.axhline(0,color=COL["gray"],lw=.8,ls="--"); ax.set_xlabel("七域宏平均绝对分差"); ax.set_ylabel("七域宏平均 Spearman ρ")
ax.set_title("231 个指标对的分歧结构"); ax.grid(color="#E2E5E9",linewidth=.6); ax.legend(frameon=False)
save(fig,"process_q1_2_pair_conflict",(7.1,5.1),"每点为一对指标；横轴是七域均值的宏平均，纵轴是七域 Spearman ρ 描述性均值。Top 10 仅按分差排序，不表示显著。")

sens={}
for label,path in [("严格 Cf/Cc 428",STRICT),("扩展规则 463",BROAD)]:
    q=csv(path/"contamination_domain_sensitivity.csv")
    sens[label]=q[(q.row_type.eq("a1_reference"))&q.dataset.eq("a1")].set_index("source_domain").reindex(domains)
fig,ax=plt.subplots(figsize=(8.1,4.7)); y=np.arange(7)
for (label,q),color,off,marker in zip(sens.items(),[COL["blue"],COL["orange"]],[-.12,.12],["o","s"]):
    xx=q.threshold_shift_clean_minus_main.to_numpy(float)*1000
    ax.scatter(xx,y+off,color=color,marker=marker,s=36,label=label,zorder=3)
    for xi,yi in zip(xx,y+off): ax.plot([0,xi],[yi,yi],color=color,alpha=.28,lw=.8)
ax.axvline(0,color=COL["dark"],lw=.8); ax.set_yticks(y,domains); ax.invert_yaxis()
ax.set_xlabel("清洁 A1 Q95 - 全量 A1 Q95 (x 1e-3)"); ax.set_title("Unicode 标记剔除后的域阈值变化")
ax.grid(axis="x",color="#E2E5E9",linewidth=.6); ax.legend(frameon=False)
save(fig,"process_q1_2_contamination_sensitivity",(8.1,4.7),"严格/扩展规则分别剔除 428/463 条 A1 标记记录后重估阈值。命中是完整性审计标记，不是恶意标签；敏感性结果不证明上游评分器免疫原文污染。")

# Results: candidate score effect, bootstrap intervals, ranking stability.
sub=summary[summary.dataset.eq("a1")].set_index("source_domain").reindex(domains); x=np.arange(7)
fig,ax=plt.subplots(figsize=(8.2,4.5))
for label,medc,lwc,upc,color,off in [
    ("高冲突","delta_q_median_high_conflict","delta_q_q25_high_conflict","delta_q_q75_high_conflict",COL["orange"],-.12),
    ("其他样本","delta_q_median_other","delta_q_q25_other","delta_q_q75_other",COL["blue"],.12)]:
    med=sub[medc].to_numpy(float); lo=med-sub[lwc].to_numpy(float); hi=sub[upc].to_numpy(float)-med
    ax.errorbar(x+off,med,yerr=np.vstack([lo,hi]),fmt="o",color=color,capsize=3,markersize=5,lw=1.1,label=label)
ax.axhline(0,color=COL["dark"],lw=.8,ls="--"); ax.set_xticks(x,domains,rotation=25,ha="right")
ax.set_ylabel("Delta Q = Q_H - Q_eq"); ax.set_title("Huber 候选分对综合分的改变量")
ax.grid(axis="y",color="#E2E5E9",linewidth=.6); ax.legend(frameon=False)
save(fig,"result_q1_2_score_effect",(8.2,4.5),"点为 ΔQ 中位数，线为第 25–75 百分位距；按 A1 域内 Q95 分组。仅比较候选分，不代表通过人工盲评效度验证。")

ext=csv(MAIN/"external_validation.csv").set_index("extension_dataset").reindex(["a2","a3"]).reset_index()
fig,axes=plt.subplots(1,2,figsize=(8.0,4.4))
for ax,val,low,high,title,scale,xlab in [
    (axes[0],"median_conflict_diff","median_diff_ci_low","median_diff_ci_high","中位 D 差值",1000,"A2/A3 - A1 中位 D (x 1e-3)"),
    (axes[1],"high_conflict_rate_diff","rate_diff_ci_low","rate_diff_ci_high","高冲突率差值",100,"A2/A3 - A1 高冲突率（百分点）")]:
    v=ext[val].to_numpy(float)*scale; lo=ext[low].to_numpy(float)*scale; hi=ext[high].to_numpy(float)*scale
    ax.errorbar(v,np.arange(2),xerr=np.vstack([v-lo,hi-v]),fmt="o",color=COL["blue"],capsize=4,markersize=5,lw=1.2)
    ax.axvline(0,color=COL["dark"],lw=.8,ls="--"); ax.set_yticks([0,1],["A2 · arxiv","A3 · github"]); ax.invert_yaxis()
    ax.set_xlabel(xlab); ax.set_title(title); ax.grid(axis="x",color="#E2E5E9",lw=.6)
fig.suptitle("同来源域扩展复核：差值区间均覆盖 0",fontsize=12)
save(fig,"result_q1_2_external_validation",(8.0,4.4),"点为扩展集减 A1 同域的差值；线段为域内有放回 bootstrap 2000 次 percentile 95% CI（seed 20260923）。A2/A3 缺原文，只能作冻结评分表条件下的数值复核。")

ranks=[]
for label,path,color in [("严格剔除 428",STRICT,COL["green"]),("扩展剔除 463",BROAD,COL["purple"])]:
    q=csv(path/"pair_ranking_sensitivity.csv"); z=q[q.comparison.eq("a1_macro_full_vs_clean")].iloc[0]
    ranks.append((label,float(z.pair_rank_spearman),int(z.top10_overlap_n),color))
for _,z in ext.iterrows(): ranks.append((f"{z.extension_dataset.upper()} 同域复核",float(z.pair_rank_spearman),int(z.top10_overlap_n),COL["orange"] if z.extension_dataset=="a2" else COL["blue"]))
fig,ax=plt.subplots(figsize=(7.7,4.5))
for label,rho,topn,color in ranks:
    xx=(1-rho)*10000; ax.scatter(xx,topn,s=64,color=color,edgecolor="white",lw=.7,zorder=3)
    right_side = label.startswith("A3")
    ax.annotate(f"{label} ρ={rho:.6f}",(xx,topn),xytext=(-6 if right_side else 6,7),
                textcoords="offset points",ha="right" if right_side else "left",fontsize=8)
ax.set_xlim(left=0); ax.set_ylim(8.4,10.6); ax.set_yticks([9,10],["9/10","10/10"])
ax.set_xlabel("1 - 指标对排序 Spearman rho (x 1e-4)"); ax.set_ylabel("Top-10 重合")
ax.set_title("指标对排序的污染敏感性与同域迁移稳定性"); ax.grid(color="#E2E5E9",lw=.6)
save(fig,"result_q1_2_pair_rank_stability",(7.7,4.5),"横轴为排序差异，纵轴为 Top-10 重合数；A2/A3 仅同域比较。稳定性不能证明原文清洁或评分有效。")

# Method-flow drawings use Matplotlib shapes so every node and edge is editable.
def draw_node(ax, xy, wh, text, kind="box", face="#F4F6F8", edge="#59636E", fs=8.5):
    x0,y0=xy; w,h=wh
    if kind=="round": art=FancyBboxPatch((x0,y0),w,h,boxstyle="round,pad=.012,rounding_size=.018",facecolor=face,edgecolor=edge,lw=1)
    elif kind=="data": art=Polygon([(x0+.025,y0),(x0+w,y0),(x0+w-.025,y0+h),(x0,y0+h)],closed=True,facecolor=face,edgecolor=edge,lw=1)
    else: art=Rectangle((x0,y0),w,h,facecolor=face,edgecolor=edge,lw=1)
    ax.add_patch(art); ax.text(x0+w/2,y0+h/2,text,ha="center",va="center",fontsize=fs,linespacing=1.2,color=COL["dark"])
    return (x0,y0,w,h)

def connect(ax,a,b,color=COL["gray"],rad=0):
    x,y,w,h=a; X,Y,W,H=b
    start=(x+w,y+h/2); end=(X,Y+H/2)
    if X<=x+w: start=(x+w/2,y); end=(X+W/2,Y+H)
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=10,lw=1,color=color,connectionstyle=f"arc3,rad={rad}"))

fig,ax=plt.subplots(figsize=(10,4.8)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
ax.set_title("F 题任务依赖总览（当前仅 Q1.2 已完成计算）",pad=12)
na=draw_node(ax,(.03,.70),(.18,.14),"A1–A3\n质量信号","data",face="#E8F1F8")
nb=draw_node(ax,(.03,.43),(.18,.14),"B 组\n训练日志","data")
nc=draw_node(ax,(.03,.16),(.18,.14),"C 组\n历史评测","data")
n1=draw_node(ax,(.29,.64),(.27,.26),"问题一\nQ1.1 质量评分（候选）\nQ1.2 冲突分析（已计算）\nQ1.3 配比与 Loss（待推进）",face="#E4F0F7",edge=COL["blue"],fs=8)
n2=draw_node(ax,(.62,.64),(.16,.26),"问题二\n标度律\n待推进")
n3=draw_node(ax,(.82,.64),(.15,.26),"问题三\n资源优化\n待推进",fs=8)
n4=draw_node(ax,(.53,.18),(.27,.25),"问题四\n技术进步分解\n与前沿预测\n待推进")
no=draw_node(ax,(.84,.18),(.13,.25),"最终结果\n参数、区间\n情景与预测","round",fs=8)
for aa,bb,rad in [(na,n1,0),(n1,n2,0),(n2,n3,0),(nc,n4,-.12),(n2,n4,.10),(n3,no,.12),(n4,no,0)]: connect(ax,aa,bb,rad=rad)
# Route B input below the Q1 block so the arrow does not cross a task node.
ax.plot([.21,.57,.61],[.50,.50,.63],color=COL["gray"],lw=1)
ax.add_patch(FancyArrowPatch((.61,.63),(.62,.68),arrowstyle="-|>",mutation_scale=10,lw=1,color=COL["gray"]))
ax.text(.5,.04,"Q1.1 候选评分尚需盲评效度证据；本阶段不把 Q1.2 候选分称为最终质量真值。",ha="center",fontsize=8,color=COL["gray"])
save(fig,"flow_overall_model",(10,4.8),"仅表达可见题面中的任务依赖与当前状态；后续问题的具体模型、参数和结果尚未声称完成。")

fig,ax=plt.subplots(figsize=(10,6.6)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
ax.set_title("Q1.2 冲突分析与污染敏感性流程",pad=12)
ni=draw_node(ax,(.03,.76),(.19,.13),"冻结公共预处理\nA1–A3 norm_*","data",face="#E8F1F8")
npair=draw_node(ax,(.28,.76),(.19,.13),"指标对秩相关\n与绝对分差")
nd=draw_node(ax,(.53,.76),(.17,.13),"样本冲突度\nD_s")
nt=draw_node(ax,(.76,.76),(.20,.13),"A1 域内 Q95\n阈值与区间",face="#E8F1F8")
for aa,bb in [(ni,npair),(npair,nd),(nd,nt)]: connect(ax,aa,bb)
ne=draw_node(ax,(.69,.48),(.27,.15),"冻结阈值/排序\n复核 A2/A3 同域",face="#E9F4EF",edge=COL["green"])
nq=draw_node(ax,(.37,.48),(.25,.15),"高/非高冲突组\n比较 Q_H - Q_eq",face="#F5EFE6",edge=COL["amber"])
nr=draw_node(ax,(.43,.22),(.31,.15),"汇总差值区间、排序稳定性\n与适用边界","round",face="#E8F1F8",edge=COL["blue"])
connect(ax,nt,ne,rad=.08); connect(ax,nt,nq,rad=.06); connect(ax,ne,nr); connect(ax,nq,nr,rad=-.06)
naudit=draw_node(ax,(.03,.45),(.25,.16),"A1 Unicode 审计\n428 / 463 标记","data",face="#FFF4E5",edge=COL["orange"])
nsens=draw_node(ax,(.03,.23),(.25,.14),"分别重估 A1 阈值、权重\nHuber δ 与指标对排序",face="#FFF8EF",edge=COL["orange"],fs=8)
nc=draw_node(ax,(.03,.05),(.25,.11),"比较全量与清洁校准","round",face="#FFF8EF",edge=COL["orange"],fs=8)
connect(ax,naudit,nsens); connect(ax,nsens,nc); connect(ax,nc,nr,color=COL["orange"],rad=-.10)
ax.text(.54,.06,"标记只用于敏感性分组；原文不进入统计，不据此认定恶意，也不证明评分器免疫污染。",ha="center",fontsize=8,color=COL["gray"])
save(fig,"flow_q1_2_model",(10,6.6),"主分支只读冻结 norm_*；污染支路仅重估 A1 校准并比较稳定性。A2/A3 无原文字段，外部复核限于冻结评分表条件下。")

contract="""# Q1.2 图表契约\n\n核心结论：冻结评分信号存在可量化分歧；剔除 A1 Unicode 标记记录后，可复核域阈值、指标对排序和候选分变化，但不能证明上游评分器免疫原文污染。\n\n- 后端：Python / Matplotlib，遵循项目科研绘图样式、导出与视觉检查流程。\n- 证据：冻结 `norm_*`；Q1.2 主结果；官方严格/扩展 Unicode 敏感性结果。绘图程序不读取 `content`。\n- 原始数据图：来源域样本量、22 个 norm 指标分布、A1 端点饱和率。\n- 过程图：A1 域内冲突度与 Q95、231 个指标对、428/463 标记剔除后的阈值变化。\n- 结果图：ΔQ 分布、A2/A3 bootstrap 差值区间、指标对排序稳定性。\n- 统计口径：箱体为 IQR；小提琴为密度；差值区间为 2000 次域内 bootstrap percentile 95% CI；不作因果或恶意污染推断。\n- 流程图：总体依赖图标明未完成分支；Q1.2 图严格对应已实现数据流。\n- 题意门禁：只将可见正式题面作为题意依据；PDF 低可见度文本留在隔离审计记录，不进入 Q1.2 模型或提示词。\n- 数据边界：A1 Unicode 命中是完整性风险标记，不是恶意标签；A2/A3 无原文字段；本敏感性分析不能证明上游评分器免疫原文污染。\n- 导出：SVG 矢量文字；PNG 按 301 DPI 导出以满足至少 300 DPI 的严格门槛，并生成灰度预览。\n"""
(FIG/"图表契约.md").write_text(contract,encoding="utf-8")
(FIG/"visual_qa.json").write_text(json.dumps({"style":style_info,"figures":qa},ensure_ascii=False,indent=2),encoding="utf-8")
if any(i["severity"]=="FAIL" for row in qa for i in row["issues"]): raise SystemExit("Visual QA has FAIL; revise plots and rerun.")
print(f"Wrote Q1.2 figures to {FIG}")
