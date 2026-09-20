# -*- coding: utf-8 -*-
"""
Matplotlib 基础演练
===================

运行方式（项目已配好 conda 环境）::

    conda activate huawei-cup
    python src/test.py
123
注意：Windows 下如果 PyCharm 的 PATH 混入了 base/CUDA 的 DLL，Matplotlib
可能在 fig.savefig() 处触发无 traceback 的 Win32 异常。脚本启动时会优先加入
当前 Conda 环境的 Library\\bin，以避免 DLL 混用。

脚本在 results/matplotlib_basics/ 下生成 7 张 PNG，每个 demo 只教一件事：

    01 画布与坐标轴    Figure / Axes 解剖、spines、grid、annotate
    02 折线图          多序列趋势、直接标注 + 图例
    03 柱状图          量级对比、顺序色（单色深浅）
    04 堆叠柱状图      部分-整体、分类色、段间留缝
    05 散点图          相关性、标记描边、序列上限
    06 直方图          分布、参考线与标注
    07 子图布局        小倍数（small multiples）、sharex/sharey

想弹窗看图，把下面的 SHOW 改成 True。
"""

from pathlib import Path
import os
import sys

# Windows 下直接从 PyCharm 启动 Conda 环境时，父进程的 PATH 可能让
# Matplotlib 误加载 base/CUDA 的 DLL，进而出现无 traceback 的 Win32 异常。
# 先显式加入当前解释器对应的 Conda DLL 目录，避免 DLL 混用。
if os.name == "nt":
    conda_dll_dir = Path(sys.prefix) / "Library" / "bin"
    if conda_dll_dir.is_dir():
        os.add_dll_directory(str(conda_dll_dir))

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# 设计令牌（design tokens）
# --------------------------------------------------------------------------
# 图表用色不靠感觉调，按"颜色承担什么职责"来取：
#   分类色 = 区分身份（谁是谁）；顺序色 = 表达量级（多深多大）。
# 这 8 个分类槽的顺序本身就是防色盲机制，务必按序取用、不要循环复用。
SURFACE = "#fcfcfb"   # 画布底色
INK = "#0b0b0b"       # 主文字
INK_2 = "#52514e"     # 次级文字（轴标签、直接标注）
MUTED = "#898781"     # 刻度等弱化文字
GRID = "#e1e0d9"      # 网格发丝线
AXIS = "#c3c2b7"      # 轴线 / 基线

SERIES = [            # 分类色：按序取，最多 8 个
    "#2a78d6",  # 1 蓝
    "#eb6834",  # 2 橙
    "#1baf7a",  # 3 青
    "#eda100",  # 4 黄
    "#e87ba4",  # 5 品红
    "#008300",  # 6 绿
    "#4a3aa7",  # 7 紫
    "#e34948",  # 8 红
]

SEQ_BLUE = [          # 顺序色：单一蓝色相，由浅到深
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_MIN = 3           # 离散量级（柱状图）最浅只能用到 SEQ_BLUE[3]，
                      # 更浅的台阶贴近底色、对比度不足 2:1

OUT = Path(__file__).resolve().parents[1] / "results" / "matplotlib_basics"
SHOW = False          # True = 每张图额外弹窗显示


# --------------------------------------------------------------------------
# 全局样式：一次配置，全脚本生效
# --------------------------------------------------------------------------
def setup_style() -> None:
    """配置 rcParams 与中文字体。这是每次画图前最该先做的事。"""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",       # 导出时自动裁掉多余白边
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,   # 中文字体下负号会变方块，必须关掉
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,       # 图例不要边框，少一层视觉噪音
        "legend.fontsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    })

    # 中文字体：Windows 上首选微软雅黑，逐个探测、找到就用
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "PingFang SC"):
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            print(f"[样式] 中文字体：{name}")
            break
    else:
        print("[样式] 警告：没找到中文字体，图中中文可能显示为方块")


def new_fig(width: float = 8.0, height: float = 4.5):
    """建一张图，返回 (fig, ax)。用面向对象的 fig/ax，别用全局状态。"""
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def dress(ax, title: str, subtitle: str = "", xlabel: str = "",
          ylabel: str = "", ygrid: bool = True) -> None:
    """统一的图表外框：标题、副标题、轴标签、网格、轴线。"""
    ax.set_title(title, loc="left", color=INK, fontsize=13,
                 fontweight="bold", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                ha="left", va="bottom", color=INK_2, fontsize=9.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ygrid:
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)         # 网格压在数据下方
    for side in ("top", "right"):      # 上、右边框是纯装饰，去掉
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)           # 去掉刻度小刺


def save(fig, name: str) -> Path:
    """导出并关闭。不 close 的话批量画图会内存泄漏。"""
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    if SHOW:
        plt.show()
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 01 画布与坐标轴
# --------------------------------------------------------------------------
def demo01_axes_anatomy() -> Path:
    """Figure 是整张画布，Axes 是里面那块坐标系——一个 Figure 可以放多个 Axes。"""
    fig, ax = new_fig(8.0, 4.5)

    x = np.linspace(0, 2 * np.pi, 200)
    ax.plot(x, np.sin(x), color=SERIES[0], linewidth=2, label="sin(x)")
    ax.plot(x, np.cos(x), color=SERIES[1], linewidth=2, label="cos(x)")

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.35, 1.35)
    dress(ax, "画布与坐标轴", "fig 是画布，ax 是坐标系；spines / grid / annotate 都挂在 ax 上",
          xlabel="相位", ylabel="幅值")
    ax.legend(loc="upper right", ncols=2)

    # annotate：xy 是箭头指的数据点，xytext 是文字落点（这里是"相对数据点偏移 30pt"）
    ax.annotate("峰值 1.0", xy=(np.pi / 2, 1.0), xytext=(20, 16),
                textcoords="offset points", color=INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1))
    return save(fig, "01_axes_anatomy")


# --------------------------------------------------------------------------
# 02 折线图：多序列 + 直接标注
# --------------------------------------------------------------------------
def demo02_line() -> Path:
    """时间趋势首选折线。2 个以上序列要图例；不超过 4 个时再在线尾直接标名字。"""
    rng = np.random.default_rng(42)
    months = np.arange(1, 13)
    online = np.array([120, 132, 128, 145, 160, 172, 168, 185, 199, 210, 226, 240], float)
    data = {
        "线上直销": online,
        "渠道分销": online * 0.64 + rng.normal(0, 8, 12),
        "门店零售": online * 0.35 + np.linspace(10, 46, 12) + rng.normal(0, 5, 12),
    }

    fig, ax = new_fig(9.0, 5.0)
    for i, (name, values) in enumerate(data.items()):
        ax.plot(months, values, color=SERIES[i], linewidth=2,
                marker="o", markersize=4, label=name)
        # 线尾直接标名字：读图的人不必在图例和数据之间来回找
        # 注意文字用墨色令牌（INK_2），身份由旁边那条彩线承担，不靠文字染色
        ax.annotate(name, xy=(months[-1], values[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center",
                    color=INK_2, fontsize=9)

    ax.set_xlim(1, 13.6)               # 右侧留白给直接标注
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}月" for m in months])
    dress(ax, "三条渠道的月度销量", "折线用于趋势；序列 ≤ 4 时线尾直接标注 + 图例双保险",
          xlabel="月份", ylabel="销量（万件）")
    ax.legend(loc="upper left")
    return save(fig, "02_line")


# --------------------------------------------------------------------------
# 03 柱状图：单序列量级 → 顺序色
# --------------------------------------------------------------------------
def demo03_bar() -> Path:
    """比大小用柱状图。量级本身有顺序，所以用单色深浅，而不是 8 种分类色。"""
    regions = ["华东", "华南", "华北", "西南", "华中", "东北", "西北", "港澳台"]
    values = np.array([482, 431, 396, 288, 265, 174, 121, 96], float)

    order = np.argsort(-values)                    # 从高到低排，读者不用自己找
    regions = [regions[i] for i in order]
    values = values[order]

    # 映射到顺序色：值越大颜色越深。最浅只到 SEQ_BLUE[SEQ_MIN]，
    # 再浅就贴底色、对比度不足 2:1 了。values 已降序，所以 i=0 取最深那阶。
    n = len(values)
    span = len(SEQ_BLUE) - 1 - SEQ_MIN
    colors = [SEQ_BLUE[len(SEQ_BLUE) - 1 - round(i * span / (n - 1))]
              for i in range(n)]

    fig, ax = new_fig(8.5, 4.5)
    bars = ax.bar(regions, values, color=colors, width=0.62)
    ax.bar_label(bars, padding=3, color=INK_2, fontsize=9,
                 labels=[f"{v:,.0f}" for v in values])

    ax.set_ylim(0, values.max() * 1.15)            # 顶部留白给数值标签
    dress(ax, "各区域销量", "单序列比大小 → 单色相由浅到深；数值直接标在柱顶",
          ylabel="销量（万件）")
    return save(fig, "03_bar")


# --------------------------------------------------------------------------
# 04 堆叠柱状图：部分-整体 → 分类色
# --------------------------------------------------------------------------
def demo04_stack() -> Path:
    """看总量怎么由几部分构成用堆叠柱。段与段之间留 2px 底色缝隙，边界才不会糊在一起。"""
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    lines = {
        "手机": np.array([210, 245, 268, 301], float),
        "穿戴": np.array([96, 112, 108, 134], float),
        "平板": np.array([64, 71, 88, 82], float),
    }

    fig, ax = new_fig(8.0, 4.5)
    bottom = np.zeros(len(quarters))
    for i, (name, values) in enumerate(lines.items()):
        ax.bar(quarters, values, bottom=bottom, width=0.58, label=name,
               color=SERIES[i], edgecolor=SURFACE, linewidth=1.5)  # 缝隙
        bottom += values

    for x, total in zip(quarters, bottom):     # 只标总量，不标每一段
        ax.text(x, total + 12, f"{total:,.0f}", ha="center",
                color=INK_2, fontsize=9)

    ax.set_ylim(0, bottom.max() * 1.15)
    dress(ax, "各季度分产品线销量", "部分-整体 → 堆叠柱 + 分类色；段间 1.5px 底色缝隙隔开",
          ylabel="销量（万件）")
    ax.legend(loc="upper left", ncols=3)
    return save(fig, "04_stack")


# --------------------------------------------------------------------------
# 05 散点图：两个变量的关系
# --------------------------------------------------------------------------
def demo05_scatter() -> Path:
    """看相关性用散点。散点要两两比较颜色，分类色最多 3 个——第 4 个起就分不清了。"""
    rng = np.random.default_rng(7)
    groups = {
        "数码": (60, 3.2, 210),
        "家居": (38, 2.4, 130),
        "服饰": (22, 1.6, 90),
    }

    fig, ax = new_fig(8.5, 5.0)
    for i, (name, (n, slope, base)) in enumerate(groups.items()):
        ad = rng.uniform(5, 60, n)
        sales = base + slope * ad + rng.normal(0, 22, n)
        ax.scatter(ad, sales, s=70, color=SERIES[i], label=name,
                   alpha=0.85, edgecolors=SURFACE, linewidths=1.2)  # 底色描边，重叠点也数得清

    ax.set_xlim(0, 66)
    dress(ax, "广告投入与销量的关系", "散点看相关性；标记 ≥8px、加底色描边，重叠处不糊",
          xlabel="广告投入（万元）", ylabel="销量（万件）")
    ax.legend(loc="upper left", ncols=3)
    return save(fig, "05_scatter")


# --------------------------------------------------------------------------
# 06 直方图：分布形态
# --------------------------------------------------------------------------
def demo06_hist() -> Path:
    """看数据分布用直方图。加一条均值参考线，形状才有参照物。"""
    rng = np.random.default_rng(2024)
    # 对数正态：典型的"长尾"分布，订单金额、用户活跃度都长这样
    orders = rng.lognormal(mean=4.0, sigma=0.55, size=2000)
    mean_value = orders.mean()
    p90 = np.percentile(orders, 90)

    fig, ax = new_fig(8.5, 4.5)
    ax.hist(orders, bins=38, color=SERIES[0], alpha=0.9,
            edgecolor=SURFACE, linewidth=0.6)

    ax.axvline(mean_value, color=SERIES[1], linewidth=2, linestyle="--")
    ax.annotate(f"均值 {mean_value:,.0f}", xy=(mean_value, ax.get_ylim()[1] * 0.86),
                xytext=(12, 0), textcoords="offset points",
                color=INK_2, fontsize=9, va="center")

    dress(ax, "订单金额分布", "2000 笔订单；对数正态长尾，均值被右侧长尾拉高",
          xlabel="订单金额（元）", ylabel="订单数")
    ax.text(0.99, 0.95, f"P90 = {p90:,.0f} 元", transform=ax.transAxes,
            ha="right", va="top", color=INK_2, fontsize=9)
    return save(fig, "06_hist")


# --------------------------------------------------------------------------
# 07 子图布局：小倍数
# --------------------------------------------------------------------------
def demo07_subplots() -> Path:
    """同一套指标要按维度拆开看时，用等分的小倍数，不要塞进一张图。"""
    rng = np.random.default_rng(11)
    months = np.arange(1, 13)
    regions = ["华东", "华南", "华北", "西南"]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5),
                             constrained_layout=True, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    trend = np.array([100, 108, 104, 119, 128, 141, 137, 152, 163, 171, 186, 198], float)
    for ax, region, scale in zip(axes.ravel(), regions, (1.0, 0.82, 0.64, 0.45)):
        values = trend * scale + rng.normal(0, 3.5, 12)
        ax.plot(months, values, color=SERIES[0], linewidth=2)
        ax.fill_between(months, values, color=SERIES[0], alpha=0.10)  # 单序列可以用面积托底
        ax.set_facecolor(SURFACE)
        ax.set_title(region, loc="left", color=INK, fontsize=11, fontweight="bold", pad=6)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    for ax in axes[-1]:                 # 只在最下一行标 x 轴刻度，避免重复
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xticklabels(["1月", "4月", "7月", "10月"])

    fig.suptitle("四个区域月度销量", x=0.005, ha="left",
                 color=INK, fontsize=13, fontweight="bold")
    fig.supxlabel("月份", color=INK_2, fontsize=10)
    fig.supylabel("销量（万件）", color=INK_2, fontsize=10)
    return save(fig, "07_subplots")


# --------------------------------------------------------------------------
DEMOS = [
    demo01_axes_anatomy,
    demo02_line,
    demo03_bar,
    demo04_stack,
    demo05_scatter,
    demo06_hist,
    demo07_subplots,
]


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[输出] {OUT}")

    for demo in DEMOS:
        path = demo()
        print(f"  ✓ {demo.__name__:<22} -> {path.name}")

    print(f"[完成] 共 {len(DEMOS)} 张图")


if __name__ == "__main__":
    main()
