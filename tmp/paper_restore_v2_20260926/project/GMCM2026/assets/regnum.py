#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字登记与核对工具

为什么需要它：
  往届优秀论文最普遍、也最致命的错误是「摘要里的数字和正文表对不上」。
  真实案例：B题优秀论文摘要问题三写 R²=0.702、RMSE=62.195，但正文表里
  这两个数分属两个不同模型——串位了。评委一眼能看出来。

  根因是"手抄数字"。本工具的做法：所有数字先登记到 CSV，写作时从 CSV 取，
  定稿前再自动核对摘要里出现的数字是否都在登记表里。

用法：
  # 1) 跑出一个数，立刻登记（推荐在求解脚本末尾调用）
  python3 assets/regnum.py add 问题一_最优目标值 1234.56 kWh \\
          --script scripts/q1_solve.py --model "问题一 完整模型" \\
          --note "对偶间隙 0.3%"

  # 2) 列出全部登记（写作时从这里取值）
  python3 assets/regnum.py list

  # 3) 定稿前核对：摘要里出现的数字是否都能在登记表里找到
  python3 assets/regnum.py check
"""

import csv
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "state", "数字登记表.csv")
TEX_PATH = os.path.join(ROOT, "paper.tex")

FIELDS = ["key", "数值", "单位", "生成脚本", "模型或场景", "登记时间", "备注"]


def _load():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _save(rows):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def add(args):
    if len(args) < 3:
        print("用法: regnum.py add <key> <数值> <单位> [--script ...] [--model ...] [--note ...]")
        return 1
    key, value, unit = args[0], args[1], args[2]
    opt = {"--script": "", "--model": "", "--note": ""}
    for i, a in enumerate(args):
        if a in opt and i + 1 < len(args):
            opt[a] = args[i + 1]

    rows = _load()
    if any(r["key"] == key for r in rows):
        print(f"⚠ key 「{key}」已存在。若要更新，请先手动删除该行——")
        print("  刻意不做自动覆盖：同名不同值往往意味着你改了模型，")
        print("  应该换个 key（如 问题一_最优目标值_v2），保留可追溯性。")
        return 1

    rows.append({
        "key": key, "数值": value, "单位": unit,
        "生成脚本": opt["--script"], "模型或场景": opt["--model"],
        "登记时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "备注": opt["--note"],
    })
    _save(rows)
    print(f"✓ 已登记：{key} = {value} {unit}")
    return 0


def list_all():
    rows = _load()
    if not rows:
        print("（登记表为空）")
        return 0
    print(f"{len(rows)} 条登记：\n")
    for r in rows:
        print(f"  {r['key']:<28} {r['数值']:>12} {r['单位']:<8} "
              f"{r['模型或场景']:<20} {r['生成脚本']}")
    return 0


def _numbers_in(text):
    """抽出文本里的数字（含小数、百分号），返回字符串集合"""
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def check():
    """核对：paper.tex 摘要段里出现的数字，是否都在登记表里"""
    if not os.path.exists(TEX_PATH):
        print(f"找不到 {TEX_PATH}")
        return 1

    with open(TEX_PATH, encoding="utf-8") as f:
        tex = f.read()

    # 截取 abstract 环境
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        print("⚠ paper.tex 里没找到 abstract 环境")
        return 1
    abstract = m.group(1)

    # 去掉注释行（% 开头到行尾），避免把注释里的说明文字当成正文数字
    abstract = "\n".join(
        line for line in abstract.splitlines() if not line.strip().startswith("%")
    )

    rows = _load()
    registered = set()
    for r in rows:
        registered |= _numbers_in(str(r["数值"]))
    # 单位、时间里的数字不应参与比对
    registered |= _numbers_in(" ".join(
        r["登记时间"] for r in rows
    ))

    found = _numbers_in(abstract)
    # 过滤明显不是结果的小整数（如"问题一"里的编号、年份）
    suspicious = sorted(
        n for n in found
        if n not in registered
        and not (n.isdigit() and len(n) <= 2)   # 1~2 位的整数多为编号/序号
        and not n.startswith("2026")
    )

    print("=" * 60)
    print("摘要数字核对")
    print("=" * 60)
    print(f"登记表条目数：{len(rows)}")
    if not rows or all(r["key"].startswith("示例_") for r in rows):
        print("\n⚠ 登记表目前只有示例行——选题后请先清理，再开始登记真实数字。")

    if not suspicious:
        print("\n✓ 摘要中的数字均可在登记表中找到（或属于序号类）")
        return 0

    print(f"\n⚠ 摘要里有 {len(suspicious)} 个数字不在登记表中：")
    for n in suspicious:
        print(f"    {n}")
    print("\n处理方式（三选一）：")
    print("  · 若这是真实结果 → 运行 `regnum.py add` 登记它")
    print("  · 若是手抄错了   → 从登记表取正确值")
    print("  · 若是序号类数字 → 忽略本条提示")
    print("\n★ 这一步的价值：往届优秀论文就是栽在摘要数字与正文表不一致上。")
    return 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 0
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "add":
        return add(args)
    if cmd == "list":
        return list_all()
    if cmd == "check":
        return check()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
