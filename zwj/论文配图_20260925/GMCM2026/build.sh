#!/bin/bash
# ============================================================
#  只编译 paper.tex，中间产物全部进 tmp/
#  用法： ./build.sh          正常编译
#         ./build.sh clean    清理全部中间产物
#
#  为什么不用 makefiles.sh：那个脚本会对目录下【每个】.tex 都跑一遍
#  完整编译链（会连带编译示例文件），且末尾 rm 会删掉根目录所有
#  中间文件，混用会导致 latexmk 判定不一致、.bbl 找错目录。
# ============================================================
set -u
cd "$(dirname "$0")"

if [ "${1:-}" = "clean" ]; then
  rm -rf tmp
  rm -f paper.aux paper.log paper.out paper.toc paper.synctex.gz \
        paper.bbl paper.blg paper.fls paper.fdb_latexmk paper.xdv
  echo "已清理中间产物"
  exit 0
fi

command -v latexmk >/dev/null || { echo "❌ 未找到 latexmk（需要 MacTeX）"; exit 1; }

echo "▶ 编译 paper.tex ..."
# -xelatex 指定引擎；-outdir=tmp 隔离产物；latexmk 自动跑
# xelatex → bibtex → xelatex → xelatex 完整链路
latexmk -xelatex -interaction=nonstopmode -outdir=tmp paper.tex > /tmp/gmcm-build.log 2>&1
RC=$?

echo
if [ $RC -ne 0 ]; then
  echo "❌ 编译失败（退出码 $RC）。错误摘要："
  grep -nE "^!|^l\.[0-9]+" /tmp/gmcm-build.log | head -20
  echo "  完整日志：tmp/paper.log"
  exit $RC
fi

# ---------------- 自检 ----------------
echo "✅ 编译成功"
PDF=tmp/paper.pdf
[ -f "$PDF" ] || PDF=paper.pdf

# 页数：从编译日志取（比 mdls 可靠，不依赖 Spotlight 索引）
PG=$(grep -oE "Output written on [^(]*\(([0-9]+) pages" tmp/paper.log 2>/dev/null \
     | grep -oE "[0-9]+" | tail -1)
echo "   页数：${PG:-?}"

# 1) 参考文献是否真的出现
if [ -f tmp/paper.blg ]; then
  N=$(grep -oE "You've used [0-9]+ entries" tmp/paper.blg | grep -oE "[0-9]+" | head -1)
  if [ "${N:-0}" = "0" ]; then
    echo "   ⚠ 参考文献【0 条】：正文还没有 \\cite，PDF 里不会有参考文献章！"
  else
    echo "   参考文献：$N 条 ✓"
  fi
fi

# 2) 交叉引用是否断裂
if [ -f tmp/paper.log ]; then
  UNDEF=$(grep -cE "LaTeX Warning: (Reference|Citation).*undefined" tmp/paper.log)
  if [ "$UNDEF" != "0" ]; then
    echo "   ⚠ 有 $UNDEF 处未定义引用，PDF 里会显示 ??"
  else
    echo "   交叉引用：无断裂 ✓"
  fi
  ERR=$(grep -cE "^! " tmp/paper.log)
  echo "   编译错误：$ERR"
fi

# 3) 表格列宽 lint
#    tabularx 必须至少含一个 X 列才会伸展到 \textwidth；没有 X 列时
#    表格按自然宽度渲染，内容全挤在左侧。纯数值表请改用
#    \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}...@{}}
echo "─── 表格列宽检查 ───"
python3 - <<'TBLCHECK'
import re, io
tex = io.open('paper.tex', encoding='utf-8').read()
bad = []
for m in re.finditer(r'\\begin\{tabularx\}\{[^}]*\}\{([^}]*)\}', tex):
    spec = m.group(1)
    if 'X' not in spec:
        line = tex[:m.start()].count('\n') + 1
        pre = tex[max(0, m.start()-200):m.start()]
        lab = re.findall(r'\\label\{([^}]*)\}', pre)
        bad.append((line, lab[-1] if lab else '?', spec))
if bad:
    print(f"   ⚠ {len(bad)} 张 tabularx 无 X 列 → 不会伸展到版心：")
    for ln, lab, spec in bad:
        print(f"      行{ln}  {lab}  {{{spec}}}")
    print("      改法：纯数值表用 tabular* 加 @{\\extracolsep{\\fill}}；")
    print("            含长文本的表给一列 X。")
else:
    print("   所有 tabularx 均含 X 列 ✓")
TBLCHECK

echo
# 4) 摘要数字一致性核对（往届优秀论文最普遍的翻车点）
if [ -f assets/regnum.py ]; then
  echo "─── 摘要数字核对 ───"
  python3 assets/regnum.py check | tail -n +4
fi

echo
echo "   PDF：$PDF"
echo "   日志：tmp/paper.log"
