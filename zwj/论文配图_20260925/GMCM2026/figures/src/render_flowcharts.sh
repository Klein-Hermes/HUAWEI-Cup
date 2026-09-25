#!/bin/bash
# 把队友的流程图 HTML 渲染成论文用的 PNG。
#
# 为什么需要这一步：交付里只有 HTML（内联 SVG），而论文 5.2.2 / 5.2.3 引用的是
# .png。本机没有 rsvg-convert / cairosvg / inkscape，唯一可用的渲染器是 Chrome。
#
# 做法：无头 Chrome 截图 → 按内容包围盒裁掉 body 的内边距 → 覆盖 figures/ 下的两个文件。
# LaTeX 正文不用改，因为引用的文件名没变。
#
# 依赖：Google Chrome（路径见下）。换机器时改 CHROME 变量即可。
# 运行：./render_flowcharts.sh
set -eu
cd "$(dirname "$0")"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# 流程图源码在团队工作区，与 LaTeX 项目不在同一棵目录树下，故用显式路径；
# 换机器时设 HUAWEI_CUP_ROOT 覆盖即可。
REPO="${HUAWEI_CUP_ROOT:-/Users/zhaomengchen/Documents/Codex/HUAWEI-Cup}"
SRC="$REPO/zwj/流程图重绘_20260924"
OUT="$(cd .. && pwd)"
TMP="$(mktemp -d)"
SCALE=3                       # 设备像素比：3 倍即约 210 dpi 于最终排版尺寸

if [ ! -x "$CHROME" ]; then
  echo "❌ 未找到 Chrome：$CHROME"
  echo "   可改用：brew install --cask google-chrome，或把 CHROME 指向其他 Chromium 内核浏览器"
  exit 1
fi
if [ ! -d "$SRC" ]; then
  echo "❌ 未找到流程图源码目录：$SRC"
  exit 1
fi

# 文件名 → 视口尺寸（CSS px）。
#   viewBox 尺寸 + body 内边距（左右各 16、上 24、下 32）；
#   宽度不足 1200 时 figure 取 viewBox 宽，超过则被 min(1200px,100%) 夹到 1200。
declare -a JOBS=(
  "q1-flow-score.html:992:896:q1-flow-score.png"
  "q1-flow-conflict.html:1232:760:q1-flow-conflict.png"
)

echo "════ 渲染流程图 ════"
for job in "${JOBS[@]}"; do
  IFS=':' read -r html w h png <<< "$job"
  raw="$TMP/${png%.png}-raw.png"
  out="$TMP/$png"

  "$CHROME" --headless --disable-gpu --hide-scrollbars --no-sandbox \
            --default-background-color=FFFFFFFF \
            --force-device-scale-factor="$SCALE" \
            --window-size="$w,$h" \
            --screenshot="$raw" \
            "file://$SRC/$html" >/dev/null 2>&1 || true

  [ -f "$raw" ] || { echo "  ✗ $html：截图失败"; continue; }

  # 裁掉四周留白（body 的 24/16/32px 内边距），按非白像素包围盒
  python3 - "$raw" "$out" <<'PY'
import sys
import numpy as np
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
a = np.asarray(im)
# 容差取 250：抗锯齿的边缘不算内容
mask = (a < 250).any(axis=-1)
rows = np.where(mask.any(axis=1))[0]
cols = np.where(mask.any(axis=0))[0]
if not len(rows) or not len(cols):
    print("      ✗ 画面全白，未裁剪"); sys.exit(1)
pad = 6                      # 留一点呼吸位，避免描边贴边
y0, y1 = max(0, rows[0] - pad), min(a.shape[0], rows[-1] + 1 + pad)
x0, x1 = max(0, cols[0] - pad), min(a.shape[1], cols[-1] + 1 + pad)
im.crop((x0, y0, x1, y1)).save(dst)
print(f"      裁剪 原始 {a.shape[1]}×{a.shape[0]} → {x1-x0}×{y1-y0}")
PY

  [ -f "$out" ] || continue
  cp "$out" "$OUT/$png"
  python3 - "$OUT/$png" <<'PY'
import sys
from PIL import Image
im = Image.open(sys.argv[1])
w, h = im.size
# 按 .92\textwidth = 430pt 排版时的有效分辨率（pt→inch 除以 72）
print(f"      ✓ {sys.argv[1].split('/')[-1]}  {w}×{h}px"
      f"  按 .92\\textwidth 折算 {w*72/430:.0f} dpi")
PY
done

rm -rf "$TMP"
echo
echo "完成。输出目录：$OUT"
