#!/bin/bash
# 一键重绘三问的全部图表。
#
# 依赖：pip install matplotlib pandas numpy
# 数据：默认从团队工作区读取（见 style.py 的 DATA_ROOT）；
#      换机器时设环境变量即可：
#        Q1_DATA_ROOT=/path/to/HUAWEI-Cup/results ./run_all.sh
#      Q2/Q3 的输入按 DATA_ROOT 的父目录自动定位：
#        <repo>/Q2/03_结果/…、<repo>/Q3/04_结果/…
set -eu
cd "$(dirname "$0")"

echo "════ 问题一图表重绘 ════"
python3 plot_q1_1.py
python3 plot_q1_2.py
python3 plot_q1_3.py
python3 plot_q1_new.py

echo
echo "════ 问题二图表重绘 ════"
python3 plot_q2.py

echo
echo "════ 问题三图表重绘 ════"
python3 plot_q3.py

echo
echo "输出目录：$(cd ../ && pwd)"
ls -1 ../q1-*.png ../q2-*.png ../q3-*.png | sed 's|.*/|  |'
echo
echo "提示：任一图出现「⚠ 文字压在数据上」都必须先修好再交付。"
