import os
import sys
import re
import glob
import shlex
from typing import Tuple

def parse_vpr_out(vtr_out_path: str) -> Tuple[float, float]:
    """
    Extract area and delay from a VTR run output file (e.g., vtr.out or run.out).

    - delay: matches "Final critical path delay (least slack): <num> ns"
             or      "Final critical path delay: <num> ns"
    - area : prefers  "Total used logic block area: <num>"
             falls back to "Total routing area: <num>"

    Returns:
        (area, delay)

    Raises:
        FileNotFoundError: if the output file does not exist.
        RuntimeError: if delay or area cannot be parsed from the file.
    """
    if not os.path.exists(vtr_out_path):
        raise FileNotFoundError(f"File not found: {vtr_out_path}")
    delay = None
    total_logic = None
    used_logic = None
    routing = None
    area = None

    with open(vtr_out_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if delay is None:
                m = re.search(r"Final critical path delay (?:\(least slack\):|:)\s*([0-9.]+)\s*ns", line, re.I)
                if m:
                    delay = float(m.group(1))
            if total_logic is None:
                # e.g. "Total logic block area (Warning, ...): 4.8774e+07"
                m = re.search(r"Total logic block area.*?:\s*([0-9.eE+-]+)", line, re.I)  # <-- fixed here
                if m:
                    total_logic = float(m.group(1))

            if used_logic is None:
                m = re.search(r"Total used logic block area:\s*([0-9.eE+-]+)", line, re.I)
                if m:
                    used_logic = float(m.group(1))

            if routing is None:
                m = re.search(r"Total\s+routing\s+area\s*:\s*([0-9][0-9,._eE+-]*)", line, re.I)
                if m:
                    raw = m.group(1)
                    cleaned = raw.replace(",", "").replace("_", "")
                    cleaned = re.match(r"[0-9.+-eE]+", cleaned).group(0)  # trim trailing punctuation
                    routing = float(cleaned)

    if delay is None:
        raise RuntimeError("Failed to parse delay (ns) from VTR output.")
    if used_logic is None:
        raise RuntimeError("Failed to parse area from VTR output.")
    if routing is None:
        return used_logic, delay

    # Guard: if total_logic is missing or zero, fall back to used_logic only
    if total_logic is None or total_logic == 0:
        return used_logic, delay

    util = used_logic / total_logic
    area = used_logic + routing * util

    return area, delay


import os

def show_vtr_results(vtr_out_path: str):
    """
    调用 parse_vpr_out() 并直接打印结果。
    """
    try:
        area, delay = parse_vpr_out(vtr_out_path)
        print("=" * 50)
        print(f"VTR 输出文件: {vtr_out_path}")
        print(f"⦿ 解析得到的总面积 (MWTA): {area:.2f}")
        print(f"⦿ 解析得到的关键路径延迟 (ns): {delay:.3f}")
        print("=" * 50)
    except FileNotFoundError:
        print(f"[错误] 文件不存在: {vtr_out_path}")
    except RuntimeError as e:
        print(f"[错误] 解析失败: {e}")
    except Exception as e:
        print(f"[未知错误] {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python parse_vpr.py <vtr_out_path>")
        sys.exit(1)

    vtr_out = sys.argv[1]
    show_vtr_results(vtr_out)