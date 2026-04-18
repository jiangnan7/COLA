#!/usr/bin/env python3
# problem_eval.py
# -*- coding: utf-8 -*-
"""VTR 与 Bambu 输出解析、MLIR 指令构造等工具。
"""

import os, re, glob, shlex
from typing import Tuple

try:
    import yaml
except Exception as e:
    raise SystemExit("请先 `pip install pyyaml`")


def run(cmd: str):
    import subprocess
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def load_params(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    space = cfg.get("space", [])
    kv = {it["name"]: it.get("default", it.get("values", [None])[0]) for it in space if isinstance(it, dict) and "name" in it}
    loop_perm = bool(kv.get("loop_interchange", False))
    unroll = int(kv.get("unroll_factor", 1))
    tile_size = int(kv.get("tile_size", 1))
    return loop_perm, unroll, tile_size


def build_mlir_cmd(mlir_opt: str, hls_opt: str, mlir_in: str, mlir_out: str, loop_perm: bool, unroll: int, tile_size: int):
    base = [
        mlir_opt,
        mlir_in,
        "--fold-memref-alias-ops",
        "--affine-loop-invariant-code-motion",
        "--affine-simplify-structures",
        "--affine-expand-index-ops",
        "--expand-strided-metadata",
        "--canonicalize",
        "--cse",
        "--affine-simplify-structures",
        "--affine-loop-fusion",
    ]
    if loop_perm:
        base.append("--affine-loop-permutation")
    base.append(f"-affine-loop-tile=tile-size={tile_size}")
    pass_pipe = f'--pass-pipeline="builtin.module(func.func(affine-loop-unroll{{unroll-factor={unroll}}}))"'
    cmd = f"{' '.join(shlex.quote(p) for p in base)} | {shlex.quote(hls_opt)} {pass_pipe} -o {shlex.quote(mlir_out)}"
    return cmd


def find_latest_bambu_xml(dir_path: str):
    print(f"[DEBUG] Scanning XML in: {dir_path}")
    if not os.path.isdir(dir_path):
        print("[DEBUG] Input directory does not exist.")
        return None
    pat = re.compile(r"^bambu_results_(\d+)\.xml$")
    candidates = []
    for name in os.listdir(dir_path):
        m = pat.match(name)
        if m:
            idx = int(m.group(1))
            p = os.path.join(dir_path, name)
            candidates.append((idx, p))
    # Prefer the largest index
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        print("[DEBUG] XML candidates (sorted):", [os.path.basename(p) for _, p in candidates])
        return candidates[0][1]
    # Fallback: sometimes there may be a plain name without index
    plain = os.path.join(dir_path, "bambu_results.xml")
    if os.path.exists(plain):
        print("[DEBUG] Fallback XML found:", os.path.basename(plain))
        return plain
    print("[DEBUG] No XML candidates found.")
    return None


def parse_bambu_cycles(xml_path: str) -> int:
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()
    m = re.search(r'<CYCLES\s+value="(\d+)"', s)
    if not m:
        raise RuntimeError(f"未在 {xml_path} 中找到 <CYCLES value=\"...\">")
    return int(m.group(1))


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
