#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bambu_pipeline.py

- Artifacts live next to the input MLIR: foo.mlir → foo.ll / foo.tb.xml
- If --top is not provided, the first func.func name is auto-detected.
- Testbench XML is generated ONLY from the manual --arg-types string you pass.
  * Float (f16/bf16/f32/f64) → "1.0"
  * Otherwise → "0"
  * For memref/tensor, the value is expanded by element count; unknown dims are 1.
- Bambu is invoked directly via a command string (no extra wrapper).

Example:
  python3 bambu_pipeline.py \
      --mlir path/to/forward.mlir \
      --bambu /path/to/bambu \
      --mliropt /path/to/mlir-opt \
      --translate /path/to/mlir-translate \
      --arg-types "%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>" \
      --device xc7vx330t-1ffg1157 --clock 2.0
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

import os
os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")

FLOAT_PAT = re.compile(r'\b(f16|bf16|f32|f64)\b')

def run_shell(cmd: str, check: bool = True, cwd: str | None = None):
    """Run a shell command and stream combined stdout/stderr."""
    result = subprocess.run(cmd, shell=True, cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="")
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")
    return result

def parse_first_top_name(mlir_text: str) -> str | None:
    """Extract the name of the first func.func as the default top."""
    m = re.search(r'func\.func\s+@([A-Za-z0-9_]+)\s*\(', mlir_text)
    return m.group(1) if m else None

def split_top_level_commas(s: str) -> list[str]:
    """Split by commas while ignoring commas inside <...>."""
    parts, cur, depth = [], [], 0
    for ch in s:
        if ch == '<':
            depth += 1
        elif ch == '>':
            depth = max(0, depth - 1)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return parts

def parse_types_from_cli(arg_types_str: str) -> list[str]:
    """
    Parse a manual --arg-types string like:
      "%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>"
    into a clean list of MLIR types:
      ["memref<1024xi32>", "memref<1024xi32>", "memref<1024xi32>"]
    """
    types: list[str] = []
    for chunk in split_top_level_commas(arg_types_str):
        # Accept forms like "%arg0: memref<...>" or just "memref<...>"
        m = re.search(r':\s*([A-Za-z0-9_<>\?\sx,]+)$', chunk)
        ty = m.group(1).strip() if m else chunk.strip()
        ty = re.sub(r'\{.*?\}', '', ty).strip()   # drop attribute blocks
        ty = re.sub(r'\s+', '', ty)               # compress spaces
        types.append(ty)
    return types

def count_num_elements(type_str: str) -> int:
    """
    Count elements for memref/tensor by reading the shape.
    Unknown dims count as 1. Scalars/non-containers return 1.
    """
    m = re.search(r'<(.*?)x[fib]', type_str)
    if not m:
        return 1
    shape_part = m.group(1)
    dims = [d for d in shape_part.split('x') if d]
    count = 1
    for d in dims:
        if d == '?' or not d.isdigit():
            count *= 1
        else:
            count *= int(d)
    return count

def is_float_type(type_str: str) -> bool:
    """Return True if type contains a float element type."""
    return bool(FLOAT_PAT.search(type_str))

def generate_testbench_xml(arg_types: list[str], xml_path: Path):
    """
    Build a minimal testbench XML.
    P0, P1, ... are on separate lines; memref/tensor expanded to element count.
    Float → "1.0"; otherwise → "1".
    """
    lines = []
    for i, t in enumerate(arg_types):
        num_elems = count_num_elements(t)
        val = "12.0" if is_float_type(t) else "1"
        vals = ",".join([val] * num_elems)
        lines.append(f'    P{i}="{vals}"')  # indentation for readability

    # Join attributes with line breaks
    attrs = "\n".join(lines)
    xml_content = f"""<?xml version="1.0"?>
<function>
  <testbench
{attrs}
  />
</function>
"""
    xml_path.write_text(xml_content, encoding="utf-8")
    print(f"[INFO] Wrote testbench XML → {xml_path}")

def main():
    ap = argparse.ArgumentParser(description="MLIR → LLVM.ll → Bambu runner (outputs next to the .mlir)")
    ap.add_argument("--mlir", type=str, required=True, help="Path to input MLIR file")
    ap.add_argument("--ll", type=str, default=None, help="Optional pre-generated LLVM .ll (skips MLIR lowering)")
    ap.add_argument("--top", type=str, default=None, help="Top function name (auto-detected if omitted)")
    ap.add_argument("--mliropt", type=str, default="mliropt", help="Path to mlir-opt (or compatible polygeist-opt)")
    ap.add_argument("--translate", type=str, default="translate", help="Path to mlir-translate")
    ap.add_argument("--bambu", type=str, help="Path to bambu binary (e.g., /path/to/bambu or AppImage)")
    ap.add_argument("--device", type=str, default="xc7vx330t-1ffg1157", help="Bambu --device")
    ap.add_argument("--clock", type=float, default=10, help="Bambu --clock-period (ns)")
    ap.add_argument("--extra-bambu", type=str, default="", help="Extra flags passed to Bambu verbatim")
    ap.add_argument("--arg-types", type=str, 
                    help="Manual parameter types string, e.g. "
                         "\"%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>\"")
    args = ap.parse_args()

    # Hard-coded paths (optional; keep or remove as you prefer)
    args.bambu = args.bambu or "/home/edalab/EDA-DSE/bambu/bambu-2024.10.AppImage"
    args.mliropt = "/home/edalab/EDA-DSE/bambu/polygeist-opt"
    args.translate = "/home/edalab/EDA-DSE/bambu/mlir-translate"

    mlir_path = Path(args.mlir).resolve()
    if not mlir_path.exists():
        print(f"ERROR: MLIR file not found: {mlir_path}", file=sys.stderr)
        sys.exit(1)

    # Output files live next to the MLIR with suffix changes
    ll_path = Path(args.ll).resolve() if args.ll else mlir_path.with_suffix(".ll")
    tb_xml_path = mlir_path.with_suffix(".tb.xml")

    # Read MLIR and determine top if needed
    mlir_text = mlir_path.read_text(encoding="utf-8", errors="ignore")
    top_name = args.top or parse_first_top_name(mlir_text)
    if not top_name:
        print("ERROR: Cannot determine top function name. Please pass --top.", file=sys.stderr)
        sys.exit(1)

    # Parse manual arg types and write XML
    try:
        manual_arg_types = parse_types_from_cli(args.arg_types)
        generate_testbench_xml(manual_arg_types, tb_xml_path)
    except Exception as e:
        print(f"[ERROR] Failed to create testbench XML from --arg-types: {e}", file=sys.stderr)
        sys.exit(1)

    # MLIR → LLVM (unless --ll provided)
    if args.ll is None:
        print("[INFO] Lowering MLIR → LLVM IR")
        hls_cmd = f"/home/edalab/EDA-DSE/bambu/polygeist-opt  --lower-affine {shlex.quote(str(mlir_path))} | /home/edalab/circt/build/bin/circt-opt --flatten-memref"
        opt_cmd = f"{shlex.quote(args.mliropt)} --lower-affine --convert-polygeist-to-llvm=\"use-c-style-memref=1\" "
        trans_cmd = f"{shlex.quote(args.translate)} --mlir-to-llvmir --opaque-pointers=0 -o {shlex.quote(str(ll_path))}"
        lower_cmd = f"{hls_cmd} | {opt_cmd} | {trans_cmd}"
        run_shell(lower_cmd, check=True)
    
    memory_policy = "NO_BRAM" if top_name == "conv" else "ALL_BRAM"

    # Build and run Bambu
    bambu_bits = [
        shlex.quote(args.bambu),
        "-v3",
        "--print-dot",
        "-lm",
        "--soft-float",
        "--compiler=I386_CLANG12",
        "-O1",
        f"--device={shlex.quote(args.device)}",
        f"--clock-period={args.clock}",
        "--experimental-setup=BAMBU-BALANCED-MP",# BAMBU-BALANCED-MP BAMBU-AREA-MP
        "--channels-number=2",
        f"--memory-allocation-policy={memory_policy}",
        "--disable-function-proxy",
        f"--top-fname={shlex.quote(top_name)}",
        "--simulate",
        "--simulator=VERILATOR",
        f"--generate-tb={shlex.quote(str(tb_xml_path))}",
        shlex.quote(str(ll_path)),
    ]
    if args.extra_bambu:
        bambu_bits.append(args.extra_bambu)

    bambu_cmd = " ".join(bambu_bits)

    print("\n[run] Launching Bambu...")
    try:
        run_shell(bambu_cmd, check=True, cwd=str(mlir_path.parent))
    except Exception as e:
        print(f"[error] Bambu failed: {e}", file=sys.stderr)
        sys.exit(2)
    print("[done] Bambu finished.")

if __name__ == "__main__":
    #python ./dse/run_bambu.py --mlir ./sample/matmul32x32/matmul.mlir --top forward --arg-types "%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>"
    main()
