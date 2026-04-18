#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import argparse
from typing import Any, Dict, List, Optional

import torch
import numpy as np

# 你的项目内模块
# from problem import Problem
from explorer import Explorer

try:
    import yaml
except Exception:
    print("Missing dependency: pyyaml. Please `pip install pyyaml`.", file=sys.stderr)
    raise

# -------- 默认硬件/数值类型 --------
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

bench_list = ['32x32']
bench_list = ['64x64']
bench_list = ['128x128']
bench_list = ['256x256']
top_list = ['matmul']

SAMPLE_ROOT = "/home/edalab/EDA-DSE/sample"


def sample_dir(top: str, bench: str) -> str:
    # e.g. "matmul" + "32x32" -> /.../sample/matmul32x32
    return os.path.join(SAMPLE_ROOT, f"{top}{bench}")


for idx in range(len(bench_list)):


    res_root     = sample_dir(top_list[0], bench_list[idx])                  # /home/edalab/EDA-DSE/sample/matmul32x32
    cfg_yaml = os.path.join(res_root, "config.yaml")
    
    if not os.path.exists(cfg_yaml):
        raise FileNotFoundError(f"No {cfg_yaml}")
    with open(cfg_yaml, "r", encoding="utf-8") as f:
        ydoc = yaml.safe_load(f) or {}
    space_spec = ydoc.get("space", [])
    kernel_top = str(ydoc.get("top", top_list[0]))


    seeds = [16, 26, 36, 46, 56, 66]
    seeds = [36]

    for seed in seeds:
        # 固定随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)

        res_dir = os.path.join(res_root, f'result_{seed}')
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)


        path    = f'../dataset/matmul_num/{bench_list[idx]}_dataset_num.csv'
        ppa_res = f'{res_dir}/{bench_list[idx]}_ppa.csv'
        pkl_path= f'../dataset/hls_latent/{bench_list[idx]}_ds_latent_epoch_end.pkl'

        os.environ["DSE_SAMPLE_DIR"] = res_root          # 供评估脚本定位样本根
        os.environ["DSE_CONFIG_YAML"] = cfg_yaml     # config.yaml 路径（可能不存在）
        os.environ["DSE_TOP"] = top_list[0]              # 顶层函数名（来自 config.yaml: top: forward）
        os.environ["DSE_RUN_ROOT"] = res_dir         # 本次结果目录（让评估器落盘在这里）
        os.environ["DSE_BENCH"] = bench_list[idx]    # e.g. '32x32'

        start_time = time.time()

        # 实例化
        explorer = Explorer(path=res_root, batch_size=4, num_init=16, max_evals=128, num_trs=4, seed=seed)
        explorer.set_search_space(space_spec)
        explorer.set_kernel_top(top_list[0]) # 单 kernel 示例
        explorer.bayes_opt()

 
        dse_time = (time.time() - start_time) / 60
        with open(f'{res_dir}/{bench_list[idx]}_runtime.txt', 'w') as file:
            file.write(f'DSE Runtime: {dse_time} minutes\n')
        os.system(f"python3 /home/edalab/EDA-DSE/sample/clean_bambu_results.py")


"""
export DSE_EVAL_WRAPPER=/path/to/run_mlir_bambu_vtr.py
export DSE_MLIR_INPUT=/path/to/forward.mlir
export DSE_MLIR_OPT=/path/to/mlir-opt
export DSE_HLS_OPT=/path/to/hls-opt
export DSE_TOP=matmul
export DSE_ARG_TYPES='["memref<32x32xf32>","memref<32x32xf32>","memref<32x32xf32>"]'
export DSE_VTR_FLOW=/path/to/vtr_flow/scripts/run_vtr_flow.py
export DSE_VTR_ARCH=/path/to/vtr_flow/arch/timing/k6_frac_N10_frac_chain_mem32K_40nm.xml
export DSE_VERILOG=/tmp/forward.v


python3 run_dse.py
"""