#!/usr/bin/env python3
# explorer_moebo.py
# -*- coding: utf-8 -*-
"""
基于 TuRBO 思想的多目标集成贝叶斯优化 (MOEBO) 参考实现：
- 目标：最大化 y=(y1,y2,y3)（在本项目中约定 y = (-cycle,-area,-delay)）。
- 四个关键点：
  1) 使用超体积(Hypervolume)作为多目标指标（含蒙特卡洛 EHVI 近似）。
  2) 多任务“高斯”代理：用协方差分解(C)做共区域化的近似，把多目标投影到无关主成分上，
     对每个分量用核回归（NW/KRR 近似 GP 的后验均值/方差）建模，再投影回去。
  3) 多个局部代理（来自多个 TR）+ 一个“全局”代理，使用多臂赌博机(MAB)按 HV 增益做自适应资源分配。
  4) 新采集函数：EHVI (Expected Hypervolume Improvement) 的批量贪心近似，带多模型协同与去重正则。

说明：
- 这是 dependency-free 的实现（仅 numpy），在工业系统可替换为 BoTorch 的 qEHVI/MGTGP 以获得更强性能。
- TR 策略、MAB、EHVI 都做了轻量近似以保证鲁棒与易用。
"""

import os, json, time, math, random, tempfile, subprocess, uuid, shlex, csv, shutil, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np

# ===================== 工具：参数空间 =====================
class Space:
    def __init__(self, items: List[Dict[str, Any]]):
        self.vars = []
        for it in items:
            if not isinstance(it, dict):
                continue
            if "name" in it and "type" in it:
                self.vars.append({
                    "name": it["name"],
                    "type": it["type"],
                    "values": it.get("values"),
                    "default": it.get("default"),
                })
        self.maps = []
        for v in self.vars:
            if v["type"] in ("bool", "int", "cat"):
                if v["type"] == "bool":
                    vals = [False, True]
                else:
                    vals = v["values"]
                    if not vals or not isinstance(vals, list):
                        raise ValueError(f"{v['name']} 缺少 values")
                self.maps.append(vals)
            elif v["type"] == "float":
                assert isinstance(v["values"], list) and len(v["values"]) == 2
                self.maps.append(None)
            else:
                raise ValueError("未知类型: " + v["type"])

    def encode01(self, x: Dict[str, Any]) -> List[float]:
        z = []
        for v, m in zip(self.vars, self.maps):
            val = x[v["name"]]
            if v["type"] in ("bool", "int", "cat"):
                idx = m.index(val)
                z.append(0.0 if len(m) == 1 else idx / (len(m) - 1))
            else:
                lo, hi = v["values"]
                z.append((val - lo) / (hi - lo))
        return z

    def decode01(self, z: List[float]) -> Dict[str, Any]:
        out = {}
        for v, m, zi in zip(self.vars, self.maps, z):
            if v["type"] in ("bool", "int", "cat"):
                idx = 0 if len(m) == 1 else int(round(zi * (len(m) - 1)))
                idx = max(0, min(len(m) - 1, idx))
                choice = m[idx]
                out[v["name"]] = bool(choice) if v["type"] == "bool" else choice
            else:
                lo, hi = v["values"]
                out[v["name"]] = lo + zi * (hi - lo)
        return out

    def sample_in_box(self, center: List[float], length: float, n: int) -> List[List[float]]:
        out = []
        half = length / 2.0
        for _ in range(n):
            z = []
            for ci in center:
                low = max(0.0, ci - half)
                high = min(1.0, ci + half)
                z.append(random.uniform(low, high))
            out.append(z)
        return out

    def sample_random_cfg(self) -> Dict[str, Any]:
        x = {}
        for v, m in zip(self.vars, self.maps):
            if v["type"] == "bool":
                x[v["name"]] = bool(random.getrandbits(1))
            elif v["type"] in ("int", "cat"):
                x[v["name"]] = random.choice(m)
            else:
                lo, hi = v["values"]
                x[v["name"]] = random.uniform(lo, hi)
        return x

# ===================== HV 与帕累托 =====================

def pareto_idx_maximize(Y: List[Tuple[float, float, float]]) -> List[int]:
    keep = []
    for i, a in enumerate(Y):
        dom = False
        for j, b in enumerate(Y):
            if j == i:
                continue
            if (b[0] >= a[0] and b[1] >= a[1] and b[2] >= a[2]) and (b != a):
                dom = True
                break
        if not dom:
            keep.append(i)
    return keep


def hypervolume_approx(Y: List[Tuple[float, float, float]]) -> float:
    if not Y:
        return 0.0
    mins = [min(y[i] for y in Y) for i in range(3)]
    maxs = [max(y[i] for y in Y) for i in range(3)]
    span = [max(1e-9, maxs[i] - mins[i]) for i in range(3)]
    Q = [
        (
            (y[0] - mins[0]) / span[0],
            (y[1] - mins[1]) / span[1],
            (y[2] - mins[2]) / span[2],
        )
        for y in Y
    ]
    K = 30
    vox = set()
    for q in Q:
        i = int(q[0] * K)
        j = int(q[1] * K)
        k = int(q[2] * K)
        for ii in range(i + 1):
            for jj in range(j + 1):
                for kk in range(k + 1):
                    vox.add((ii, jj, kk))
    return len(vox) / (K + 1) ** 3


def ehvi_mc(cands_mu: np.ndarray, cands_sd: np.ndarray, Y_ref: List[Tuple[float, float, float]], n_samp: int = 64) -> np.ndarray:
    """蒙特卡洛 EHVI：
    - cands_mu, cands_sd: (B,3)
    - Y_ref: 当前已评估点（在“最大化”空间）
    返回每个候选的 EHVI 估计 (B,)
    """
    if len(Y_ref) == 0:
        # 没有参考帕累托时，退化为期望体积（越大越好）
        base_hv = 0.0
    else:
        base_hv = hypervolume_approx(Y_ref)

    B = cands_mu.shape[0]
    out = np.zeros(B, dtype=float)
    # 独立正态采样（忽略跨目标协方差，已在多任务代理里吸收一部分相关性）
    for b in range(B):
        mu = cands_mu[b]
        sd = np.maximum(cands_sd[b], 1e-8)
        hv_sum = 0.0
        for _ in range(n_samp):
            ys = np.random.normal(mu, sd, size=(1, 3))
            Y_new = Y_ref + [tuple(float(v) for v in ys[0])]
            hv_sum += max(0.0, hypervolume_approx(Y_new) - base_hv)
        out[b] = hv_sum / n_samp
    return out

# ===================== 近似 MGP 代理 =====================
class SurrogateMGP:
    """多任务核回归 + 协方差解耦。兼容小数据、无外部依赖。
    - 拟合：Y 的目标间协方差 C≈LL^T，令 Y_tilde = Y @ (L^-1)^T；
      对每个维度做 NW 核回归；
    - 预测：先在各维度上预测均值/方差，再投影回原目标空间。
    """

    def __init__(self, dim: int, kappa: float = 1.2):
        self.dim = dim
        self.kappa = kappa
        self.X: List[List[float]] = []
        self.Y: List[Tuple[float, float, float]] = []
        # 任务相关性分解矩阵
        self.L = np.eye(3)
        self.L_inv_T = np.eye(3)
        # 高斯核带宽（经验）
        self.s2 = 0.25 ** 2

    def fit(self, X: List[List[float]], Y: List[Tuple[float, float, float]]):
        self.X = [list(x) for x in X]
        self.Y = [tuple(y) for y in Y]
        if len(Y) >= 3:
            Yarr = np.array(Y)
            C = np.cov(Yarr.T) + 1e-6 * np.eye(3)
            try:
                self.L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # 退化时改用对角
                self.L = np.diag(np.sqrt(np.maximum(np.diag(C), 1e-8)))
        else:
            self.L = np.eye(3)
        self.L_inv_T = np.linalg.pinv(self.L.T)  # 稳定一些

    def _kernel_weights(self, z: List[float]) -> np.ndarray:
        if not self.X:
            return np.ones(1)
        X = np.array(self.X)
        d = np.sum((X - np.array(z)) ** 2, axis=1)
        w = np.exp(-d / (2 * self.s2)) + 1e-12
        return w / np.sum(w)

    def predict(self, Z: List[List[float]]) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        mu_list = []
        sd_list = []
        if not self.X:
            mu_list = [(0.0, 0.0, 0.0) for _ in Z]
            sd_list = [(0.6, 0.6, 0.6) for _ in Z]
            return mu_list, sd_list

        Y = np.array(self.Y)
        Y_tilde = Y @ self.L_inv_T
        for z in Z:
            w = self._kernel_weights(z)
            # 对每个“无关任务分量”做核回归
            m_tilde = np.sum(Y_tilde * w[:, None], axis=0)
            v_tilde = np.sum(((Y_tilde - m_tilde) ** 2) * w[:, None], axis=0) + 1e-6
            # 投影回原目标
            m = m_tilde @ self.L.T
            # 方差近似：只保留对角项（忽略交叉项）
            Sd_diag = np.sum((self.L ** 2) * v_tilde[None, :], axis=1)
            s = np.sqrt(np.maximum(Sd_diag, 1e-9))
            mu_list.append(tuple(m.tolist()))
            sd_list.append(tuple(s.tolist()))
        return mu_list, sd_list

    def score_ucb(self, Z: List[List[float]]) -> List[float]:
        mu, sd = self.predict(Z)
        scores = []
        for m, s in zip(mu, sd):
            mbar = (m[0] + m[1] + m[2]) / 3.0
            sbar = (s[0] + s[1] + s[2]) / 3.0
            scores.append(mbar + self.kappa * sbar)
        return scores

# ===================== TR 与多臂赌博机 =====================
@dataclass
class TR:
    center: List[float]
    length: float
    succ: int = 0
    fail: int = 0


class Bandit:
    def __init__(self, k: int, c: float = 0.8, window: int = 10):
        self.k = k
        self.c = c
        self.w = window
        self.hist = [[] for _ in range(k)]
        self.n = [0] * k

    def update(self, arm: int, reward: float):
        self.hist[arm].append(reward)
        if len(self.hist[arm]) > self.w:
            self.hist[arm] = self.hist[arm][-self.w :]
        self.n[arm] += 1

    def select(self) -> int:
        t = sum(self.n) + 1
        best = 0
        bestv = -1e9
        for i in range(self.k):
            mean = (sum(self.hist[i]) / len(self.hist[i])) if self.hist[i] else 0.0
            bonus = self.c * math.sqrt(math.log(max(2, t)) / (self.n[i] + 1e-9))
            v = mean + bonus
            if v > bestv:
                bestv = v
                best = i
        return best

# ===================== 外部评估封装 =====================

def eval_one_cfg(cfg: Dict[str, Any]) -> Tuple[float, float, float, str]:
    def _tail_json(stdout: str) -> Optional[Dict[str, Any]]:
        for line in stdout.strip().splitlines()[::-1]:
            t = line.strip()
            if t.startswith("{") and t.endswith("}"):
                try:
                    return json.loads(t)
                except Exception:
                    pass
        return None
    
    bambu = os.getenv("DSE_BAMBU_WRAPPER")
    mlir_input = os.getenv("DSE_MLIR_INPUT")

    mlir_opt  = os.getenv("DSE_MLIR_OPT") 
    circt_opt   = os.getenv("DSE_CIRCT_OPT") 
 
    input_dir = os.path.dirname(os.path.abspath(mlir_input)) if mlir_input else os.getcwd()
    opt_mlir  = os.path.join(input_dir, f"opt.mlir")

    def _as_int(name, default=1):
        v = int(round(float(cfg.get(name, default))))
        return max(1, v)

    tile = _as_int("tile", 1)
    unroll = _as_int("unroll_factor", 1)
    do_inter = bool(cfg.get("loop_interchange", False))

    base_passes = [
        "--fold-memref-alias-ops",
        "--expand-strided-metadata",
        "--canonicalize",
        "--cse",
        "--affine-simplify-structures",
        "--affine-loop-fusion",
    ]
    interchange_pass = "--affine-loop-permutation"
    if do_inter:
        base_passes.append(interchange_pass)

    base_passes += [
        "--affine-loop-invariant-code-motion",
        "--affine-simplify-structures",
        "--affine-expand-index-ops",
    ]
    if tile > 1:
        base_passes.append(f"--affine-loop-tile=\"tile-size={tile}\"")

    unroll_flag = f'--pass-pipeline="builtin.module(func.func(affine-loop-unroll{{unroll-factor={unroll}}}))"'

    cmd = (
        f"{shlex.quote(mlir_opt)} {' '.join(base_passes)} {shlex.quote(mlir_input)} "
        f"| {shlex.quote(mlir_opt)} {unroll_flag} -o {shlex.quote(opt_mlir)}"
    )
    print("[DEBUG] MLIR pipeline:\n ", cmd)
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError("MLIR/HLS pipeline failed:\n" + p.stdout)
    
    # Stage-1: Bambu
    must_bambu = ["DSE_MLIR_INPUT", "DSE_MLIR_OPT", "DSE_HLS_OPT", "DSE_TOP", "DSE_ARG_TYPES"]
    miss = [k for k in must_bambu if not os.getenv(k)]
    if miss:
        raise RuntimeError("Missing environment variables (Bambu stage): " + ", ".join(miss))
    
    arg_types = shlex.quote(os.getenv("DSE_ARG_TYPES"))


    cmd1 = (
        f"python3 {shlex.quote(bambu)} "
        f"--mlir {shlex.quote(opt_mlir)} "
        f"--top {shlex.quote(os.getenv('DSE_TOP'))} "
        f"--arg-types {arg_types} ".strip()
    )
    print("[DEBUG] Running Bambu command:")
    print("        ", cmd1)

    try:
        p1 = subprocess.run(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print("[ERROR] Bambu command exceeded 30 minutes and was killed.")
        out1 = "[TIMEOUT] Command terminated after 30 minutes."
    out1 = p1.stdout
    
    
    
    # with open("bambu_stdout.log", "w", encoding="utf-8") as f:
    #     f.write(out1)
    if p1.returncode != 0:
        print("[ERROR] Bambu stage failed. See bambu_stdout.log for details.")
        raise RuntimeError(f"Bambu stage failed:\n{out1}")
    
    js1 = _tail_json(out1) or {}

    # cycle
    if "cycle" in js1:
        cycle = float(js1["cycle"])
    else:
        try:
            from problem_eval import find_latest_bambu_xml, parse_bambu_cycles

            xmlp = find_latest_bambu_xml(input_dir)
            if not xmlp:
                raise RuntimeError("Cannot find Bambu result XML near DSE_MLIR_INPUT or CWD, and stdout has no JSON['cycle'].")
            cycle = float(parse_bambu_cycles(xmlp))
            print(f"[DEBUG] Cycle parsed from XML: {cycle}")
        except Exception:
            raise RuntimeError("No cycles.")
   
    verilog_path = os.path.splitext(mlir_input)[0] + ".v"
    print(f"[DEBUG] Derived verilog path: {verilog_path}")

    # Stage-2: VTR
    must_vtr = ["DSE_VTR_ARCH"]
    miss2 = [k for k in must_vtr if not os.getenv(k)]
    if miss2:
        raise RuntimeError("Missing environment variables (VTR stage): " + ", ".join(miss2))
    
    cmd2 = (
        f"/home/edalab/vtr-verilog-to-routing/vtr_flow/scripts/run_vtr_flow.py "
        f"{shlex.quote(verilog_path)} "
        f"{shlex.quote(os.getenv('DSE_VTR_ARCH'))} "
    )
    p2 = subprocess.run(cmd2, shell=True, cwd=input_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out2 = p2.stdout
    if p2.returncode != 0:
        raise RuntimeError(f"VTR Failed: \n{out2}")

    vpr_out = os.path.join(input_dir, "temp", "vpr.out")
    print(f"[DEBUG] Using conventional VTR out path: {vpr_out}")

    if not vpr_out or not os.path.exists(vpr_out):
        print("[ERROR] VTR output file not found.")
        print(f"[DEBUG] Expected path: {vpr_out}")
        # optional: show what's inside input_dir/temp to help debugging
        temp_dir = os.path.join(input_dir, "temp")
        if os.path.isdir(temp_dir):
            print(f"[DEBUG] Listing '{temp_dir}':", os.listdir(temp_dir))
        else:
            print(f"[DEBUG] Temp directory does not exist: {temp_dir}")
        raise RuntimeError("VTR stage did not provide JSON (area, delay), and no parsable output file was found. "
                        "Set DSE_vpr_out or ensure input_dir/temp/vtr.out exists.")
    from problem_eval import parse_vpr_out
    area, delay = parse_vpr_out(vpr_out)
    print(f"[DEBUG] Parsed VTR metrics -> area: {area}, delay: {delay}")

    return float(cycle), float(area), float(delay)

# ===================== Explorer (MOEBO) =====================
class Explorer:
    def __init__(
        self,
        path: str = "",
        batch_size: int = 4,
        num_init: int = 16,
        max_evals: int = 128,
        num_trs: int = 4,
        kappa: float = 1.2,
        seed: int = 42,
        problem=None,
    ):
        random.seed(seed)
        np.random.seed(seed)
        self.problem = problem 
        self.batch_size = batch_size
        self.num_init = num_init
        self.max_evals = max_evals
        self.num_trs = num_trs
        self.kappa = kappa

        # TuRBO 参数
        self.init_L = 0.8
        self.min_L = 0.05
        self.max_L = 1.6
        self.succ_t = 3
        self.fail_t = 5

        self.space_spec: List[Dict[str, Any]] = []
        self.kernel_top: Optional[str] = None

        # 数据缓存
        self.Z: List[List[float]] = []  # 0..1 编码
        self.Y: List[Tuple[float, float, float]] = []  # maximize: (-cycle, -area, -delay)
        self.raw: List[Dict[str, Any]] = []  # 原始 config
        
        self.run_root = path
        self.csv_path = os.path.join(self.run_root, f"{seed}_trials.csv")
        self.sum_csv  = os.path.join(self.run_root, "run_summary.csv")
        self.bandit_csv = os.path.join(self.run_root, "bandit_history.csv")
        self.rtl_dir  = os.path.join(self.run_root, "rtl")
        self.eval_count = 0  
        self._csv_header_written = False
        self._trial_fieldnames: List[str] = []

    def set_search_space(self, space: List[Dict[str, Any]]):
        self.space_spec = space

    def set_kernel_top(self, top: str):
        self.kernel_top = top

    def _ensure_csv_headers(self):
        if self._csv_header_written:
            return
        # 固定字段
        fixed = [
            "ts","iter","phase","arm","arm_label",
            "tr_len","tr_center","cfg_hash",
            "cycle","area","delay",
            "hv_before","hv_after","hvi",
            "ehvi","score","penalty","dmin",
            "mu_cycle","mu_area","mu_delay",
            "sd_cycle","sd_area","sd_delay",
        ]
        # 参数列按 space 定义展开，顺序稳定
        param_cols = [f"param_{v['name']}" for v in self.space.vars]
        self._trial_fieldnames = fixed + param_cols

        # trials
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._trial_fieldnames).writeheader()
        # summary
        if not os.path.exists(self.sum_csv):
            with open(self.sum_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=[
                    "iter","arm","arm_label","trl_list","hv_before","hv_after","gain_batch","pareto_size"
                ]).writeheader()
        # bandit
        if not os.path.exists(self.bandit_csv):
            with open(self.bandit_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=["iter","arm","reward","n_pulls"]).writeheader()

        self._csv_header_written = True

    def _log_trial(self, row: Dict[str, Any]):
        self._ensure_csv_headers()
        for k in self._trial_fieldnames:
            row.setdefault(k, "")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._trial_fieldnames).writerow(row)

    def _log_iter_summary(self, it: int, arm: int, hv_before: float, hv_after: float, gain: float):
        with open(self.sum_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=[
                "iter","arm","arm_label","trl_list","hv_before","hv_after","gain_batch","pareto_size"
            ]).writerow({
                "iter": it,
                "arm": arm,
                "arm_label": ("G" if arm==self.num_trs else str(arm)),
                "trl_list": json.dumps([round(t.length,4) for t in self.trs]),
                "hv_before": hv_before,
                "hv_after": hv_after,
                "gain_batch": gain,
                "pareto_size": len(pareto_idx_maximize(self.Y)),
            })

    def _log_bandit(self, it: int, arm: int, reward: float, n_pulls: int):
        with open(self.bandit_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["iter","arm","reward","n_pulls"]).writerow({
                "iter": it, "arm": arm, "reward": reward, "n_pulls": n_pulls
            })
    # ======= 公开主流程 =======
    def bayes_opt(self):
        if not self.space_spec:
            try:
                self.space_spec = json.loads(os.getenv("DSE_SPACE_JSON", "[]"))
            except Exception:
                raise RuntimeError("No (space_spec)")

        self.space = Space(self.space_spec)

        # 初始化 TR 与代理（最后一个 arm 作为“全局”代理，不绑定 TR）
        self.trs = [
            TR(center=self.space.encode01(self.space.sample_random_cfg()), length=self.init_L)
            for _ in range(self.num_trs)
        ]
        self.bandit = Bandit(self.num_trs + 1, c=0.8, window=10)  # +1: global arm
        self.surrogates = [SurrogateMGP(dim=len(self.space.vars), kappa=self.kappa) for _ in range(self.num_trs + 1)]

        # ------ 初始探索 ------
        while len(self.Z) < self.num_init and len(self.Z) < self.max_evals:
            cfg = self.space.sample_random_cfg()
            y = self._eval_cfg(cfg)
            if y is None:
                continue
     
            hv_before = hypervolume_approx(self.Y)
            hv_after = hypervolume_approx(self.Y + [y])
            hvi = hv_after - hv_before

            self._append(cfg, y)
            print(f"[INIT {len(self.Z)}/{self.num_init}] y={y} (maximize: -cycle,-area,-delay)")

            self._log_trial({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "iter": 0,
                "phase": "init",
                "arm": "init",
                "arm_label": "init",
                "tr_len": "",
                "tr_center": "",
                "cfg_hash": hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:10],
                # y 是 (-cycle,-area,-delay) → 自然量要取负
                "cycle": -y[0], "area": -y[1], "delay": -y[2],
                "hv_before": hv_before, "hv_after": hv_after, "hvi": hvi,
                "ehvi": "", "score": "", "penalty": "", "dmin": "",
                "mu_cycle": "", "mu_area": "", "mu_delay": "",
                "sd_cycle": "", "sd_area": "", "sd_delay": "",
                **{f"param_{k}": v for k, v in cfg.items()},
            })

        hv = hypervolume_approx(self.Y)
        print(f"[HV] init = {hv:.6f}")
        best_idx = max(range(len(self.Y)), key=lambda i: self.Y[i])
        for tr in self.trs:
            tr.center = self.Z[best_idx][:]
        # ------ 迭代 ------
        it = 0
        while len(self.Z) < self.max_evals:
            it += 1
            # 拟合每个 TR 的局部代理；最后一个代理用全量数据（全局）
            for i, tr in enumerate(self.trs):
                half = tr.length / 2.0
                def in_tr(z):
                    return all(abs(zi - ci) <= half + 1e-12 for zi, ci in zip(z, tr.center))
                idx = [j for j, z in enumerate(self.Z) if in_tr(z)]
                X = [self.Z[j] for j in idx] if idx else self.Z
                Y = [self.Y[j] for j in idx] if idx else self.Y
                self.surrogates[i].fit(X, Y)
            # 全局代理
            self.surrogates[-1].fit(self.Z, self.Y)

            arm = self.bandit.select() 
            arm_label = "G" if arm == self.num_trs else str(arm)

            # 生成候选
            if arm < self.num_trs:
                tr = self.trs[arm]
                cands_z = self.space.sample_in_box(tr.center, tr.length, self.batch_size * 60)
            else:
                # 全局随机候选
                cands_z = [self.space.encode01(self.space.sample_random_cfg()) for _ in range(self.batch_size * 60)]

            # 去重 & 降密
            uniq = []
            seen = set()
            for z in cands_z:
                cfg = self.space.decode01(z)
                key = json.dumps(cfg, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((z, cfg))
                if len(uniq) >= self.batch_size * 12:
                    break
            if not uniq:
                # 重置
                for tr in self.trs:
                    tr.center = self.space.encode01(self.space.sample_random_cfg())
                continue

            Z_batch = [u[0] for u in uniq]
            mu, sd = self.surrogates[arm].predict(Z_batch)
            mu_arr = np.array(mu)
            sd_arr = np.array(sd)
            # EHVI 打分 + 去重正则
            ehvi = ehvi_mc(mu_arr, sd_arr, self.Y, n_samp=48)
            # 距离惩罚，鼓励分散
            Z_arr = np.array(Z_batch)
            if len(self.Z) > 0:
                Z_exist = np.array(self.Z)
                dmin = np.min(np.sqrt(((Z_arr[:, None, :] - Z_exist[None, :, :]) ** 2).sum(axis=2)), axis=1)
                penalty = 0.02 / (dmin + 1e-6)
            else:
                penalty = np.zeros(len(uniq))
            scores = ehvi - penalty

            pick_idx = list(np.argsort(-scores)[: self.batch_size])
            picked = [uniq[i] for i in pick_idx]

            old_hv = hypervolume_approx(self.Y)
            success = False
            for loc, (z, cfg) in enumerate(picked):
                i_pick   = pick_idx[loc]
                mu_i     = mu_arr[i_pick]
                sd_i     = sd_arr[i_pick]
                ehvi_i   = float(ehvi[i_pick])
                score_i  = float(scores[i_pick])
                penalty_i= float(penalty[i_pick])
                dmin_i   = float(dmin[i_pick])

                # 评估前的 hv_before（每个样本单独计算）
                hv_before = hypervolume_approx(self.Y)

                y = self._eval_cfg(cfg)
                if y is None:
                    continue

                hv_after = hypervolume_approx(self.Y + [y])
                hvi = hv_after - hv_before

                # 缓存/打印
                self.Z.append(z)
                self.raw.append(cfg)
                self.Y.append(y)
                print(f"[IT {it}] arm={arm_label}  y={y}  cfg={cfg}")
                success = True

                # 记录（自然量：取负）
                self._log_trial({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "iter": it,
                    "phase": "opt",
                    "arm": arm,
                    "arm_label": arm_label,
                    "tr_len": (self.trs[arm].length if arm < self.num_trs else ""),
                    "tr_center": (json.dumps(self.trs[arm].center) if arm < self.num_trs else ""),
                    "cfg_hash": hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:10],
                    "cycle": -y[0], "area": -y[1], "delay": -y[2],
                    "hv_before": hv_before, "hv_after": hv_after, "hvi": hvi,
                    "ehvi": ehvi_i, "score": score_i, "penalty": penalty_i, "dmin": dmin_i,
                    "mu_cycle": mu_i[0], "mu_area": mu_i[1], "mu_delay": mu_i[2],
                    "sd_cycle": sd_i[0], "sd_area": sd_i[1], "sd_delay": sd_i[2],
                    **{f"param_{k}": v for k, v in cfg.items()},
                })

            new_hv = hypervolume_approx(self.Y)
            gain = max(0.0, new_hv - old_hv)
            self.bandit.update(arm, gain)
            self._log_bandit(it, arm, gain, self.bandit.n[arm])

            # 调整 TR
            if arm < self.num_trs:
                tr = self.trs[arm]
                if success and gain > 0:
                    tr.succ += 1
                    tr.fail = 0
                else:
                    tr.fail += 1
                if tr.succ >= self.succ_t:
                    tr.length = min(self.max_L, tr.length * 2.0)
                    tr.succ = 0
                if tr.fail >= self.fail_t:
                    tr.length = max(self.min_L, tr.length / 2.0)
                    tr.fail = 0
                    best_idx = max(range(len(self.Y)), key=lambda i: self.Y[i])
                    tr.center = self.Z[best_idx][:]

            pf = pareto_idx_maximize(self.Y)
            print(
                f"[IT {it}] HV={new_hv:.6f}  arm={'G' if arm==self.num_trs else arm}  TRL={[round(t.length,3) for t in self.trs]}  gain={gain:.6f}  |Pareto|={len(pf)}"
            )
            self._log_iter_summary(it, arm, old_hv, new_hv, gain)
        # 导出结果
        pf = pareto_idx_maximize(self.Y)
        result = []
        for i in pf:
            y = self.Y[i]
            metrics = {"cycle": -y[0], "area": -y[1], "delay": -y[2]}
            result.append({"config": self.raw[i], "metrics": metrics})
        result.sort(key=lambda d: (-d["metrics"]["cycle"], -d["metrics"]["area"], -d["metrics"]["delay"]))
        pareto_json = os.path.join(self.run_root, "moebo_pareto.json")
        with open(pareto_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("[DONE] Save moebo_pareto.json")

    # ======= 内部方法 =======
    def _append(self, cfg: Dict[str, Any], y: Tuple[float, float, float]):
        self.Z.append(self.space.encode01(cfg))
        self.raw.append(cfg)
        self.Y.append(y)

    def _eval_cfg(self, cfg: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:

        try:
            cycle, area, delay = eval_one_cfg(cfg)
        except Exception as e:
            print(f"[WARN] 评估失败，跳过：{e}")

            return None
        finally:
            pass
        return (-float(cycle), -float(area), -float(delay))
