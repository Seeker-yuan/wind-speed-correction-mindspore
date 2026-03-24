import argparse
import importlib.util
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


def _load_core_module():
    base = Path(__file__).resolve().parent
    candidates = [base / "预测_mindspore.py", base / "汇总预测.py"]
    src = None
    for c in candidates:
        if c.exists():
            src = c
            break
    if src is None:
        raise FileNotFoundError("未找到核心脚本：预测_mindspore.py 或 汇总预测.py")

    spec = importlib.util.spec_from_file_location("core_pipeline", str(src))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载核心脚本: {src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _load_core_module()


@dataclass
class EvalResult:
    method: str
    rmse: float
    mae: float
    boundary_jump_rate: float
    peak_memory_mb: float


def _make_predictor(n_neighbors: int, hidden_size: int, seq_len: int):
    """兼容 MindSpore 主模型与 sklearn 回退模型的构造参数。"""
    try:
        return core.MindSporeWindPredictor(
            n_neighbors=n_neighbors,
            hidden_size=hidden_size,
            seq_len=seq_len,
        )
    except TypeError:
        return core.MindSporeWindPredictor(
            n_neighbors=n_neighbors,
            hidden_size=hidden_size,
        )


class _FlatMLPWrapper:
    """在无 MindSpore 环境下，兼容 (N, nodes, seq_len) 输入的回退模型。"""

    def __init__(self, hidden_size: int, random_state: int = 42):
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_size, hidden_size, 32),
            max_iter=200,
            random_state=random_state,
        )

    def fit(self, X, y, **kwargs):
        x2 = X.reshape(X.shape[0], -1)
        self.model.fit(x2, y)

    def predict(self, X, **kwargs):
        x2 = X.reshape(X.shape[0], -1)
        pred = self.model.predict(x2)
        if pred.ndim == 1:
            pred = pred[:, None]
        return pred.astype(np.float32)


def _fill_nan_targets(y: np.ndarray):
    y2 = y.copy().astype(np.float32)
    col_mean = np.nanmean(y2, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    inds = np.where(~np.isfinite(y2))
    if inds[0].size > 0:
        y2[inds] = col_mean[inds[1]]
    return y2


def _find_blocks(mask_1d: np.ndarray):
    idx = np.where(mask_1d)[0]
    if len(idx) == 0:
        return []
    blocks = []
    s = int(idx[0])
    e = int(idx[0])
    for x in idx[1:]:
        x = int(x)
        if x == e + 1:
            e = x
        else:
            blocks.append((s, e))
            s = x
            e = x
    blocks.append((s, e))
    return blocks


def build_holdout_mask(panel: pd.DataFrame, seq_len: int, missing_rate: float,
                      block_min: int, block_max: int, rng: np.random.Generator):
    """基于完整观测构造连续缺失块，统一作为对比评测集。"""
    n_t, n_nodes = panel.shape
    target_points = int(np.floor(np.isfinite(panel.to_numpy()).sum() * missing_rate))
    mask = np.zeros((n_t, n_nodes), dtype=bool)
    selected = 0

    valid_start_lo = max(seq_len, 1)
    valid_start_hi = n_t - 2
    if valid_start_lo >= valid_start_hi:
        raise ValueError("时序长度过短，无法构造评测缺失块")

    all_cols = np.arange(n_nodes)
    max_trials = target_points * 30 + 5000
    trials = 0

    arr = panel.to_numpy(dtype=np.float32)
    while selected < target_points and trials < max_trials:
        trials += 1
        j = int(rng.choice(all_cols))
        start = int(rng.integers(valid_start_lo, valid_start_hi + 1))
        length = int(rng.integers(block_min, block_max + 1))
        end = min(start + length - 1, n_t - 2)
        if end < start:
            continue

        seg = arr[start:end + 1, j]
        right_obs = arr[end + 1, j]
        if not np.isfinite(seg).all() or (not np.isfinite(right_obs)):
            continue
        if mask[start:end + 1, j].any():
            continue

        mask[start:end + 1, j] = True
        selected += (end - start + 1)

    if selected < max(10, int(target_points * 0.6)):
        raise RuntimeError("评测缺失块构造失败，请调低 missing_rate 或缩短 block_max")

    return mask


def apply_mask(panel: pd.DataFrame, holdout_mask: np.ndarray):
    masked = panel.copy()
    arr = masked.to_numpy(dtype=np.float32)
    arr[holdout_mask] = np.nan
    return pd.DataFrame(arr, index=panel.index, columns=panel.columns)


def _calc_dist_matrix(machine_names):
    n = len(machine_names)
    dist = np.zeros((n, n), dtype=np.float32)
    for i, ni in enumerate(machine_names):
        lon1, lat1 = core.pos[ni]
        for j, nj in enumerate(machine_names):
            if i == j:
                continue
            lon2, lat2 = core.pos[nj]
            dist[i, j] = core.haversine(lon1, lat1, lon2, lat2)
    return dist


def method_knn_interpolation(panel_masked: pd.DataFrame, machine_names, k=5):
    """传统 KNN 插值：同一时刻的空间近邻加权，兜底使用时间向前填充。"""
    arr = panel_masked.to_numpy(dtype=np.float32)
    out = arr.copy()
    n_t, n_nodes = out.shape
    dist = _calc_dist_matrix(machine_names)

    for t in range(n_t):
        row = out[t]
        miss_idx = np.where(~np.isfinite(row))[0]
        if len(miss_idx) == 0:
            continue
        obs_idx = np.where(np.isfinite(row))[0]
        for j in miss_idx:
            if len(obs_idx) > 0:
                nn = obs_idx[np.argsort(dist[j, obs_idx])[:k]]
                w = 1.0 / (dist[j, nn] + 1e-6)
                w = w / w.sum()
                row[j] = float(np.dot(w, row[nn]))
        out[t] = row

    df = pd.DataFrame(out, index=panel_masked.index, columns=panel_masked.columns)
    df = df.ffill().bfill().fillna(0.0)
    return df.astype(np.float32)


def _fit_ar_coeff(series: np.ndarray, seq_len: int):
    xs, ys = [], []
    for t in range(seq_len, len(series)):
        win = series[t - seq_len:t]
        y = series[t]
        if np.isfinite(win).all() and np.isfinite(y):
            xs.append(win)
            ys.append(y)
    if len(xs) < max(30, seq_len * 5):
        return None
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return coef.astype(np.float32)


def method_single_machine_ar(panel_masked: pd.DataFrame, seq_len=6):
    """单机自回归基线：每台风机仅用自身历史窗口递推补全。"""
    out = panel_masked.copy()
    n_t, n_nodes = out.shape
    arr = out.to_numpy(dtype=np.float32)

    coefs = []
    for j in range(n_nodes):
        coef = _fit_ar_coeff(arr[:, j], seq_len=seq_len)
        coefs.append(coef)

    for t in range(seq_len, n_t):
        for j in range(n_nodes):
            if np.isfinite(arr[t, j]):
                continue
            win = arr[t - seq_len:t, j]
            if np.isfinite(win).all() and (coefs[j] is not None):
                x = np.concatenate([win, np.array([1.0], dtype=np.float32)])
                arr[t, j] = float(np.dot(x, coefs[j]))

    filled = pd.DataFrame(arr, index=out.index, columns=out.columns)
    filled = filled.ffill().bfill().fillna(0.0)
    return filled.clip(lower=0.0).astype(np.float32)


def method_stgcn_no_mask_no_blend(panel_masked: pd.DataFrame, machine_names,
                                  seq_len=6, epochs=10, hidden_size=64, adjacency=None):
    if adjacency is None:
        raise ValueError("method_stgcn_no_mask_no_blend 需要预先构建 adjacency")
    x_train, y_train, _ = core.build_global_training_set(panel_masked, seq_len=seq_len)
    if getattr(core, "MINDSPORE_AVAILABLE", False):
        model = _make_predictor(
            n_neighbors=len(machine_names), hidden_size=hidden_size, seq_len=seq_len
        )
        y_fit = y_train
    else:
        model = _FlatMLPWrapper(hidden_size=hidden_size)
        y_fit = _fill_nan_targets(y_train)
    model.fit(
        x_train, y_fit,
        epochs=epochs,
        batch_size=min(64, len(x_train)),
        verbose=False,
        adjacency=adjacency,
        mask=None
    )
    return core.synchronous_fill_panel(panel_masked, model, adjacency, seq_len=seq_len)


def method_invention(panel_masked: pd.DataFrame, machine_names,
                     seq_len=6, epochs=10, hidden_size=64, adjacency=None):
    if adjacency is None:
        raise ValueError("method_invention 需要预先构建 adjacency")
    x_train, y_train, m_train = core.build_global_training_set(panel_masked, seq_len=seq_len)
    if getattr(core, "MINDSPORE_AVAILABLE", False):
        model = _make_predictor(
            n_neighbors=len(machine_names), hidden_size=hidden_size, seq_len=seq_len
        )
        y_fit = y_train
    else:
        model = _FlatMLPWrapper(hidden_size=hidden_size)
        y_fit = _fill_nan_targets(y_train)
    model.fit(
        x_train, y_fit,
        epochs=epochs,
        batch_size=min(64, len(x_train)),
        verbose=False,
        adjacency=adjacency,
        mask=m_train
    )
    pred = core.synchronous_fill_panel(panel_masked, model, adjacency, seq_len=seq_len)
    return core.apply_bidirectional_blending(panel_masked, pred)


def calc_metrics(panel_true: pd.DataFrame, panel_pred: pd.DataFrame,
                 holdout_mask: np.ndarray, jump_threshold=1.0):
    true_arr = panel_true.to_numpy(dtype=np.float32)
    pred_arr = panel_pred.to_numpy(dtype=np.float32)

    y_true = true_arr[holdout_mask]
    y_pred = pred_arr[holdout_mask]
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    jumps = 0
    blocks_total = 0
    for j in range(holdout_mask.shape[1]):
        blocks = _find_blocks(holdout_mask[:, j])
        for s_idx, e_idx in blocks:
            if e_idx + 1 >= holdout_mask.shape[0]:
                continue
            right_true = true_arr[e_idx + 1, j]
            end_pred = pred_arr[e_idx, j]
            if np.isfinite(right_true) and np.isfinite(end_pred):
                blocks_total += 1
                if abs(end_pred - right_true) > jump_threshold:
                    jumps += 1
    jump_rate = float((jumps / max(1, blocks_total)) * 100.0)
    return rmse, mae, jump_rate


def run_with_memory(method_name: str, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    print(f"[DONE] {method_name} 运行完成, 用时 {elapsed:.1f}s, 峰值内存 {peak_mb:.1f} MB")
    return result, peak_mb


def evaluate_once(panel_true: pd.DataFrame, machine_names, holdout_mask,
                  seq_len=6, epochs=10, hidden_size=64, jump_threshold=1.0):
    panel_masked = apply_mask(panel_true, holdout_mask)
    shared_adjacency = core.build_global_adjacency(
        machine_names, panel_masked,
        w_geo=0.4, w_dtw=0.3, w_corr=0.3,
        downsample_step=3, use_cache=False
    )

    methods = [
        ("传统 KNN 插值", lambda: method_knn_interpolation(panel_masked, machine_names, k=5)),
        ("单机自回归（AR代理）", lambda: method_single_machine_ar(panel_masked, seq_len=seq_len)),
        ("常规 STGCN（无掩码与双向补偿）",
         lambda: method_stgcn_no_mask_no_blend(panel_masked, machine_names,
                                               seq_len=seq_len, epochs=epochs,
                                               hidden_size=hidden_size,
                                               adjacency=shared_adjacency)),
        ("本发明方法",
         lambda: method_invention(panel_masked, machine_names,
                                  seq_len=seq_len, epochs=epochs,
                                  hidden_size=hidden_size,
                                  adjacency=shared_adjacency)),
    ]

    rows = []
    for method_name, runner in methods:
        pred_panel, peak_mb = run_with_memory(method_name, runner)
        rmse, mae, jump_rate = calc_metrics(
            panel_true, pred_panel, holdout_mask, jump_threshold=jump_threshold
        )
        rows.append(EvalResult(
            method=method_name,
            rmse=rmse,
            mae=mae,
            boundary_jump_rate=jump_rate,
            peak_memory_mb=peak_mb,
        ))
    return rows


def summarize_results(all_runs):
    records = []
    for run in all_runs:
        for r in run:
            records.append({
                "对比算法模型": r.method,
                "RMSE (m/s)": r.rmse,
                "MAE (m/s)": r.mae,
                "并发断层边界突变率 (%)": r.boundary_jump_rate,
                "框架级内存/占用 (MB)": r.peak_memory_mb,
            })
    df = pd.DataFrame(records)
    out = df.groupby("对比算法模型", as_index=False).mean(numeric_only=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="一键生成专利对比表评测数据")
    parser.add_argument("--data-dir", default="wind_data", help="原始数据目录")
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--missing-rate", type=float, default=0.08,
                        help="人工挖洞比例，建议 0.05~0.12")
    parser.add_argument("--block-min", type=int, default=4)
    parser.add_argument("--block-max", type=int, default=18)
    parser.add_argument("--jump-threshold", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=1,
                        help="重复次数，建议最终报告用 3~5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="专利对比结果_自动评测.csv")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not core.os.path.isabs(data_dir):
        data_dir = core.os.path.join(core.BASE_DIR, data_dir)

    print("=" * 72)
    print("统一评测口径：人工挖洞 + 四方法同集对比")
    print("=" * 72)
    core.preload_machines(data_dir, keep_cols=("OBS",))
    machine_names = sorted(core.machines.keys())
    panel_true = core.build_global_panel(machine_names)

    all_runs = []
    for r in range(args.repeats):
        print(f"\n[RUN {r + 1}/{args.repeats}] 构造评测缺失块...")
        rng = np.random.default_rng(args.seed + r)
        holdout_mask = build_holdout_mask(
            panel_true,
            seq_len=args.seq_len,
            missing_rate=args.missing_rate,
            block_min=args.block_min,
            block_max=args.block_max,
            rng=rng,
        )
        run_rows = evaluate_once(
            panel_true,
            machine_names,
            holdout_mask,
            seq_len=args.seq_len,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            jump_threshold=args.jump_threshold,
        )
        all_runs.append(run_rows)

    summary = summarize_results(all_runs)
    out_path = args.output
    if not core.os.path.isabs(out_path):
        out_path = core.os.path.join(core.BASE_DIR, out_path)
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 180)
    print("\n[对比结果汇总]")
    print(summary.round(4))
    print(f"\n[OK] 已导出: {out_path}")


if __name__ == "__main__":
    main()
