# 汇总预测_mindspore版.py
"""
基于MindSpore/深度学习的风速预测系统
改进版：使用神经网络替代Ridge回归
"""

import math
import os
import json
import pandas as pd
import numpy as np
import time
from fastdtw import fastdtw

# 当前脚本所在目录，所有路径基于此构建
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 导入MindSpore时空图神经网络模型
try:
    from mindspore_gnn_model import MindSporeWindPredictor
    MINDSPORE_AVAILABLE = True
    print("[OK] MindSpore ST-GNN 4-layer GCN + Attention")
except ImportError:
    from sklearn.neural_network import MLPRegressor
    class MindSporeWindPredictor:
        def __init__(self, n_neighbors=4, hidden_size=64):
            self.model = MLPRegressor(hidden_layer_sizes=(hidden_size, hidden_size, 32),
                                      max_iter=200, random_state=42)
        def fit(self, X, y, epochs=30, batch_size=32, verbose=False, **kw):
            self.model.fit(X, y)
        def predict(self, X, **kw):
            return self.model.predict(X)
    MINDSPORE_AVAILABLE = False
    print("[WARN] MindSpore not available, using sklearn fallback")

# 全局变量
machines = {}  # {fname: DataFrame(index=timestamp, cols=['OBS'])}
pos = {}       # {fname: [lon, lat]}

def preload_machines(original_dir, keep_cols=('OBS',)):
    """一次性把目录内所有机位读入内存"""
    global machines, pos
    machines.clear()
    pos.clear()

    for fname in os.listdir(original_dir):
        if not fname.endswith('.xlsx') or fname.startswith('~$'):
            continue
        path = os.path.join(original_dir, fname)
        df = pd.read_excel(path)
        position = []
        
        # 规范时间戳
        df['time'] = pd.to_datetime(df['time'])
        df['dtime'] = pd.to_numeric(df['dtime'], errors='coerce')
        df['timestamp'] = df['time'].dt.normalize() + pd.to_timedelta(df['dtime'], unit='h')
        df = df.set_index('timestamp')
        
        position.append(float(df['lon'].iloc[0]))
        position.append(float(df['lat'].iloc[0]))
        
        # 只留需要的列
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols].copy()

        # 统一成 float32
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

        machines[fname] = df
        pos[fname] = position

    print(f"[OK] 已预加载 {len(machines)} 台风机")


def load_one(fname):
    """从内存缓存取数据"""
    if fname not in machines:
        raise KeyError(f"machines 中没有 {fname}，请先调用 preload_machines()")
    return machines[fname]


def build_global_panel(machine_names):
    """把全场风机按统一小时索引拼接成面板数据: index=时间, columns=风机"""
    starts = [machines[name].index.min() for name in machine_names]
    ends = [machines[name].index.max() for name in machine_names]
    global_index = pd.date_range(min(starts), max(ends), freq='h')

    panel = pd.DataFrame(index=global_index)
    for name in machine_names:
        panel[name] = machines[name]['OBS'].reindex(global_index)
    return panel


def _fill_for_features(df):
    """仅使用历史信息补全特征，避免时间泄露。"""
    filled = df.copy()
    filled = filled.ffill()
    filled = filled.fillna(0.0)
    return filled


def _geo_similarity(machine_names):
    n = len(machine_names)
    dist = np.zeros((n, n), dtype=np.float32)

    for i, ni in enumerate(machine_names):
        lon1, lat1 = pos[ni]
        for j, nj in enumerate(machine_names):
            if i == j:
                continue
            lon2, lat2 = pos[nj]
            dist[i, j] = haversine(lon1, lat1, lon2, lat2)

    sigma = np.median(dist[dist > 0]) + 1e-8
    sim = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


def _corr_similarity(panel_filled):
    corr = panel_filled.corr(method='pearson').to_numpy(dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0)
    # [-1, 1] 映射到 [0, 1]
    sim = (corr + 1.0) / 2.0
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def _dtw_similarity(panel_filled, downsample_step=3):
    arr = panel_filled.to_numpy(dtype=np.float32)
    arr = arr[::downsample_step] if downsample_step > 1 else arr
    n = arr.shape[1]

    dtw_dist = np.zeros((n, n), dtype=np.float32)
    upper_vals = []

    for i in range(n):
        for j in range(i + 1, n):
            d, _ = fastdtw(arr[:, i], arr[:, j], dist=scalar_distance)
            dtw_dist[i, j] = d
            dtw_dist[j, i] = d
            upper_vals.append(d)

    scale = np.median(upper_vals) + 1e-8 if upper_vals else 1.0
    sim = np.exp(-dtw_dist / scale).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


def _build_adj_cache_meta(machine_names, panel, w_geo, w_dtw, w_corr, downsample_step):
    return {
        'machine_names': list(machine_names),
        'n_nodes': int(len(machine_names)),
        'panel_start': str(panel.index.min()),
        'panel_end': str(panel.index.max()),
        'w_geo': float(w_geo),
        'w_dtw': float(w_dtw),
        'w_corr': float(w_corr),
        'downsample_step': int(downsample_step),
    }


def _load_cached_adjacency(cache_np_path, cache_meta_path, expected_meta):
    if not (os.path.exists(cache_np_path) and os.path.exists(cache_meta_path)):
        return None
    try:
        with open(cache_meta_path, 'r', encoding='utf-8') as f:
            saved_meta = json.load(f)
        if saved_meta != expected_meta:
            return None
        adj = np.load(cache_np_path)
        if adj.shape != (expected_meta['n_nodes'], expected_meta['n_nodes']):
            return None
        return adj.astype(np.float32)
    except Exception:
        return None


def _save_cached_adjacency(cache_np_path, cache_meta_path, meta, adjacency):
    os.makedirs(os.path.dirname(cache_np_path), exist_ok=True)
    np.save(cache_np_path, adjacency.astype(np.float32))
    with open(cache_meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_global_adjacency(machine_names, panel, w_geo=0.4, w_dtw=0.3, w_corr=0.3,
                           downsample_step=3, use_cache=True):
    """构建多视图融合邻接矩阵 A_global"""
    cache_np_path = os.path.join(BASE_DIR, 'cache', 'A_global.npy')
    cache_meta_path = os.path.join(BASE_DIR, 'cache', 'A_global.meta.json')
    expected_meta = _build_adj_cache_meta(
        machine_names, panel, w_geo, w_dtw, w_corr, downsample_step
    )

    if use_cache:
        cached = _load_cached_adjacency(cache_np_path, cache_meta_path, expected_meta)
        if cached is not None:
            print(f"[OK] 已加载缓存邻接矩阵: {cache_np_path}")
            return cached

    panel_filled = _fill_for_features(panel)
    a_geo = _geo_similarity(machine_names)
    a_dtw = _dtw_similarity(panel_filled, downsample_step=downsample_step)
    a_corr = _corr_similarity(panel_filled)

    a_global = w_geo * a_geo + w_dtw * a_dtw + w_corr * a_corr
    a_global = np.clip(a_global, 0.0, None).astype(np.float32)

    # 加自环并行归一化
    a_global = a_global + np.eye(a_global.shape[0], dtype=np.float32)
    deg = a_global.sum(axis=1, keepdims=True) + 1e-8
    a_global = a_global / deg

    if use_cache:
        _save_cached_adjacency(cache_np_path, cache_meta_path, expected_meta, a_global)
        print(f"[OK] 已保存邻接矩阵缓存: {cache_np_path}")

    return a_global.astype(np.float32)


def build_global_training_set(panel, seq_len=6):
    """构造全局训练集:
    X: (N_samples, N_nodes, seq_len)
    y: (N_samples, N_nodes)
    mask: (N_samples, N_nodes)
    """
    feat_panel = _fill_for_features(panel)
    x_list, y_list, m_list = [], [], []

    for t in range(seq_len, len(panel)):
        x_win = feat_panel.iloc[t - seq_len:t].to_numpy(dtype=np.float32).T
        y_vec = panel.iloc[t].to_numpy(dtype=np.float32)
        m_vec = (~np.isnan(y_vec)).astype(np.float32)
        if m_vec.sum() < 1:
            continue
        x_list.append(x_win)
        y_list.append(y_vec)
        m_list.append(m_vec)

    if not x_list:
        raise ValueError("全局训练样本为空，请检查数据是否全部缺失")

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    m = np.array(m_list, dtype=np.float32)
    return x, y, m


def synchronous_fill_panel(panel, model, adjacency, seq_len=6):
    """全场同步逐时推理补全"""
    filled_panel = panel.copy()

    for t in range(seq_len, len(filled_panel)):
        miss_mask = filled_panel.iloc[t].isna().to_numpy()
        if not miss_mask.any():
            continue

        ctx = filled_panel.iloc[t - seq_len:t]
        ctx = _fill_for_features(ctx)
        x = ctx.to_numpy(dtype=np.float32).T[np.newaxis, :, :]
        pred_all = model.predict(x, adjacency=adjacency)[0]

        row = filled_panel.iloc[t].to_numpy(dtype=np.float32)
        row[miss_mask] = pred_all[miss_mask]
        filled_panel.iloc[t] = row

    # 处理最开头不足 seq_len 的少量缺失
    filled_panel = _fill_for_features(filled_panel)
    return filled_panel


def save_global_filled_results(machine_names, panel_raw, panel_filled, out_dir):
    """按风机拆分导出补全结果"""
    os.makedirs(out_dir, exist_ok=True)

    for name in machine_names:
        raw = panel_raw[name]
        new = panel_filled[name]
        out_df = pd.DataFrame(index=panel_filled.index)
        out_df['OBS_raw'] = raw.astype('float32')
        out_df['OBS'] = new.astype('float32')
        out_df['filled'] = (raw.isna() & new.notna()).astype('int8')
        out_df.to_excel(os.path.join(out_dir, name))


def run_global_synchronous_pipeline(original_dir, seq_len=6, epochs=10,
                                    hidden_size=64, w_geo=0.4, w_dtw=0.3, w_corr=0.3,
                                    dtw_downsample_step=3, use_adj_cache=True):
    """两阶段全局方案:
    1) 全局关系图构建 (A_global)
    2) 全局模型训练 + 同步逐时补全
    """
    print(f"\n{'='*70}")
    print("阶段1/2: 全局关系图构建")
    print(f"{'='*70}")
    preload_machines(original_dir, keep_cols=('OBS',))
    machine_names = sorted(machines.keys())
    panel_raw = build_global_panel(machine_names)
    adjacency = build_global_adjacency(machine_names, panel_raw,
                                       w_geo=w_geo, w_dtw=w_dtw, w_corr=w_corr,
                                       downsample_step=dtw_downsample_step,
                                       use_cache=use_adj_cache)

    print(f"\n{'='*70}")
    print("阶段2/2: 全局同步模型训练与补全")
    print(f"{'='*70}")
    x_train, y_train, m_train = build_global_training_set(panel_raw, seq_len=seq_len)
    print(f"训练样本数: {x_train.shape[0]}, 节点数: {x_train.shape[1]}, 序列长度: {x_train.shape[2]}")

    model = MindSporeWindPredictor(n_neighbors=len(machine_names),
                                   hidden_size=hidden_size, seq_len=seq_len)
    model.fit(x_train, y_train, epochs=epochs,
              batch_size=min(64, len(x_train)),
              verbose=True, adjacency=adjacency, mask=m_train)

    panel_filled = synchronous_fill_panel(panel_raw, model, adjacency, seq_len=seq_len)
    out_dir = os.path.join(BASE_DIR, 'cleaned_data')
    save_global_filled_results(machine_names, panel_raw, panel_filled, out_dir)
    print(f"[OK] 全场同步补全完成，结果输出到: {out_dir}")
    return panel_raw, panel_filled


def haversine(lon1, lat1, lon2, lat2, radius=6371.0):
    """计算两经纬度点的大圆距离（km）"""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def scalar_distance(x, y):
    return abs(x - y)


def generate_damage_report(original_dir):
    """生成风机缺损率报告"""
    damage_rates = []

    for fname in os.listdir(original_dir):
        if not fname.endswith('.xlsx'):
            continue

        try:
            df = pd.read_excel(os.path.join(original_dir, fname))
            df['time'] = pd.to_datetime(df['time'])
            df['dtime'] = pd.to_numeric(df['dtime'], errors='coerce')
            df['timestamp'] = df['time'].dt.normalize() + pd.to_timedelta(df['dtime'], unit='h')
            df = df.set_index('timestamp')

            total_points = len(df)
            missing_points = df['OBS'].isna().sum()
            damage_rate = missing_points / total_points

            machine_id = fname.split('_')[1].split('.')[0]

            damage_rates.append({
                'machine_id': machine_id,
                'damage_rate': damage_rate,
                'missing_count': missing_points,
                'total_count': total_points
            })

        except Exception as e:
            print(f"处理 {fname} 时出错: {e}")

    damage_rates.sort(key=lambda x: x['damage_rate'], reverse=True)
    report_df = pd.DataFrame(damage_rates)

    output_path = os.path.join(BASE_DIR, "缺损率报告_neural.xlsx")
    report_df.to_excel(output_path, index=False)

    print(f"[OK] 缺损率报告已生成: {output_path}")
    return report_df


if __name__ == "__main__":
    # 数据目录（相对路径，无需修改）
    DIR = os.path.join(BASE_DIR, "wind_data")
    
    # 全局同步模型配置参数
    SEQ_LEN = 6
    EPOCHS = 10
    HIDDEN_SIZE = 64
    W_GEO, W_DTW, W_CORR = 0.4, 0.3, 0.3
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║       风能发电机风速误差校正系统                  ║
    ║                                                                ║
    ║       基于 MindSpore                     ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    t_start = time.time()

    # 全局两阶段流程：关系图构建 + 同步补全
    run_global_synchronous_pipeline(
        DIR,
        seq_len=SEQ_LEN,
        epochs=EPOCHS,
        hidden_size=HIDDEN_SIZE,
        w_geo=W_GEO,
        w_dtw=W_DTW,
        w_corr=W_CORR
    )
    
    # 生成缺损率报告
    print("\n生成缺损率报告...")
    damage_report = generate_damage_report(DIR)
    
    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    print("\n" + "="*70)
    print("[OK] 所有任务完成！")
    print("="*70)
