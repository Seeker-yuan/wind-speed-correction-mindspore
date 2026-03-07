# 汇总预测_mindspore版.py
"""
基于MindSpore/深度学习的风速预测系统
改进版：使用神经网络替代Ridge回归
"""

import math
import os
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw

# 尝试导入MindSpore模型
try:
    from mindspore_model_v2 import MindSporeWindPredictor
    MINDSPORE_AVAILABLE = True
    print("✓ 使用MindSpore模型")
except ImportError:
    from wind_neural_model import WindSpeedNeuralNetwork as MindSporeWindPredictor
    MINDSPORE_AVAILABLE = False
    print("⚠ MindSpore不可用，使用通用神经网络模型")

# 全局变量
machines = {}  # {fname: DataFrame(index=timestamp, cols=['OBS'])}
pos = {}       # {fname: [lon, lat]}

def preload_machines(original_dir, keep_cols=('OBS',)):
    """一次性把目录内所有机位读入内存"""
    global machines, pos
    machines.clear()
    pos.clear()

    for fname in os.listdir(original_dir):
        if not fname.endswith('.xlsx'):
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

    print(f"✓ 已预加载 {len(machines)} 台风机")


def load_one(fname):
    """从内存缓存取数据"""
    if fname not in machines:
        raise KeyError(f"machines 中没有 {fname}，请先调用 preload_machines()")
    return machines[fname]


def haversine(lon1, lat1, lon2, lat2, radius=6371.0):
    """计算两经纬度点的大圆距离（km）"""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*radius*math.asin(math.sqrt(a))


def pearson_r(x: np.ndarray, y: np.ndarray):
    """返回皮尔逊相关系数"""
    r = np.corrcoef(x, y)[0, 1]
    return float(r)


def scalar_distance(x, y):
    return abs(x - y)


def get_candidates_for_block_cached(target_id, block_start, block_end, k, n):
    """获取候选风机（基于DTW、Pearson、地理距离）"""
    tgt = load_one(target_id)
    win_start, win_end = block_start - pd.Timedelta(hours=k), block_start - pd.Timedelta(hours=1)

    tgt_lon, tgt_lat = pos[target_id][0], pos[target_id][1]

    # 目标窗口校验
    req_tgt = pd.date_range(win_start, win_end, freq='H')
    if not req_tgt.isin(tgt.index).all(): 
        return [], [], []
    q = tgt.loc[win_start:win_end, 'OBS']
    if len(q) < k or q.isna().any(): 
        return [], [], []

    dtw_cands = []
    pearson_cands = []
    pos_cands = []
    
    for fname, df in machines.items():
        if fname == target_id: 
            continue
        
        # 候选全段覆盖且无 NaN
        req_full = pd.date_range(win_start, block_end, freq='H')
        if not req_full.isin(df.index).all(): 
            continue
        s_full = df.loc[req_full, 'OBS']
        if s_full.isna().any(): 
            continue

        s_win = df.loc[win_start:win_end, 'OBS']
        lon2, lat2 = pos[fname][0], pos[fname][1]

        dtw_dist, _ = fastdtw(q.to_numpy(), s_win.to_numpy(), dist=scalar_distance)
        pearson_dist = pearson_r(q.to_numpy(), s_win.to_numpy())
        pos_dist = haversine(tgt_lon, tgt_lat, lon2, lat2)

        dtw_cands.append((dtw_dist, fname))
        pearson_cands.append((pearson_dist, fname))
        pos_cands.append((pos_dist, fname))

    dtw_cands.sort(key=lambda x: x[0])
    pearson_cands.sort(key=lambda x: x[0], reverse=True)  # Pearson越大越好
    pos_cands.sort(key=lambda x: x[0])

    return dtw_cands[:n], pearson_cands[:n], pos_cands[:n]


def predict_block_gap_neural(target_id, block_start, block_end, k=24, n=4, 
                             use_ensemble=True, epochs=30):
    """
    使用神经网络预测缺失块
    
    Args:
        target_id: 目标风机ID
        block_start: 块起始时间
        block_end: 块结束时间
        k: 历史窗口长度（小时）
        n: 候选风机数量
        use_ensemble: 是否使用集成（DTW+Pearson+Position三路融合）
        epochs: 神经网络训练轮数
        
    Returns:
        dict: {timestamp: predicted_value}
    """
    tgt = load_one(target_id)
    dtw_meta, pearson_meta, pos_meta = get_candidates_for_block_cached(
        target_id, block_start, block_end, k, n
    )
    
    if not dtw_meta:
        print(f"  [BLOCK {block_start}~{block_end}] 候选不足，跳过")
        return {}

    win_start = block_start - pd.Timedelta(hours=k)
    win_end = block_start - pd.Timedelta(hours=1)
    times = pd.date_range(win_start, win_end, freq='H')

    # 准备三路预测
    predictions_list = []
    
    if use_ensemble:
        methods = [
            ('DTW', dtw_meta),
            ('Pearson', pearson_meta),
            ('Position', pos_meta)
        ]
    else:
        methods = [('DTW', dtw_meta)]  # 只用DTW
    
    for method_name, meta in methods:
        cand_names = [name for _, name in meta]
        cand_dfs = [load_one(name) for name in cand_names]
        
        # 训练集（窗口内）
        X = np.vstack([df.loc[times, 'OBS'].to_numpy() for df in cand_dfs]).T  # (k, n)
        y = tgt.loc[times, 'OBS'].to_numpy()
        
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X, y = X[mask], y[mask]
        
        if len(X) < 10:  # 数据太少
            continue
        
        # 创建并训练神经网络
        model = MindSporeWindPredictor(n_neighbors=len(cand_names), hidden_size=64)
        model.fit(X, y, epochs=epochs, batch_size=min(32, len(X)), verbose=False)
        
        # 预测块内数据
        pred_times = pd.date_range(block_start, block_end, freq='H')
        Xp = np.vstack([df.loc[pred_times, 'OBS'].to_numpy() for df in cand_dfs]).T
        preds = model.predict(Xp)
        
        predictions_list.append(dict(zip(pred_times, preds)))
    
    # 融合多路预测
    if not predictions_list:
        return {}
    
    all_times = set()
    for pred_dict in predictions_list:
        all_times.update(pred_dict.keys())
    
    final_preds = {}
    for t in all_times:
        vals = [pred_dict.get(t) for pred_dict in predictions_list if t in pred_dict]
        if vals:
            final_preds[t] = float(np.mean(vals))
    
    return final_preds


def group_consecutive_hours(ts_index):
    """将连续的小时分组"""
    if len(ts_index) == 0: 
        return []
    ts_sorted = ts_index.sort_values()
    blocks, start, prev = [], ts_sorted[0], ts_sorted[0]
    
    for t in ts_sorted[1:]:
        if t - prev != pd.Timedelta(hours=1):
            blocks.append((start, prev))
            start = t
        prev = t
    blocks.append((start, prev))
    return blocks


def fill_machine_neural(fname, k=24, n=4, start_time=None, fill_hour=None, 
                       use_ensemble=True, epochs=30):
    """
    使用神经网络补全风机数据
    
    Args:
        fname: 文件名
        k: 历史窗口长度
        n: 候选风机数量
        start_time: 补全起始时间
        fill_hour: 补全小时数
        use_ensemble: 是否使用集成模型
        epochs: 训练轮数
    """
    df = load_one(fname)

    # 建立原始列与标记列
    if 'OBS_raw' not in df.columns:
        df['OBS_raw'] = df['OBS'].copy()
    if 'filled' not in df.columns:
        df['filled'] = 0
    df['filled'] = df['filled'].astype('int8')

    # 计算补全窗口
    if start_time is not None:
        begin_ts = pd.to_datetime(start_time).floor('H')
    else:
        begin_ts = df.index.min()

    if fill_hour is not None:
        end_ts = begin_ts + pd.Timedelta(hours=int(fill_hour) - 1)
    else:
        end_ts = df.index.max()

    miss_all = df.index[df['OBS'].isna()]
    miss = miss_all[(miss_all >= begin_ts) & (miss_all <= end_ts)]

    if len(miss) == 0:
        print(f"✓ {fname} 在 [{begin_ts} ~ {end_ts}] 无需补全")
        temp_path = r"D:\project\风能ui设计\cleaned_data"
        out_path = os.path.join(temp_path, fname)
        df.to_excel(out_path)
        return df

    # 连续分块
    blocks = []
    for bstart, bend in sorted(group_consecutive_hours(miss)):
        bstart = max(bstart, begin_ts)
        bend = min(bend, end_ts)
        if bstart <= bend:
            blocks.append((bstart, bend))

    print(f"\n{'='*70}")
    print(f"处理风机: {fname}")
    print(f"缺失块数量: {len(blocks)}")
    print(f"{'='*70}")

    # 逐块预测
    for idx, (bstart, bend) in enumerate(blocks, 1):
        print(f"\n[{idx}/{len(blocks)}] 补全 {bstart} ~ {bend}...")
        
        preds = predict_block_gap_neural(
            fname, bstart, bend, k=k, n=n, 
            use_ensemble=use_ensemble, epochs=epochs
        )
        
        # 写回
        for t, v in preds.items():
            if pd.isna(df.at[t, 'OBS']):
                df.at[t, 'OBS'] = v
                df.at[t, 'filled'] = 1
        
        if preds:
            print(f"  ✓ 已补全 {len(preds)} 个数据点")

    # 导出
    temp_path = r"C:\Users\31876\Desktop\风能ui设计\cleaned_data"
    out_path = os.path.join(temp_path, fname)
    df.to_excel(out_path)
    print(f"\n✓ {fname} 已保存到 {out_path}")
    return df


def fill_directory_neural(original_dir, k=24, n=4, start_time=None, fill_hour=None,
                         use_ensemble=True, epochs=30):
    """
    批量补全目录中的所有风机数据
    
    Args:
        original_dir: 数据目录
        k: 历史窗口长度
        n: 候选风机数量
        start_time: 补全起始时间
        fill_hour: 补全小时数
        use_ensemble: 是否使用集成
        epochs: 神经网络训练轮数
    """
    print(f"\n{'='*70}")
    print("开始批量补全（神经网络版）")
    print(f"{'='*70}")
    print(f"模型类型: {'集成模型 (DTW+Pearson+Position)' if use_ensemble else '单一DTW模型'}")
    print(f"历史窗口: {k}小时")
    print(f"候选数量: {n}台")
    print(f"训练轮数: {epochs}轮")
    print(f"{'='*70}\n")
    
    preload_machines(original_dir, keep_cols=('OBS',))
    
    total = len(machines)
    for idx, fname in enumerate(machines.keys(), 1):
        print(f"\n[{idx}/{total}] 开始处理...")
        fill_machine_neural(
            fname, k=k, n=n, start_time=start_time, fill_hour=fill_hour,
            use_ensemble=use_ensemble, epochs=epochs
        )
    
    print(f"\n{'='*70}")
    print("✓ 全部风机处理完成！")
    print(f"{'='*70}")


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

    output_path = r"C:\Users\31876\Desktop\风能ui设计\缺损率报告_neural.xlsx"
    report_df.to_excel(output_path, index=False)

    print(f"✓ 缺损率报告已生成: {output_path}")
    return report_df


if __name__ == "__main__":
    # 数据目录
    DIR = r"C:\Users\31876\Desktop\风能ui设计\wind_data"
    
    # 配置参数
    WINDOW_SIZE = 24      # 历史窗口（小时）
    N_NEIGHBORS = 4       # 候选风机数量
    USE_ENSEMBLE = True   # 使用集成模型
    EPOCHS = 30           # 训练轮数
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║       风能发电机风速误差校正系统 (神经网络版)                    ║
    ║                                                                ║
    ║       基于 MindSpore / PyTorch / sklearn                       ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # 批量补全
    fill_directory_neural(
        DIR, 
        k=WINDOW_SIZE, 
        n=N_NEIGHBORS,
        use_ensemble=USE_ENSEMBLE,
        epochs=EPOCHS
    )
    
    # 生成缺损率报告
    print("\n生成缺损率报告...")
    damage_report = generate_damage_report(DIR)
    
    print("\n" + "="*70)
    print("✓ 所有任务完成！")
    print("="*70)
