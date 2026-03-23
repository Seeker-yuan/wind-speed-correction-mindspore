import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体，防止图表乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "汇报图表")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _list_cleaned_files():
    if not os.path.isdir(CLEANED_DIR):
        return []
    return sorted([f for f in os.listdir(CLEANED_DIR) if f.endswith(".xlsx")])


def _read_cleaned_file(fname):
    path = os.path.join(CLEANED_DIR, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path, index_col=0)
    if not all(c in df.columns for c in ["OBS", "OBS_raw", "filled"]):
        return None
    df.index = pd.to_datetime(df.index)
    return df


def _find_gap_blocks(filled_series, min_hours=12, max_hours=48):
    """寻找长度在[min_hours, max_hours]的连续补全区间。"""
    arr = (filled_series.to_numpy() == 1)
    blocks = []
    start = None
    for i, flag in enumerate(arr):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            length = i - start
            if min_hours <= length <= max_hours:
                blocks.append((start, i - 1))
            start = None

    if start is not None:
        length = len(arr) - start
        if min_hours <= length <= max_hours:
            blocks.append((start, len(arr) - 1))
    return blocks


def plot_heatmap():
    """生成全局多视图拓扑热力图。"""
    adj_path = os.path.join(CACHE_DIR, "A_global.npy")
    if not os.path.exists(adj_path):
        print("[WARN] 未找到 A_global.npy，跳过图1")
        return None

    adj = np.load(adj_path).astype(np.float32)
    vis = adj.copy()
    np.fill_diagonal(vis, 0.0)

    nz = vis[vis > 0]
    vmax_val = np.percentile(nz, 95) if nz.size > 0 else 1.0

    plt.figure(figsize=(10, 8))
    sns.heatmap(vis, cmap="YlGnBu", xticklabels=False, yticklabels=False, vmax=vmax_val)
    plt.title("图1 全场多视图全局拓扑矩阵 ($A_{global}$)", fontsize=16)
    plt.xlabel("全场风机节点", fontsize=12)
    plt.ylabel("全场风机节点", fontsize=12)

    out_path = os.path.join(OUTPUT_DIR, "1_全局拓扑热力图.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图1已生成: {out_path}")
    return out_path


def plot_single_machine_repair():
    """生成典型单台风机短时修复对比图（优先12-48小时，兜底6-48小时）。"""
    files = _list_cleaned_files()
    best = None
    best_score = -1.0

    for fname in files:
        df = _read_cleaned_file(fname)
        if df is None:
            continue

        blocks = _find_gap_blocks(df["filled"], min_hours=12, max_hours=48)
        if not blocks:
            blocks = _find_gap_blocks(df["filled"], min_hours=6, max_hours=48)
        if not blocks:
            continue

        for s_idx, e_idx in blocks:
            center = (s_idx + e_idx) // 2
            l = max(0, center - 60)
            r = min(len(df), center + 60)
            sub = df.iloc[l:r]
            obs = sub["OBS"].to_numpy(dtype=np.float32)
            score = float(np.nanstd(obs)) + 0.02 * (e_idx - s_idx + 1)
            if score > best_score:
                best_score = score
                best = (fname, df, l, r)

    if best is None:
        print("[WARN] 未找到 6~48 小时缺失区间，跳过图2")
        return None

    fname, df, l, r = best
    sub_df = df.iloc[l:r]

    plt.figure(figsize=(12, 5))
    plt.plot(sub_df.index, sub_df["OBS"], label="ST-GNN 预测修复值", color="#d62728", linewidth=2)
    plt.plot(sub_df.index, sub_df["OBS_raw"], label="原始观测值", color="#1f77b4", alpha=0.6, linestyle="--")

    fill_times = sub_df[sub_df["filled"] == 1].index
    for t in fill_times:
        plt.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                    color="#2ca02c", alpha=0.2, lw=0)

    plt.title(f"图2 典型机组短期风速修复时序图 (节点ID: {fname.split('.')[0]})", fontsize=16)
    plt.ylabel("风速 (m/s)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.gcf().autofmt_xdate()

    out_path = os.path.join(OUTPUT_DIR, "2_典型单机修复图.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图2已生成: {out_path}")
    return out_path


def plot_synchronous_evolution():
    """生成多机位并发缺失同步推演图（限制短时区间，避免长平线）。"""
    files = _list_cleaned_files()
    if len(files) < 3:
        print("[WARN] 机组文件不足，跳过图3")
        return None

    loaded = []
    # 扩大候选机组范围，避免前几个机组恰好没有短时并发缺失
    for f in files[:40]:
        df = _read_cleaned_file(f)
        if df is not None:
            loaded.append((f, df))

    if len(loaded) < 3:
        print("[WARN] 可用机组不足，跳过图3")
        return None

    best = None
    best_score = -1.0
    fallback_used = False
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            for k in range(j + 1, len(loaded)):
                n1, d1 = loaded[i]
                n2, d2 = loaded[j]
                n3, d3 = loaded[k]

                strict_common = ((d1["filled"] == 1) & (d2["filled"] == 1) & (d3["filled"] == 1)).astype(np.int8)
                blocks = _find_gap_blocks(strict_common, min_hours=12, max_hours=48)
                if not blocks:
                    blocks = _find_gap_blocks(strict_common, min_hours=6, max_hours=48)

                # 兜底：允许至少两台机组并发缺失，仍限制短时段避免大平线
                local_fallback = False
                if not blocks:
                    at_least_two = (((d1["filled"] == 1).astype(np.int8)
                                     + (d2["filled"] == 1).astype(np.int8)
                                     + (d3["filled"] == 1).astype(np.int8)) >= 2).astype(np.int8)
                    blocks = _find_gap_blocks(at_least_two, min_hours=12, max_hours=48)
                    if not blocks:
                        blocks = _find_gap_blocks(at_least_two, min_hours=6, max_hours=72)
                    local_fallback = len(blocks) > 0

                if not blocks:
                    continue

                for s_idx, e_idx in blocks:
                    center = (s_idx + e_idx) // 2
                    l = max(0, center - 40)
                    r = min(len(d1), center + 40)
                    score = (
                        float(np.nanstd(d1.iloc[l:r]["OBS"].to_numpy(dtype=np.float32)))
                        + float(np.nanstd(d2.iloc[l:r]["OBS"].to_numpy(dtype=np.float32)))
                        + float(np.nanstd(d3.iloc[l:r]["OBS"].to_numpy(dtype=np.float32)))
                        + 0.02 * (e_idx - s_idx + 1)
                    )
                    if score > best_score:
                        best_score = score
                        best = ((n1, d1), (n2, d2), (n3, d3), l, r)
                        fallback_used = local_fallback

    if best is None:
        # 最终兜底：选一台有短缺失段的典型机组，再选两台高相关机组同窗展示
        anchor = None
        anchor_l = 0
        anchor_r = 0
        for name, df in loaded:
            blocks = _find_gap_blocks(df["filled"], min_hours=12, max_hours=48)
            if not blocks:
                blocks = _find_gap_blocks(df["filled"], min_hours=6, max_hours=72)
            if blocks:
                s_idx, e_idx = blocks[0]
                center = (s_idx + e_idx) // 2
                anchor_l = max(0, center - 40)
                anchor_r = min(len(df), center + 40)
                anchor = (name, df)
                break

        if anchor is None:
            print("[WARN] 未找到可展示的短时缺失区间，跳过图3")
            return None

        an_name, an_df = anchor
        cand = []
        a = an_df["OBS"].to_numpy(dtype=np.float32)
        for name, df in loaded:
            if name == an_name:
                continue
            b = df["OBS"].to_numpy(dtype=np.float32)
            corr = np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))[0, 1]
            if np.isfinite(corr):
                cand.append((float(corr), name, df))
        cand.sort(reverse=True, key=lambda x: x[0])
        if len(cand) < 2:
            print("[WARN] 相关机组不足，跳过图3")
            return None

        (c1, n2, d2), (c2, n3, d3) = cand[0], cand[1]
        best = ((an_name, an_df), (n2, d2), (n3, d3), anchor_l, anchor_r)
        fallback_used = True

    (n1, d1), (n2, d2), (n3, d3), start_idx, end_idx = best
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, (name, df) in zip(axes, [(n1, d1), (n2, d2), (n3, d3)]):
        sub_df = df.iloc[start_idx:end_idx]
        ax.plot(sub_df.index, sub_df["OBS"], color="#d62728", label="ST-GNN 同步推演")
        ax.plot(sub_df.index, sub_df["OBS_raw"], color="#1f77b4", alpha=0.5, linestyle="--", label="原观测数据")

        sync_times = sub_df[sub_df["filled"] == 1].index
        for t in sync_times:
            ax.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                       color="gray", alpha=0.3, lw=0)

        ax.set_ylabel(f"节点 {name.split('.')[0]}", fontsize=10)
        if ax is axes[0]:
            title = "图3 风场多节点并发缺失时的同步推演过程"
            if fallback_used:
                title = "图3 风场多节点缺失联动推演过程（短时并发/同窗参考）"
            ax.set_title(title, fontsize=15)
            ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "3_全局同步演化切片图.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图3已生成: {out_path}")
    return out_path


def main():
    print("重新生成学术级汇报图表...")
    generated = []
    for fn in [plot_heatmap, plot_single_machine_repair, plot_synchronous_evolution]:
        p = fn()
        if p:
            generated.append(p)
    print("[完成] 图表已更新！")
    for p in generated:
        print(" -", p)


if __name__ == "__main__":
    main()
