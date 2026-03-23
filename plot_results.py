import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chinese font fallback for Windows charts
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
REPORT_PATH = os.path.join(BASE_DIR, "缺损率报告_neural.xlsx")
OUT_DIR = os.path.join(BASE_DIR, "汇报图表")
os.makedirs(OUT_DIR, exist_ok=True)


def _list_cleaned_files():
    if not os.path.isdir(CLEANED_DIR):
        return []
    return sorted([f for f in os.listdir(CLEANED_DIR) if f.endswith(".xlsx")])


def _contiguous_segments(idxs):
    """Convert sorted integer indices to contiguous [start, end] segments."""
    if len(idxs) == 0:
        return []
    segs = []
    start = int(idxs[0])
    prev = int(idxs[0])
    for x in idxs[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
            continue
        segs.append((start, prev))
        start = x
        prev = x
    segs.append((start, prev))
    return segs


def _series_std(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    return float(np.std(arr))


def plot_topology_heatmap():
    """图1: 全局拓扑热力图（回应: 前段找联系）"""
    adj_path = os.path.join(CACHE_DIR, "A_global.npy")
    if not os.path.exists(adj_path):
        print("[WARN] 未找到 A_global.npy，跳过图1")
        return None

    adj = np.load(adj_path).astype(np.float32)
    vis = adj.copy()
    np.fill_diagonal(vis, 0.0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(vis, cmap="YlGnBu", xticklabels=False, yticklabels=False)
    plt.title("图1 全场多视图全局拓扑矩阵 A_global", fontsize=15)
    plt.xlabel("风机节点")
    plt.ylabel("风机节点")

    out = os.path.join(OUT_DIR, "1_全局拓扑热力图.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图1已生成: {out}")
    return out


def _find_best_single_machine_window(files, min_seg_len=6, context=48):
    """Pick the most informative single-machine window for visualization."""
    best = None
    best_score = -1.0

    for fname in files:
        path = os.path.join(CLEANED_DIR, fname)
        df = pd.read_excel(path, index_col=0)
        if not all(c in df.columns for c in ["OBS", "OBS_raw", "filled"]):
            continue

        fill_idx = np.where(df["filled"].to_numpy() == 1)[0]
        segs = _contiguous_segments(fill_idx)
        for s, e in segs:
            seg_len = e - s + 1
            if seg_len < min_seg_len:
                continue

            left = max(0, s - context)
            right = min(len(df), e + context + 1)
            sub = df.iloc[left:right]

            # Require observable values before and after missing segment
            pre_raw = df.iloc[max(0, s - context):s]["OBS_raw"].to_numpy(dtype=np.float32)
            post_raw = df.iloc[e + 1:min(len(df), e + context + 1)]["OBS_raw"].to_numpy(dtype=np.float32)
            pre_valid = int(np.isfinite(pre_raw).sum())
            post_valid = int(np.isfinite(post_raw).sum())
            if pre_valid < 6 or post_valid < 6:
                continue

            # Score: prioritize larger dynamics and sufficiently long repaired segment
            obs_std = _series_std(sub["OBS"].to_numpy(dtype=np.float32))
            raw_std = _series_std(np.concatenate([pre_raw, post_raw]))
            score = 0.7 * obs_std + 0.3 * raw_std + 0.01 * seg_len
            if score > best_score:
                best_score = score
                best = (fname, df, left, right, s, e)

    return best


def plot_single_machine_repair():
    """图2: 单机长序列修复图（回应: 补得准）"""
    files = _list_cleaned_files()
    if not files:
        print("[WARN] cleaned_data 无xlsx文件，跳过图2")
        return None

    best = _find_best_single_machine_window(files, min_seg_len=6, context=48)
    if best is None:
        print("[WARN] 未找到满足展示条件的机组片段，跳过图2")
        return None

    fname, df, left, right, seg_s, seg_e = best
    sub = df.iloc[left:right].copy()

    ts = pd.to_datetime(sub.index)
    plt.figure(figsize=(14, 5))
    plt.plot(ts, sub["OBS"], color="#d62728", linewidth=2, label="ST-GNN补全后")
    plt.plot(ts, sub["OBS_raw"], color="#1f77b4", linewidth=1.4, linestyle="--", alpha=0.7, label="原始观测")

    # highlight missing-repaired region
    mask = (sub["filled"] == 1).to_numpy()
    for i, flag in enumerate(mask):
        if flag:
            t = ts[i]
            plt.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                        color="#2ca02c", alpha=0.18, lw=0)

    plt.title(f"图2 单机位长序列修复效果（{fname.split('.')[0]}）", fontsize=15)
    plt.xlabel("时间")
    plt.ylabel("风速 (m/s)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    out = os.path.join(OUT_DIR, f"2_单机长序列修复_{fname.split('.')[0]}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图2已生成: {out}")

    # Boundary zoom view for presentation clarity
    zoom = 24
    l1, r1 = max(0, seg_s - zoom), min(len(df), seg_s + zoom + 1)
    l2, r2 = max(0, seg_e - zoom), min(len(df), seg_e + zoom + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, (l, r), ttl in [
        (axes[0], (l1, r1), "缺失段起点附近"),
        (axes[1], (l2, r2), "缺失段终点附近"),
    ]:
        win = df.iloc[l:r].copy()
        ts = pd.to_datetime(win.index)
        ax.plot(ts, win["OBS"], color="#d62728", lw=2, label="ST-GNN补全后")
        ax.plot(ts, win["OBS_raw"], color="#1f77b4", lw=1.3, ls="--", alpha=0.7, label="原始观测")
        flags = (win["filled"] == 1).to_numpy()
        for i, flag in enumerate(flags):
            if flag:
                t = ts[i]
                ax.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                           color="#2ca02c", alpha=0.18, lw=0)
        ax.set_title(ttl)
        ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"图2b 单机修复边界放大（{fname.split('.')[0]}）", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_zoom = os.path.join(OUT_DIR, f"2b_单机边界放大_{fname.split('.')[0]}.png")
    plt.savefig(out_zoom, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图2b已生成: {out_zoom}")
    return out


def _load_three_sync_candidates(files):
    loaded = []
    for fname in files:
        path = os.path.join(CLEANED_DIR, fname)
        df = pd.read_excel(path, index_col=0)
        if all(c in df.columns for c in ["OBS", "OBS_raw", "filled"]):
            loaded.append((fname, df))
        if len(loaded) == 6:
            break
    return loaded


def _best_sync_triplet_window(candidates, min_seg_len=3, context=36):
    """Find a 3-machine synchronized missing segment with strongest dynamics."""
    best = None
    best_score = -1.0

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                n1, d1 = candidates[i]
                n2, d2 = candidates[j]
                n3, d3 = candidates[k]

                m1 = d1["filled"].to_numpy() == 1
                m2 = d2["filled"].to_numpy() == 1
                m3 = d3["filled"].to_numpy() == 1
                common = np.where(m1 & m2 & m3)[0]
                segs = _contiguous_segments(common)
                for s, e in segs:
                    seg_len = e - s + 1
                    if seg_len < min_seg_len:
                        continue

                    left = max(0, s - context)
                    right = min(len(d1), e + context + 1)

                    # Ensure each machine has observed raw values before and after segment
                    valid_ok = True
                    dyn_score = 0.0
                    for df in [d1, d2, d3]:
                        pre = df.iloc[max(0, s - context):s]["OBS_raw"].to_numpy(dtype=np.float32)
                        post = df.iloc[e + 1:min(len(df), e + context + 1)]["OBS_raw"].to_numpy(dtype=np.float32)
                        if np.isfinite(pre).sum() < 4 or np.isfinite(post).sum() < 4:
                            valid_ok = False
                            break
                        dyn_score += _series_std(df.iloc[left:right]["OBS"].to_numpy(dtype=np.float32))

                    if not valid_ok:
                        continue

                    score = dyn_score + 0.05 * seg_len
                    if score > best_score:
                        best_score = score
                        best = ((n1, d1), (n2, d2), (n3, d3), left, right, s, e)

    return best


def plot_synchronous_slice():
    """图3: 多机并发缺失同步推演切片图（回应: 后段同步走）"""
    files = _list_cleaned_files()
    if len(files) < 3:
        print("[WARN] cleaned_data 文件数不足，跳过图3")
        return None

    cand = _load_three_sync_candidates(files)
    if len(cand) < 3:
        print("[WARN] 可用机组不足，跳过图3")
        return None

    best = _best_sync_triplet_window(cand, min_seg_len=3, context=36)
    if best is None:
        print("[WARN] 未找到满足展示条件的并发缺失片段，跳过图3")
        return None

    (n1, d1), (n2, d2), (n3, d3), left, right, seg_s, seg_e = best

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, name, df in zip(axes, [n1, n2, n3], [d1, d2, d3]):
        sub = df.iloc[left:right].copy()
        ts = pd.to_datetime(sub.index)
        ax.plot(ts, sub["OBS"], color="#d62728", lw=1.8, label="ST-GNN同步补全")
        ax.plot(ts, sub["OBS_raw"], color="#1f77b4", lw=1.2, ls="--", alpha=0.7, label="原始观测")

        flags = (sub["filled"] == 1).to_numpy()
        for i, flag in enumerate(flags):
            if flag:
                t = ts[i]
                ax.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                           color="gray", alpha=0.25, lw=0)

        ax.set_ylabel(f"{name.split('.')[0]}\n风速")
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].set_title("图3 多机并发缺失时段的全场同步推演切片", fontsize=15)
    axes[0].legend(loc="upper right")
    plt.xlabel("时间")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "3_并发缺失同步推演切片图.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图3已生成: {out}")

    # Boundary zoom for synchronized segment
    zoom = 18
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharey="row")
    for row, (name, df) in enumerate([(n1, d1), (n2, d2), (n3, d3)]):
        for col, center in enumerate([seg_s, seg_e]):
            l = max(0, center - zoom)
            r = min(len(df), center + zoom + 1)
            win = df.iloc[l:r].copy()
            ts = pd.to_datetime(win.index)
            ax = axes[row, col]
            ax.plot(ts, win["OBS"], color="#d62728", lw=1.8)
            ax.plot(ts, win["OBS_raw"], color="#1f77b4", lw=1.1, ls="--", alpha=0.7)
            flags = (win["filled"] == 1).to_numpy()
            for i, flag in enumerate(flags):
                if flag:
                    t = ts[i]
                    ax.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                               color="gray", alpha=0.22, lw=0)
            if row == 0:
                ax.set_title("并发段起点放大" if col == 0 else "并发段终点放大")
            if col == 0:
                ax.set_ylabel(name.split(".")[0])
            ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("图3b 多机并发缺失段边界放大", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_zoom = os.path.join(OUT_DIR, "3b_并发缺失边界放大图.png")
    plt.savefig(out_zoom, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图3b已生成: {out_zoom}")
    return out


def plot_macro_summary():
    """图4: 宏观量化汇总图（缺损率分布+Top10）"""
    if not os.path.exists(REPORT_PATH):
        print("[WARN] 未找到缺损率报告，跳过图4")
        return None

    df = pd.read_excel(REPORT_PATH)
    if "damage_rate" not in df.columns:
        print("[WARN] 报告缺少damage_rate列，跳过图4")
        return None

    rates = df["damage_rate"].to_numpy(dtype=float)
    top = df.sort_values("damage_rate", ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(rates, bins=20, color="#4C78A8", alpha=0.85)
    axes[0].set_title("图4a 全场机组缺损率分布")
    axes[0].set_xlabel("缺损率")
    axes[0].set_ylabel("机组数量")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].barh(top["machine_id"].astype(str), top["damage_rate"], color="#F58518")
    axes[1].invert_yaxis()
    axes[1].set_title("图4b 缺损率 Top10 机组")
    axes[1].set_xlabel("缺损率")
    axes[1].grid(True, axis="x", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "4_宏观量化汇总图.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图4已生成: {out}")
    return out


def export_metrics_text():
    """导出简要量化指标文本，便于粘贴PPT"""
    files = _list_cleaned_files()
    if not files:
        return None

    total_points = 0
    filled_points = 0
    for fname in files:
        df = pd.read_excel(os.path.join(CLEANED_DIR, fname), index_col=0)
        if "filled" not in df.columns:
            continue
        total_points += len(df)
        filled_points += int((df["filled"] == 1).sum())

    txt = []
    txt.append("汇报指标摘要")
    txt.append(f"机组数量: {len(files)}")
    txt.append(f"总时间点: {total_points}")
    txt.append(f"补全点数: {filled_points}")
    txt.append(f"补全占比: {filled_points / max(total_points, 1):.4%}")

    out = os.path.join(OUT_DIR, "5_汇报指标摘要.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    print(f"[OK] 指标摘要已导出: {out}")
    return out


def main():
    print("=" * 60)
    print("正在生成汇报图表（全局同步 ST-GNN）")
    print("=" * 60)
    generated = []
    for fn in [
        plot_topology_heatmap,
        plot_single_machine_repair,
        plot_synchronous_slice,
        plot_macro_summary,
        export_metrics_text,
    ]:
        p = fn()
        if p:
            generated.append(p)

    print("\n[完成] 已生成以下文件:")
    for p in generated:
        print(" -", p)


if __name__ == "__main__":
    main()
