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

# =========================
# Audit 固定模式参数
# =========================
AUDIT_SINGLE_MACHINE = "id_53588.xlsx"
AUDIT_SINGLE_START = "2024-01-30 00:00:00"
AUDIT_SINGLE_END = "2024-03-02 23:00:00"

AUDIT_SYNC_MACHINES = ["id_53575.xlsx", "id_53588.xlsx", "id_53590.xlsx"]
AUDIT_SYNC_START = "2024-01-30 12:00:00"
AUDIT_SYNC_END = "2024-03-02 11:00:00"


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


def plot_topology_heatmap():
    """图1: 全局拓扑热力图（固定显示策略）"""
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
    sns.heatmap(
        vis,
        cmap="YlGnBu",
        xticklabels=False,
        yticklabels=False,
        vmax=vmax_val,
    )
    plt.title("图1 全场多视图全局拓扑矩阵 A_global", fontsize=15)
    plt.xlabel("风机节点")
    plt.ylabel("风机节点")

    out = os.path.join(OUT_DIR, "1_全局拓扑热力图.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图1已生成: {out}")
    return out


def plot_single_machine_repair_audit():
    """图2: 固定机组固定时段修复图"""
    df = _read_cleaned_file(AUDIT_SINGLE_MACHINE)
    if df is None:
        print(f"[WARN] 未找到审计机组 {AUDIT_SINGLE_MACHINE}，跳过图2")
        return None

    sub = df.loc[pd.to_datetime(AUDIT_SINGLE_START):pd.to_datetime(AUDIT_SINGLE_END)].copy()
    if len(sub) == 0:
        print("[WARN] 审计时段无数据，跳过图2")
        return None

    plt.figure(figsize=(14, 5))
    plt.plot(sub.index, sub["OBS"], color="#d62728", linewidth=1.8, label="ST-GNN补全后")
    plt.plot(sub.index, sub["OBS_raw"], color="#1f77b4", linewidth=1.2, linestyle="--", alpha=0.75, label="原始观测")

    flags = (sub["filled"] == 1).to_numpy()
    for i, flag in enumerate(flags):
        if flag:
            t = sub.index[i]
            plt.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                        color="#2ca02c", alpha=0.18, lw=0)

    plt.title(f"图2 固定审计机组修复曲线（{AUDIT_SINGLE_MACHINE.split('.')[0]}）", fontsize=15)
    plt.xlabel("时间")
    plt.ylabel("风速 (m/s)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.gcf().autofmt_xdate()

    out = os.path.join(OUT_DIR, "2_单机长序列修复_AUDIT.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图2已生成: {out}")
    return out


def plot_single_machine_boundary_zoom_audit():
    """图2b: 固定机组边界放大，展示缝合连续性"""
    df = _read_cleaned_file(AUDIT_SINGLE_MACHINE)
    if df is None:
        print(f"[WARN] 未找到审计机组 {AUDIT_SINGLE_MACHINE}，跳过图2b")
        return None

    sub = df.loc[pd.to_datetime(AUDIT_SINGLE_START):pd.to_datetime(AUDIT_SINGLE_END)].copy()
    if len(sub) == 0:
        print("[WARN] 审计时段无数据，跳过图2b")
        return None

    filled_pos = np.where(sub["filled"].to_numpy() == 1)[0]
    if len(filled_pos) == 0:
        print("[WARN] 固定审计时段无补全点，跳过图2b")
        return None

    seg_start = int(filled_pos[0])
    seg_end = int(filled_pos[-1])
    zoom = 24

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, center, ttl in [
        (axes[0], seg_start, "缺失段起点附近"),
        (axes[1], seg_end, "缺失段终点附近"),
    ]:
        l = max(0, center - zoom)
        r = min(len(sub), center + zoom + 1)
        win = sub.iloc[l:r]
        ax.plot(win.index, win["OBS"], color="#d62728", lw=1.8, label="ST-GNN补全后")
        ax.plot(win.index, win["OBS_raw"], color="#1f77b4", lw=1.2, ls="--", alpha=0.75, label="原始观测")
        flags = (win["filled"] == 1).to_numpy()
        for i, flag in enumerate(flags):
            if flag:
                t = win.index[i]
                ax.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                           color="#2ca02c", alpha=0.18, lw=0)
        ax.set_title(ttl)
        ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("图2b 固定审计机组边界放大", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.gcf().autofmt_xdate()

    out = os.path.join(OUT_DIR, "2b_单机边界放大_AUDIT.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图2b已生成: {out}")
    return out


def plot_synchronous_slice_audit():
    """图3: 固定三机固定时段同步演化图"""
    dfs = []
    for n in AUDIT_SYNC_MACHINES:
        df = _read_cleaned_file(n)
        if df is None:
            print(f"[WARN] 未找到审计机组 {n}，跳过图3")
            return None
        dfs.append((n, df.loc[pd.to_datetime(AUDIT_SYNC_START):pd.to_datetime(AUDIT_SYNC_END)].copy()))

    if any(len(df) == 0 for _, df in dfs):
        print("[WARN] 固定同步时段无数据，跳过图3")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, (name, sub) in zip(axes, dfs):
        ax.plot(sub.index, sub["OBS"], color="#d62728", lw=1.8, label="ST-GNN同步补全")
        ax.plot(sub.index, sub["OBS_raw"], color="#1f77b4", lw=1.2, ls="--", alpha=0.75, label="原始观测")

        flags = (sub["filled"] == 1).to_numpy()
        for i, flag in enumerate(flags):
            if flag:
                t = sub.index[i]
                ax.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                           color="gray", alpha=0.25, lw=0)

        ax.set_ylabel(f"{name.split('.')[0]}\n风速")
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].set_title("图3 固定审计时段并发缺失同步推演", fontsize=15)
    axes[0].legend(loc="upper right")
    plt.xlabel("时间")
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    out = os.path.join(OUT_DIR, "3_并发缺失同步推演切片图_AUDIT.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图3已生成: {out}")
    return out


def plot_synchronous_boundary_zoom_audit():
    """图3b: 固定同步段边界放大"""
    pairs = []
    for n in AUDIT_SYNC_MACHINES:
        df = _read_cleaned_file(n)
        if df is None:
            print(f"[WARN] 未找到审计机组 {n}，跳过图3b")
            return None
        sub = df.loc[pd.to_datetime(AUDIT_SYNC_START):pd.to_datetime(AUDIT_SYNC_END)].copy()
        if len(sub) == 0:
            print("[WARN] 固定同步时段无数据，跳过图3b")
            return None
        pairs.append((n, sub))

    ref = pairs[0][1]
    idx = np.where(ref["filled"].to_numpy() == 1)[0]
    if len(idx) == 0:
        print("[WARN] 固定同步时段无补全点，跳过图3b")
        return None

    seg_start = int(idx[0])
    seg_end = int(idx[-1])
    zoom = 18

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharey="row")
    for r, (name, sub) in enumerate(pairs):
        for c, center in enumerate([seg_start, seg_end]):
            l = max(0, center - zoom)
            rr = min(len(sub), center + zoom + 1)
            win = sub.iloc[l:rr]

            ax = axes[r, c]
            ax.plot(win.index, win["OBS"], color="#d62728", lw=1.7)
            ax.plot(win.index, win["OBS_raw"], color="#1f77b4", lw=1.1, ls="--", alpha=0.75)
            flags = (win["filled"] == 1).to_numpy()
            for i, flag in enumerate(flags):
                if flag:
                    t = win.index[i]
                    ax.axvspan(t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30),
                               color="gray", alpha=0.22, lw=0)

            if r == 0:
                ax.set_title("并发段起点放大" if c == 0 else "并发段终点放大")
            if c == 0:
                ax.set_ylabel(name.split(".")[0])
            ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("图3b 固定审计同步段边界放大", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.gcf().autofmt_xdate()

    out = os.path.join(OUT_DIR, "3b_并发缺失边界放大图_AUDIT.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图3b已生成: {out}")
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
    txt.append(f"审计机组: {AUDIT_SINGLE_MACHINE}")
    txt.append(f"审计时段: {AUDIT_SINGLE_START} -> {AUDIT_SINGLE_END}")
    txt.append(f"审计并发机组: {', '.join(AUDIT_SYNC_MACHINES)}")
    txt.append(f"审计并发时段: {AUDIT_SYNC_START} -> {AUDIT_SYNC_END}")

    out = os.path.join(OUT_DIR, "5_汇报指标摘要.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    print(f"[OK] 指标摘要已导出: {out}")
    return out


def main():
    print("=" * 60)
    print("正在生成汇报图表（Audit 固定模式）")
    print("=" * 60)

    generated = []
    for fn in [
        plot_topology_heatmap,
        plot_single_machine_repair_audit,
        plot_single_machine_boundary_zoom_audit,
        plot_synchronous_slice_audit,
        plot_synchronous_boundary_zoom_audit,
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
