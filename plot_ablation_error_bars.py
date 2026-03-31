from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
RAW_PATH = BASE / "ablation_fill_numbers_raw.txt"
SEED47_PATH = BASE / "ablation_seed47.txt"
OUT_DIR = BASE / "汇报图表"
OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "图9_消融实验误差棒对比图.png"

METHOD_ORDER = [
    "本发明完整方法",
    "w/o 掩码阻断",
    "w/o 双向边界平滑",
    "w/o 多视图融合",
]


def parse_line(line):
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None
    method = parts[0]
    seed_part = parts[1]
    try:
        seed = int(seed_part.split("=")[1])
        rmse = float(parts[2])
        mae = float(parts[3])
        jump = float(parts[4])
    except Exception:
        return None
    return method, seed, rmse, mae, jump


def load_records():
    records = []
    for p in [RAW_PATH, SEED47_PATH]:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            rec = parse_line(line)
            if rec is not None:
                records.append(rec)
    return records


def build_stats(records):
    # Prefer seed set [42, 43, 45, 46, 47] to avoid the known overflow seed 44.
    wanted = {42, 43, 45, 46, 47}
    bucket = {k: [] for k in METHOD_ORDER}

    for method, seed, rmse, mae, jump in records:
        if method not in bucket or seed not in wanted:
            continue
        arr = np.array([rmse, mae, jump], dtype=float)
        if not np.isfinite(arr).all():
            continue
        bucket[method].append(arr)

    means, stds = {}, {}
    for m in METHOD_ORDER:
        data = np.array(bucket[m], dtype=float)
        if data.shape[0] < 3:
            raise RuntimeError(f"方法 {m} 的有效样本不足，当前仅 {data.shape[0]} 个")
        means[m] = data.mean(axis=0)
        stds[m] = data.std(axis=0, ddof=1)
    return means, stds


def annotate(ax, bars, values, dy):
    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + dy,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    records = load_records()
    if not records:
        raise FileNotFoundError("未找到消融原始结果文件")

    means, stds = build_stats(records)

    # Chinese font fallback list
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200, facecolor="white")

    labels = METHOD_ORDER
    x = np.arange(len(labels))
    width = 0.36

    rmse_mean = np.array([means[m][0] for m in labels])
    mae_mean = np.array([means[m][1] for m in labels])
    rmse_std = np.array([stds[m][0] for m in labels])
    mae_std = np.array([stds[m][1] for m in labels])

    b1 = ax1.bar(
        x - width / 2,
        rmse_mean,
        width,
        yerr=rmse_std,
        capsize=4,
        label="RMSE",
        color="#4E79A7",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )
    b2 = ax1.bar(
        x + width / 2,
        mae_mean,
        width,
        yerr=mae_std,
        capsize=4,
        label="MAE",
        color="#F28E2B",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )

    ax1.set_title("(a) 精度指标对比（含误差棒）", fontsize=13)
    ax1.set_ylabel("误差值 (m/s)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(1.0, 1.6)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(frameon=True)
    annotate(ax1, b1, rmse_mean, dy=0.01)
    annotate(ax1, b2, mae_mean, dy=0.01)

    jump_mean = np.array([means[m][2] for m in labels])
    jump_std = np.array([stds[m][2] for m in labels])

    colors = []
    for m in labels:
        if m == "w/o 双向边界平滑":
            colors.append("#D62728")  # red highlight
        elif m == "本发明完整方法":
            colors.append("#2CA02C")  # green highlight
        else:
            colors.append("#9AA0A6")

    b3 = ax2.bar(
        x,
        jump_mean,
        yerr=jump_std,
        capsize=4,
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )

    ax2.set_title("(b) 并发断层边界突变率对比（含误差棒）", fontsize=13)
    ax2.set_ylabel("突变率 (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    # Fix y-range for cross-version comparability in review materials.
    ax2.set_ylim(0, 55)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    for rect, val in zip(b3, jump_mean):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.8,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("多模块消融实验对比（5种子均值±标准差）", fontsize=15, y=1.02)
    fig.text(
        0.5,
        0.01,
        "注：误差棒表示样本标准差（n=5，剔除异常溢出种子后补跑有效种子）。",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[OK] saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
