from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
REPORT_PATH = BASE / "缺损率报告_neural.xlsx"
OUT_DIR = BASE / "汇报图表"
OUT_DIR.mkdir(exist_ok=True)

OUT_A4 = OUT_DIR / "4_宏观量化汇总图_A4论文版.png"
OUT_A4_PDF = OUT_DIR / "4_宏观量化汇总图_A4论文版.pdf"
OUT_ROOT = BASE / "image-2.png"


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"未找到缺损率报告: {REPORT_PATH}")

    df = pd.read_excel(REPORT_PATH)
    needed = {"machine_id", "damage_rate"}
    if not needed.issubset(df.columns):
        raise RuntimeError("缺损率报告字段缺失，至少需要 machine_id 与 damage_rate")

    df = df[["machine_id", "damage_rate"]].dropna().copy()
    df["damage_rate"] = pd.to_numeric(df["damage_rate"], errors="coerce")
    df = df.dropna(subset=["damage_rate"]).sort_values("damage_rate", ascending=False)

    top_n = 10
    top = df.head(top_n).sort_values("damage_rate", ascending=True)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27), dpi=320, facecolor="white")

    # Left: overall distribution histogram
    ax1.hist(df["damage_rate"].to_numpy(dtype=float), bins=20, color="#4E79A7", edgecolor="white", alpha=0.9)
    ax1.set_title("(a) 全场机组缺损率分布", fontsize=24)
    ax1.set_xlabel("缺损率", fontsize=18)
    ax1.set_ylabel("机组数量", fontsize=18)
    ax1.tick_params(axis="both", labelsize=16)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Right: top-N missing rate ranking
    bars = ax2.barh(
        top["machine_id"].astype(str).to_numpy(),
        top["damage_rate"].to_numpy(dtype=float),
        color="#F28E2B",
        edgecolor="white",
        alpha=0.95,
    )
    ax2.set_title(f"(b) 缺损率最高机组 Top{top_n}", fontsize=24)
    ax2.set_xlabel("缺损率", fontsize=18)
    ax2.tick_params(axis="both", labelsize=16)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    max_rate = float(top["damage_rate"].max()) if not top.empty else 0.0
    ax2.set_xlim(0.0, max_rate * 1.10 if max_rate > 0 else 1.0)

    for b in bars:
        w = float(b.get_width())
        ax2.text(
            w + max_rate * 0.012,
            b.get_y() + b.get_height() / 2,
            f"{w:.4f}",
            va="center",
            ha="left",
            fontsize=14,
        )

    fig.suptitle("全场缺损率统计概览", fontsize=30, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_A4, dpi=420, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_A4_PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_ROOT, dpi=360, bbox_inches="tight", facecolor="white")

    print(f"[OK] saved: {OUT_A4}")
    print(f"[OK] saved: {OUT_A4_PDF}")
    print(f"[OK] saved: {OUT_ROOT}")


if __name__ == "__main__":
    main()
