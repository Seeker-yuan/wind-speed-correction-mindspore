from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed data provided by the user
labels = ["传统 KNN", "单机 LSTM", "常规 STGCN", "本发明方法"]
rmse = np.array([1.9570, 1.6520, 1.4344, 1.4354], dtype=float)
mae = np.array([1.3828, 1.2133, 1.0335, 1.0392], dtype=float)
jump = np.array([49.62, 51.43, 41.44, 0.27], dtype=float)

base = Path(__file__).resolve().parent
out_dir = base / "汇报图表"
out_dir.mkdir(exist_ok=True)
out_png = out_dir / "图10_基线模型性能对比图.png"
out_svg = out_dir / "图10_基线模型性能对比图.svg"
out_png_600 = out_dir / "图10_基线模型性能对比图_审查版_600dpi.png"

# Font fallback for Chinese rendering across machines
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=220, facecolor="white")

x = np.arange(len(labels))
w = 0.36

# Left panel: grouped bars for RMSE / MAE
bars_rmse = ax1.bar(
    x - w / 2,
    rmse,
    width=w,
    color="#4E79A7",
    edgecolor="black",
    linewidth=0.7,
    label="RMSE",
)
bars_mae = ax1.bar(
    x + w / 2,
    mae,
    width=w,
    color="#F28E2B",
    edgecolor="black",
    linewidth=0.7,
    label="MAE",
)

ax1.set_title("(a) 基线模型误差指标对比", fontsize=13)
ax1.set_ylabel("误差值 (m/s)", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=10, ha="right")
ax1.set_ylim(0.9, 2.1)
ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.legend(frameon=True)

for bars in (bars_rmse, bars_mae):
    for b in bars:
        h = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# Right panel: boundary jump rate with strong contrast
colors = ["#B94E48", "#C96A62", "#8C8C8C", "#1F8A5B"]
bars_jump = ax2.bar(
    x,
    jump,
    color=colors,
    edgecolor="black",
    linewidth=0.7,
)

ax2.set_title("(b) 并发断层边界突变率对比", fontsize=13)
ax2.set_ylabel("突变率 (%)", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=10, ha="right")
ax2.set_ylim(0, 56)
ax2.grid(axis="y", linestyle="--", alpha=0.35)

# Add simple legend proxies for visual semantics
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#B94E48", edgecolor="black", label="常规/高风险"),
    Patch(facecolor="#1F8A5B", edgecolor="black", label="本发明/低突变"),
]
ax2.legend(handles=legend_handles, frameon=True, loc="upper right")

for b in bars_jump:
    h = b.get_height()
    ax2.text(
        b.get_x() + b.get_width() / 2,
        h + 0.8,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.suptitle("基线模型性能对比（专利说明书附图）", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
fig.savefig(out_png_600, dpi=600, bbox_inches="tight", facecolor="white")

print(f"[OK] saved: {out_png}")
print(f"[OK] saved: {out_svg}")
print(f"[OK] saved: {out_png_600}")
