from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

A4_LANDSCAPE = (11.69, 8.27)
FIG_DPI = 360
FS_SUPTITLE = 36
FS_SUBTITLE = 22
FS_AXIS = 24
FS_TICK = 18
FS_LEGEND = 20
FS_ANNOT = 18

# Fixed data provided by the user
labels = ["传统 KNN", "单机 LSTM", "常规 STGCN", "本发明方法"]
tick_labels = ["传统\nKNN", "单机\nLSTM", "常规\nSTGCN", "本发明\n方法"]
rmse = np.array([1.9570, 1.6520, 1.4344, 1.4354], dtype=float)
mae = np.array([1.3828, 1.2133, 1.0335, 1.0392], dtype=float)
jump = np.array([49.62, 51.43, 41.44, 0.27], dtype=float)

base = Path(__file__).resolve().parent
out_dir = base / "汇报图表"
out_dir.mkdir(exist_ok=True)
out_png_a4 = out_dir / "图10_基线模型性能对比图_A4论文版.png"
out_pdf_a4 = out_dir / "图10_基线模型性能对比图_A4论文版.pdf"

# Font fallback for Chinese rendering across machines
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=A4_LANDSCAPE, dpi=FIG_DPI, facecolor="white")

x = np.arange(len(labels))
w = 0.36

# Left panel: grouped bars for RMSE / MAE
bars_rmse = ax1.bar(
    x - w / 2,
    rmse,
    width=w,
    color="#4E79A7",
    edgecolor="black",
    linewidth=1.0,
    label="RMSE",
)
bars_mae = ax1.bar(
    x + w / 2,
    mae,
    width=w,
    color="#F28E2B",
    edgecolor="black",
    linewidth=1.0,
    label="MAE",
)

ax1.set_title("(a) 基线模型误差指标对比", fontsize=FS_SUBTITLE, pad=10)
ax1.set_ylabel("误差值 (m/s)", fontsize=FS_AXIS)
ax1.set_xticks(x)
ax1.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=FS_TICK)
ax1.set_ylim(0.9, 2.1)
ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.tick_params(axis="y", labelsize=FS_TICK)
ax1.legend(frameon=True, fontsize=FS_LEGEND)

for bars in (bars_rmse, bars_mae):
    for b in bars:
        h = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=FS_ANNOT,
        )

# Right panel: boundary jump rate with strong contrast
colors = ["#B94E48", "#C96A62", "#8C8C8C", "#1F8A5B"]
bars_jump = ax2.bar(
    x,
    jump,
    color=colors,
    edgecolor="black",
    linewidth=1.0,
)

ax2.set_title("(b) 并发断层边界突变率对比", fontsize=FS_SUBTITLE, pad=10)
ax2.set_ylabel("突变率 (%)", fontsize=FS_AXIS)
ax2.set_xticks(x)
ax2.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=FS_TICK)
ax2.set_ylim(0, 56)
ax2.grid(axis="y", linestyle="--", alpha=0.35)
ax2.tick_params(axis="y", labelsize=FS_TICK)

# Right panel uses 4 colors; provide complete legend entries.
legend_handles = [
    Patch(facecolor="#B94E48", edgecolor="black", label="传统 KNN"),
    Patch(facecolor="#C96A62", edgecolor="black", label="单机 LSTM"),
    Patch(facecolor="#8C8C8C", edgecolor="black", label="常规 STGCN"),
    Patch(facecolor="#1F8A5B", edgecolor="black", label="本发明方法"),
]
ax2.legend(
    handles=legend_handles,
    frameon=True,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.0,
    fontsize=FS_LEGEND,
)

for b in bars_jump:
    h = b.get_height()
    ax2.text(
        b.get_x() + b.get_width() / 2,
        h + 0.8,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=FS_ANNOT,
    )

fig.suptitle("基线模型性能对比（专利说明书附图）", fontsize=FS_SUPTITLE, y=0.98)
fig.tight_layout(rect=[0.02, 0.03, 0.93, 0.94], w_pad=2.8)
fig.savefig(out_png_a4, dpi=420, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf_a4, bbox_inches="tight", facecolor="white")

print(f"[OK] saved: {out_png_a4}")
print(f"[OK] saved: {out_pdf_a4}")
