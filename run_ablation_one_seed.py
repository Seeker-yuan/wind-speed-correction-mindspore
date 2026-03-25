import importlib.util
from pathlib import Path

import numpy as np

base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ab_module", str(base / "run_ablation_numbers.py"))
if spec is None or spec.loader is None:
    raise ImportError("无法加载 run_ablation_numbers.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

ev = ab.ev
core = ab.core


def main(seed=47):
    core.preload_machines("wind_data", keep_cols=("OBS",))
    if not getattr(core, "MINDSPORE_AVAILABLE", False):
        raise RuntimeError("当前未启用MindSpore主路径")

    machine_names = sorted(core.machines.keys())[:24]
    panel_true = core.build_global_panel(machine_names)

    rng = np.random.default_rng(seed)
    holdout_mask = ev.build_holdout_mask(
        panel_true,
        seq_len=6,
        missing_rate=0.08,
        block_min=4,
        block_max=12,
        rng=rng,
    )

    rows = [
        ("本发明完整方法", *ab.run_invention(panel_true, machine_names, holdout_mask)),
        ("w/o 掩码阻断", *ab.run_wo_mask(panel_true, machine_names, holdout_mask)),
        ("w/o 双向边界平滑", *ab.run_wo_blending(panel_true, machine_names, holdout_mask)),
        ("w/o 多视图融合", *ab.run_wo_multiview(panel_true, machine_names, holdout_mask)),
    ]

    out = base / "ablation_seed47.txt"
    lines = []
    for name, rmse, mae, jump in rows:
        lines.append(f"{name}\tseed={seed}\t{rmse:.4f}\t{mae:.4f}\t{jump:.4f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines:
        print(line)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main(47)
