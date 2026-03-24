import importlib.util
from pathlib import Path

import numpy as np

base = Path(__file__).resolve().parent
eval_path = base / "evaluate_table_metrics.py"
spec = importlib.util.spec_from_file_location("ev_module", str(eval_path))
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载评测脚本: {eval_path}")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

core = ev.core
core.preload_machines('wind_data', keep_cols=('OBS',))
machine_names = sorted(core.machines.keys())[:24]
panel_true = core.build_global_panel(machine_names)

rng = np.random.default_rng(42)
mask = ev.build_holdout_mask(
    panel_true,
    seq_len=6,
    missing_rate=0.08,
    block_min=4,
    block_max=12,
    rng=rng,
)
rows = ev.evaluate_once(
    panel_true,
    machine_names,
    mask,
    seq_len=6,
    epochs=2,
    hidden_size=32,
    jump_threshold=1.0,
)

lines = []
for r in rows:
    line = f"{r.method}\t{r.rmse:.4f}\t{r.mae:.4f}\t{r.boundary_jump_rate:.4f}\t{r.peak_memory_mb:.2f}"
    print(line)
    lines.append(line)

with open("quick_fill_numbers_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
