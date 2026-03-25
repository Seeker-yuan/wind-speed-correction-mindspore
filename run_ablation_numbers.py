import importlib.util
from pathlib import Path

import numpy as np

# Load evaluate script by path to avoid module resolution issues.
base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ev_module", str(base / "evaluate_table_metrics.py"))
if spec is None or spec.loader is None:
    raise ImportError("无法加载 evaluate_table_metrics.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

core = ev.core


def _build_adjacency(panel_masked, machine_names, w_geo, w_dtw, w_corr):
    return core.build_global_adjacency(
        machine_names,
        panel_masked,
        w_geo=w_geo,
        w_dtw=w_dtw,
        w_corr=w_corr,
        downsample_step=3,
        use_cache=False,
    )


def run_invention(panel_true, machine_names, holdout_mask, seq_len=6, epochs=2, hidden_size=32):
    panel_masked = ev.apply_mask(panel_true, holdout_mask)
    adjacency = _build_adjacency(panel_masked, machine_names, 0.4, 0.3, 0.3)
    pred = ev.method_invention(
        panel_masked,
        machine_names,
        seq_len=seq_len,
        epochs=epochs,
        hidden_size=hidden_size,
        adjacency=adjacency,
    )
    return ev.calc_metrics(panel_true, pred, holdout_mask, jump_threshold=1.0)


def run_wo_mask(panel_true, machine_names, holdout_mask, seq_len=6, epochs=2, hidden_size=32):
    """w/o 掩码阻断: 训练时不传mask，但保留边界平滑。"""
    panel_masked = ev.apply_mask(panel_true, holdout_mask)
    adjacency = _build_adjacency(panel_masked, machine_names, 0.4, 0.3, 0.3)

    x_train, y_train, _ = core.build_global_training_set(panel_masked, seq_len=seq_len)

    if getattr(core, "MINDSPORE_AVAILABLE", False):
        model = ev._make_predictor(n_neighbors=len(machine_names), hidden_size=hidden_size, seq_len=seq_len)
        y_fit = y_train
    else:
        model = ev._FlatMLPWrapper(hidden_size=hidden_size)
        y_fit = ev._fill_nan_targets(y_train)

    model.fit(
        x_train,
        y_fit,
        epochs=epochs,
        batch_size=min(64, len(x_train)),
        verbose=False,
        adjacency=adjacency,
        mask=None,
    )

    pred0 = core.synchronous_fill_panel(panel_masked, model, adjacency, seq_len=seq_len)
    pred = core.apply_bidirectional_blending(panel_masked, pred0)
    return ev.calc_metrics(panel_true, pred, holdout_mask, jump_threshold=1.0)


def run_wo_blending(panel_true, machine_names, holdout_mask, seq_len=6, epochs=2, hidden_size=32):
    """w/o 双向边界平滑: 保留掩码和多视图，只去掉边界补偿。"""
    panel_masked = ev.apply_mask(panel_true, holdout_mask)
    adjacency = _build_adjacency(panel_masked, machine_names, 0.4, 0.3, 0.3)

    x_train, y_train, m_train = core.build_global_training_set(panel_masked, seq_len=seq_len)

    if getattr(core, "MINDSPORE_AVAILABLE", False):
        model = ev._make_predictor(n_neighbors=len(machine_names), hidden_size=hidden_size, seq_len=seq_len)
        y_fit = y_train
    else:
        model = ev._FlatMLPWrapper(hidden_size=hidden_size)
        y_fit = ev._fill_nan_targets(y_train)

    model.fit(
        x_train,
        y_fit,
        epochs=epochs,
        batch_size=min(64, len(x_train)),
        verbose=False,
        adjacency=adjacency,
        mask=m_train,
    )

    pred = core.synchronous_fill_panel(panel_masked, model, adjacency, seq_len=seq_len)
    return ev.calc_metrics(panel_true, pred, holdout_mask, jump_threshold=1.0)


def run_wo_multiview(panel_true, machine_names, holdout_mask, seq_len=6, epochs=2, hidden_size=32):
    """w/o 多视图融合: 仅地理矩阵，保留掩码与边界平滑。"""
    panel_masked = ev.apply_mask(panel_true, holdout_mask)
    adjacency = _build_adjacency(panel_masked, machine_names, 1.0, 0.0, 0.0)

    x_train, y_train, m_train = core.build_global_training_set(panel_masked, seq_len=seq_len)

    if getattr(core, "MINDSPORE_AVAILABLE", False):
        model = ev._make_predictor(n_neighbors=len(machine_names), hidden_size=hidden_size, seq_len=seq_len)
        y_fit = y_train
    else:
        model = ev._FlatMLPWrapper(hidden_size=hidden_size)
        y_fit = ev._fill_nan_targets(y_train)

    model.fit(
        x_train,
        y_fit,
        epochs=epochs,
        batch_size=min(64, len(x_train)),
        verbose=False,
        adjacency=adjacency,
        mask=m_train,
    )

    pred0 = core.synchronous_fill_panel(panel_masked, model, adjacency, seq_len=seq_len)
    pred = core.apply_bidirectional_blending(panel_masked, pred0)
    return ev.calc_metrics(panel_true, pred, holdout_mask, jump_threshold=1.0)


def main():
    core.preload_machines("wind_data", keep_cols=("OBS",))

    if not getattr(core, "MINDSPORE_AVAILABLE", False):
        raise RuntimeError(
            "当前未启用MindSpore主路径，消融证据强度不足。请先修复主路径环境后再运行。"
        )

    # Keep exactly the same quick-eval scope used before for consistency.
    machine_names = sorted(core.machines.keys())[:24]
    panel_true = core.build_global_panel(machine_names)

    seeds = [42, 43, 44, 45, 46]
    names = ["本发明完整方法", "w/o 掩码阻断", "w/o 双向边界平滑", "w/o 多视图融合"]
    metrics = {k: [] for k in names}

    for sd in seeds:
        rng = np.random.default_rng(sd)
        holdout_mask = ev.build_holdout_mask(
            panel_true,
            seq_len=6,
            missing_rate=0.08,
            block_min=4,
            block_max=12,
            rng=rng,
        )

        metrics["本发明完整方法"].append(
            run_invention(panel_true, machine_names, holdout_mask)
        )
        metrics["w/o 掩码阻断"].append(
            run_wo_mask(panel_true, machine_names, holdout_mask)
        )
        metrics["w/o 双向边界平滑"].append(
            run_wo_blending(panel_true, machine_names, holdout_mask)
        )
        metrics["w/o 多视图融合"].append(
            run_wo_multiview(panel_true, machine_names, holdout_mask)
        )

    raw_lines = []
    summary_lines = []
    for name in names:
        arr = np.asarray(metrics[name], dtype=np.float64)
        # arr shape: (n_seeds, 3) => rmse, mae, jump
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(3, dtype=np.float64)

        for i, sd in enumerate(seeds):
            raw_lines.append(
                f"{name}\tseed={sd}\t{arr[i,0]:.4f}\t{arr[i,1]:.4f}\t{arr[i,2]:.4f}"
            )

        summary_lines.append(
            f"{name}\t{mean[0]:.4f}±{std[0]:.4f}\t{mean[1]:.4f}±{std[1]:.4f}\t{mean[2]:.4f}±{std[2]:.4f}"
        )

    out_raw = base / "ablation_fill_numbers_raw.txt"
    out_sum = base / "ablation_fill_numbers_out.txt"
    out_raw.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    out_sum.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("[RAW]")
    for line in raw_lines:
        print(line)
    print("[SUMMARY]")
    for line in summary_lines:
        print(line)
    print(f"[OK] {out_raw}")
    print(f"[OK] {out_sum}")


if __name__ == "__main__":
    main()
