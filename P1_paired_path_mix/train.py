import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from scipy.signal import welch
from torch.utils.data import DataLoader

from dataset import (
    BalancedTrainANCDataset,
    GridSpliceANCDataset,
    apply_dynamic_path,
    infer_num_paths,
)
from model import TimeDomainANC


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv(filepath, rows, fieldnames):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(filepath, row, fieldnames):
    ensure_dir(os.path.dirname(filepath))
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(filepath, model, optimizer, seed, step, best_val_nr, args, val_metrics):
    ensure_dir(os.path.dirname(filepath))
    torch.save(
        {
            "seed": seed,
            "step": step,
            "best_val_nr": best_val_nr,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "validation_metrics": val_metrics,
        },
        filepath,
    )


def compute_nr_db(energy_d, energy_e):
    return 10.0 * math.log10((energy_d + 1e-12) / (energy_e + 1e-12))


def evaluate_grid(model, data_loader, device, sample_rate=48000):
    model.eval()
    rows = []
    path_to_nrs = defaultdict(list)
    total_energy_d = 0.0
    total_energy_e = 0.0
    worst_record = None

    dataset = data_loader.dataset

    with torch.no_grad():
        for batch in data_loader:
            x, sh, d, path_indices, window_indices = batch
            x = x.to(device, non_blocking=True)
            sh = sh.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)

            y = model(x)
            anti_noise = apply_dynamic_path(y, sh)
            error = d - anti_noise

            energy_d = torch.sum(d * d, dim=1)
            energy_e = torch.sum(error * error, dim=1)
            nr_values = 10.0 * torch.log10(
                (energy_d + 1e-12) / (energy_e + 1e-12)
            )

            x_np = x.detach().cpu().numpy()
            d_np = d.detach().cpu().numpy()
            e_np = error.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            nr_np = nr_values.detach().cpu().numpy()
            energy_d_np = energy_d.detach().cpu().numpy()
            energy_e_np = energy_e.detach().cpu().numpy()
            path_np = path_indices.cpu().numpy()
            window_np = window_indices.cpu().numpy()

            for item_index in range(len(nr_np)):
                path_index = int(path_np[item_index])
                window_index = int(window_np[item_index])
                nr_db = float(nr_np[item_index])
                start_seconds = float(dataset.window_starts_seconds[window_index])

                row = {
                    "path_index_zero_based": path_index,
                    "scene_number_one_based": path_index + 1,
                    "window_index": window_index,
                    "window_start_seconds": start_seconds,
                    "nr_db": nr_db,
                    "disturbance_energy": float(energy_d_np[item_index]),
                    "residual_energy": float(energy_e_np[item_index]),
                    "is_negative_nr": int(nr_db < 0.0),
                }
                rows.append(row)
                path_to_nrs[path_index].append(nr_db)

                total_energy_d += float(energy_d_np[item_index])
                total_energy_e += float(energy_e_np[item_index])

                if worst_record is None or nr_db < worst_record["nr_db"]:
                    worst_record = {
                        "nr_db": nr_db,
                        "path_index": path_index,
                        "window_index": window_index,
                        "window_start_seconds": start_seconds,
                        "x": x_np[item_index].copy(),
                        "d": d_np[item_index].copy(),
                        "e": e_np[item_index].copy(),
                        "y": y_np[item_index].copy(),
                    }

    sample_nrs = [row["nr_db"] for row in rows]
    path_rows = []
    for path_index in sorted(path_to_nrs):
        values = path_to_nrs[path_index]
        path_rows.append(
            {
                "path_index_zero_based": path_index,
                "scene_number_one_based": path_index + 1,
                "mean_nr_db": float(np.mean(values)),
                "std_nr_db": float(np.std(values)),
                "min_window_nr_db": float(np.min(values)),
                "max_window_nr_db": float(np.max(values)),
                "negative_window_count": int(np.sum(np.asarray(values) < 0.0)),
                "num_windows": len(values),
            }
        )

    path_mean_nrs = [row["mean_nr_db"] for row in path_rows]
    metrics = {
        "mean_sample_nr_db": float(np.mean(sample_nrs)),
        "std_sample_nr_db": float(np.std(sample_nrs)),
        "pooled_energy_nr_db": compute_nr_db(total_energy_d, total_energy_e),
        "worst_sample_nr_db": float(np.min(sample_nrs)),
        "best_sample_nr_db": float(np.max(sample_nrs)),
        "worst_path_mean_nr_db": float(np.min(path_mean_nrs)),
        "best_path_mean_nr_db": float(np.max(path_mean_nrs)),
        "negative_sample_count": int(np.sum(np.asarray(sample_nrs) < 0.0)),
        "negative_path_count": int(np.sum(np.asarray(path_mean_nrs) < 0.0)),
        "num_samples": len(sample_nrs),
        "num_paths": len(path_rows),
    }

    return metrics, rows, path_rows, worst_record


def save_heatmap(rows, filepath, title):
    path_indices = sorted({row["path_index_zero_based"] for row in rows})
    window_indices = sorted({row["window_index"] for row in rows})

    matrix = np.full((len(path_indices), len(window_indices)), np.nan, dtype=np.float32)
    path_to_row = {path_index: index for index, path_index in enumerate(path_indices)}
    window_to_col = {window_index: index for index, window_index in enumerate(window_indices)}

    starts = {}
    for row in rows:
        matrix[
            path_to_row[row["path_index_zero_based"]],
            window_to_col[row["window_index"]],
        ] = row["nr_db"]
        starts[row["window_index"]] = row["window_start_seconds"]

    fig, axis = plt.subplots(figsize=(2.2 * len(window_indices) + 2, 0.65 * len(path_indices) + 2.5))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Noise Reduction (dB)")

    axis.set_xticks(range(len(window_indices)))
    axis.set_xticklabels([f"{starts[index]:g}s" for index in window_indices])
    axis.set_yticks(range(len(path_indices)))
    axis.set_yticklabels([f"Scene {index + 1}" for index in path_indices])
    axis.set_xlabel("Fixed time-window start")
    axis.set_ylabel("Acoustic path")
    axis.set_title(title)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            axis.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    ensure_dir(os.path.dirname(filepath))
    fig.savefig(filepath, dpi=220)
    plt.close(fig)


def save_worst_waveform(worst_record, filepath, sample_rate, title):
    d = worst_record["d"]
    e = worst_record["e"]
    time_ms = np.arange(len(d)) * 1000.0 / sample_rate

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(time_ms, d, label="Disturbance d(n)", alpha=0.7)
    axis.plot(time_ms, e, label="Residual e(n)", alpha=0.8)
    axis.axvline(500.0, linestyle="--", linewidth=1.2, label="Noise switch at 0.5 s")
    axis.set_xlim(0.0, len(d) * 1000.0 / sample_rate)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Amplitude")
    axis.set_title(
        f"{title}\nWorst sample: Scene {worst_record['path_index'] + 1}, "
        f"window {worst_record['window_start_seconds']:g}s, "
        f"NR={worst_record['nr_db']:.2f} dB"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")
    fig.tight_layout()
    ensure_dir(os.path.dirname(filepath))
    fig.savefig(filepath, dpi=220)
    plt.close(fig)


def save_worst_psd(worst_record, filepath, sample_rate, title):
    d = worst_record["d"]
    e = worst_record["e"]
    frequencies_d, psd_d = welch(d, fs=sample_rate, nperseg=2048)
    frequencies_e, psd_e = welch(e, fs=sample_rate, nperseg=2048)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.semilogx(frequencies_d[1:], 10.0 * np.log10(psd_d[1:] + 1e-20), label="Disturbance d(n)")
    axis.semilogx(frequencies_e[1:], 10.0 * np.log10(psd_e[1:] + 1e-20), label="Residual e(n)")
    axis.set_xlim(20, sample_rate / 2)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (dB/Hz)")
    axis.set_title(
        f"{title}\nWorst sample PSD, NR={worst_record['nr_db']:.2f} dB"
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    ensure_dir(os.path.dirname(filepath))
    fig.savefig(filepath, dpi=220)
    plt.close(fig)


def save_evaluation_artifacts(output_dir, name, metrics, rows, path_rows, worst_record, sample_rate):
    ensure_dir(output_dir)

    sample_fields = [
        "path_index_zero_based",
        "scene_number_one_based",
        "window_index",
        "window_start_seconds",
        "nr_db",
        "disturbance_energy",
        "residual_energy",
        "is_negative_nr",
    ]
    path_fields = [
        "path_index_zero_based",
        "scene_number_one_based",
        "mean_nr_db",
        "std_nr_db",
        "min_window_nr_db",
        "max_window_nr_db",
        "negative_window_count",
        "num_windows",
    ]

    write_csv(os.path.join(output_dir, f"{name}_samples.csv"), rows, sample_fields)
    write_csv(os.path.join(output_dir, f"{name}_paths.csv"), path_rows, path_fields)

    with open(os.path.join(output_dir, f"{name}_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    save_heatmap(
        rows,
        os.path.join(output_dir, f"{name}_nr_heatmap.png"),
        title=f"{name}: path × fixed-window NR",
    )
    save_worst_waveform(
        worst_record,
        os.path.join(output_dir, f"{name}_worst_waveform.png"),
        sample_rate,
        title=name,
    )
    save_worst_psd(
        worst_record,
        os.path.join(output_dir, f"{name}_worst_psd.png"),
        sample_rate,
        title=name,
    )


def scan_noise_names(dataset_dir):
    noise_dir = os.path.join(dataset_dir, "NOISE")
    names = sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(noise_dir)
        if filename.lower().endswith(".wav")
    )
    if len(names) < 3:
        raise RuntimeError(
            f"至少需要 3 个噪声文件，实际只找到 {len(names)} 个: {names}"
        )
    return names


def build_data_loaders(args, seed, device):
    all_noise_names = scan_noise_names(args.dataset_dir)
    train_noises = all_noise_names[:-2]
    test_noises = all_noise_names[-2:]

    num_paths = infer_num_paths(args.dataset_dir)
    train_count = int(num_paths * 0.8)
    train_paths = list(range(train_count))
    unseen_paths = list(range(train_count, num_paths))

    if len(train_noises) == 1:
        validation_pair = (train_noises[0], train_noises[0])
    else:
        validation_pair = (train_noises[0], train_noises[-1])
    test_pair = (test_noises[0], test_noises[1])

    train_dataset = BalancedTrainANCDataset(
        dataset_dir=args.dataset_dir,
        noise_names=train_noises,
        path_indices=train_paths,
        segment_duration=args.segment_duration,
        sample_rate=args.sample_rate,
        repeats_per_combination=args.repeats_per_combination,
        train_start_seconds=args.train_start_seconds,
        fallback_skip_seconds=args.fallback_skip_seconds,
        path_mix_probability=args.path_mix_probability,
        path_mix_alpha_min=args.path_mix_alpha_min,
        path_mix_alpha_max=args.path_mix_alpha_max,
    )

    validation_dataset = GridSpliceANCDataset(
        dataset_dir=args.dataset_dir,
        noise_pair=validation_pair,
        path_indices=train_paths,
        window_starts_seconds=args.grid_windows,
        segment_duration=args.segment_duration,
        sample_rate=args.sample_rate,
    )

    final_seen_dataset = GridSpliceANCDataset(
        dataset_dir=args.dataset_dir,
        noise_pair=test_pair,
        path_indices=train_paths,
        window_starts_seconds=args.grid_windows,
        segment_duration=args.segment_duration,
        sample_rate=args.sample_rate,
    )

    final_unseen_dataset = GridSpliceANCDataset(
        dataset_dir=args.dataset_dir,
        noise_pair=test_pair,
        path_indices=unseen_paths,
        window_starts_seconds=args.grid_windows,
        segment_duration=args.segment_duration,
        sample_rate=args.sample_rate,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    final_seen_loader = DataLoader(
        final_seen_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    final_unseen_loader = DataLoader(
        final_unseen_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    split_info = {
        "all_noises": all_noise_names,
        "train_noises": train_noises,
        "validation_pair": list(validation_pair),
        "test_pair": list(test_pair),
        "train_paths_zero_based": train_paths,
        "unseen_paths_zero_based": unseen_paths,
        "num_path_noise_combinations": len(train_dataset.combinations),
        "samples_per_data_cycle": len(train_dataset),
        "path_mix_probability": train_dataset.path_mix_probability,
        "path_mix_alpha_range": [
            train_dataset.path_mix_alpha_min,
            train_dataset.path_mix_alpha_max,
        ],
        "mixed_repeats_per_combination": (
            train_dataset.mixed_repeats_per_combination
        ),
        "original_repeats_per_combination": (
            train_dataset.original_repeats_per_combination
        ),
        "mixed_samples_per_data_cycle": (
            len(train_dataset.combinations)
            * train_dataset.mixed_repeats_per_combination
        ),
        "original_samples_per_data_cycle": (
            len(train_dataset.combinations)
            * train_dataset.original_repeats_per_combination
        ),
    }

    return (
        train_loader,
        validation_loader,
        final_seen_loader,
        final_unseen_loader,
        split_info,
    )


def print_metrics(prefix, metrics):
    print(
        f"{prefix}: mean={metrics['mean_sample_nr_db']:.3f} dB, "
        f"worst_path={metrics['worst_path_mean_nr_db']:.3f} dB, "
        f"worst_sample={metrics['worst_sample_nr_db']:.3f} dB, "
        f"negative_samples={metrics['negative_sample_count']}/{metrics['num_samples']}"
    )


def train_one_seed(args, seed, device):
    print("\n" + "=" * 88)
    print(f"开始 P1 成对路径插值训练，seed={seed}")
    print("=" * 88)

    set_seed(seed)
    seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
    checkpoint_dir = os.path.join(seed_dir, "checkpoints")
    evaluation_dir = os.path.join(seed_dir, "evaluations")
    ensure_dir(checkpoint_dir)
    ensure_dir(evaluation_dir)

    (
        train_loader,
        validation_loader,
        final_seen_loader,
        final_unseen_loader,
        split_info,
    ) = build_data_loaders(args, seed, device)

    with open(os.path.join(seed_dir, "data_split.json"), "w", encoding="utf-8") as file:
        json.dump(split_info, file, ensure_ascii=False, indent=2)

    print(f"训练噪声: {split_info['train_noises']}")
    print(f"验证拼接噪声: {split_info['validation_pair']}")
    print(f"最终测试拼接噪声: {split_info['test_pair']}")
    print(f"训练路径: {split_info['train_paths_zero_based']}")
    print(f"未见路径: {split_info['unseen_paths_zero_based']}")
    print(f"路径×噪声组合数: {split_info['num_path_noise_combinations']}")
    print(f"一个完整均衡数据周期样本数: {split_info['samples_per_data_cycle']}")
    print(
        "成对路径插值: "
        f"p={split_info['path_mix_probability']:.2f}, "
        f"alpha范围={split_info['path_mix_alpha_range']}, "
        f"每组合原始/插值="
        f"{split_info['original_repeats_per_combination']}/"
        f"{split_info['mixed_repeats_per_combination']}"
    )

    model = TimeDomainANC(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        num_layers=10,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    history_path = os.path.join(seed_dir, "validation_history.csv")
    history_fields = [
        "seed",
        "step",
        "recent_train_loss",
        "val_mean_sample_nr_db",
        "val_worst_path_mean_nr_db",
        "val_worst_sample_nr_db",
        "val_negative_sample_count",
        "val_negative_path_count",
    ]

    train_iterator = iter(train_loader)
    loss_since_log = 0.0
    steps_since_log = 0
    loss_since_eval = 0.0
    steps_since_eval = 0
    best_val_nr = -float("inf")
    best_step = 0

    for step in range(1, args.max_steps + 1):
        try:
            x, sh, d = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, sh, d = next(train_iterator)

        model.train()
        x = x.to(device, non_blocking=True)
        sh = sh.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        anti_noise = apply_dynamic_path(y, sh)
        error = d - anti_noise
        loss = torch.mean(error * error)
        loss.backward()
        optimizer.step()

        loss_since_log += float(loss.item())
        steps_since_log += 1
        loss_since_eval += float(loss.item())
        steps_since_eval += 1

        if step % args.log_interval == 0:
            average_loss = loss_since_log / max(1, steps_since_log)
            print(f"seed={seed} step={step:05d}/{args.max_steps} train_loss={average_loss:.8f}")
            loss_since_log = 0.0
            steps_since_log = 0

        if step % args.eval_interval == 0 or step == args.max_steps:
            recent_train_loss = loss_since_eval / max(1, steps_since_eval)
            val_metrics, val_rows, val_path_rows, val_worst = evaluate_grid(
                model, validation_loader, device, args.sample_rate
            )
            print_metrics(f"seed={seed} step={step:05d} validation", val_metrics)

            append_csv(
                history_path,
                {
                    "seed": seed,
                    "step": step,
                    "recent_train_loss": recent_train_loss,
                    "val_mean_sample_nr_db": val_metrics["mean_sample_nr_db"],
                    "val_worst_path_mean_nr_db": val_metrics["worst_path_mean_nr_db"],
                    "val_worst_sample_nr_db": val_metrics["worst_sample_nr_db"],
                    "val_negative_sample_count": val_metrics["negative_sample_count"],
                    "val_negative_path_count": val_metrics["negative_path_count"],
                },
                history_fields,
            )

            save_checkpoint(
                os.path.join(checkpoint_dir, "last.pt"),
                model,
                optimizer,
                seed,
                step,
                best_val_nr,
                args,
                val_metrics,
            )

            if val_metrics["mean_sample_nr_db"] > best_val_nr:
                best_val_nr = val_metrics["mean_sample_nr_db"]
                best_step = step
                save_checkpoint(
                    os.path.join(checkpoint_dir, "best.pt"),
                    model,
                    optimizer,
                    seed,
                    step,
                    best_val_nr,
                    args,
                    val_metrics,
                )
                save_evaluation_artifacts(
                    evaluation_dir,
                    "validation_best",
                    val_metrics,
                    val_rows,
                    val_path_rows,
                    val_worst,
                    args.sample_rate,
                )
                print(f"  保存新的 best.pt: step={step}, val_NR={best_val_nr:.3f} dB")

            loss_since_eval = 0.0
            steps_since_eval = 0

    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\n载入最佳模型: seed={seed}, best_step={checkpoint['step']}, best_val_NR={checkpoint['best_val_nr']:.3f} dB")

    seen_metrics, seen_rows, seen_path_rows, seen_worst = evaluate_grid(
        model, final_seen_loader, device, args.sample_rate
    )
    unseen_metrics, unseen_rows, unseen_path_rows, unseen_worst = evaluate_grid(
        model, final_unseen_loader, device, args.sample_rate
    )

    save_evaluation_artifacts(
        evaluation_dir,
        "final_unseen_noise_seen_paths",
        seen_metrics,
        seen_rows,
        seen_path_rows,
        seen_worst,
        args.sample_rate,
    )
    save_evaluation_artifacts(
        evaluation_dir,
        "final_unseen_noise_unseen_paths",
        unseen_metrics,
        unseen_rows,
        unseen_path_rows,
        unseen_worst,
        args.sample_rate,
    )

    print_metrics("FINAL unseen noise + seen paths", seen_metrics)
    print_metrics("FINAL unseen noise + unseen paths", unseen_metrics)

    summary = {
        "seed": seed,
        "best_step": int(checkpoint["step"]),
        "best_validation_nr_db": float(checkpoint["best_val_nr"]),
        "seen_paths": seen_metrics,
        "unseen_paths": unseen_metrics,
    }
    with open(os.path.join(seed_dir, "seed_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return summary


def aggregate_seed_results(summaries, output_dir):
    rows = []
    for summary in summaries:
        rows.append(
            {
                "seed": summary["seed"],
                "best_step": summary["best_step"],
                "best_validation_nr_db": summary["best_validation_nr_db"],
                "seen_mean_nr_db": summary["seen_paths"]["mean_sample_nr_db"],
                "seen_worst_path_nr_db": summary["seen_paths"]["worst_path_mean_nr_db"],
                "seen_negative_samples": summary["seen_paths"]["negative_sample_count"],
                "unseen_mean_nr_db": summary["unseen_paths"]["mean_sample_nr_db"],
                "unseen_worst_path_nr_db": summary["unseen_paths"]["worst_path_mean_nr_db"],
                "unseen_negative_samples": summary["unseen_paths"]["negative_sample_count"],
            }
        )

    fieldnames = list(rows[0].keys())
    write_csv(os.path.join(output_dir, "multi_seed_summary.csv"), rows, fieldnames)

    metric_keys = [
        "best_validation_nr_db",
        "seen_mean_nr_db",
        "seen_worst_path_nr_db",
        "unseen_mean_nr_db",
        "unseen_worst_path_nr_db",
    ]
    recommended = max(rows, key=lambda row: row["best_validation_nr_db"])
    aggregate = {
        "num_seeds": len(rows),
        "seeds": [row["seed"] for row in rows],
        "recommended_seed_by_validation": recommended["seed"],
        "recommended_best_step": recommended["best_step"],
    }
    for key in metric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    with open(os.path.join(output_dir, "multi_seed_aggregate.json"), "w", encoding="utf-8") as file:
        json.dump(aggregate, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("三随机种子汇总")
    print("=" * 88)
    print(
        f"Seen paths: {aggregate['seen_mean_nr_db']['mean']:.3f} ± "
        f"{aggregate['seen_mean_nr_db']['std']:.3f} dB"
    )
    print(
        f"Unseen paths: {aggregate['unseen_mean_nr_db']['mean']:.3f} ± "
        f"{aggregate['unseen_mean_nr_db']['std']:.3f} dB"
    )
    print(
        f"Unseen worst-path mean: {aggregate['unseen_worst_path_nr_db']['mean']:.3f} ± "
        f"{aggregate['unseen_worst_path_nr_db']['std']:.3f} dB"
    )
    print(
        "后续实验建议使用验证集选出的模型，而不是测试集最高的模型: "
        f"seed={aggregate['recommended_seed_by_validation']}, "
        f"step={aggregate['recommended_best_step']}"
    )


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repository_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description="P1 paired-path interpolation experiment for CCF ANC")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/cha_ccf/code/CCF_DEEPANC_2026/dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(script_dir, "outputs_p1"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--segment-duration", type=float, default=1.0)
    parser.add_argument("--repeats-per-combination", type=int, default=32)
    parser.add_argument(
        "--path-mix-probability",
        type=float,
        default=0.5,
        help="训练样本中成对路径插值所占比例，0表示关闭",
    )
    parser.add_argument(
        "--path-mix-alpha-min",
        type=float,
        default=0.2,
        help="路径插值系数alpha的下界",
    )
    parser.add_argument(
        "--path-mix-alpha-max",
        type=float,
        default=0.8,
        help="路径插值系数alpha的上界",
    )
    parser.add_argument("--train-start-seconds", type=float, default=30.0)
    parser.add_argument("--fallback-skip-seconds", type=float, default=20.0)
    parser.add_argument(
        "--grid-windows",
        type=float,
        nargs="+",
        default=[20.0, 22.0, 24.0, 26.0],
    )
    return parser.parse_args()


def validate_args(args):
    if args.max_steps <= 0:
        raise ValueError("max_steps 必须大于 0")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if not 0.0 <= args.path_mix_probability <= 1.0:
        raise ValueError("path_mix_probability 必须位于 [0, 1]")
    if not (
        0.0
        <= args.path_mix_alpha_min
        < args.path_mix_alpha_max
        <= 1.0
    ):
        raise ValueError(
            "必须满足 0 <= path_mix_alpha_min < "
            "path_mix_alpha_max <= 1"
        )
    if not os.path.isdir(args.dataset_dir):
        raise FileNotFoundError(f"数据集目录不存在: {args.dataset_dir}")

    latest_validation_end = max(args.grid_windows) + args.segment_duration
    if args.train_start_seconds < latest_validation_end:
        raise ValueError(
            "train_start_seconds 必须晚于所有验证窗口的结束时间。"
            f"当前训练起点={args.train_start_seconds}s, "
            f"最晚验证结束={latest_validation_end}s"
        )


def main():
    args = parse_args()
    validate_args(args)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"数据集目录: {args.dataset_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"随机种子: {args.seeds}")
    print(f"固定训练步数: {args.max_steps}")
    print(f"每 {args.eval_interval} 步验证一次")
    print(
        "P1路径插值参数: "
        f"p={args.path_mix_probability}, "
        f"alpha=[{args.path_mix_alpha_min}, {args.path_mix_alpha_max}]"
    )

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    summaries = []
    for seed in args.seeds:
        summaries.append(train_one_seed(args, seed, device))

    aggregate_seed_results(summaries, args.output_dir)


if __name__ == "__main__":
    main()
