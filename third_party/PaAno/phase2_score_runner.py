import argparse
import datetime as dt
import json
import os
import random
import socket
import threading
import time
from pathlib import Path

for _env_name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_env_name, "1")

import numpy as np
import pandas as pd
import torch

from main import AnomalyDetection
from main_rescnn import AnomalyDetection as ResCNNAnomalyDetection
from phase2_metric_common import (
    apply_train_zscore,
    build_metric_metadata,
    compute_metric_context,
    compute_sliding_window,
    is_paano_method,
    load_split,
    metadata_path,
    metric_window_mode,
    point_score_path,
    read_file_list,
    save_point_scores_csv,
    write_metadata_json,
    write_score_coverage,
    write_score_manifest,
)
from utils.data_preprocess import preprocess_to_patches
from utils.evaluation import distribute_patch_scores_to_points

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


_THREADPOOL_CONTROLLER = None

PAANO_METHOD_CONFIGS = {
    "paano": {
        "anchor_augmentation": "none",
        "positive_mode": "temporal",
        "positive_radius": 2,
        "time_warp_negatives": False,
    },
    "paano_aug_amp": {
        "anchor_augmentation": "amplitude",
        "positive_mode": "temporal",
        "positive_radius": 2,
        "time_warp_negatives": False,
    },
    "paano_aug_multi": {
        "anchor_augmentation": "multi",
        "positive_mode": "temporal",
        "positive_radius": 2,
        "time_warp_negatives": False,
    },
    "paano_aug_mask": {
        "anchor_augmentation": "mask",
        "positive_mode": "temporal",
        "positive_radius": 2,
        "time_warp_negatives": False,
    },
    "paano_wider_radius": {
        "anchor_augmentation": "none",
        "positive_mode": "temporal",
        "positive_radius": 32,
        "time_warp_negatives": False,
    },
    "paano_shape_pos": {
        "anchor_augmentation": "none",
        "positive_mode": "shape",
        "positive_radius": 2,
        "time_warp_negatives": False,
    },
    "paano_time_warp_neg": {
        "anchor_augmentation": "amplitude",
        "positive_mode": "temporal",
        "positive_radius": 2,
        "time_warp_negatives": True,
    },
}

RESCNN_METHOD_CONFIGS = {
    "rescnn_cnrv": {
        "use_revin": False,
    },
}

CLASSICAL_METHOD_CONFIGS = {
    "sbd_knn": {
        "base_method": "sbd_knn",
        "patch_normalize": False,
    },
    "sbd_knn_revin": {
        "base_method": "sbd_knn",
        "patch_normalize": True,
    },
    "kshape_proto": {
        "base_method": "kshape_proto",
        "patch_normalize": False,
    },
    "kshape_proto_revin": {
        "base_method": "kshape_proto",
        "patch_normalize": True,
    },
    "matrix_profile": {
        "base_method": "matrix_profile",
        "patch_normalize": False,
    },
}


def log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def configure_cpu_thread_limits(cpu_threads: int) -> None:
    global _THREADPOOL_CONTROLLER
    threads = max(1, int(cpu_threads))
    for env_name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[env_name] = str(threads)
    torch.set_num_threads(threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(2, threads)))
        except RuntimeError:
            pass
    if threadpool_limits is not None:
        try:
            _THREADPOOL_CONTROLLER = threadpool_limits(limits=threads)
        except Exception:
            _THREADPOOL_CONTROLLER = None


def parse_cpu_affinity(spec: str) -> list[int]:
    cores: set[int] = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid CPU affinity range: {token}")
            cores.update(range(start, end + 1))
        else:
            cores.add(int(token))
    if not cores:
        raise ValueError("CPU affinity spec did not contain any CPU ids")
    return sorted(cores)


def configure_cpu_affinity(cpu_affinity: str | None) -> list[int] | None:
    if not cpu_affinity:
        return None
    if not hasattr(os, "sched_setaffinity"):
        return None
    cores = parse_cpu_affinity(cpu_affinity)
    os.sched_setaffinity(0, set(cores))
    return sorted(os.sched_getaffinity(0))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 score-only runner")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            *PAANO_METHOD_CONFIGS.keys(),
            *RESCNN_METHOD_CONFIGS.keys(),
            *CLASSICAL_METHOD_CONFIGS.keys(),
        ],
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--artifact_root", required=True)
    parser.add_argument("--patch_size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--track",
        required=True,
        choices=["strict_external_zscore", "paano_native"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--cpu_affinity", type=str, default=None)
    parser.add_argument("--metric_version", type=str, default="opt_mem", choices=["opt", "opt_mem"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--reference_chunk_size", type=int, default=1024)
    parser.add_argument("--mirror_root", type=str, default=None)
    parser.add_argument("--kshape_k", type=int, default=16)
    parser.add_argument("--kshape_max_refs", type=int, default=500)
    parser.add_argument("--patch_normalize", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_subset_dir(data_dir: Path, selected_files: list[str], subset_root: Path) -> Path:
    dataset_name = data_dir.name
    subset_dir = subset_root / dataset_name
    subset_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(selected_files)
    for existing in subset_dir.glob("*.csv"):
        if existing.name not in wanted:
            existing.unlink()
    for file_name in selected_files:
        source = data_dir / file_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing input file: {source}")
        target = subset_dir / file_name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
    return subset_dir


def build_train_normal_mask(train_labels: np.ndarray, patch_size: int) -> np.ndarray:
    patch_labels = np.lib.stride_tricks.sliding_window_view(train_labels, patch_size)
    mask = patch_labels.max(axis=1) <= 0
    if mask.any():
        return mask
    return np.ones_like(mask, dtype=bool)


def write_run_metadata(run_root: Path, args, selected_files: list[str], device: torch.device) -> None:
    method_config = PAANO_METHOD_CONFIGS.get(args.method, {})
    rescnn_config = RESCNN_METHOD_CONFIGS.get(args.method, {})
    classical_config = CLASSICAL_METHOD_CONFIGS.get(args.method, {})
    metadata = {
        "host": socket.gethostname(),
        "stage": "score_only",
        "method": args.method,
        "track": args.track,
        "data_dir": str(Path(args.data_dir).resolve()),
        "artifact_root": str(Path(args.artifact_root).resolve()),
        "mirror_root": str(Path(args.mirror_root).resolve()) if args.mirror_root else None,
        "run_root": str(run_root.resolve()),
        "patch_size": int(args.patch_size),
        "seed": int(args.seed),
        "device": str(device),
        "selected_files": selected_files,
        "num_iters": int(args.num_iters),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "top_k": int(args.top_k),
        "kshape_k": int(args.kshape_k),
        "kshape_max_refs": int(args.kshape_max_refs),
        "cpu_threads": int(args.cpu_threads),
        "cpu_affinity": args.cpu_affinity,
        "anchor_augmentation": method_config.get("anchor_augmentation"),
        "positive_mode": method_config.get("positive_mode"),
        "positive_radius": method_config.get("positive_radius"),
        "time_warp_negatives": method_config.get("time_warp_negatives"),
        "patch_normalize": bool(args.patch_normalize or classical_config.get("patch_normalize", False)),
        "base_method": classical_config.get("base_method", args.method),
        "model_family": "rescnn" if args.method in RESCNN_METHOD_CONFIGS else "paano",
        "use_revin": rescnn_config.get("use_revin"),
        "metric_version_recorded_only": args.metric_version,
        "skip_existing": bool(args.skip_existing),
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


def refresh_score_outputs(
    artifact_root: Path,
    run_root: Path,
    mirror_root: Path | None,
    selected_files: list[str],
) -> int:
    rows = write_score_manifest(run_root)
    write_score_coverage(artifact_root, mirror_root, selected_files)
    return len(rows)


def _ncc_c_torch(
    query_fft: torch.Tensor,
    reference_fft: torch.Tensor,
    query_norm: torch.Tensor,
    reference_norm: torch.Tensor,
    patch_length: int,
) -> torch.Tensor:
    cross_corr = torch.fft.irfft(
        query_fft[:, None, :] * reference_fft[None, :, :],
        n=(1 << (2 * patch_length - 1).bit_length()),
        dim=-1,
    )
    cross_corr = torch.cat([cross_corr[..., -(patch_length - 1):], cross_corr[..., :patch_length]], dim=-1)
    denom = query_norm[:, None, None] * reference_norm[None, :, None]
    return cross_corr / denom


def sbd_knn_patch_scores(
    query_patches: torch.Tensor,
    reference_patches: torch.Tensor,
    top_k: int,
    device: torch.device,
    query_batch_size: int,
    reference_chunk_size: int,
) -> np.ndarray:
    if query_patches.ndim != 3 or reference_patches.ndim != 3:
        raise ValueError("Expected patch tensors with shape [N, C, L]")
    if query_patches.shape[1] != 1 or reference_patches.shape[1] != 1:
        raise NotImplementedError("Task 2 score runner currently supports univariate TSB-AD-U data only")

    query = query_patches[:, 0, :].to(dtype=torch.float32)
    reference = reference_patches[:, 0, :].to(dtype=torch.float32)
    patch_length = query.shape[-1]
    fft_size = 1 << (2 * patch_length - 1).bit_length()

    reference = reference.to(device)
    reference_norm = reference.norm(dim=1).clamp_min(1e-12)
    reference_fft = torch.fft.rfft(reference, n=fft_size, dim=-1).conj()
    effective_top_k = min(top_k, reference.shape[0])

    all_scores: list[np.ndarray] = []
    for start in range(0, query.shape[0], query_batch_size):
        query_batch = query[start:start + query_batch_size].to(device)
        query_norm = query_batch.norm(dim=1).clamp_min(1e-12)
        query_fft = torch.fft.rfft(query_batch, n=fft_size, dim=-1)
        best_dists = torch.full((query_batch.shape[0], effective_top_k), float("inf"), dtype=torch.float32, device=device)

        for ref_start in range(0, reference.shape[0], reference_chunk_size):
            ref_fft_chunk = reference_fft[ref_start:ref_start + reference_chunk_size]
            ref_norm_chunk = reference_norm[ref_start:ref_start + reference_chunk_size]
            cross_corr = torch.fft.irfft(query_fft[:, None, :] * ref_fft_chunk[None, :, :], n=fft_size, dim=-1)
            cross_corr = torch.cat([cross_corr[..., -(patch_length - 1):], cross_corr[..., :patch_length]], dim=-1)
            denom = query_norm[:, None, None] * ref_norm_chunk[None, :, None]
            ncc = cross_corr / denom
            dist = 1.0 - ncc.amax(dim=-1)
            combined = torch.cat([best_dists, dist], dim=1)
            best_dists = torch.topk(combined, k=effective_top_k, largest=False, dim=1).values

        all_scores.append(best_dists.mean(dim=1).detach().cpu().numpy())

    return np.concatenate(all_scores, axis=0).astype(np.float32)


def _per_patch_znorm(patches: torch.Tensor) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError(f"Expected patch tensor with shape [N, C, L], got {tuple(patches.shape)}")
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (patches - mean) / std


def _patch_normalize_enabled(args) -> bool:
    return bool(args.patch_normalize or CLASSICAL_METHOD_CONFIGS.get(args.method, {}).get("patch_normalize", False))


def _classical_base_method(method: str) -> str:
    return CLASSICAL_METHOD_CONFIGS.get(method, {}).get("base_method", method)


def _prepare_classical_data(args, file_path: Path):
    train_data, train_labels, test_data, _, _, full_labels = load_split(file_path)
    train_data, test_data = apply_train_zscore(train_data, test_data)
    full_data = np.concatenate([train_data, test_data], axis=0)
    sliding_window = compute_sliding_window(full_data)
    train_patches = preprocess_to_patches(train_data, patch_size=args.patch_size, stride=1)
    query_patches = preprocess_to_patches(full_data, patch_size=args.patch_size, stride=1)
    if _patch_normalize_enabled(args):
        train_patches = _per_patch_znorm(train_patches)
        query_patches = _per_patch_znorm(query_patches)
    normal_mask = build_train_normal_mask(train_labels, args.patch_size)
    reference_patches = train_patches[torch.from_numpy(normal_mask)]
    return full_labels, sliding_window, reference_patches, query_patches


def _has_valid_score_artifacts(
    run_root: Path,
    file_name: str,
    expected_points: int | None = None,
) -> bool:
    score_csv = point_score_path(run_root, file_name)
    meta_json = metadata_path(run_root, file_name)
    if not score_csv.exists() or not meta_json.exists():
        return False
    try:
        metadata = json.loads(meta_json.read_text())
        score_df = pd.read_csv(score_csv)
    except Exception:
        return False
    if "Anomaly scores" not in score_df.columns or "True Labels" not in score_df.columns:
        return False
    if expected_points is not None and len(score_df) != int(expected_points):
        return False
    if int(metadata.get("num_points", -1)) != len(score_df):
        return False
    if str(metadata.get("file")) != file_name:
        return False
    return True


def _write_score_artifacts(
    run_root: Path,
    file_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    metadata: dict[str, object],
) -> None:
    save_point_scores_csv(point_score_path(run_root, file_name), labels, scores)
    write_metadata_json(metadata_path(run_root, file_name), metadata)


def _sync_paano_scores_once(
    run_root: Path,
    internal_root: Path,
    data_dir: Path,
    selected_files: list[str],
    artifact_root: Path,
    mirror_root: Path | None,
    metadata_cache: dict[str, dict[str, object]],
    method: str,
    track: str,
) -> int:
    dataset_name = data_dir.name
    copied = 0
    updated = False

    for file_name in selected_files:
        source_score = internal_root / "scores" / dataset_name / Path(file_name).stem / "point_scores.csv"
        if not source_score.is_file():
            continue

        dest_score = point_score_path(run_root, file_name)
        dest_meta = metadata_path(run_root, file_name)
        should_copy = (
            not dest_score.exists()
            or not dest_meta.exists()
            or source_score.stat().st_mtime_ns > dest_score.stat().st_mtime_ns
        )
        if not should_copy:
            copied += 1
            continue

        dest_score.write_bytes(source_score.read_bytes())
        metric_metadata = metadata_cache.get(file_name)
        if metric_metadata is None:
            sliding_window, full_labels, window_mode = compute_metric_context(data_dir / file_name, method=method, track=track)
            metric_metadata = build_metric_metadata(
                file_name=file_name,
                method=method,
                track=track,
                source_file=data_dir / file_name,
                sliding_window=sliding_window,
                num_points=len(full_labels),
                window_mode=window_mode,
            )
            metadata_cache[file_name] = metric_metadata
        write_metadata_json(dest_meta, metric_metadata)
        copied += 1
        updated = True

    if updated:
        refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
    return copied


def start_paano_score_sync(
    run_root: Path,
    internal_root: Path,
    data_dir: Path,
    selected_files: list[str],
    artifact_root: Path,
    mirror_root: Path | None,
    method: str,
    track: str,
    interval_sec: int = 30,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    metadata_cache: dict[str, dict[str, object]] = {}
    dataset_name = data_dir.name

    def _beat() -> None:
        while not stop_event.wait(interval_sec):
            internal_count = sum(1 for _ in (internal_root / "scores" / dataset_name).glob("*/point_scores.csv"))
            exported_count = _sync_paano_scores_once(
                run_root=run_root,
                internal_root=internal_root,
                data_dir=data_dir,
                selected_files=selected_files,
                artifact_root=artifact_root,
                mirror_root=mirror_root,
                method=method,
                metadata_cache=metadata_cache,
                track=track,
            )
            log(
                "heartbeat paano_score "
                f"internal_scores={internal_count} "
                f"exported_scores={exported_count} "
                f"internal_root={internal_root}"
            )

    thread = threading.Thread(target=_beat, name="paano-score-sync", daemon=True)
    thread.start()
    return stop_event, thread


def run_paano_score(args, data_dir: Path, selected_files: list[str], run_root: Path, device: torch.device) -> int:
    subset_dir = prepare_subset_dir(data_dir, selected_files, run_root / "_subset")
    internal_root = run_root / "_internal"
    internal_root.mkdir(parents=True, exist_ok=True)
    method_config = PAANO_METHOD_CONFIGS[args.method]
    log(
        f"starting paano_score track={args.track} files={len(selected_files)} "
        f"subset_dir={subset_dir} internal_root={internal_root} device={device} "
        f"anchor_augmentation={method_config['anchor_augmentation']} "
        f"positive_mode={method_config['positive_mode']} "
        f"positive_radius={method_config['positive_radius']} "
        f"time_warp_negatives={method_config['time_warp_negatives']}"
    )

    experiment = AnomalyDetection(
        data_dir=str(subset_dir),
        output_dir=None,
        artifact_root=str(internal_root),
        patch_size=args.patch_size,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        random_seed=args.seed,
        device=device,
        use_revin=(args.track == "paano_native"),
        anchor_augmentation=method_config["anchor_augmentation"],
        positive_mode=method_config["positive_mode"],
        positive_radius=method_config["positive_radius"],
        time_warp_negatives=method_config["time_warp_negatives"],
        cpu_threads=args.cpu_threads,
        metric_version=args.metric_version,
        evaluation_mode="score_only",
    )

    artifact_root = Path(args.artifact_root).resolve()
    mirror_root = Path(args.mirror_root).resolve() if args.mirror_root else None
    sync_stop, sync_thread = start_paano_score_sync(
        run_root=run_root,
        internal_root=internal_root,
        data_dir=data_dir,
        selected_files=selected_files,
        artifact_root=artifact_root,
        mirror_root=mirror_root,
        method=args.method,
        track=args.track,
    )
    started_at = time.monotonic()
    try:
        experiment.run()
    finally:
        sync_stop.set()
        sync_thread.join(timeout=1.0)

    exported_count = _sync_paano_scores_once(
        run_root=run_root,
        internal_root=internal_root,
        data_dir=data_dir,
        selected_files=selected_files,
        artifact_root=artifact_root,
        mirror_root=mirror_root,
        method=args.method,
        metadata_cache={},
        track=args.track,
    )
    refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
    log(
        f"completed paano_score track={args.track} files={exported_count}/{len(selected_files)} "
        f"elapsed={time.monotonic() - started_at:.1f}s"
    )
    return exported_count


def run_rescnn_score(args, data_dir: Path, selected_files: list[str], run_root: Path, device: torch.device) -> int:
    subset_dir = prepare_subset_dir(data_dir, selected_files, run_root / "_subset")
    internal_root = run_root / "_internal"
    internal_root.mkdir(parents=True, exist_ok=True)
    method_config = RESCNN_METHOD_CONFIGS[args.method]
    log(
        f"starting rescnn_score track={args.track} files={len(selected_files)} "
        f"subset_dir={subset_dir} internal_root={internal_root} device={device} "
        f"use_revin={method_config['use_revin']}"
    )

    experiment = ResCNNAnomalyDetection(
        data_dir=str(subset_dir),
        output_dir=None,
        artifact_root=str(internal_root),
        patch_size=args.patch_size,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        random_seed=args.seed,
        device=device,
        use_revin=bool(method_config["use_revin"]),
        cpu_threads=args.cpu_threads,
        metric_version=args.metric_version,
        evaluation_mode="score_only",
    )

    artifact_root = Path(args.artifact_root).resolve()
    mirror_root = Path(args.mirror_root).resolve() if args.mirror_root else None
    sync_stop, sync_thread = start_paano_score_sync(
        run_root=run_root,
        internal_root=internal_root,
        data_dir=data_dir,
        selected_files=selected_files,
        artifact_root=artifact_root,
        mirror_root=mirror_root,
        method=args.method,
        track=args.track,
    )
    started_at = time.monotonic()
    try:
        experiment.run()
    finally:
        sync_stop.set()
        sync_thread.join(timeout=1.0)

    exported_count = _sync_paano_scores_once(
        run_root=run_root,
        internal_root=internal_root,
        data_dir=data_dir,
        selected_files=selected_files,
        artifact_root=artifact_root,
        mirror_root=mirror_root,
        method=args.method,
        metadata_cache={},
        track=args.track,
    )
    refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
    log(
        f"completed rescnn_score track={args.track} files={exported_count}/{len(selected_files)} "
        f"elapsed={time.monotonic() - started_at:.1f}s"
    )
    return exported_count


def run_sbd_knn_score(args, data_dir: Path, selected_files: list[str], run_root: Path, device: torch.device) -> int:
    artifact_root = Path(args.artifact_root).resolve()
    mirror_root = Path(args.mirror_root).resolve() if args.mirror_root else None
    completed = 0

    for idx, file_name in enumerate(selected_files, start=1):
        file_started = time.monotonic()
        full_labels, sliding_window, reference_patches, query_patches = _prepare_classical_data(args, data_dir / file_name)
        if args.skip_existing and _has_valid_score_artifacts(run_root, file_name, expected_points=len(full_labels)):
            completed += 1
            refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
            log(f"[sbd_knn_score] skip idx={idx}/{len(selected_files)} file={file_name} existing_score_artifacts=1")
            continue

        log(f"[sbd_knn_score] start idx={idx}/{len(selected_files)} file={file_name}")
        log(
            f"[sbd_knn_score] prepared file={file_name} refs={reference_patches.shape[0]} "
            f"queries={query_patches.shape[0]} points={len(full_labels)} sliding_window={sliding_window} "
            f"patch_normalize={_patch_normalize_enabled(args)}"
        )
        patch_scores = sbd_knn_patch_scores(
            query_patches=query_patches,
            reference_patches=reference_patches,
            top_k=args.top_k,
            device=device,
            query_batch_size=args.query_batch_size,
            reference_chunk_size=args.reference_chunk_size,
        )
        point_scores = distribute_patch_scores_to_points(
            patch_scores,
            patch_size=args.patch_size,
            num_points=len(full_labels),
        )
        metric_metadata = build_metric_metadata(
            file_name=file_name,
            method=args.method,
            track=args.track,
            source_file=data_dir / file_name,
            sliding_window=sliding_window,
            num_points=len(full_labels),
            window_mode=metric_window_mode(args.method, args.track),
        )
        _write_score_artifacts(run_root, file_name, full_labels, point_scores, metric_metadata)
        completed += 1
        manifest_rows = refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
        log(
            f"[sbd_knn_score] done idx={idx}/{len(selected_files)} file={file_name} "
            f"elapsed={time.monotonic() - file_started:.2f}s manifest_rows={manifest_rows}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return completed


def run_kshape_proto_score(args, data_dir: Path, selected_files: list[str], run_root: Path, device: torch.device) -> int:
    from tslearn.clustering import KShape

    artifact_root = Path(args.artifact_root).resolve()
    mirror_root = Path(args.mirror_root).resolve() if args.mirror_root else None
    completed = 0

    for idx, file_name in enumerate(selected_files, start=1):
        file_started = time.monotonic()
        full_labels, sliding_window, reference_patches, query_patches = _prepare_classical_data(args, data_dir / file_name)
        if args.skip_existing and _has_valid_score_artifacts(run_root, file_name, expected_points=len(full_labels)):
            completed += 1
            refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
            log(f"[kshape_proto_score] skip idx={idx}/{len(selected_files)} file={file_name} existing_score_artifacts=1")
            continue

        log(f"[kshape_proto_score] start idx={idx}/{len(selected_files)} file={file_name}")
        reference_np = reference_patches[:, 0, :].cpu().numpy().astype(np.float64)
        refs_original = int(reference_np.shape[0])
        if refs_original > int(args.kshape_max_refs):
            subsample_stride = int(np.ceil(refs_original / int(args.kshape_max_refs)))
            reference_np_fit = reference_np[::subsample_stride]
        else:
            subsample_stride = 1
            reference_np_fit = reference_np
        refs_subsampled = int(reference_np_fit.shape[0])
        effective_k = max(1, min(int(args.kshape_k), refs_subsampled))
        log(
            f"[kshape_proto_score] prepared file={file_name} refs_original={refs_original} "
            f"refs_subsampled={refs_subsampled} stride={subsample_stride} "
            f"queries={query_patches.shape[0]} effective_k={effective_k} points={len(full_labels)} "
            f"patch_normalize={_patch_normalize_enabled(args)}"
        )
        model = KShape(n_clusters=effective_k, random_state=args.seed, verbose=False)
        model.fit(reference_np_fit)
        proto = np.asarray(model.cluster_centers_, dtype=np.float32).reshape(effective_k, 1, args.patch_size)
        proto_patches = torch.from_numpy(proto)
        patch_scores = sbd_knn_patch_scores(
            query_patches=query_patches,
            reference_patches=proto_patches,
            top_k=1,
            device=device,
            query_batch_size=args.query_batch_size,
            reference_chunk_size=max(1, min(args.reference_chunk_size, effective_k)),
        )
        point_scores = distribute_patch_scores_to_points(
            patch_scores,
            patch_size=args.patch_size,
            num_points=len(full_labels),
        )
        metric_metadata = build_metric_metadata(
            file_name=file_name,
            method=args.method,
            track=args.track,
            source_file=data_dir / file_name,
            sliding_window=sliding_window,
            num_points=len(full_labels),
            window_mode=metric_window_mode(args.method, args.track),
        )
        metric_metadata.update(
            {
                "effective_k": int(effective_k),
                "num_train_normal_patches": refs_original,
                "num_train_normal_patches_subsampled": refs_subsampled,
                "kshape_subsample_stride": subsample_stride,
                "kshape_max_refs": int(args.kshape_max_refs),
                "prototype_method": "kshape",
            }
        )
        _write_score_artifacts(run_root, file_name, full_labels, point_scores, metric_metadata)
        completed += 1
        manifest_rows = refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
        log(
            f"[kshape_proto_score] done idx={idx}/{len(selected_files)} file={file_name} "
            f"elapsed={time.monotonic() - file_started:.2f}s effective_k={effective_k} "
            f"refs_original={refs_original} refs_subsampled={refs_subsampled} "
            f"stride={subsample_stride} manifest_rows={manifest_rows}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return completed


def run_matrix_profile_score(args, data_dir: Path, selected_files: list[str], run_root: Path, _device: torch.device) -> int:
    import stumpy

    artifact_root = Path(args.artifact_root).resolve()
    mirror_root = Path(args.mirror_root).resolve() if args.mirror_root else None
    completed = 0

    for idx, file_name in enumerate(selected_files, start=1):
        file_started = time.monotonic()
        train_data, _, test_data, _, _, full_labels = load_split(data_dir / file_name)
        train_data, test_data = apply_train_zscore(train_data, test_data)
        full_data = np.concatenate([train_data, test_data], axis=0)
        if args.skip_existing and _has_valid_score_artifacts(run_root, file_name, expected_points=len(full_labels)):
            completed += 1
            refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
            log(f"[matrix_profile_score] skip idx={idx}/{len(selected_files)} file={file_name} existing_score_artifacts=1")
            continue

        log(f"[matrix_profile_score] start idx={idx}/{len(selected_files)} file={file_name}")
        if full_data.ndim != 1:
            raise NotImplementedError("matrix_profile in Task 2 score runner currently supports univariate TSB-AD-U data only")
        mp = stumpy.stump(full_data.astype(np.float64), m=args.patch_size)
        patch_scores = np.asarray(mp[:, 0], dtype=np.float32)
        point_scores = distribute_patch_scores_to_points(
            patch_scores,
            patch_size=args.patch_size,
            num_points=len(full_labels),
        )
        sliding_window = compute_sliding_window(full_data)
        metric_metadata = build_metric_metadata(
            file_name=file_name,
            method="matrix_profile",
            track=args.track,
            source_file=data_dir / file_name,
            sliding_window=sliding_window,
            num_points=len(full_labels),
            window_mode=metric_window_mode("matrix_profile", args.track),
        )
        _write_score_artifacts(run_root, file_name, full_labels, point_scores, metric_metadata)
        completed += 1
        manifest_rows = refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
        log(
            f"[matrix_profile_score] done idx={idx}/{len(selected_files)} file={file_name} "
            f"elapsed={time.monotonic() - file_started:.2f}s manifest_rows={manifest_rows}"
        )

    return completed


def main():
    args = parse_args()
    affinity = configure_cpu_affinity(args.cpu_affinity)
    data_dir = Path(args.data_dir).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    run_root = artifact_root / args.track / args.method
    run_root.mkdir(parents=True, exist_ok=True)
    mirror_root = Path(args.mirror_root).resolve() if args.mirror_root else None

    device = resolve_device(args.device)
    set_seed(args.seed)
    configure_cpu_thread_limits(args.cpu_threads)

    selected_files = read_file_list(args.file_list)
    write_run_metadata(run_root, args, selected_files, device)
    refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
    log(
        f"score_runner_start method={args.method} track={args.track} files={len(selected_files)} "
        f"device={device} cpu_threads={args.cpu_threads} cpu_affinity={affinity} "
        f"artifact_root={artifact_root} mirror_root={mirror_root} skip_existing={args.skip_existing}"
    )

    if is_paano_method(args.method):
        completed = run_paano_score(args, data_dir, selected_files, run_root, device)
    elif args.method in RESCNN_METHOD_CONFIGS:
        completed = run_rescnn_score(args, data_dir, selected_files, run_root, device)
    elif _classical_base_method(args.method) == "sbd_knn":
        completed = run_sbd_knn_score(args, data_dir, selected_files, run_root, device)
    elif _classical_base_method(args.method) == "kshape_proto":
        completed = run_kshape_proto_score(args, data_dir, selected_files, run_root, device)
    else:
        completed = run_matrix_profile_score(args, data_dir, selected_files, run_root, device)

    manifest_rows = refresh_score_outputs(artifact_root, run_root, mirror_root, selected_files)
    log(
        f"completed score_runner method={args.method} track={args.track} "
        f"completed_files={completed} manifest_rows={manifest_rows}"
    )


if __name__ == "__main__":
    main()
