#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from model_dual import build_encoder
try:
    from score_variants import get_last_score_metadata, score_with_variant
except ModuleNotFoundError:
    get_last_score_metadata = None
    score_with_variant = None
from score_variants_15c import (
    PATCH_LEVEL_VARIANTS,
    POINT_LEVEL_VARIANTS,
    VARIANTS as VARIANTS_15C,
    VariantTimeoutError,
    score_with_variant_15c,
)
from utils.data_preprocess import create_patch_tensor_and_indices
from utils.evaluation import distribute_patch_scores_to_points
from utils.metrics import generate_curve, get_metrics


ALL_VARIANTS = [
    "c1_euclidean",
    "c2_e2a_euclidean",
    "c2_cosine_orig",
    "c3_e2a_combo",
    "c4_euclidean_clean",
    "c5_euclidean_median",
    "c6_point_euclidean",
    "c7_point_euclid_s32",
    "c8_point_euclid_s128",
    "c9_point_lof_mag",
    "c10_temporal_lof",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 15c scorer runner")
    p.add_argument("--scorer_variant", type=str, required=True, choices=ALL_VARIANTS)
    p.add_argument("--precomputed_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--metric_version", type=str, default="opt", choices=["opt", "opt_mem"])
    p.add_argument("--metrics_mode", type=str, default="full", choices=["full", "vuspr_only"])
    p.add_argument("--timeout_seconds", type=float, default=120.0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--tsb_split", type=str, default="TSB-AD-U", choices=["TSB-AD-U", "TSB-AD-M"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--selected_ids", type=Path, default=None, help="Optional text file listing file_ids (or file names) to score")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_scores", action="store_true", default=True)
    p.add_argument("--no_save_scores", action="store_true")
    return p.parse_args()


SUMMARY_COLUMNS = [
    "file",
    "Category",
    "GlobalIndex",
    "ShardIndex",
    "AUC-ROC",
    "AUC-PR",
    "VUS-PR",
    "VUS-ROC",
    "BestF1",
    "RangeF1",
    "BestLoss",
    "ScorerVariantRequested",
    "ScorerVariantResolved",
    "ScorerFallbackUsed",
    "ScorerFallbackReason",
]


def _configure_cpu_threads(cpu_threads: int) -> None:
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


def _load_pt(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu").float()


def _load_optional_pt(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    return _load_pt(path)


def _load_array_compat(file_dir: Path, preferred_name: str, legacy_name: str) -> np.ndarray:
    preferred = file_dir / preferred_name
    if preferred.exists():
        return np.load(preferred)
    return np.load(file_dir / legacy_name)


def _ensure_finite(x: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values in {label}")
    return arr


def _fallback_variant(requested: str) -> str | None:
    if requested == "c1_euclidean":
        return None
    if requested == "c2_cosine_orig":
        # This variant exists purely to measure the original cosine baseline vs
        # our Euclidean c2; a silent fallback would contaminate the comparison.
        return None
    if requested in {"c2_e2a_euclidean", "c3_e2a_combo"}:
        return "c2_e2a_euclidean"
    return "c1_euclidean"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.exists():
        return None
    return _sha256_bytes(path.read_bytes())


def _repo_head_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""
    return out


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value.expanduser().resolve())
        else:
            out[key] = value
    return out


def _write_selected_file_ids(path: Path, file_dirs: list[Path]) -> str:
    payload = "".join(f"{p.name}\n" for p in file_dirs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return _sha256_bytes(payload.encode("utf-8"))


def _write_run_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    precomputed_dir: Path,
    file_dirs: list[Path],
    device: torch.device,
    worker_devices: list[str],
) -> None:
    selected_file_ids_path = output_dir / "selected_file_ids.txt"
    selected_file_ids_sha = _write_selected_file_ids(selected_file_ids_path, file_dirs)
    precompute_manifest_path = precomputed_dir / "run_manifest.json"
    manifest = {
        "kind": "task15c_scorer",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "repo_head": _repo_head_sha(Path(__file__).resolve().parent),
        "args": _jsonable_args(args),
        "precomputed_dir": str(precomputed_dir),
        "precompute_manifest_path": str(precompute_manifest_path) if precompute_manifest_path.exists() else None,
        "precompute_manifest_sha256": _sha256_file(precompute_manifest_path),
        "selected_ids_path": str(args.selected_ids.expanduser().resolve()) if args.selected_ids is not None else None,
        "selected_ids_sha256": _sha256_file(args.selected_ids),
        "selected_file_ids_path": str(selected_file_ids_path),
        "selected_file_ids_sha256": selected_file_ids_sha,
        "selected_file_count": int(len(file_dirs)),
        "device_requested": str(args.device),
        "device_resolved": str(device),
        "num_workers_requested": int(args.num_workers),
        "worker_devices": list(worker_devices),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _normalize_for_revin(train_raw: np.ndarray, full_raw: np.ndarray, use_revin: bool) -> tuple[np.ndarray, np.ndarray]:
    train_raw = np.asarray(train_raw, dtype=np.float32)
    full_raw = np.asarray(full_raw, dtype=np.float32)

    if use_revin:
        return train_raw, full_raw

    test_raw = full_raw[len(train_raw) :]
    mean = np.mean(train_raw, axis=0, keepdims=True).astype(np.float32)
    std = np.std(train_raw, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std == 0.0, 1e-8, std)

    train_proc = ((train_raw - mean) / std).astype(np.float32)
    test_proc = ((test_raw - mean) / std).astype(np.float32)
    full_proc = np.concatenate([train_proc, test_proc], axis=0)
    return train_proc, full_proc


def _score_c3_combo(
    file_dir: Path,
    meta: dict[str, Any],
    seed: int,
    patch_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_raw = np.load(file_dir / "train_raw.npy")
    test_raw = _load_array_compat(file_dir, "full_series_raw.npy", "test_raw.npy")

    use_revin = bool(meta.get("use_revin", True))
    train_proc, full_proc = _normalize_for_revin(train_raw, test_raw, use_revin=use_revin)

    train_patches, _ = create_patch_tensor_and_indices(train_proc, int(patch_size), 1)
    full_patches, _ = create_patch_tensor_and_indices(full_proc, int(patch_size), 1)

    in_channels = int(train_patches.shape[1])
    encoder = build_encoder(
        in_channels=in_channels,
        use_revin=use_revin,
        encoder_variant=str(meta.get("encoder_variant", "apure")),
        agree_mode=str(meta.get("agree_mode", "off")),
        agree_dim=int(meta.get("agree_dim", 64)),
        sharp_mode=str(meta.get("sharp_mode", "off")),
        sharp_dim=int(meta.get("sharp_dim", 64)),
    ).to(device)

    ckpt_path = file_dir / "trained_encoder.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for c3: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(state)
    encoder.eval()

    with torch.inference_mode():
        if score_with_variant is None or get_last_score_metadata is None:
            raise RuntimeError(
                "c3_e2a_combo requires the score_variants module, which is not "
                "available on this installation. Use c1/c2/c4/c5 or c6-c10 instead."
            )
        patch_scores = score_with_variant(
            "s7_combo",
            encoder,
            train_patches,
            full_patches,
            test_raw,
            train_raw,
            device,
            batch_size=int(batch_size),
        )

    patch_scores = _ensure_finite(np.asarray(patch_scores, dtype=np.float32), "c3.patch_scores")
    t_test = int(np.asarray(test_raw).reshape(-1).shape[0])
    point_scores = distribute_patch_scores_to_points(patch_scores, int(patch_size), t_test)
    point_scores = _ensure_finite(point_scores, "c3.point_scores")

    scorer_meta = get_last_score_metadata()
    info = {
        "native_level": "patch",
        "native_len": int(patch_scores.shape[0]),
        "point_len": int(point_scores.shape[0]),
        "combo_requested": "s7_combo",
        "combo_resolved": str(scorer_meta.get("resolved_variant", "s7_combo")),
        "combo_fallback_used": bool(scorer_meta.get("fallback_used", False)),
        "combo_fallback_reason": str(scorer_meta.get("fallback_reason", "")),
    }
    return point_scores, info


def _load_payload_for_15c(file_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "train_raw": np.load(file_dir / "train_raw.npy"),
        "test_raw": _load_array_compat(file_dir, "full_series_raw.npy", "test_raw.npy"),
    }

    # patch-level (required for C1/C2/C4/C5)
    payload["train_patch_embeds"] = _load_pt(file_dir / "train_patch_embeds.pt")
    payload["test_patch_embeds"] = _load_pt(file_dir / "test_patch_embeds.pt")
    payload["patch_bank"] = _load_optional_pt(file_dir / "patch_bank.pt")

    # optional unnormalized patch-level
    payload["train_patch_embeds_unnorm"] = _load_optional_pt(file_dir / "train_patch_embeds_unnorm.pt")
    payload["test_patch_embeds_unnorm"] = _load_optional_pt(file_dir / "test_patch_embeds_unnorm.pt")
    payload["patch_bank_unnorm"] = _load_optional_pt(file_dir / "patch_bank_unnorm.pt")

    # point-level (required for C6/C7/C8/C9/C10)
    train_point = _load_optional_pt(file_dir / "train_point_embeds.pt")
    test_point = _load_optional_pt(file_dir / "test_point_embeds.pt")
    point_bank = _load_optional_pt(file_dir / "point_bank.pt")

    if train_point is not None:
        payload["train_point_embeds"] = train_point
    if test_point is not None:
        payload["test_point_embeds"] = test_point
    if point_bank is not None:
        payload["point_bank"] = point_bank

    payload["train_point_embeds_unnorm"] = _load_optional_pt(file_dir / "train_point_embeds_unnorm.pt")
    payload["test_point_embeds_unnorm"] = _load_optional_pt(file_dir / "test_point_embeds_unnorm.pt")
    payload["point_bank_unnorm"] = _load_optional_pt(file_dir / "point_bank_unnorm.pt")

    return payload


def _score_file(
    variant: str,
    file_dir: Path,
    meta: dict[str, Any],
    seed: int,
    patch_size: int,
    batch_size: int,
    timeout_seconds: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    if variant == "c3_e2a_combo":
        return _score_c3_combo(file_dir, meta, seed, patch_size, batch_size, device)

    if variant not in VARIANTS_15C:
        raise ValueError(f"Unknown variant for score_with_variant_15c: {variant}")

    payload = _load_payload_for_15c(file_dir)

    output = score_with_variant_15c(
        variant=variant,
        payload=payload,
        seed=int(seed),
        patch_size=int(patch_size),
        batch_size=int(batch_size),
        device=device,
        timeout_seconds=float(timeout_seconds),
    )

    point_scores = _ensure_finite(output.point_scores, f"{variant}.point_scores")

    if variant in POINT_LEVEL_VARIANTS:
        if output.native_level != "point":
            raise ValueError(f"{variant} must be native point-level")

    if variant in PATCH_LEVEL_VARIANTS:
        if output.native_level != "patch":
            raise ValueError(f"{variant} must be native patch-level")

    return point_scores, dict(output.info)


def _load_saved_point_scores(score_path: Path) -> np.ndarray:
    df = pd.read_csv(score_path)
    if "Anomaly scores" not in df.columns:
        raise ValueError(f"Missing 'Anomaly scores' column in {score_path}")
    return _ensure_finite(df["Anomaly scores"].to_numpy(dtype=np.float32), "saved_point_scores")


def _score_or_resume_file(
    variant: str,
    file_dir: Path,
    global_index: int,
    shard_index: int,
    seed: int,
    patch_size: int,
    batch_size: int,
    timeout_seconds: float,
    metric_version: str,
    metrics_mode: str,
    save_scores: bool,
    overwrite: bool,
    output_dir: Path,
    device: torch.device,
    tsb_split: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta = json.loads((file_dir / "metadata.json").read_text())
    file_name = str(meta.get("file_name", f"{file_dir.name}.csv"))
    source = str(meta.get("source", "UNKNOWN"))
    file_id = str(meta.get("file_id", file_dir.name))

    score_path = output_dir / "scores" / tsb_split / file_id / "point_scores.csv"
    t0 = time.perf_counter()

    resolved_variant = variant
    fallback_used = False
    fallback_reason = ""
    timeout_used = False
    skip_reason = ""
    native_info: dict[str, Any] = {}

    try:
        if score_path.exists() and (not overwrite):
            point_scores = _load_saved_point_scores(score_path)
            native_info["resumed_from_saved_scores"] = True
        else:
            point_scores, native_info = _score_file(
                variant=variant,
                file_dir=file_dir,
                meta=meta,
                seed=int(seed),
                patch_size=int(patch_size),
                batch_size=int(batch_size),
                timeout_seconds=float(timeout_seconds),
                device=device,
            )
    except VariantTimeoutError as exc:
        timeout_count = 1
        timeout_used = True
        fb = _fallback_variant(variant)
        if fb is None:
            elapsed = time.perf_counter() - t0
            timing_row = {
                "file": file_name,
                "source": source,
                "variant_requested": variant,
                "variant_resolved": resolved_variant,
                "timeout_used": True,
                "fallback_used": False,
                "fallback_reason": "",
                "skipped": True,
                "skip_reason": str(exc),
                "elapsed_s": elapsed,
            }
            timing_row["timeout_count"] = timeout_count
            timing_row["fallback_count"] = 0
            timing_row["skipped_count"] = 1
            return None, timing_row

        fallback_used = True
        resolved_variant = fb
        fallback_reason = f"timeout: {type(exc).__name__}: {exc}"
        try:
            point_scores, native_info = _score_file(
                variant=fb,
                file_dir=file_dir,
                meta=meta,
                seed=int(seed),
                patch_size=int(patch_size),
                batch_size=int(batch_size),
                timeout_seconds=float(timeout_seconds),
                device=device,
            )
        except Exception as fallback_exc:
            elapsed = time.perf_counter() - t0
            timing_row = {
                "file": file_name,
                "source": source,
                "variant_requested": variant,
                "variant_resolved": resolved_variant,
                "timeout_used": True,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "skipped": True,
                "skip_reason": f"fallback_failed: {type(fallback_exc).__name__}: {fallback_exc}",
                "elapsed_s": elapsed,
            }
            timing_row["timeout_count"] = timeout_count
            timing_row["fallback_count"] = 1
            timing_row["skipped_count"] = 1
            return None, timing_row
    except Exception as exc:
        fb = _fallback_variant(variant)
        if fb is None:
            elapsed = time.perf_counter() - t0
            timing_row = {
                "file": file_name,
                "source": source,
                "variant_requested": variant,
                "variant_resolved": resolved_variant,
                "timeout_used": False,
                "fallback_used": False,
                "fallback_reason": "",
                "skipped": True,
                "skip_reason": f"{type(exc).__name__}: {exc}",
                "elapsed_s": elapsed,
            }
            timing_row["timeout_count"] = 0
            timing_row["fallback_count"] = 0
            timing_row["skipped_count"] = 1
            return None, timing_row

        fallback_used = True
        fallback_reason = f"{type(exc).__name__}: {exc}"
        resolved_variant = fb
        try:
            point_scores, native_info = _score_file(
                variant=fb,
                file_dir=file_dir,
                meta=meta,
                seed=int(seed),
                patch_size=int(patch_size),
                batch_size=int(batch_size),
                timeout_seconds=float(timeout_seconds),
                device=device,
            )
        except Exception as fallback_exc:
            elapsed = time.perf_counter() - t0
            timing_row = {
                "file": file_name,
                "source": source,
                "variant_requested": variant,
                "variant_resolved": resolved_variant,
                "timeout_used": timeout_used,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "skipped": True,
                "skip_reason": f"fallback_failed: {type(fallback_exc).__name__}: {fallback_exc}",
                "elapsed_s": elapsed,
            }
            timing_row["timeout_count"] = 0
            timing_row["fallback_count"] = 1
            timing_row["skipped_count"] = 1
            return None, timing_row

    point_scores = _ensure_finite(point_scores, "point_scores")
    labels = _ensure_finite(
        _load_array_compat(file_dir, "full_series_labels.npy", "full_labels.npy"),
        "full_labels",
    )

    if int(point_scores.shape[0]) != int(labels.shape[0]):
        elapsed = time.perf_counter() - t0
        timing_row = {
            "file": file_name,
            "source": source,
            "variant_requested": variant,
            "variant_resolved": resolved_variant,
            "timeout_used": timeout_used,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "skipped": True,
            "skip_reason": f"length mismatch scores={point_scores.shape[0]} labels={labels.shape[0]}",
            "elapsed_s": elapsed,
        }
        timing_row["timeout_count"] = int(timeout_used)
        timing_row["fallback_count"] = int(fallback_used)
        timing_row["skipped_count"] = 1
        return None, timing_row

    sliding_window = int(meta.get("sliding_window_raw", 125))
    if metrics_mode == "vuspr_only":
        _, _, _, _, _, _, vus_roc, vus_pr = generate_curve(
            labels,
            point_scores,
            sliding_window,
            metric_version,
            250,
        )
        metrics = {
            "AUC-ROC": float("nan"),
            "AUC-PR": float("nan"),
            "VUS-PR": float(vus_pr),
            "VUS-ROC": float(vus_roc),
            "Standard-F1": float("nan"),
            "R-based-F1": float("nan"),
        }
    else:
        metrics = get_metrics(
            point_scores,
            labels,
            slidingWindow=sliding_window,
            pred=None,
            version=metric_version,
            thre=250,
        )

    elapsed = time.perf_counter() - t0

    if save_scores and (overwrite or (not score_path.exists())):
        score_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "True Labels": labels,
                "Anomaly scores": point_scores,
            }
        ).to_csv(score_path, index=False)

    summary_row = {
        "file": file_name,
        "Category": source,
        "GlobalIndex": int(global_index),
        "ShardIndex": int(shard_index),
        "AUC-ROC": float(metrics["AUC-ROC"]),
        "AUC-PR": float(metrics["AUC-PR"]),
        "VUS-PR": float(metrics["VUS-PR"]),
        "VUS-ROC": float(metrics["VUS-ROC"]),
        "BestF1": float(metrics["Standard-F1"]),
        "RangeF1": float(metrics["R-based-F1"]),
        "BestLoss": float("nan"),
        "ScorerVariantRequested": variant,
        "ScorerVariantResolved": resolved_variant,
        "ScorerFallbackUsed": bool(fallback_used),
        "ScorerFallbackReason": fallback_reason,
    }

    timing_row = {
        "file": file_name,
        "source": source,
        "variant_requested": variant,
        "variant_resolved": resolved_variant,
        "timeout_used": bool(timeout_used),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "skipped": False,
        "skip_reason": skip_reason,
        "elapsed_s": float(elapsed),
        "native_level": str(native_info.get("native_level", "")),
        "native_len": int(native_info.get("native_len", -1)),
        "point_len": int(native_info.get("point_len", -1)),
        "pruned_count": int(native_info.get("pruned_count", 0)),
        "pruned_fraction": float(native_info.get("pruned_fraction", 0.0)),
        "faiss_available": bool(native_info.get("faiss_available", False)),
        "raw_std": float(native_info.get("raw_std", float("nan"))),
        "smoothed_std": float(native_info.get("smoothed_std", float("nan"))),
    }
    timing_row["timeout_count"] = int(timeout_used)
    timing_row["fallback_count"] = int(fallback_used)
    timing_row["skipped_count"] = 0
    return summary_row, timing_row


def _split_round_robin(items: list[Any], num_shards: int) -> list[list[Any]]:
    shard_count = max(1, int(num_shards))
    shards: list[list[Any]] = [[] for _ in range(shard_count)]
    for index, item in enumerate(items):
        shards[index % shard_count].append(item)
    return [shard for shard in shards if shard]


def _empty_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def _resolve_worker_devices(device_arg: str, requested_workers: int, file_count: int) -> list[str]:
    worker_budget = max(1, min(int(requested_workers), int(file_count)))
    if device_arg == "cpu":
        return ["cpu"] * worker_budget
    if device_arg == "cuda":
        visible = torch.cuda.device_count()
        if visible <= 0:
            return ["cpu"] * worker_budget
        worker_budget = max(1, min(worker_budget, visible))
        return [f"cuda:{idx}" for idx in range(worker_budget)]
    # Explicit custom device strings should not be replicated automatically.
    return [device_arg]


def _score_shard_worker(
    shard_index: int,
    total_shards: int,
    shard_entries: list[tuple[int, str]],
    variant: str,
    seed: int,
    patch_size: int,
    batch_size: int,
    timeout_seconds: float,
    metric_version: str,
    metrics_mode: str,
    save_scores: bool,
    overwrite: bool,
    output_dir_str: str,
    device_name: str,
    cpu_threads: int,
    tsb_split: str,
) -> dict[str, Any]:
    _configure_cpu_threads(int(cpu_threads))
    output_dir = Path(output_dir_str)
    device = torch.device(device_name if device_name != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))

    summary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for global_index, file_dir_str in shard_entries:
        summary_row, timing_row = _score_or_resume_file(
            variant=variant,
            file_dir=Path(file_dir_str),
            global_index=int(global_index),
            shard_index=int(shard_index),
            seed=int(seed),
            patch_size=int(patch_size),
            batch_size=int(batch_size),
            timeout_seconds=float(timeout_seconds),
            metric_version=str(metric_version),
            metrics_mode=str(metrics_mode),
            save_scores=bool(save_scores),
            overwrite=bool(overwrite),
            output_dir=output_dir,
            device=device,
            tsb_split=str(tsb_split),
        )
        if summary_row is not None:
            summary_rows.append(summary_row)
        timing_rows.append(timing_row)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    shard_metrics_dir = output_dir / "metrics" / tsb_split / f"shard_{shard_index:02d}_of_{total_shards:02d}"
    shard_metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows) if summary_rows else _empty_summary_df()
    summary_path = shard_metrics_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    timing_df = pd.DataFrame(timing_rows) if timing_rows else _empty_timing_df()
    timing_path = output_dir / f"timing_details_shard_{shard_index:02d}_of_{total_shards:02d}.csv"
    timing_df.to_csv(timing_path, index=False)

    return {
        "shard_index": int(shard_index),
        "summary_path": str(summary_path),
        "timing_path": str(timing_path),
        "completed_files": int(len(summary_rows)),
        "skipped_files": int(sum(int(row.get("skipped_count", 0)) for row in timing_rows)),
        "timeout_count": int(sum(int(row.get("timeout_count", 0)) for row in timing_rows)),
        "fallback_count": int(sum(int(row.get("fallback_count", 0)) for row in timing_rows)),
        "device": device_name,
    }


def _empty_timing_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "file",
            "source",
            "variant_requested",
            "variant_resolved",
            "timeout_used",
            "fallback_used",
            "fallback_reason",
            "skipped",
            "skip_reason",
            "elapsed_s",
            "native_level",
            "native_len",
            "point_len",
            "pruned_count",
            "pruned_fraction",
            "faiss_available",
            "raw_std",
            "smoothed_std",
        ]
    )


def _merge_shard_outputs(
    shard_results: list[dict[str, Any]],
    output_dir: Path,
    variant: str,
    seed: int,
    requested_files: int,
    tsb_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    timing_frames: list[pd.DataFrame] = []
    for result in sorted(shard_results, key=lambda item: int(item["shard_index"])):
        summary_path = Path(result["summary_path"])
        timing_path = Path(result["timing_path"])
        if summary_path.exists():
            summary_frames.append(pd.read_csv(summary_path))
        if timing_path.exists():
            timing_frames.append(pd.read_csv(timing_path))

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else _empty_summary_df()
    if not summary_df.empty and "GlobalIndex" in summary_df.columns:
        summary_df = summary_df.sort_values(["GlobalIndex", "file"]).reset_index(drop=True)

    timing_df = pd.concat(timing_frames, ignore_index=True) if timing_frames else _empty_timing_df()
    if not timing_df.empty and {"skipped", "elapsed_s"} <= set(timing_df.columns):
        timing_df = timing_df.sort_values(["source", "file"]).reset_index(drop=True)

    canonical_metrics_dir = output_dir / "metrics" / tsb_split / "shard_00_of_01"
    canonical_metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = canonical_metrics_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    timing_df.to_csv(output_dir / "timing_details.csv", index=False)

    completed_files = int((~timing_df["skipped"]).sum()) if not timing_df.empty else 0
    skipped_files = int(timing_df["skipped"].sum()) if not timing_df.empty else 0
    timeout_count = int(timing_df["timeout_used"].fillna(False).astype(bool).sum()) if not timing_df.empty else 0
    fallback_count = int(timing_df["fallback_used"].fillna(False).astype(bool).sum()) if not timing_df.empty else 0
    summary_payload = {
        "variant": variant,
        "seed": int(seed),
        "requested_files": int(requested_files),
        "completed_files": int(completed_files),
        "skipped_files": int(skipped_files),
        "timeout_count": int(timeout_count),
        "fallback_count": int(fallback_count),
        "mean_elapsed_s": float(timing_df.loc[~timing_df["skipped"], "elapsed_s"].mean())
        if (not timing_df.empty and completed_files > 0)
        else float("nan"),
        "mean_vuspr": float(summary_df["VUS-PR"].mean()) if not summary_df.empty else float("nan"),
    }
    pd.DataFrame([summary_payload]).to_csv(output_dir / "timing_summary.csv", index=False)
    return summary_df, timing_df


def main() -> None:
    args = parse_args()
    variant = str(args.scorer_variant)
    tsb_split = str(args.tsb_split)

    precomputed_dir = args.precomputed_dir.expanduser().resolve()
    if not precomputed_dir.exists():
        raise FileNotFoundError(f"precomputed_dir not found: {precomputed_dir}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_scores = bool(args.save_scores) and (not bool(args.no_save_scores))

    file_dirs = sorted([p for p in precomputed_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists()])

    if args.selected_ids is not None:
        sel_path = args.selected_ids.expanduser().resolve()
        if not sel_path.exists():
            raise FileNotFoundError(f"selected_ids not found: {sel_path}")
        selected = set()
        for line in sel_path.read_text().splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            selected.add(name)
        file_dirs = [p for p in file_dirs if (p.name in selected) or (f"{p.name}.csv" in selected)]

    if args.limit and int(args.limit) > 0:
        file_dirs = file_dirs[: int(args.limit)]

    worker_devices = _resolve_worker_devices(args.device, int(args.num_workers), len(file_dirs))
    worker_count = max(1, len(worker_devices))
    worker_cpu_threads = max(1, int(args.cpu_threads) // worker_count)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    _write_run_manifest(
        output_dir=output_dir,
        args=args,
        precomputed_dir=precomputed_dir,
        file_dirs=file_dirs,
        device=device,
        worker_devices=worker_devices,
    )

    print(
        f"[run_scorer_15c] variant={variant} seed={args.seed} files={len(file_dirs)} "
        f"timeout_seconds={args.timeout_seconds} device={device} workers={worker_count}"
    )
    file_entries = [(idx, str(file_dir)) for idx, file_dir in enumerate(file_dirs, start=1)]
    shards = _split_round_robin(file_entries, worker_count)
    shard_results: list[dict[str, Any]] = []

    if len(shards) == 1:
        shard_results.append(
            _score_shard_worker(
                shard_index=0,
                total_shards=1,
                shard_entries=shards[0],
                variant=variant,
                seed=int(args.seed),
                patch_size=int(args.patch_size),
                batch_size=int(args.batch_size),
                timeout_seconds=float(args.timeout_seconds),
                metric_version=str(args.metric_version),
                metrics_mode=str(args.metrics_mode),
                save_scores=bool(save_scores),
                overwrite=bool(args.overwrite),
                output_dir_str=str(output_dir),
                device_name=worker_devices[0],
                cpu_threads=int(worker_cpu_threads),
                tsb_split=tsb_split,
            )
        )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(shards), mp_context=ctx) as pool:
            futures = []
            for shard_index, shard_entries in enumerate(shards):
                device_name = worker_devices[shard_index % len(worker_devices)]
                futures.append(
                    pool.submit(
                        _score_shard_worker,
                        shard_index,
                        len(shards),
                        shard_entries,
                        variant,
                        int(args.seed),
                        int(args.patch_size),
                        int(args.batch_size),
                        float(args.timeout_seconds),
                        str(args.metric_version),
                        str(args.metrics_mode),
                        bool(save_scores),
                        bool(args.overwrite),
                        str(output_dir),
                        device_name,
                        int(worker_cpu_threads),
                        tsb_split,
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                shard_results.append(result)
                print(
                    f"[run_scorer_15c] shard={result['shard_index']:02d} "
                    f"device={result['device']} completed={result['completed_files']} "
                    f"skipped={result['skipped_files']}"
                )

    summary_df, timing_df = _merge_shard_outputs(
        shard_results=shard_results,
        output_dir=output_dir,
        variant=variant,
        seed=int(args.seed),
        requested_files=len(file_dirs),
        tsb_split=tsb_split,
    )

    completed_files = int((~timing_df["skipped"]).sum()) if not timing_df.empty else 0
    skipped_files = int(timing_df["skipped"].sum()) if not timing_df.empty else 0
    timeout_count = int(timing_df["timeout_used"].fillna(False).astype(bool).sum()) if not timing_df.empty else 0
    fallback_count = int(timing_df["fallback_used"].fillna(False).astype(bool).sum()) if not timing_df.empty else 0

    summary_path = output_dir / "metrics" / tsb_split / "shard_00_of_01" / "summary_metrics.csv"
    print(f"[run_scorer_15c] wrote: {summary_path}")
    print(
        f"[run_scorer_15c] completed={completed_files} skipped={skipped_files} "
        f"timeouts={timeout_count} fallbacks={fallback_count} workers={worker_count}"
    )


if __name__ == "__main__":
    main()
