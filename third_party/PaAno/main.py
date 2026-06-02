import os
import random
import time
import warnings
import json
import hashlib
from collections import defaultdict
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import UndefinedMetricWarning

from model import PatchEncoder
from model_transformer import PatchHybridTransformerEncoder, PatchTransformerEncoder
from train import train_model
from utils.data_preprocess import *
from utils.utils import *
from utils.evaluation import *
from utils.metrics import get_metrics

warnings.simplefilter("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import wandb
except ImportError:
    wandb = None

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


_THREADPOOL_CONTROLLER = None


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


class AnomalyDetection:
    def __init__(self, data_dir, output_dir=None, patch_size=64,
                 num_iters=None, lr=1e-4, batch_size=512, random_seed=2000,
                 device=None, see_loss=False, use_revin=False,
                 encoder_type="cnn", d_model=128, n_heads=4, n_layers=2,
                 sub_patch_size=8, pooling="cls", pos_encoding="learned", stem_norm="bn", stem_depth=2,
                 anchor_augmentation="none", positive_mode="temporal",
                 positive_radius=2, time_warp_negatives=False, koleo_weight=0.0,
                 mask_prediction=False, mask_prediction_weight=0.5,
                 mask_ratio_min=0.2, mask_ratio_max=0.4,
                 ema_teacher=False, ema_teacher_weight=0.5, ema_use_teacher_bank=False,
                 ema_teacher_positive=False, ema_tau_start=0.996, ema_tau_end=0.999,
                 use_wandb=False, wandb_project="PaAno", wandb_entity=None,
                 wandb_run_name=None, wandb_group=None, wandb_tags=None,
                 wandb_mode="online", wandb_log_every=1,
                 artifact_root=None, num_shards=1, shard_index=0,
                 data_cache_mode="series",
                 selected_870=None, checkpoint_interval=0,
                 save_final_checkpoint=False,
                 draw_prediction=False, cpu_threads=1, metric_version="opt",
                 evaluation_mode="inline"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.num_iters = num_iters
        self.lr = lr
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.see_loss = see_loss
        self.use_revin = use_revin
        if encoder_type not in {"cnn", "transformer", "hybrid"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if pos_encoding not in {"learned", "sinusoidal", "rope", "sinusoidal_rope"}:
            raise ValueError(f"Unsupported pos_encoding: {pos_encoding}")
        self.encoder_type = encoder_type
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.sub_patch_size = int(sub_patch_size)
        self.pooling = pooling
        self.pos_encoding = pos_encoding
        if stem_norm not in {"bn", "gn"}:
            raise ValueError(f"Unsupported stem_norm: {stem_norm}")
        if int(stem_depth) not in {2, 3}:
            raise ValueError(f"stem_depth must be 2 or 3, got {stem_depth}")
        self.stem_norm = stem_norm
        self.stem_depth = int(stem_depth)
        self.anchor_augmentation = anchor_augmentation
        if positive_mode not in {"temporal", "shape"}:
            raise ValueError(f"Unsupported positive_mode: {positive_mode}")
        if int(positive_radius) < 1:
            raise ValueError(f"positive_radius must be >= 1, got {positive_radius}")
        self.positive_mode = positive_mode
        self.positive_radius = int(positive_radius)
        self.time_warp_negatives = bool(time_warp_negatives)
        if float(koleo_weight) < 0:
            raise ValueError(f"koleo_weight must be >= 0, got {koleo_weight}")
        self.koleo_weight = float(koleo_weight)
        if float(mask_prediction_weight) < 0:
            raise ValueError(f"mask_prediction_weight must be >= 0, got {mask_prediction_weight}")
        if not (0.0 < float(mask_ratio_min) < 1.0):
            raise ValueError(f"mask_ratio_min must be in (0, 1), got {mask_ratio_min}")
        if not (0.0 < float(mask_ratio_max) < 1.0):
            raise ValueError(f"mask_ratio_max must be in (0, 1), got {mask_ratio_max}")
        if float(mask_ratio_max) < float(mask_ratio_min):
            raise ValueError(
                f"mask_ratio_max must be >= mask_ratio_min, got {mask_ratio_max} < {mask_ratio_min}"
            )
        if float(ema_teacher_weight) < 0:
            raise ValueError(f"ema_teacher_weight must be >= 0, got {ema_teacher_weight}")
        if ema_use_teacher_bank and not ema_teacher:
            raise ValueError("ema_use_teacher_bank requires ema_teacher=True")
        if not (0.0 < float(ema_tau_start) < 1.0):
            raise ValueError(f"ema_tau_start must be in (0, 1), got {ema_tau_start}")
        if not (0.0 < float(ema_tau_end) < 1.0):
            raise ValueError(f"ema_tau_end must be in (0, 1), got {ema_tau_end}")
        if float(ema_tau_end) < float(ema_tau_start):
            raise ValueError(f"ema_tau_end must be >= ema_tau_start, got {ema_tau_end} < {ema_tau_start}")
        self.mask_prediction = bool(mask_prediction)
        self.mask_prediction_weight = float(mask_prediction_weight)
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)
        self.ema_teacher = bool(ema_teacher)
        self.ema_teacher_weight = float(ema_teacher_weight)
        self.ema_use_teacher_bank = bool(ema_use_teacher_bank)
        self.ema_teacher_positive = bool(ema_teacher_positive)
        self.ema_tau_start = float(ema_tau_start)
        self.ema_tau_end = float(ema_tau_end)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_group = wandb_group
        self.wandb_tags = wandb_tags or []
        self.wandb_mode = wandb_mode
        self.wandb_log_every = wandb_log_every
        self.num_shards = num_shards
        self.shard_index = shard_index
        if data_cache_mode not in {"none", "series", "patches"}:
            raise ValueError(f"Unsupported data_cache_mode: {data_cache_mode}")
        self.data_cache_mode = data_cache_mode
        self.save_final_checkpoint = bool(save_final_checkpoint)
        self.draw_prediction = draw_prediction
        self.cpu_threads = max(1, int(cpu_threads))
        self.metric_version = metric_version
        if evaluation_mode not in {"inline", "score_only"}:
            raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")
        self.evaluation_mode = evaluation_mode
        self.checkpoint_interval = int(checkpoint_interval)
        if self.checkpoint_interval < 0:
            raise ValueError(f"checkpoint_interval must be >= 0, got {checkpoint_interval}")
        if self.checkpoint_interval > 0 and self.evaluation_mode != "inline":
            raise ValueError("checkpoint_interval requires evaluation_mode='inline'")
        self.selected_870 = selected_870
        self.selected_file_names = self._load_selected_file_names(selected_870)

        self.directory_name = os.path.basename(self.data_dir.rstrip('/'))
        resolved_artifact_root = artifact_root or output_dir
        if self.checkpoint_interval > 0 and not resolved_artifact_root:
            raise ValueError("checkpoint_interval requires --artifact_root or --output_dir")
        self.artifact_root = Path(resolved_artifact_root) if resolved_artifact_root else None
        if self.artifact_root is not None:
            self.checkpoint_root = self.artifact_root / "checkpoints" / self.directory_name
            self.score_root = self.artifact_root / "scores" / self.directory_name
            self.figure_root = self.artifact_root / "figures" / self.directory_name if self.draw_prediction else None
            self.metric_root = self.artifact_root / "metrics" / self.directory_name if self.evaluation_mode != "score_only" else None
            self.wandb_root = self.artifact_root / "logs" / "wandb" if self.use_wandb else None
            for path in [self.checkpoint_root, self.figure_root, self.score_root, self.metric_root, self.wandb_root]:
                if path is None:
                    continue
                path.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_root = None
            self.figure_root = None
            self.score_root = None
            self.metric_root = None
            self.wandb_root = None
        self.variant_name = self._resolve_variant_name()
        self.checkpoint_metrics_path = self.artifact_root / "checkpoints.csv" if self.artifact_root is not None else None
        self.timing_details_path = self.artifact_root / "timing_details.csv" if self.artifact_root is not None else None
        self.timing_summary_path = self.artifact_root / "timing_summary.csv" if self.artifact_root is not None else None
        self.dataset_metadata_root = self._resolve_dataset_metadata_root()
        self.checkpoint_rows = []
        self.checkpoint_index = {}
        self.timing_rows = []
        self.timing_index = {}
        self._load_existing_diagnostic_outputs()

    def _resolve_variant_name(self):
        if self.artifact_root is not None:
            return self.artifact_root.name
        if self.output_dir:
            return Path(self.output_dir).name
        return self.directory_name

    def _load_selected_file_names(self, selected_path):
        if not selected_path:
            return None
        selected_file = Path(selected_path).expanduser()
        if not selected_file.exists():
            raise FileNotFoundError(f"selected_870 file not found: {selected_file}")
        selected_names = []
        seen = set()
        for line in selected_file.read_text().splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            if name not in seen:
                selected_names.append(name)
                seen.add(name)
        if not selected_names:
            raise ValueError(f"selected_870 file is empty: {selected_file}")
        return set(selected_names)

    def _load_existing_diagnostic_outputs(self):
        if self.checkpoint_metrics_path is not None and self.checkpoint_metrics_path.exists():
            try:
                checkpoint_df = pd.read_csv(self.checkpoint_metrics_path)
            except Exception:
                checkpoint_df = None
            if checkpoint_df is not None and not checkpoint_df.empty:
                self.checkpoint_rows = checkpoint_df.to_dict("records")
                self.checkpoint_index = {
                    (str(row["file"]), int(row["iteration"])): row
                    for row in self.checkpoint_rows
                    if "file" in row and "iteration" in row
                }
        if self.timing_details_path is not None and self.timing_details_path.exists():
            try:
                timing_df = pd.read_csv(self.timing_details_path)
            except Exception:
                timing_df = None
            if timing_df is not None and not timing_df.empty:
                self.timing_rows = timing_df.to_dict("records")
                self.timing_index = {
                    str(row["file"]): row
                    for row in self.timing_rows
                    if "file" in row
                }

    def _resolve_dataset_metadata_root(self):
        data_dir_key = hashlib.sha1(str(Path(self.data_dir).expanduser().resolve()).encode("utf-8")).hexdigest()[:16]
        metadata_root = Path.home() / ".cache" / "paano" / "dataset_metadata" / f"{self.directory_name}_{data_dir_key}"
        metadata_root.mkdir(parents=True, exist_ok=True)
        return metadata_root

    def _metadata_cache_path(self, file_name):
        return self.dataset_metadata_root / f"{self._sanitize_stem(file_name)}.json"

    def _source_signature(self, file_path):
        stat = file_path.stat()
        return {
            "source_file": str(file_path.resolve()),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
        }

    def _sliding_window_from_data(self, full_data):
        if full_data.ndim == 1:
            sliding_input = full_data.reshape(-1, 1)
        else:
            sliding_input = full_data[:, 0].reshape(-1, 1)
        return int(find_length_rank(sliding_input, rank=1))

    def _load_cached_file_metadata(self, file_path):
        cache_path = self._metadata_cache_path(file_path.name)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text())
        except Exception:
            return None
        if int(payload.get("cache_version", -1)) != 1:
            return None
        signature = self._source_signature(file_path)
        for key, value in signature.items():
            if payload.get(key) != value:
                return None
        for required_key in ["train_mean", "train_std", "sliding_window_raw", "sliding_window_train_zscore"]:
            if required_key not in payload:
                return None
        return payload

    def _build_file_metadata(self, file_path, train_data, test_data):
        train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
        train_std = np.std(train_data, axis=0, keepdims=True).astype(np.float32)
        train_std = np.where(train_std == 0.0, 1e-8, train_std)
        train_z = ((train_data - train_mean) / train_std).astype(np.float32)
        test_z = ((test_data - train_mean) / train_std).astype(np.float32)
        full_raw = np.concatenate([train_data, test_data], axis=0)
        full_z = np.concatenate([train_z, test_z], axis=0)

        payload = {
            "cache_version": 1,
            **self._source_signature(file_path),
            "train_length": int(len(train_data)),
            "num_points": int(len(full_raw)),
            "num_channels": 1 if full_raw.ndim == 1 else int(full_raw.shape[1]),
            "train_mean": np.asarray(train_mean, dtype=np.float32).reshape(-1).tolist(),
            "train_std": np.asarray(train_std, dtype=np.float32).reshape(-1).tolist(),
            "sliding_window_raw": self._sliding_window_from_data(full_raw),
            "sliding_window_train_zscore": self._sliding_window_from_data(full_z),
        }
        return payload

    def _write_file_metadata_cache(self, file_name, payload):
        cache_path = self._metadata_cache_path(file_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2))
        os.replace(temp_path, cache_path)

    def _get_file_metadata(self, file_path, train_data, test_data):
        payload = self._load_cached_file_metadata(file_path)
        if payload is not None:
            return payload
        payload = self._build_file_metadata(file_path, train_data, test_data)
        self._write_file_metadata_cache(file_path.name, payload)
        return payload

    def _list_csv_files(self):
        csv_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        if self.selected_file_names is not None:
            csv_files = [f for f in csv_files if f in self.selected_file_names]
        if self.num_shards <= 1:
            return csv_files
        return [f for idx, f in enumerate(csv_files) if idx % self.num_shards == self.shard_index]

    def _sanitize_stem(self, file_name):
        return Path(file_name).stem

    def _wandb_run_name_for_file(self, file_name):
        stem = self._sanitize_stem(file_name)
        if self.wandb_run_name:
            return f"{self.wandb_run_name}-{stem}"
        return stem

    def _artifact_paths(self, file_name):
        stem = self._sanitize_stem(file_name)
        checkpoint_dir = self.checkpoint_root / stem if self.checkpoint_root is not None else None
        score_dir = self.score_root / stem if self.score_root is not None else None
        figure_path = self.figure_root / f"{stem}.png" if self.figure_root is not None else None
        return {
            "stem": stem,
            "checkpoint_dir": checkpoint_dir,
            "best_ckpt": checkpoint_dir / "best_trained_encoder.pth" if checkpoint_dir is not None else None,
            "final_ckpt": checkpoint_dir / "trained_encoder.pth" if checkpoint_dir is not None and self.save_final_checkpoint else None,
            "score_file": score_dir / "point_scores.csv" if score_dir is not None else None,
            "figure_path": figure_path,
        }

    def _diagnostic_iterations(self):
        if self.checkpoint_interval <= 0:
            return []
        iterations = [0]
        iterations.extend(range(self.checkpoint_interval, self.num_iters, self.checkpoint_interval))
        if iterations[-1] != self.num_iters:
            iterations.append(self.num_iters)
        return iterations

    def _diagnostic_file_complete(self, file_name):
        if self.checkpoint_interval <= 0:
            return False
        required_iterations = self._diagnostic_iterations()
        if not required_iterations:
            return False
        if file_name not in self.timing_index:
            return False
        return all((file_name, iteration) in self.checkpoint_index for iteration in required_iterations)

    def _persist_checkpoint_metrics(self):
        if self.checkpoint_metrics_path is None:
            return
        checkpoint_df = pd.DataFrame(self.checkpoint_rows)
        checkpoint_df = checkpoint_df.sort_values(["file", "iteration"]).reset_index(drop=True)
        self.checkpoint_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_df.to_csv(self.checkpoint_metrics_path, index=False)

    def _persist_timing_details(self):
        if self.timing_details_path is None:
            return
        timing_df = pd.DataFrame(self.timing_rows)
        timing_df = timing_df.sort_values(["file"]).reset_index(drop=True)
        self.timing_details_path.parent.mkdir(parents=True, exist_ok=True)
        timing_df.to_csv(self.timing_details_path, index=False)

    def _persist_timing_summary(self):
        if self.timing_summary_path is None or not self.timing_rows:
            return
        total_train_time_s = float(sum(float(row["train_time_cumulative_s"]) for row in self.timing_rows))
        total_wall_time_s = float(sum(float(row["file_wall_time_s"]) for row in self.timing_rows))
        total_iterations = float(sum(int(row["iterations"]) for row in self.timing_rows))
        peak_gpu_memory_mb = float(max(float(row["peak_gpu_memory_mb"]) for row in self.timing_rows))
        timing_summary_df = pd.DataFrame([{
            "variant": self.variant_name,
            "num_files": int(len(self.timing_rows)),
            "mean_time_per_iter_ms": float((total_train_time_s / max(total_iterations, 1.0)) * 1000.0),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "total_time_50files_s": total_wall_time_s,
            "total_train_time_s": total_train_time_s,
        }])
        self.timing_summary_path.parent.mkdir(parents=True, exist_ok=True)
        timing_summary_df.to_csv(self.timing_summary_path, index=False)

    def _append_checkpoint_metrics(self, file_name, iteration, results, cumulative_train_time_s):
        if self.checkpoint_metrics_path is None:
            return
        key = (file_name, int(iteration))
        row = {
            "file": file_name,
            "iteration": int(iteration),
            "VUS-PR": float(results["VUS-PR"]),
            "VUS-ROC": float(results["VUS-ROC"]),
            "Standard-F1": float(results["Standard-F1"]),
            "R-based-F1": float(results["R-based-F1"]),
            "wall_time_cumulative_s": float(cumulative_train_time_s),
        }
        self.checkpoint_index[key] = row
        existing_idx = None
        for idx, existing in enumerate(self.checkpoint_rows):
            if str(existing.get("file")) == file_name and int(existing.get("iteration")) == int(iteration):
                existing_idx = idx
                break
        if existing_idx is None:
            self.checkpoint_rows.append(row)
        else:
            self.checkpoint_rows[existing_idx] = row
        self._persist_checkpoint_metrics()

    def _append_timing_detail(self, file_name, category, global_index, train_info, file_wall_time_s):
        if self.timing_details_path is None:
            return
        row = {
            "file": file_name,
            "source": category,
            "global_index": int(global_index),
            "variant": self.variant_name,
            "iterations": int(train_info["iterations"]),
            "mean_train_iter_ms": float(train_info["mean_train_iter_ms"]),
            "train_time_cumulative_s": float(train_info["train_time_cumulative_s"]),
            "file_wall_time_s": float(file_wall_time_s),
            "peak_gpu_memory_mb": float(train_info["peak_gpu_memory_mb"]),
        }
        self.timing_index[file_name] = row
        existing_idx = None
        for idx, existing in enumerate(self.timing_rows):
            if str(existing.get("file")) == file_name:
                existing_idx = idx
                break
        if existing_idx is None:
            self.timing_rows.append(row)
        else:
            self.timing_rows[existing_idx] = row
        self._persist_timing_details()
        self._persist_timing_summary()

    def _results_from_checkpoint_record(self, checkpoint_record):
        return {
            "VUS-PR": float(checkpoint_record["VUS-PR"]),
            "VUS-ROC": float(checkpoint_record["VUS-ROC"]),
            "Standard-F1": float(checkpoint_record["Standard-F1"]),
            "R-based-F1": float(checkpoint_record["R-based-F1"]),
            "AUC-ROC": float("nan"),
            "AUC-PR": float("nan"),
        }

    def _evaluate_current_model(self, scoring_model, train_patches, score_patch_tensor, full_labels, slidingWindow, compute_metrics=True):
        memory_bank, _ = create_memory_bank(
            scoring_model,
            train_patches,
            self.device,
            num_cores=0.1,
            batch_size=self.batch_size,
        )
        all_scores = calculate_anomaly_scores(
            scoring_model,
            score_patch_tensor,
            memory_bank,
            top_k=3,
            device=self.device,
            batch_size=self.batch_size,
        )
        dist_scores = distribute_patch_scores_to_points(
            all_scores,
            patch_size=self.patch_size,
            num_points=len(full_labels),
        )
        results = None
        if compute_metrics:
            results = get_metrics(
                dist_scores,
                full_labels,
                slidingWindow=slidingWindow,
                pred=None,
                version=self.metric_version,
                thre=250,
            )
        return dist_scores, results

    def _try_resume_diagnostic_file(self, file_name, category, global_index, artifact_paths):
        if not self._diagnostic_file_complete(file_name):
            return False
        score_file = artifact_paths["score_file"]
        best_ckpt = artifact_paths["best_ckpt"]
        if score_file is not None and not score_file.exists():
            return False
        if best_ckpt is not None and not best_ckpt.exists():
            return False
        final_record = self.checkpoint_index.get((file_name, int(self.num_iters)))
        if final_record is None:
            return False
        results = self._results_from_checkpoint_record(final_record)
        print("    >> Existing diagnostic checkpoint outputs found. Reusing saved metrics.")
        self._record_results(file_name, category, global_index, results, float("nan"))
        return True

    def _init_wandb_for_file(self, file_name, category, global_index, total_csv_files):
        if not self.use_wandb:
            return None
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or run without --use_wandb.")

        config = {
            "data_dir": self.data_dir,
            "dataset_name": self.directory_name,
            "file_name": file_name,
            "category": category,
            "global_file_index": global_index,
            "total_csv_files": total_csv_files,
            "patch_size": self.patch_size,
            "num_iters": self.num_iters,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "seed": self.random_seed,
            "use_revin": self.use_revin,
            "encoder_type": self.encoder_type,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "sub_patch_size": self.sub_patch_size,
            "pooling": self.pooling,
            "pos_encoding": self.pos_encoding,
            "stem_norm": self.stem_norm,
            "stem_depth": self.stem_depth,
            "anchor_augmentation": self.anchor_augmentation,
            "positive_mode": self.positive_mode,
            "positive_radius": self.positive_radius,
            "time_warp_negatives": self.time_warp_negatives,
            "koleo_weight": self.koleo_weight,
            "mask_prediction": self.mask_prediction,
            "mask_prediction_weight": self.mask_prediction_weight,
            "mask_ratio_min": self.mask_ratio_min,
            "mask_ratio_max": self.mask_ratio_max,
            "ema_teacher": self.ema_teacher,
            "ema_teacher_weight": self.ema_teacher_weight,
            "ema_use_teacher_bank": self.ema_use_teacher_bank,
            "ema_teacher_positive": self.ema_teacher_positive,
            "ema_tau_start": self.ema_tau_start,
            "ema_tau_end": self.ema_tau_end,
            "device": str(self.device),
            "num_shards": self.num_shards,
            "shard_index": self.shard_index,
            "cpu_threads": self.cpu_threads,
        }

        return wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self._wandb_run_name_for_file(file_name),
            group=self.wandb_group or self.directory_name,
            tags=self.wandb_tags,
            mode=self.wandb_mode,
            dir=str(self.wandb_root) if self.wandb_root is not None else None,
            config=config,
            reinit=True,
        )

    def _save_prediction_figure(self, file_name, train_data, full_data, full_labels, dist_scores):
        if not self.draw_prediction or self.figure_root is None:
            return

        is_univariate = full_data.ndim == 1 or (full_data.ndim == 2 and full_data.shape[1] == 1)
        if not is_univariate:
            return

        signal = full_data.reshape(-1) if full_data.ndim == 1 else full_data[:, 0]
        labels = full_labels.reshape(-1)
        scores = np.asarray(dist_scores).reshape(-1)
        split_idx = len(train_data)

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        x = np.arange(len(signal))

        axes[0].plot(x, signal, color='steelblue', linewidth=1.0, label='signal')
        axes[0].fill_between(x, signal.min(), signal.max(), where=labels > 0, color='tomato', alpha=0.18, label='anomaly label')
        axes[0].axvline(split_idx, color='black', linestyle='--', linewidth=1.0, label='train/test split')
        axes[0].set_ylabel('Signal')
        axes[0].set_title(file_name)
        axes[0].legend(loc='upper right')

        axes[1].plot(x, scores, color='darkorange', linewidth=1.0, label='anomaly score')
        axes[1].fill_between(x, 0, scores.max() if scores.size else 1.0, where=labels > 0, color='tomato', alpha=0.18, label='anomaly label')
        axes[1].axvline(split_idx, color='black', linestyle='--', linewidth=1.0, label='train/test split')
        axes[1].set_ylabel('Score')
        axes[1].set_xlabel('Time Index')
        axes[1].legend(loc='upper right')

        fig.tight_layout()
        figure_path = self.figure_root / f"{self._sanitize_stem(file_name)}.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

    def _record_results(self, file_name, category, global_index, results, best_loss):
        self.dis_aurocs.append(results['AUC-ROC'])
        self.dis_auprcs.append(results['AUC-PR'])
        self.dis_vuspr.append(results['VUS-PR'])
        self.dis_vusroc.append(results['VUS-ROC'])
        self.dis_f1.append(results['Standard-F1'])
        self.dis_Rfl.append(results['R-based-F1'])
        self.results_by_category[category].append(results)
        self.summary_rows.append({
            'file': file_name,
            'Category': category,
            'GlobalIndex': global_index,
            'ShardIndex': self.shard_index,
            'AUC-ROC': results['AUC-ROC'],
            'AUC-PR': results['AUC-PR'],
            'VUS-PR': results['VUS-PR'],
            'VUS-ROC': results['VUS-ROC'],
            'BestF1': results['Standard-F1'],
            'RangeF1': results['R-based-F1'],
            'BestLoss': best_loss,
        })
        self._persist_summary()

    def _persist_summary(self):
        if self.metric_root is None or not self.summary_rows:
            return
        summary_df = pd.DataFrame(self.summary_rows)
        shard_name = f"shard_{self.shard_index:02d}_of_{self.num_shards:02d}"
        shard_dir = self.metric_root / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(shard_dir / 'summary_metrics.csv', index=False)

    def _prepare_file_bundle(self, file_name, patch_creator=None, cache_patches=False):
        file_path = Path(self.data_dir) / file_name
        train_data, train_labels, test_data, test_labels = load_and_split_data(str(file_path))

        train_data = np.asarray(train_data, dtype=np.float32)
        train_labels = np.asarray(train_labels, dtype=np.float32)
        test_data = np.asarray(test_data, dtype=np.float32)
        test_labels = np.asarray(test_labels, dtype=np.float32)

        file_metadata = self._get_file_metadata(file_path, train_data, test_data)
        train_mean = np.asarray(file_metadata["train_mean"], dtype=np.float32)
        train_std = np.asarray(file_metadata["train_std"], dtype=np.float32)

        if self.use_revin is False:
            train_data = (train_data - train_mean) / train_std
            test_data = (test_data - train_mean) / train_std

        full_data = np.concatenate([train_data, test_data], axis=0)
        full_labels = np.concatenate([train_labels, test_labels], axis=0)

        bundle = {
            "category": file_name.split('_')[1],
            "train_data": train_data,
            "test_data": test_data,
            "test_labels": test_labels,
            "full_data": full_data,
            "full_labels": full_labels,
            "slidingWindow": int(
                file_metadata["sliding_window_raw"]
                if self.use_revin
                else file_metadata["sliding_window_train_zscore"]
            ),
        }

        if cache_patches:
            if patch_creator is None:
                raise ValueError("patch_creator is required when cache_patches=True")
            full_patch_tensor, full_patch_indices = patch_creator.create_patches(full_data)
            train_patch_count = patch_window_count(len(train_data), self.patch_size, patch_creator.s)
            bundle.update({
                "train_patch_count": train_patch_count,
                "full_patch_tensor": full_patch_tensor,
                "full_patch_indices": full_patch_indices,
            })

        return bundle

    def _preload_file_cache(self, csv_files, patch_creator):
        if self.data_cache_mode == "none":
            return None

        print(f"Preloading shard data with cache_mode={self.data_cache_mode}...")
        cache = {}
        cache_patches = self.data_cache_mode == "patches"
        for idx, file_name in enumerate(csv_files, start=1):
            cache[file_name] = self._prepare_file_bundle(
                file_name,
                patch_creator=patch_creator,
                cache_patches=cache_patches,
            )
            if idx % 50 == 0 or idx == len(csv_files):
                print(f"    >> Preloaded {idx}/{len(csv_files)} files")
        return cache

    def _try_resume_completed_file(self, file_name, category, global_index, full_labels, slidingWindow, artifact_paths):
        score_file = artifact_paths["score_file"]
        best_ckpt = artifact_paths["best_ckpt"]
        figure_path = artifact_paths["figure_path"]

        if score_file is None or best_ckpt is None:
            return False
        if not (score_file.exists() and best_ckpt.exists()):
            return False
        if self.draw_prediction and self.directory_name == "TSB-AD-U" and figure_path is not None and not figure_path.exists():
            return False

        try:
            score_df = pd.read_csv(score_file)
        except Exception:
            return False
        if 'Anomaly scores' not in score_df.columns or len(score_df) != len(full_labels):
            return False

        dist_scores = score_df['Anomaly scores'].to_numpy(dtype=np.float32)
        if self.evaluation_mode == "score_only":
            print("    >> Existing score artifacts found. Reusing saved scores and checkpoints.")
            return True

        results = get_metrics(
            dist_scores,
            full_labels,
            slidingWindow=slidingWindow,
            pred=None,
            version=self.metric_version,
            thre=250,
        )
        print("    >> Existing artifacts found. Reusing saved scores and checkpoints.")
        print(
            f"    >> AUC-ROC: {results['AUC-ROC']:.4f}, AUC-PR: {results['AUC-PR']:.4f}, "
            f"VUS-PR: {results['VUS-PR']:.4f}, VUS-ROC: {results['VUS-ROC']:.4f}, "
            f"BestF1: {results['Standard-F1']:.4f}, RangeF1: {results['R-based-F1']:.4f}"
        )
        self._record_results(file_name, category, global_index, results, float('nan'))
        return True

    def run(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        configure_cpu_thread_limits(self.cpu_threads)

        self.dis_aurocs = []
        self.dis_auprcs = []
        self.dis_vuspr = []
        self.dis_vusroc = []
        self.dis_f1 = []
        self.dis_Rfl = []
        self.results_by_category = defaultdict(list)
        self.summary_rows = []

        all_csv_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        if self.selected_file_names is not None:
            missing_files = sorted(self.selected_file_names.difference(all_csv_files))
            if missing_files:
                raise FileNotFoundError(
                    f"{len(missing_files)} files listed in selected_870 were not found under {self.data_dir}: "
                    f"{missing_files[:5]}"
                )
        csv_files = self._list_csv_files()
        total_all_csv_files = len(all_csv_files)
        total_csv_files = len(csv_files)
        global_index_by_file = {file_name: idx + 1 for idx, file_name in enumerate(all_csv_files)}
        if self.selected_file_names is None:
            print(f"PaAno is running... (found {total_all_csv_files} files total, shard {self.shard_index + 1}/{self.num_shards} has {total_csv_files} files)")
        else:
            print(
                "PaAno is running... "
                f"(found {total_all_csv_files} files total, selected {len(self.selected_file_names)}, "
                f"shard {self.shard_index + 1}/{self.num_shards} has {total_csv_files} files)"
            )

        if total_csv_files == 0:
            print("No files assigned to this shard.")
            return

        patch_creator = PatchCreator(L=self.patch_size, s=1, random_seed=self.random_seed)
        file_cache = self._preload_file_cache(csv_files, patch_creator)

        for shard_idx, file_name in enumerate(csv_files, start=1):
            global_index = global_index_by_file[file_name]
            print(f"\033[1m══ Running on shard file ({shard_idx}/{total_csv_files}) global ({global_index}/{total_all_csv_files})\033[0m : {file_name}")
            category = file_name.split('_')[1]
            artifact_paths = self._artifact_paths(file_name)

            if self._try_resume_diagnostic_file(file_name, category, global_index, artifact_paths):
                continue

            prepared = file_cache[file_name] if file_cache is not None else self._prepare_file_bundle(file_name)
            train_data = prepared["train_data"]
            full_data = prepared["full_data"]
            full_labels = prepared["full_labels"]
            slidingWindow = prepared["slidingWindow"]
            if self.checkpoint_interval <= 0 and self._try_resume_completed_file(
                file_name,
                category,
                global_index,
                full_labels,
                slidingWindow,
                artifact_paths,
            ):
                continue

            if self.data_cache_mode == "patches":
                train_patch_count = prepared["train_patch_count"]
                full_patch_tensor = prepared["full_patch_tensor"]
                full_patch_indices = prepared["full_patch_indices"]
                train_loader, test_loader = patch_creator.create_dataloaders_from_patches(
                    full_patch_tensor[:train_patch_count],
                    full_patch_indices[:train_patch_count],
                    full_patch_tensor,
                    full_patch_indices,
                    batch_size=self.batch_size,
                )
                train_patches = full_patch_tensor[:train_patch_count]
            else:
                train_loader, test_loader, _ = patch_creator.create_dataloaders(
                    train_data,
                    full_data,
                    full_labels,
                    batch_size=self.batch_size,
                )
                train_patches = train_loader.dataset.data

            score_patch_tensor = (
                prepared["full_patch_tensor"]
                if self.data_cache_mode == "patches"
                else test_loader.dataset.data
            )
            in_channels = train_patches.shape[1]
            print(f"[init] inferred in_channels = {in_channels}")
            if self.encoder_type == "transformer":
                model = PatchTransformerEncoder(
                    in_channels=in_channels,
                    patch_size=self.patch_size,
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_layers=self.n_layers,
                    sub_patch_size=self.sub_patch_size,
                    pooling=self.pooling,
                    pos_encoding=self.pos_encoding,
                    use_revin=self.use_revin,
                ).to(self.device)
            elif self.encoder_type == "hybrid":
                model = PatchHybridTransformerEncoder(
                    in_channels=in_channels,
                    patch_size=self.patch_size,
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_layers=self.n_layers,
                    sub_patch_size=self.sub_patch_size,
                    pooling=self.pooling,
                    pos_encoding=self.pos_encoding,
                    stem_norm=self.stem_norm,
                    stem_depth=self.stem_depth,
                    use_revin=self.use_revin,
                ).to(self.device)
            else:
                model = PatchEncoder(in_channels=in_channels, use_revin=self.use_revin).to(self.device)

            wandb_run = self._init_wandb_for_file(file_name, category, global_index, total_all_csv_files)
            try:
                file_wall_start = time.perf_counter()

                if self.checkpoint_interval > 0:
                    print("    >> Scoring random initialization (iteration 0)...")
                    _, init_results = self._evaluate_current_model(
                        model,
                        train_patches,
                        score_patch_tensor,
                        full_labels,
                        slidingWindow,
                    )
                    self._append_checkpoint_metrics(
                        file_name=file_name,
                        iteration=0,
                        results=init_results,
                        cumulative_train_time_s=0.0,
                    )

                def checkpoint_callback(iteration, model, teacher_model, cumulative_train_time_s):
                    scoring_model = teacher_model if self.ema_use_teacher_bank and teacher_model is not None else model
                    _, checkpoint_results = self._evaluate_current_model(
                        scoring_model,
                        train_patches,
                        score_patch_tensor,
                        full_labels,
                        slidingWindow,
                    )
                    self._append_checkpoint_metrics(
                        file_name=file_name,
                        iteration=iteration,
                        results=checkpoint_results,
                        cumulative_train_time_s=cumulative_train_time_s,
                    )

                train_info = train_model(
                    model,
                    train_loader,
                    train_patches,
                    self.device,
                    num_iter=self.num_iters,
                    pretext_step=self.patch_size,
                    lr=self.lr,
                    see_loss=self.see_loss,
                    wandb_run=wandb_run,
                    wandb_prefix="train",
                    log_every=self.wandb_log_every,
                    best_checkpoint_path=artifact_paths["best_ckpt"],
                    final_checkpoint_path=artifact_paths["final_ckpt"],
                    anchor_augmentation=self.anchor_augmentation,
                    positive_mode=self.positive_mode,
                    positive_radius=self.positive_radius,
                    time_warp_negatives=self.time_warp_negatives,
                    koleo_weight=self.koleo_weight,
                    mask_prediction=self.mask_prediction,
                    mask_prediction_weight=self.mask_prediction_weight,
                    mask_ratio_min=self.mask_ratio_min,
                    mask_ratio_max=self.mask_ratio_max,
                    ema_teacher=self.ema_teacher,
                    ema_teacher_weight=self.ema_teacher_weight,
                    ema_teacher_positive=self.ema_teacher_positive,
                    ema_tau_start=self.ema_tau_start,
                    ema_tau_end=self.ema_tau_end,
                    checkpoint_interval=self.checkpoint_interval,
                    checkpoint_callback=checkpoint_callback if self.checkpoint_interval > 0 else None,
                )

                scoring_model = model
                if self.ema_use_teacher_bank and train_info.get("teacher_model") is not None:
                    scoring_model = train_info["teacher_model"]
                dist_scores, results = self._evaluate_current_model(
                    scoring_model,
                    train_patches,
                    score_patch_tensor,
                    full_labels,
                    slidingWindow,
                    compute_metrics=not (self.evaluation_mode == "score_only" and self.checkpoint_interval <= 0),
                )

                if self.checkpoint_interval > 0:
                    self._append_checkpoint_metrics(
                        file_name=file_name,
                        iteration=self.num_iters,
                        results=results,
                        cumulative_train_time_s=float(train_info["train_time_cumulative_s"]),
                    )

                if self.score_root is not None:
                    file_score_dir = self.score_root / artifact_paths["stem"]
                    file_score_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        'True Labels': full_labels,
                        'Anomaly scores': dist_scores,
                    }).to_csv(file_score_dir / 'point_scores.csv', index=False)

                self._save_prediction_figure(file_name, train_data, full_data, full_labels, dist_scores)
                file_wall_time_s = time.perf_counter() - file_wall_start
                self._append_timing_detail(
                    file_name=file_name,
                    category=category,
                    global_index=global_index,
                    train_info=train_info,
                    file_wall_time_s=file_wall_time_s,
                )

                if self.evaluation_mode == "score_only":
                    print("    >> Anomaly detection completed. Scores saved for deferred exact evaluation.")
                    if wandb_run is not None:
                        wandb_run.log({
                            "train/best_loss_final": float(train_info['best_loss']),
                        }, step=self.num_iters + 1)
                        wandb_run.summary["final/score_only"] = True
                        wandb_run.summary["train/best_loss_final"] = float(train_info['best_loss'])
                    continue

                print("    >> Anomaly detection completed. Calculating the score...")
                print("    [Anomaly Detection Results]")
                print(
                    f"    >> AUC-ROC: {results['AUC-ROC']:.4f}, AUC-PR: {results['AUC-PR']:.4f}, "
                    f"VUS-PR: {results['VUS-PR']:.4f}, VUS-ROC: {results['VUS-ROC']:.4f}, "
                    f"BestF1: {results['Standard-F1']:.4f}, RangeF1: {results['R-based-F1']:.4f}"
                )

                if wandb_run is not None:
                    wandb_run.log({
                        "eval/auc_roc": float(results['AUC-ROC']),
                        "eval/auc_pr": float(results['AUC-PR']),
                        "eval/vus_pr": float(results['VUS-PR']),
                        "eval/vus_roc": float(results['VUS-ROC']),
                        "eval/best_f1": float(results['Standard-F1']),
                        "eval/range_f1": float(results['R-based-F1']),
                        "train/best_loss_final": float(train_info['best_loss']),
                    }, step=self.num_iters + 1)
                    wandb_run.summary["final/auc_roc"] = float(results['AUC-ROC'])
                    wandb_run.summary["final/auc_pr"] = float(results['AUC-PR'])
                    wandb_run.summary["final/vus_pr"] = float(results['VUS-PR'])
                    wandb_run.summary["final/vus_roc"] = float(results['VUS-ROC'])
                    wandb_run.summary["final/best_f1"] = float(results['Standard-F1'])
                    wandb_run.summary["final/range_f1"] = float(results['R-based-F1'])

                self._record_results(file_name, category, global_index, results, float(train_info['best_loss']))
            finally:
                if wandb_run is not None:
                    wandb_run.finish()

        if self.summary_rows:
            dist_auroc, dist_auprc, dist_vuspr, dist_vusroc, dist_F1, dist_RF1 = map(
                np.nanmean,
                [self.dis_aurocs, self.dis_auprcs, self.dis_vuspr, self.dis_vusroc, self.dis_f1, self.dis_Rfl],
            )
            print(
                f"PaAno's Averaged Final Results: AUROC={dist_auroc:.4f}, AUPRC={dist_auprc:.4f}, "
                f"VUSPR={dist_vuspr:.4f}, VUSROC={dist_vusroc:.4f}, F1-Score={dist_F1:.4f}, RangeF1 = {dist_RF1:.4f}"
            )
            self._persist_summary()
        self._persist_timing_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PaAno Anomaly Detection")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--artifact_root', type=str, default=None)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_iters', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--see_loss', dest='see_loss', action='store_true')
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--use_revin', action='store_true', help='Use RevIN')
    parser.add_argument('--encoder_type', type=str, default='cnn', choices=['cnn', 'transformer', 'hybrid'])
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--sub_patch_size', type=int, default=8)
    parser.add_argument('--pooling', type=str, default='cls', choices=['cls', 'mean'])
    parser.add_argument('--pos_encoding', type=str, default='learned', choices=['learned', 'sinusoidal', 'rope', 'sinusoidal_rope'])
    parser.add_argument('--stem_norm', type=str, default='bn', choices=['bn', 'gn'])
    parser.add_argument('--stem_depth', type=int, default=2, choices=[2, 3])
    parser.add_argument('--anchor_augmentation', type=str, default='none', choices=['none', 'amplitude', 'multi', 'mask'])
    parser.add_argument('--positive_mode', type=str, default='temporal', choices=['temporal', 'shape'])
    parser.add_argument('--positive_radius', type=int, default=2)
    parser.add_argument('--time_warp_negatives', action='store_true')
    parser.add_argument('--koleo_weight', type=float, default=0.0)
    parser.add_argument('--mask_prediction', action='store_true')
    parser.add_argument('--mask_prediction_weight', type=float, default=0.5)
    parser.add_argument('--mask_ratio_min', type=float, default=0.2)
    parser.add_argument('--mask_ratio_max', type=float, default=0.4)
    parser.add_argument('--ema_teacher', action='store_true')
    parser.add_argument('--ema_teacher_weight', type=float, default=0.5)
    parser.add_argument('--ema_use_teacher_bank', action='store_true')
    parser.add_argument('--ema_teacher_positive', action='store_true')
    parser.add_argument('--ema_tau_start', type=float, default=0.996)
    parser.add_argument('--ema_tau_end', type=float, default=0.999)
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='PaAno')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, default='')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_log_every', type=int, default=1)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--data_cache_mode', type=str, default='series', choices=['none', 'series', 'patches'])
    parser.add_argument('--selected_870', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=0)
    parser.add_argument('--save_final_checkpoint', action='store_true')
    parser.add_argument('--draw_prediction', action='store_true')
    parser.add_argument('--cpu_threads', type=int, default=1)
    parser.add_argument('--metric_version', type=str, default='opt', choices=['opt', 'opt_mem'])
    parser.add_argument('--evaluation_mode', type=str, default='inline', choices=['inline', 'score_only'])

    args = parser.parse_args()
    wandb_tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]

    experiment = AnomalyDetection(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        artifact_root=args.artifact_root,
        patch_size=args.patch_size,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        see_loss=args.see_loss,
        random_seed=args.seed,
        use_revin=args.use_revin,
        encoder_type=args.encoder_type,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        sub_patch_size=args.sub_patch_size,
        pooling=args.pooling,
        pos_encoding=args.pos_encoding,
        stem_norm=args.stem_norm,
        stem_depth=args.stem_depth,
        anchor_augmentation=args.anchor_augmentation,
        positive_mode=args.positive_mode,
        positive_radius=args.positive_radius,
        time_warp_negatives=args.time_warp_negatives,
        koleo_weight=args.koleo_weight,
        mask_prediction=args.mask_prediction,
        mask_prediction_weight=args.mask_prediction_weight,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        ema_teacher=args.ema_teacher,
        ema_teacher_weight=args.ema_teacher_weight,
        ema_use_teacher_bank=args.ema_use_teacher_bank,
        ema_teacher_positive=args.ema_teacher_positive,
        ema_tau_start=args.ema_tau_start,
        ema_tau_end=args.ema_tau_end,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_tags=wandb_tags,
        wandb_mode=args.wandb_mode,
        wandb_log_every=args.wandb_log_every,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        data_cache_mode=args.data_cache_mode,
        selected_870=args.selected_870,
        checkpoint_interval=args.checkpoint_interval,
        save_final_checkpoint=args.save_final_checkpoint,
        draw_prediction=args.draw_prediction,
        cpu_threads=args.cpu_threads,
        metric_version=args.metric_version,
        evaluation_mode=args.evaluation_mode,
    )
    experiment.run()
