import os
import random
import warnings
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

from rescnn_model import ResCNNEncoder
from train_rescnn import train_model
from utils.data_preprocess import *
from utils.evaluation import *
from utils.metrics import get_metrics
from utils.utils import *

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
    def __init__(
        self,
        data_dir,
        output_dir=None,
        patch_size=64,
        num_iters=None,
        lr=1e-4,
        batch_size=512,
        random_seed=2000,
        device=None,
        see_loss=False,
        use_revin=False,
        koleo_weight=0.0,
        use_wandb=False,
        wandb_project="PaAno",
        wandb_entity=None,
        wandb_run_name=None,
        wandb_group=None,
        wandb_tags=None,
        wandb_mode="online",
        wandb_log_every=1,
        artifact_root=None,
        num_shards=1,
        shard_index=0,
        draw_prediction=False,
        cpu_threads=1,
        metric_version="opt",
        evaluation_mode="inline",
    ):
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
        if float(koleo_weight) < 0:
            raise ValueError(f"koleo_weight must be >= 0, got {koleo_weight}")
        self.koleo_weight = float(koleo_weight)
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
        self.draw_prediction = draw_prediction
        self.cpu_threads = max(1, int(cpu_threads))
        self.metric_version = metric_version
        if evaluation_mode not in {"inline", "score_only"}:
            raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")
        self.evaluation_mode = evaluation_mode

        self.directory_name = os.path.basename(self.data_dir.rstrip("/"))
        self.artifact_root = Path(artifact_root) if artifact_root else None
        if self.artifact_root is not None:
            self.checkpoint_root = self.artifact_root / "checkpoints" / self.directory_name
            self.figure_root = self.artifact_root / "figures" / self.directory_name
            self.score_root = self.artifact_root / "scores" / self.directory_name
            self.metric_root = self.artifact_root / "metrics" / self.directory_name
            self.wandb_root = self.artifact_root / "logs" / "wandb"
            for path in [self.checkpoint_root, self.figure_root, self.score_root, self.metric_root, self.wandb_root]:
                path.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_root = None
            self.figure_root = None
            self.score_root = None
            self.metric_root = None
            self.wandb_root = None

    def _list_csv_files(self):
        csv_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".csv")])
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
            "final_ckpt": checkpoint_dir / "trained_encoder.pth" if checkpoint_dir is not None else None,
            "score_file": score_dir / "point_scores.csv" if score_dir is not None else None,
            "figure_path": figure_path,
        }

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
            "koleo_weight": self.koleo_weight,
            "device": str(self.device),
            "num_shards": self.num_shards,
            "shard_index": self.shard_index,
            "cpu_threads": self.cpu_threads,
            "model_family": "rescnn",
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

        axes[0].plot(x, signal, color="steelblue", linewidth=1.0, label="signal")
        axes[0].fill_between(x, signal.min(), signal.max(), where=labels > 0, color="tomato", alpha=0.18, label="anomaly label")
        axes[0].axvline(split_idx, color="black", linestyle="--", linewidth=1.0, label="train/test split")
        axes[0].set_ylabel("Signal")
        axes[0].set_title(file_name)
        axes[0].legend(loc="upper right")

        axes[1].plot(x, scores, color="darkorange", linewidth=1.0, label="anomaly score")
        axes[1].fill_between(x, 0, scores.max() if scores.size else 1.0, where=labels > 0, color="tomato", alpha=0.18, label="anomaly label")
        axes[1].axvline(split_idx, color="black", linestyle="--", linewidth=1.0, label="train/test split")
        axes[1].set_ylabel("Score")
        axes[1].set_xlabel("Time Index")
        axes[1].legend(loc="upper right")

        fig.tight_layout()
        figure_path = self.figure_root / f"{self._sanitize_stem(file_name)}.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

    def _record_results(self, file_name, category, global_index, results, best_loss):
        self.dis_aurocs.append(results["AUC-ROC"])
        self.dis_auprcs.append(results["AUC-PR"])
        self.dis_vuspr.append(results["VUS-PR"])
        self.dis_vusroc.append(results["VUS-ROC"])
        self.dis_f1.append(results["Standard-F1"])
        self.dis_Rfl.append(results["R-based-F1"])
        self.results_by_category[category].append(results)
        self.summary_rows.append(
            {
                "file": file_name,
                "Category": category,
                "GlobalIndex": global_index,
                "ShardIndex": self.shard_index,
                "AUC-ROC": results["AUC-ROC"],
                "AUC-PR": results["AUC-PR"],
                "VUS-PR": results["VUS-PR"],
                "VUS-ROC": results["VUS-ROC"],
                "BestF1": results["Standard-F1"],
                "RangeF1": results["R-based-F1"],
                "BestLoss": best_loss,
            }
        )
        self._persist_summary()

    def _persist_summary(self):
        if self.metric_root is None or not self.summary_rows:
            return
        summary_df = pd.DataFrame(self.summary_rows)
        shard_name = f"shard_{self.shard_index:02d}_of_{self.num_shards:02d}"
        shard_dir = self.metric_root / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(shard_dir / "summary_metrics.csv", index=False)

    def _try_resume_completed_file(self, file_name, category, global_index, full_labels, slidingWindow, artifact_paths):
        score_file = artifact_paths["score_file"]
        best_ckpt = artifact_paths["best_ckpt"]
        final_ckpt = artifact_paths["final_ckpt"]
        figure_path = artifact_paths["figure_path"]

        if score_file is None or best_ckpt is None or final_ckpt is None:
            return False
        if not (score_file.exists() and best_ckpt.exists() and final_ckpt.exists()):
            return False
        if self.draw_prediction and self.directory_name == "TSB-AD-U" and figure_path is not None and not figure_path.exists():
            return False

        try:
            score_df = pd.read_csv(score_file)
        except Exception:
            return False
        if "Anomaly scores" not in score_df.columns or len(score_df) != len(full_labels):
            return False

        dist_scores = score_df["Anomaly scores"].to_numpy(dtype=np.float32)
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
        self._record_results(file_name, category, global_index, results, float("nan"))
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

        all_csv_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".csv")])
        csv_files = self._list_csv_files()
        total_all_csv_files = len(all_csv_files)
        total_csv_files = len(csv_files)
        print(f"ResCNN is running... (found {total_all_csv_files} files total, shard {self.shard_index + 1}/{self.num_shards} has {total_csv_files} files)")

        if total_csv_files == 0:
            print("No files assigned to this shard.")
            return

        for shard_idx, file_name in enumerate(csv_files, start=1):
            file_path = os.path.join(self.data_dir, file_name)
            global_index = all_csv_files.index(file_name) + 1
            print(f"\033[1m══ Running on shard file ({shard_idx}/{total_csv_files}) global ({global_index}/{total_all_csv_files})\033[0m : {file_name}")

            train_data, train_labels, test_data, test_labels = load_and_split_data(file_path)
            train_data = np.array(train_data, dtype=np.float32)
            test_data = np.array(test_data, dtype=np.float32)
            test_labels = np.array(test_labels, dtype=np.float32)

            train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
            train_std = np.std(train_data, axis=0, keepdims=True).astype(np.float32)
            train_std = np.where(train_std == 0.0, 1e-8, train_std)

            if self.use_revin is False:
                train_data = (train_data - train_mean) / train_std
                test_data = (test_data - train_mean) / train_std

            full_data = np.concatenate([train_data, test_data], axis=0)
            full_labels = np.concatenate([train_labels, test_labels], axis=0)

            if full_data.ndim == 1:
                sliding_input = full_data.reshape(-1, 1)
            else:
                sliding_input = full_data[:, 0].reshape(-1, 1)

            slidingWindow = find_length_rank(sliding_input, rank=1)
            category = file_name.split("_")[1]
            artifact_paths = self._artifact_paths(file_name)
            if self._try_resume_completed_file(file_name, category, global_index, full_labels, slidingWindow, artifact_paths):
                continue

            patch_creator = PatchCreator(L=self.patch_size, s=1, random_seed=self.random_seed)
            train_loader, test_loader, _ = patch_creator.create_dataloaders(
                train_data, full_data, full_labels, batch_size=self.batch_size
            )

            xb, _ = next(iter(train_loader))
            in_channels = xb.shape[1]
            print(f"[init] inferred in_channels = {in_channels}")
            model = ResCNNEncoder(in_channels=in_channels, use_revin=self.use_revin).to(self.device)

            wandb_run = self._init_wandb_for_file(file_name, category, global_index, total_all_csv_files)
            try:
                train_patches = preprocess_to_patches(train_data, patch_size=self.patch_size, stride=1)
                train_info = train_model(
                    model,
                    train_loader,
                    train_patches,
                    self.device,
                    num_iter=self.num_iters,
                    lr=self.lr,
                    see_loss=self.see_loss,
                    wandb_run=wandb_run,
                    wandb_prefix="train",
                    log_every=self.wandb_log_every,
                    best_checkpoint_path=artifact_paths["best_ckpt"],
                    final_checkpoint_path=artifact_paths["final_ckpt"],
                    koleo_weight=self.koleo_weight,
                )

                memory_bank, _ = create_memory_bank(model, train_loader, self.device, num_cores=0.1)
                all_scores = calculate_anomaly_scores(model, test_loader, memory_bank, top_k=3, device=self.device)
                dist_scores = distribute_patch_scores_to_points(all_scores, patch_size=self.patch_size, num_points=len(full_labels))

                if self.score_root is not None:
                    file_score_dir = self.score_root / artifact_paths["stem"]
                    file_score_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(
                        {
                            "True Labels": full_labels,
                            "Anomaly scores": dist_scores,
                        }
                    ).to_csv(file_score_dir / "point_scores.csv", index=False)

                self._save_prediction_figure(file_name, train_data, full_data, full_labels, dist_scores)

                if self.evaluation_mode == "score_only":
                    print("    >> Anomaly detection completed. Scores saved for deferred exact evaluation.")
                    if wandb_run is not None:
                        wandb_run.log({"train/best_loss_final": float(train_info["best_loss"])}, step=self.num_iters + 1)
                        wandb_run.summary["final/score_only"] = True
                        wandb_run.summary["train/best_loss_final"] = float(train_info["best_loss"])
                    continue

                print("    >> Anomaly detection completed. Calculating the score...")
                results = get_metrics(
                    dist_scores,
                    full_labels,
                    slidingWindow=slidingWindow,
                    pred=None,
                    version=self.metric_version,
                    thre=250,
                )
                print("    [Anomaly Detection Results]")
                print(
                    f"    >> AUC-ROC: {results['AUC-ROC']:.4f}, AUC-PR: {results['AUC-PR']:.4f}, "
                    f"VUS-PR: {results['VUS-PR']:.4f}, VUS-ROC: {results['VUS-ROC']:.4f}, "
                    f"BestF1: {results['Standard-F1']:.4f}, RangeF1: {results['R-based-F1']:.4f}"
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "eval/auc_roc": float(results["AUC-ROC"]),
                            "eval/auc_pr": float(results["AUC-PR"]),
                            "eval/vus_pr": float(results["VUS-PR"]),
                            "eval/vus_roc": float(results["VUS-ROC"]),
                            "eval/best_f1": float(results["Standard-F1"]),
                            "eval/range_f1": float(results["R-based-F1"]),
                            "train/best_loss_final": float(train_info["best_loss"]),
                        },
                        step=self.num_iters + 1,
                    )
                    wandb_run.summary["final/auc_roc"] = float(results["AUC-ROC"])
                    wandb_run.summary["final/auc_pr"] = float(results["AUC-PR"])
                    wandb_run.summary["final/vus_pr"] = float(results["VUS-PR"])
                    wandb_run.summary["final/vus_roc"] = float(results["VUS-ROC"])
                    wandb_run.summary["final/best_f1"] = float(results["Standard-F1"])
                    wandb_run.summary["final/range_f1"] = float(results["R-based-F1"])

                self._record_results(file_name, category, global_index, results, float(train_info["best_loss"]))
            finally:
                if wandb_run is not None:
                    wandb_run.finish()

        if self.summary_rows:
            dist_auroc, dist_auprc, dist_vuspr, dist_vusroc, dist_F1, dist_RF1 = map(
                np.mean,
                [self.dis_aurocs, self.dis_auprcs, self.dis_vuspr, self.dis_vusroc, self.dis_f1, self.dis_Rfl],
            )
            print(
                f"ResCNN Averaged Final Results: AUROC={dist_auroc:.4f}, AUPRC={dist_auprc:.4f}, "
                f"VUSPR={dist_vuspr:.4f}, VUSROC={dist_vusroc:.4f}, F1-Score={dist_F1:.4f}, RangeF1 = {dist_RF1:.4f}"
            )
            self._persist_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ResCNN anomaly detection")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--artifact_root", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--see_loss", dest="see_loss", action="store_true")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--use_revin", action="store_true", help="Use RevIN")
    parser.add_argument("--koleo_weight", type=float, default=0.0)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="PaAno")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--draw_prediction", action="store_true")
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--metric_version", type=str, default="opt", choices=["opt", "opt_mem"])
    parser.add_argument("--evaluation_mode", type=str, default="inline", choices=["inline", "score_only"])

    args = parser.parse_args()
    wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]

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
        koleo_weight=args.koleo_weight,
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
        draw_prediction=args.draw_prediction,
        cpu_threads=args.cpu_threads,
        metric_version=args.metric_version,
        evaluation_mode=args.evaluation_mode,
    )
    experiment.run()
