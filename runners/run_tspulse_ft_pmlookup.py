# -*- coding: utf-8 -*-
"""
Re-run TSPulse_FT on TSB-AD-U-Eva with per-dataset optimal `prediction_mode`
selected from the TSB-AD tuning_results CSV (paper protocol).

This is a chunked batch runner; one process is intended to handle a slice of
the file list and pin to a single GPU. Launch 8 such processes in parallel
across GPUs 0-7 to cover Eva-350 in ~30-60 min.

Why this script exists:
- Run_Detector_U.py uses the default HP (prediction_mode='time') for all fids,
  giving VUS-PR ~0.486.
- The TSPulse paper reports VUS-PR ~0.52 by selecting the best `prediction_mode`
  per-dataset based on tuning results.
- The `tutorials/TSPulse.py` tutorial exposes `--tuning_results` for single
  files but is not chunkable.

This wrapper keeps the same per-fid logic as the tutorial (load CSV, slice
train, set per-dataset prediction_mode, call run_TSPulse_FT) but iterates over
a slice of a file list and writes each score as `<basename>.npy` to a target dir.
"""
import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv

TSB_AD_REPO = os.environ.get(
    "PAIAD_TSB_AD_REPO",
    str(Path(__file__).resolve().parents[2] / "third_party" / "TSB-AD"),
)
sys.path.insert(0, TSB_AD_REPO)

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.model_wrapper import (
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
    run_Semisupervise_AD,
    run_Unsupervise_AD,
)
from TSB_AD.utils.slidingWindows import find_length_rank


def get_dataset_name(filename: str) -> str:
    return os.path.basename(filename).split("_")[1]


def select_best_mode_by_dataset(
    filename: str,
    target_col: str = "file_name",
    metric_col: str = "VUS-PR",
    mode_col: str = "MODE",
    greater_is_better: bool = True,
) -> dict:
    """Replicates `tutorials/TSPulse.py::select_best_mode_by_dataset`."""
    df = pd.read_csv(filename, sep=",", header="infer", index_col=None)
    perf, cnt = {}, {}
    for tf, m, mode in zip(df[target_col], df[metric_col], df[mode_col]):
        ds = get_dataset_name(tf)
        perf.setdefault(ds, {}).setdefault(mode, 0.0)
        cnt.setdefault(ds, {}).setdefault(mode, 0)
        perf[ds][mode] += float(m)
        cnt[ds][mode] += 1
    best = {}
    for ds in perf:
        modes = list(perf[ds].keys())
        avgs = [perf[ds][k] / cnt[ds][k] for k in modes]
        idx = int(np.argmax(avgs)) if greater_is_better else int(np.argmin(avgs))
        best[ds] = modes[idx]
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=str(tsbad_dataset_dir()))
    parser.add_argument("--file_list", type=str, default=str(tsbad_eva_csv()))
    parser.add_argument("--score_dir", type=str, required=True,
                        help="Where to save .npy scores (a subdir <out_subdir> will be created).")
    parser.add_argument("--AD_Name", type=str, default="TSPulse_FT",
                        help="Detector name used to look up HP dict + Pool; should be a real entry like TSPulse_FT.")
    parser.add_argument("--out_subdir", type=str, default=None,
                        help="Subdir name under score_dir to write .npy into. Defaults to --AD_Name.")
    parser.add_argument("--tuning_results", type=str,
                        default=os.environ.get(
                            "PAIAD_TSPULSE_TUNING",
                            str(Path(__file__).resolve().parents[2] / "third_party" / "TSB-AD" / "benchmark_exp" / "benchmark_tuning_results" / "tsb_ad_u_tuning_TSPulse.csv"),
                        ))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--log_suffix", type=str, default="",
                        help="Optional suffix on the log filename so chunks don't clobber each other.")
    args = parser.parse_args()
    # Seeding (same as Run_Detector_U.py / tutorials/TSPulse.py).
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    out_subdir = args.out_subdir or args.AD_Name
    target_dir = os.path.join(args.score_dir, out_subdir)
    os.makedirs(target_dir, exist_ok=True)
    log_path = os.path.join(target_dir, f"000_run_{out_subdir}{args.log_suffix}.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.info(f"=== Run start: AD={args.AD_Name} start={args.start} end={args.end} ===")
    logging.info(f"tuning_results={args.tuning_results}")

    files = pd.read_csv(args.file_list)["file_name"].values.tolist()
    end = len(files) if args.end == -1 else args.end
    files = files[args.start:end]
    logging.info(f"Will process {len(files)} files (slice {args.start}:{end} of {len(files) + args.start})")

    base_HP = dict(Optimal_Uni_algo_HP_dict[args.AD_Name])
    base_mode = base_HP.get("prediction_mode", "time")
    logging.info(f"Base HP: {base_HP}")

    lookup = select_best_mode_by_dataset(args.tuning_results)
    logging.info(f"Per-dataset best modes: {lookup}")
    print(f"Per-dataset best modes: {lookup}")

    n_done = 0
    n_skip = 0
    n_fail = 0
    for filename in files:
        out_path = os.path.join(target_dir, filename.replace(".csv", ".npy"))
        if os.path.exists(out_path):
            logging.info(f"SKIP existing: {filename}")
            n_skip += 1
            continue
        t0 = time.time()
        try:
            df = pd.read_csv(os.path.join(args.dataset_dir, filename)).dropna()
            data = df.iloc[:, 0:-1].values.astype(float)
            _ = df["Label"].astype(int).to_numpy()  # not used here, keep for parity

            _ = find_length_rank(data, rank=1)
            train_index = filename.split(".")[0].split("_")[-3]
            data_train = data[: int(train_index), :]

            ds = get_dataset_name(filename)
            mode = lookup.get(ds, base_mode)
            HP = dict(base_HP)
            HP["prediction_mode"] = mode

            if args.AD_Name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(args.AD_Name, data_train, data, **HP)
            elif args.AD_Name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(args.AD_Name, data, **HP)
            else:
                raise RuntimeError(f"{args.AD_Name} is not in any pool")

            if isinstance(output, np.ndarray):
                np.save(out_path, output)
                logging.info(
                    f"Success at {filename} (mode={mode}) | Time cost: {time.time() - t0:.3f}s at length {len(data)}"
                )
                n_done += 1
            else:
                logging.warning(f"Failed at {filename}: {output}")
                n_fail += 1
        except Exception as e:
            logging.error(f"Exception at {filename}: {type(e).__name__}: {e}")
            n_fail += 1

    logging.info(f"=== Run done: done={n_done} skip={n_skip} fail={n_fail} ===")
    print(f"=== Run done: done={n_done} skip={n_skip} fail={n_fail} ===")


if __name__ == "__main__":
    main()
