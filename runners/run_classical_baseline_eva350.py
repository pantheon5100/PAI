# -*- coding: utf-8 -*-
"""Classical (non-SSL) baseline runner on TSB-AD-U-Eva (350 fids).

Wraps TSB-AD's `Run_Detector_U.py` flow but adds --start/--end chunking so we
can run N-way parallel. Each fid produces one .npy score (test-only).

Usage (one chunk):
  python run_classical_baseline_eva350.py --AD_Name IForest \
    --score_dir outputs/score/<AD_Name> --start 0 --end 44

Methods available from TSB-AD's HP_list:
  IForest, Sub_PCA, KShapeAD, MatrixProfile, LOF, POLY
"""
import argparse, os, random, time, logging, traceback, sys
from pathlib import Path

# Strictly limit threads BEFORE numpy/sklearn/numba import (defends against BLAS,
# sklearn, numba/stumpy spawning many internal threads despite shell OMP_NUM_THREADS).
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
          'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
          'NUMBA_NUM_THREADS', 'TBB_NUM_THREADS'):
    os.environ[v] = '1'

import numpy as np
import pandas as pd
import torch
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)  # belt-and-suspenders: caps blas / openmp pools at runtime
except ImportError:
    pass

TSB_AD_REPO = os.environ.get(
    "PAIAD_TSB_AD_REPO",
    str(Path(__file__).resolve().parents[2] / "third_party" / "TSB-AD"),
)
sys.path.insert(0, TSB_AD_REPO)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv

from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.model_wrapper import (
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
    run_Semisupervise_AD,
    run_Unsupervise_AD,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', default=tsbad_dataset_dir())
    ap.add_argument('--file_list', default=tsbad_eva_csv())
    ap.add_argument('--score_dir', required=True)
    ap.add_argument('--AD_Name', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--log_suffix', type=str, default='')
    args = ap.parse_args()
    seed = args.seed
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)

    target_dir = Path(args.score_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / f"000_run_{args.AD_Name}{args.log_suffix}.log"
    logging.basicConfig(filename=str(log_path), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.info(f"=== Run start: AD={args.AD_Name} start={args.start} end={args.end} ===")

    file_list = pd.read_csv(args.file_list)['file_name'].values.tolist()
    end = len(file_list) if args.end == -1 else args.end
    file_list = file_list[args.start:end]
    logging.info(f"Will process {len(file_list)} files")

    Optimal_Det_HP = dict(Optimal_Uni_algo_HP_dict[args.AD_Name])
    logging.info(f"HP: {Optimal_Det_HP}")

    n_done = n_skip = n_fail = 0
    for filename in file_list:
        out_path = target_dir / filename.replace('.csv', '.npy')
        if out_path.exists():
            logging.info(f"SKIP existing: {filename}")
            n_skip += 1
            continue
        t0 = time.time()
        try:
            df = pd.read_csv(os.path.join(args.dataset_dir, filename)).dropna()
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()
            train_index = int(filename.split('.')[0].split('_')[-3])
            data_train = data[:train_index, :]

            if args.AD_Name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
            elif args.AD_Name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
            else:
                raise RuntimeError(f"{args.AD_Name} is not in any pool")

            if isinstance(output, np.ndarray):
                np.save(out_path, output)
                logging.info(f"Success at {filename} | Time {time.time()-t0:.2f}s len {len(data)}")
                n_done += 1
            else:
                logging.warning(f"Failed at {filename}: {output}")
                n_fail += 1
        except Exception as e:
            logging.error(f"Exception at {filename}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            n_fail += 1

    logging.info(f"=== Run done: done={n_done} skip={n_skip} fail={n_fail} ===")
    print(f"=== {args.AD_Name} done: {n_done} done, {n_skip} skip, {n_fail} fail ===")


if __name__ == '__main__':
    main()
