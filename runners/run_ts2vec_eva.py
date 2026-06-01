"""TS2Vec adapter for TSB-AD-U-Eva evaluation.

Standalone runner; doesn't modify the TSB-AD codebase or the TS2Vec repo. Reads
fids from TSB-AD-U-Eva.csv, fits TS2Vec per-fid on the train portion (semi-
supervised: uses the train_index encoded in the fid name), computes per-point
anomaly score via the masked-vs-unmasked representation reconstruction trick
from `ts2vec/tasks/anomaly_detection.py:eval_anomaly_detection`, and saves as
.npy in the same per-fid format Run_Detector_U.py uses for TSPulse.

Score = | enc(x, mask='mask_last') - enc(x) |  summed across feature dim,
        moving-average baselined (train+test 21-window MA).

Hyperparameters: TS2Vec defaults (output=320, hidden=64, depth=10, lr=0.001,
                                  batch=8, n_iters=200).
"""
from __future__ import annotations
import argparse, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import bottleneck as bn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv

# Repo paths
TS2VEC_REPO = os.environ.get(
    "PAIAD_TS2VEC_REPO",
    str(Path(__file__).resolve().parents[2] / "third_party" / "ts2vec"),
)
sys.path.insert(0, TS2VEC_REPO)


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def compute_ts2vec_score(model, train_data, test_data):
    """Replicates ts2vec/tasks/anomaly_detection.py per-point score logic
    but returns the continuous anomaly score (not the binary flag).

    train_data, test_data: 1D float numpy arrays.
    Returns: 1D float array of length len(test_data), the per-point anomaly score.
    """
    full = np.concatenate([train_data, test_data]).reshape(1, -1, 1).astype(np.float32)
    full_repr = model.encode(
        full,
        mask='mask_last',
        causal=True,
        sliding_length=1,
        sliding_padding=200,
        batch_size=256,
    ).squeeze()

    full_repr_wom = model.encode(
        full,
        causal=True,
        sliding_length=1,
        sliding_padding=200,
        batch_size=256,
    ).squeeze()

    train_n = len(train_data)
    test_n = len(test_data)

    train_err = np.abs(full_repr_wom[:train_n] - full_repr[:train_n]).sum(axis=1)
    test_err = np.abs(full_repr_wom[train_n:train_n + test_n] - full_repr[train_n:train_n + test_n]).sum(axis=1)

    ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
    train_err_adj = (train_err - ma[:train_n]) / np.where(ma[:train_n] != 0, ma[:train_n], 1.0)
    test_err_adj = (test_err - ma[train_n:train_n + test_n]) / np.where(ma[train_n:train_n + test_n] != 0,
                                                                          ma[train_n:train_n + test_n], 1.0)

    test_err_adj = np.nan_to_num(test_err_adj, nan=0.0, posinf=0.0, neginf=0.0)
    return test_err_adj.astype(np.float32)


def run_one(filename, dataset_dir, score_dir, n_iters=200, batch_size=8, log_handle=None):
    """Process one fid. Idempotent (skips if .npy exists)."""
    fid = filename.replace('.csv', '')
    out_path = Path(score_dir) / f"{fid}.npy"
    if out_path.exists():
        return ('SKIP_DONE', fid, 0.0)

    try:
        df = pd.read_csv(Path(dataset_dir) / filename).dropna()
        data = df.iloc[:, 0:-1].values.astype(np.float64)
        if data.shape[1] > 1:
            data = data[:, 0:1]
        train_index = int(filename.split('.')[0].split('_')[-3])
        train_data = data[:train_index, 0]
        test_data = data[train_index:, 0]
        full_test_n = len(test_data)

        from ts2vec import TS2Vec

        # zero-mean, unit-std the train portion (TS2Vec default needs std-norm input)
        # Use train statistics so we don't leak test-side info.
        mu = float(train_data.mean())
        sigma = float(train_data.std() + 1e-6)
        train_norm = ((train_data - mu) / sigma).astype(np.float32)
        test_norm = ((test_data - mu) / sigma).astype(np.float32)

        t0 = time.time()
        model = TS2Vec(
            input_dims=1,
            output_dims=320,
            hidden_dims=64,
            depth=10,
            device='cuda',
            lr=0.001,
            batch_size=batch_size,
            max_train_length=3000,
        )
        # Train on train portion shaped (1, T, 1)
        model.fit(train_norm.reshape(1, -1, 1), n_iters=n_iters, verbose=False)

        score = compute_ts2vec_score(model, train_norm, test_norm)
        if len(score) != full_test_n:
            # Pad/truncate to match expected length
            if len(score) < full_test_n:
                score = np.concatenate([score, np.full(full_test_n - len(score), score[-1] if len(score) > 0 else 0.0, dtype=np.float32)])
            else:
                score = score[:full_test_n]

        np.save(out_path, score)
        del model
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        if log_handle is not None:
            log_handle.write(f"OK {fid} elapsed={elapsed:.1f}s n_test={full_test_n}\n")
            log_handle.flush()
        return ('OK', fid, elapsed)
    except Exception as e:
        if log_handle is not None:
            log_handle.write(f"FAIL {fid}: {e}\n{traceback.format_exc()}\n")
            log_handle.flush()
        return ('FAIL', fid, str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file_list', default=tsbad_eva_csv(), help='CSV with file_name column')
    ap.add_argument('--dataset_dir', default=tsbad_dataset_dir(), help='dir containing the data CSVs')
    ap.add_argument('--score_dir', required=True, help='output dir for .npy scores')
    ap.add_argument('--log_path', required=True)
    ap.add_argument('--n_iters', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    args = ap.parse_args()

    Path(args.score_dir).mkdir(parents=True, exist_ok=True)
    file_list = pd.read_csv(args.file_list)['file_name'].tolist()
    end = len(file_list) if args.end == -1 else args.end
    file_list = file_list[args.start:end]
    print(f"[ts2vec] {len(file_list)} fids on this chunk; gpu={torch.cuda.is_available()}")
    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    log_handle = open(args.log_path, 'a', buffering=1)
    log_handle.write(f"\n=== Run start: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_handle.write(f"file_list={args.file_list} score_dir={args.score_dir} n_iters={args.n_iters} start={args.start} end={args.end}\n\n")
    for i, fname in enumerate(file_list):
        status, fid, info = run_one(fname, args.dataset_dir, args.score_dir,
                                    n_iters=args.n_iters, batch_size=args.batch_size,
                                    log_handle=log_handle)
        if status == 'OK': n_ok += 1
        elif status.startswith('SKIP'): n_skip += 1
        else: n_fail += 1
        if (i + 1) % 5 == 0 or status == 'FAIL':
            print(f"  [{i+1}/{len(file_list)}] {status} {fid} info={info}", flush=True)
    log_handle.write(f"=== Run end: ok={n_ok} skip={n_skip} fail={n_fail} wall={time.time()-t_start:.0f}s ===\n")
    log_handle.close()
    print(f"[ts2vec DONE] ok={n_ok} skip={n_skip} fail={n_fail} in {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
