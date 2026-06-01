"""TS2Vec + PAI representation scoring.

For each fid:
  1. Train TS2Vec on the train portion.
  2. Encode the full signal (train + test concat) → per-timestep representation (T, 320)
  3. Build PaAno-style kmeans bank from train representations
  4. Score test representations via mean-of-top-3 euclidean distance to bank
  5. Also compute magG, T2 (raw-signal magnitude statistics) for fusion experiments
  6. Save:
       <fid>_eu.npy         — uniform encoder-distance score (test-only)
       <fid>_magG.npy       — point-scale raw amplitude (test-only)
       <fid>_T2.npy         — window-scale raw amplitude (test-only)
       <fid>_train_stats.npz — train statistics for z-normalization (eu, magG, T2 mean/std)

Fusion is computed at aggregation time by reading these
per-fid arrays; this lets us produce multiple fusion variants without re-running
the encoder.

Uses the same bank-distance scoring primitive as PaAno:
  bank = _kmeans_centers(train_patch, seed) → ~0.1·n_train clusters (max 500)
  score = _euclidean_topk(test, bank, top_k=3, agg='mean')

Per-timestep representation is treated as a "patch" with patch_size=1 (no
distribution-to-points needed).
"""
from __future__ import annotations
import argparse, os, sys, time, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import bottleneck as bn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv

# Repo paths — TS2Vec only on sys.path (PaAno code copied inline below to avoid
# `utils` package name collision)
TS2VEC_REPO = os.environ.get(
    "PAIAD_TS2VEC_REPO",
    str(Path(__file__).resolve().parents[2] / "third_party" / "ts2vec"),
)
sys.path.insert(0, TS2VEC_REPO)

EPS = 1e-12
T2_WINDOW = 32

warnings.filterwarnings('ignore')


# ============================================================================
# PaAno-equivalent uniform scoring (copied from PaAno/score_variants_15c.py
# verbatim to avoid path collision with TS2Vec's utils package).
# ============================================================================
from sklearn.cluster import MiniBatchKMeans


def _as_tensor(x):
    if torch.is_tensor(x):
        return x.detach().to(dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _baseline_bank_size(num_samples: int) -> int:
    if num_samples <= 1:
        return num_samples
    k = int(round(0.1 * num_samples))
    min_cores_eff = min(500, max(1, num_samples - 1))
    return max(min_cores_eff, min(k, num_samples - 1))


def _kmeans_centers(features, random_state: int, bank_size=None):
    x = _as_tensor(features)
    n = int(x.shape[0])
    if n <= 1:
        return x
    k = _baseline_bank_size(n) if bank_size is None else int(bank_size)
    k = max(1, min(k, n - 1))
    mbk = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        random_state=int(random_state),
        batch_size=max(8192, k),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    )
    mbk.fit(x.numpy())
    return torch.as_tensor(mbk.cluster_centers_, dtype=torch.float32)


def _euclidean_topk(query, bank, top_k: int, batch_size: int, device, agg='mean'):
    q = _as_tensor(query)
    b = _as_tensor(bank)
    if int(b.shape[0]) == 0:
        raise ValueError("Empty bank")
    k = max(1, min(int(top_k), int(b.shape[0])))
    b = b.to(device=device, dtype=torch.float32)
    out = []
    step = max(1, int(batch_size))
    for start in range(0, int(q.shape[0]), step):
        end = min(int(q.shape[0]), start + step)
        qb = q[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        d = torch.cdist(qb, b, p=2)
        top = d.topk(k=k, dim=1, largest=False).values
        if agg == 'mean':
            score = top.mean(dim=1)
        elif agg == 'median':
            score = top.median(dim=1).values
        else:
            raise ValueError(f"Unsupported agg={agg}")
        out.append(score.cpu())
    return torch.cat(out, dim=0).numpy().astype(np.float32, copy=False)


def _magG(x_train, x_test):
    """Point-scale raw amplitude: |x − median(x_train)| / MAD(x_train)."""
    xt = np.asarray(x_train, dtype=np.float64).reshape(-1)
    xq = np.asarray(x_test, dtype=np.float64).reshape(-1)
    med = float(np.median(xt))
    mad = float(np.median(np.abs(xt - med)) + EPS)
    return (np.abs(xq - med) / mad).astype(np.float32)


def _T2_meanshift(x_train, x_test, W):
    """Window-scale raw amplitude: |window_mean(x) − median(x_train)| / MAD(x_train)."""
    xt = np.asarray(x_train, dtype=np.float64).reshape(-1)
    xq = np.asarray(x_test, dtype=np.float64).reshape(-1)
    med = float(np.median(xt))
    mad = float(np.median(np.abs(xt - med)) + EPS)
    n = len(xq)
    cs = np.concatenate([[0.0], np.cumsum(xq)])
    out = np.zeros(n, dtype=np.float32)
    for t in range(n):
        a = max(0, t - W); b = min(n, t + W + 1)
        win_mean = (cs[b] - cs[a]) / (b - a)
        out[t] = abs(win_mean - med) / mad
    return out


def run_one(filename, dataset_dir, score_dir, n_iters=200, batch_size=8, seed=2024, log_handle=None):
    fid = filename.replace('.csv', '')
    out_eu = Path(score_dir) / f"{fid}_eu.npy"
    out_magG = Path(score_dir) / f"{fid}_magG.npy"
    out_T2 = Path(score_dir) / f"{fid}_T2.npy"
    out_stats = Path(score_dir) / f"{fid}_train_stats.npz"
    if out_eu.exists() and out_magG.exists() and out_T2.exists() and out_stats.exists():
        return ('SKIP_DONE', fid, 0.0)

    try:
        df = pd.read_csv(Path(dataset_dir) / filename).dropna()
        data = df.iloc[:, 0:-1].values.astype(np.float64)
        if data.shape[1] > 1:
            data = data[:, 0:1]
        train_index = int(filename.split('.')[0].split('_')[-3])
        train_data = data[:train_index, 0]
        test_data = data[train_index:, 0]

        # === 1. Train TS2Vec on z-normed train ===
        from ts2vec import TS2Vec
        mu = float(train_data.mean()); sigma = float(train_data.std() + 1e-6)
        train_norm = ((train_data - mu) / sigma).astype(np.float32)
        test_norm = ((test_data - mu) / sigma).astype(np.float32)

        t0 = time.time()
        model = TS2Vec(
            input_dims=1, output_dims=320, hidden_dims=64, depth=10,
            device='cuda', lr=0.001, batch_size=batch_size, max_train_length=3000,
        )
        model.fit(train_norm.reshape(1, -1, 1), n_iters=n_iters, verbose=False)

        # === 2. Encode full signal → per-timestep representations ===
        full = np.concatenate([train_norm, test_norm]).reshape(1, -1, 1).astype(np.float32)
        full_repr = model.encode(
            full, causal=True, sliding_length=1, sliding_padding=200, batch_size=256,
        ).squeeze()
        # Shape: (T_full, 320)
        train_repr = full_repr[:len(train_norm)].astype(np.float32)
        test_repr = full_repr[len(train_norm):].astype(np.float32)

        # === 3. PaAno-style scoring: kmeans bank + euclidean topk-3 ===
        # (functions copied verbatim from PaAno/score_variants_15c.py — see top of file)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_t = torch.from_numpy(train_repr)
        test_t = torch.from_numpy(test_repr)
        # Build bank from train representations
        bank = _kmeans_centers(train_t, random_state=seed)
        # Score test representations via mean-of-top-3 distance to bank centers
        eu_test = _euclidean_topk(test_t, bank, top_k=3, batch_size=512, device=device, agg='mean')
        # Also score train (for z-norm calibration)
        eu_train = _euclidean_topk(train_t, bank, top_k=3, batch_size=512, device=device, agg='mean')

        # === 4. Magnitude statistics (raw signal, train-calibrated) ===
        magG_test = _magG(train_data, test_data)
        magG_train = _magG(train_data, train_data)
        T2_test = _T2_meanshift(train_data, test_data, T2_WINDOW)
        T2_train = _T2_meanshift(train_data, train_data, T2_WINDOW)

        # === 5. Train stats for z-normalization ===
        train_stats = {
            'eu_mean': float(np.mean(eu_train)), 'eu_std': float(np.std(eu_train) + 1e-9),
            'magG_mean': float(np.mean(magG_train)), 'magG_std': float(np.std(magG_train) + 1e-9),
            'T2_mean': float(np.mean(T2_train)), 'T2_std': float(np.std(T2_train) + 1e-9),
            'bank_size': int(bank.shape[0]),
            'n_train': int(len(train_data)),
            'n_test': int(len(test_data)),
        }

        # === 6. Save ===
        np.save(out_eu, eu_test.astype(np.float32))
        np.save(out_magG, magG_test.astype(np.float32))
        np.save(out_T2, T2_test.astype(np.float32))
        np.savez(out_stats, **train_stats)

        del model
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        if log_handle is not None:
            log_handle.write(f"OK {fid} elapsed={elapsed:.1f}s n_test={len(test_data)} bank={int(bank.shape[0])}\n")
            log_handle.flush()
        return ('OK', fid, elapsed)
    except Exception as e:
        if log_handle is not None:
            log_handle.write(f"FAIL {fid}: {e}\n{traceback.format_exc()}\n")
            log_handle.flush()
        return ('FAIL', fid, str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file_list', default=tsbad_eva_csv())
    ap.add_argument('--dataset_dir', default=tsbad_dataset_dir())
    ap.add_argument('--score_dir', required=True)
    ap.add_argument('--log_path', required=True)
    ap.add_argument('--n_iters', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    args = ap.parse_args()

    Path(args.score_dir).mkdir(parents=True, exist_ok=True)
    file_list = pd.read_csv(args.file_list)['file_name'].tolist()
    end = len(file_list) if args.end == -1 else args.end
    file_list = file_list[args.start:end]
    print(f"[ts2vec_uniform] {len(file_list)} fids; gpu={torch.cuda.is_available()}")
    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    log_handle = open(args.log_path, 'a', buffering=1)
    log_handle.write(f"\n=== Run start: {time.strftime('%Y-%m-%d %H:%M:%S')} start={args.start} end={end} ===\n")
    for i, fname in enumerate(file_list):
        status, fid, info = run_one(fname, args.dataset_dir, args.score_dir,
                                    n_iters=args.n_iters, batch_size=args.batch_size,
                                    seed=args.seed, log_handle=log_handle)
        if status == 'OK': n_ok += 1
        elif status.startswith('SKIP'): n_skip += 1
        else: n_fail += 1
        if (i + 1) % 5 == 0 or status == 'FAIL':
            print(f"  [{i+1}/{len(file_list)}] {status} {fid} info={info}", flush=True)
    log_handle.write(f"=== Run end: ok={n_ok} skip={n_skip} fail={n_fail} wall={time.time()-t_start:.0f}s ===\n")
    log_handle.close()
    print(f"[ts2vec_uniform DONE] ok={n_ok} skip={n_skip} fail={n_fail} in {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
