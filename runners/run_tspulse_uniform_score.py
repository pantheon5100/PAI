"""TSPulse + PAI representation scoring.

Same protocol as TS2Vec/DCdetector uniform-score, but uses TSPulse's encoder
backbone hidden states instead.

For each fid:
  1. Load pretrained TSPulse-r1 model
  2. Run inference on the FULL signal (train + test concat, sliding windows of 96)
       with output_hidden_states=True
  3. Extract backbone last hidden state (per-patch representations)
  4. Expand patches back to per-position representation:
       Each patch covers `patch_length` timesteps; replicate the patch embedding
       to all positions in the patch
  5. Build PaAno-style kmeans bank from train representations
  6. Score test representations: mean of top-3 euclidean distance to bank
  7. Compute magG, T2 per fid
  8. Save: <fid>_eu.npy, _magG.npy, _T2.npy, _train_stats.npz

This runner uses the zero-shot TSPulse encoder. Fine-tuned TSPulse scoring is
kept separate in `run_tspulse_ft_pmlookup.py`.
"""
from __future__ import annotations
import argparse, os, sys, time, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

EPS = 1e-12
T2_WINDOW = 32
warnings.filterwarnings('ignore')


# === PaAno scoring (verbatim) ===

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
        n_clusters=k, init="k-means++", random_state=int(random_state),
        batch_size=max(8192, k), max_iter=50, n_init=1, reassignment_ratio=0.01,
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
        else:
            raise ValueError(f"Unsupported agg={agg}")
        out.append(score.cpu())
    return torch.cat(out, dim=0).numpy().astype(np.float32, copy=False)


def _magG(x_train, x_test):
    xt = np.asarray(x_train, dtype=np.float64).reshape(-1)
    xq = np.asarray(x_test, dtype=np.float64).reshape(-1)
    med = float(np.median(xt))
    mad = float(np.median(np.abs(xt - med)) + EPS)
    return (np.abs(xq - med) / mad).astype(np.float32)


def _T2_meanshift(x_train, x_test, W):
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


# === TSPulse representation extraction ===

def extract_tspulse_representations(model, data_full, win_size, batch_size, device):
    """Extract per-position representations from TSPulse backbone.

    data_full: (T, 1) numpy
    Returns: (T, d_model) per-position representations

    Strategy: chunk data into non-overlapping windows of size `win_size` (default 96),
    pass through model.backbone with output_hidden_states=True. Take the last hidden
    state which is per-PATCH (B, num_patches, d_model). Expand patches to per-position
    by replicating each patch's embedding.
    """
    model.eval()
    n = data_full.shape[0]
    pad = (win_size - (n % win_size)) % win_size
    if pad > 0:
        data_padded = np.concatenate([data_full, np.tile(data_full[-1:], (pad, 1))], axis=0)
    else:
        data_padded = data_full

    # Build windows: (n_windows, win_size, 1)
    n_windows = data_padded.shape[0] // win_size
    windows = data_padded.reshape(n_windows, win_size, -1).astype(np.float32)

    # Run backbone in batches
    rep_chunks = []
    config = model.config
    patch_length = getattr(config, 'patch_length', 8)
    num_patches = win_size // patch_length

    with torch.no_grad():
        for batch_start in range(0, n_windows, batch_size):
            batch_end = min(batch_start + batch_size, n_windows)
            x = torch.from_numpy(windows[batch_start:batch_end]).to(device)
            try:
                output = model.backbone(x, output_hidden_states=True, return_dict=True)
                # last_hidden_state shape: (B, num_patches, num_channels, d_model) or (B, num_channels, num_patches, d_model)
                hs = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output[0]
            except Exception as e:
                # Fallback: try without return_dict
                output = model.backbone(x, output_hidden_states=True)
                hs = output[0] if isinstance(output, tuple) else output

            # Reduce to (B, num_patches, d_model) regardless of input layout
            if hs.dim() == 4:
                # (B, num_patches, num_channels, d_model) — squeeze channels
                if hs.shape[1] == num_patches:
                    hs = hs.mean(dim=2)  # collapse channel
                elif hs.shape[2] == num_patches:
                    hs = hs.mean(dim=1)  # collapse first non-batch dim
                else:
                    hs = hs.flatten(1, 2)  # last resort
            hs = hs.cpu().numpy()  # (B, num_patches, d_model) or similar
            rep_chunks.append(hs)

    # Concatenate batches
    rep = np.concatenate(rep_chunks, axis=0)  # (n_windows, num_patches, d_model)
    # Expand each patch to patch_length positions
    rep_per_pos = np.repeat(rep, patch_length, axis=1)  # (n_windows, win_size, d_model)
    # Flatten windows to per-position
    rep_per_pos = rep_per_pos.reshape(-1, rep_per_pos.shape[-1])  # (n_windows * win_size, d_model)
    # Trim padding
    return rep_per_pos[:n]


def run_one(filename, dataset_dir, score_dir, model, device, seed=2024, log_handle=None):
    fid = filename.replace('.csv', '')
    out_eu = Path(score_dir) / f"{fid}_eu.npy"
    out_magG = Path(score_dir) / f"{fid}_magG.npy"
    out_T2 = Path(score_dir) / f"{fid}_T2.npy"
    out_stats = Path(score_dir) / f"{fid}_train_stats.npz"
    if all(p.exists() for p in [out_eu, out_magG, out_T2, out_stats]):
        return ('SKIP_DONE', fid, 0.0)
    try:
        df = pd.read_csv(Path(dataset_dir) / filename).dropna()
        data = df.iloc[:, 0:-1].values.astype(np.float64)
        if data.shape[1] > 1:
            data = data[:, 0:1]
        train_index = int(filename.split('.')[0].split('_')[-3])
        train_data = data[:train_index]
        test_data = data[train_index:]

        # MinMax-norm using train
        scaler = StandardScaler()
        scaler.fit(train_data)
        full_norm = scaler.transform(np.concatenate([train_data, test_data])).astype(np.float32)

        # TSPulse needs context_length=512 (per its training); aggr_win_size=96 is at scoring step
        win_size = int(model.config.context_length)
        t0 = time.time()
        full_rep = extract_tspulse_representations(model, full_norm, win_size=win_size, batch_size=4, device=device)
        train_rep = full_rep[:len(train_data)]
        test_rep = full_rep[len(train_data):len(train_data) + len(test_data)]

        train_rep_t = torch.from_numpy(train_rep)
        test_rep_t = torch.from_numpy(test_rep)
        bank = _kmeans_centers(train_rep_t, random_state=seed)
        eu_test = _euclidean_topk(test_rep_t, bank, top_k=3, batch_size=512, device=device, agg='mean')
        eu_train = _euclidean_topk(train_rep_t, bank, top_k=3, batch_size=512, device=device, agg='mean')

        magG_test = _magG(train_data, test_data)
        magG_train = _magG(train_data, train_data)
        T2_test = _T2_meanshift(train_data, test_data, T2_WINDOW)
        T2_train = _T2_meanshift(train_data, train_data, T2_WINDOW)

        train_stats = {
            'eu_mean': float(np.mean(eu_train)), 'eu_std': float(np.std(eu_train) + 1e-9),
            'magG_mean': float(np.mean(magG_train)), 'magG_std': float(np.std(magG_train) + 1e-9),
            'T2_mean': float(np.mean(T2_train)), 'T2_std': float(np.std(T2_train) + 1e-9),
            'bank_size': int(bank.shape[0]),
            'n_train': int(len(train_data)),
            'n_test': int(len(test_data)),
        }

        np.save(out_eu, eu_test.astype(np.float32))
        np.save(out_magG, magG_test.astype(np.float32))
        np.save(out_T2, T2_test.astype(np.float32))
        np.savez(out_stats, **train_stats)

        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        if log_handle is not None:
            log_handle.write(f"OK {fid} elapsed={elapsed:.1f}s n_test={len(test_data)} bank={int(bank.shape[0])} d_model={train_rep.shape[1]}\n")
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
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    args = ap.parse_args()

    Path(args.score_dir).mkdir(parents=True, exist_ok=True)
    file_list = pd.read_csv(args.file_list)['file_name'].tolist()
    end = len(file_list) if args.end == -1 else args.end
    file_list = file_list[args.start:end]
    print(f"[tspulse_uniform] {len(file_list)} fids; gpu={torch.cuda.is_available()}")

    # Load TSPulse model once (ZS, no per-fid finetune)
    print("[tspulse_uniform] loading model...")
    from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
    model = TSPulseForReconstruction.from_pretrained("ibm-granite/granite-timeseries-tspulse-r1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"[tspulse_uniform] model loaded on {device}, d_model={model.config.d_model}, patch_length={model.config.patch_length}")

    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    log_handle = open(args.log_path, 'a', buffering=1)
    log_handle.write(f"\n=== Run start: {time.strftime('%Y-%m-%d %H:%M:%S')} start={args.start} end={end} ===\n")
    for i, fname in enumerate(file_list):
        status, fid, info = run_one(fname, args.dataset_dir, args.score_dir, model, device,
                                    seed=args.seed, log_handle=log_handle)
        if status == 'OK': n_ok += 1
        elif status.startswith('SKIP'): n_skip += 1
        else: n_fail += 1
        if (i + 1) % 5 == 0 or status == 'FAIL':
            print(f"  [{i+1}/{len(file_list)}] {status} {fid} info={info}", flush=True)
    log_handle.write(f"=== Run end: ok={n_ok} skip={n_skip} fail={n_fail} wall={time.time()-t_start:.0f}s ===\n")
    log_handle.close()
    print(f"[tspulse_uniform DONE] ok={n_ok} skip={n_skip} fail={n_fail} in {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
