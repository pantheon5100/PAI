"""DCdetector + PAI representation scoring.

Same protocol as TS2Vec uniform-score, but uses DCdetector's encoder representations
instead of TS2Vec's.

For DCdetector, the "representation" we extract is the output of the
`embedding_window_size` layer — a per-position embedding produced after the
RevIN normalization but BEFORE the dual-attention head. This is the cleanest
"representation" the model exposes and is what the attention paths operate on.

For each fid:
  1. Train DCdetector on train sliding windows.
  2. Extract per-position representations from the trained model:
       For each non-overlapping 100-sample window of (train+test), run
       model.embedding_window_size(x_revin_norm) → (win_size, d_model)
       Concat to per-position representation (T_full, 256)
  3. Build PaAno-style kmeans bank from train representations
  4. Score test representations via mean-of-top-3 euclidean distance to bank
  5. Compute magG, T2 (raw-signal magnitude statistics) for fusion
  6. Save: <fid>_eu.npy, _magG.npy, _T2.npy, _train_stats.npz

NOTE: DCdetector has issues with very small train sets. We require
train_n >= win_size; otherwise we skip with a default-zero score.
"""
from __future__ import annotations
import argparse, os, sys, time, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv
from sklearn.cluster import MiniBatchKMeans

DCDETECTOR_REPO = os.environ.get(
    "PAIAD_DCDETECTOR_REPO",
    str(Path(__file__).resolve().parents[2] / "third_party" / "KDD2023-DCdetector"),
)
sys.path.insert(0, DCDETECTOR_REPO)

EPS = 1e-12
T2_WINDOW = 32

warnings.filterwarnings('ignore')


# === PaAno-equivalent scoring (copied verbatim from PaAno score_variants_15c.py) ===

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


# === Magnitude statistics ===

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


# === DCdetector training ===

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, x, win_size, step=1):
        self.x = x.astype(np.float32)
        self.win_size = win_size
        self.step = step

    def __len__(self):
        return max(0, (len(self.x) - self.win_size) // self.step + 1)

    def __getitem__(self, i):
        s = i * self.step
        return self.x[s:s + self.win_size]


def train_dcdetector(model, optimizer, train_loader, win_size, n_epochs, device):
    model.train()
    for epoch in range(n_epochs):
        for batch in train_loader:
            x = batch.float().to(device)
            optimizer.zero_grad()
            series, prior = model(x)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                norm = torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)
                series_loss += (
                    torch.mean(my_kl_loss(series[u], (prior[u] / norm).detach()))
                    + torch.mean(my_kl_loss((prior[u] / norm).detach(), series[u]))
                )
                prior_loss += (
                    torch.mean(my_kl_loss((prior[u] / norm), series[u].detach()))
                    + torch.mean(my_kl_loss(series[u].detach(), (prior[u] / norm)))
                )
            series_loss /= len(prior)
            prior_loss /= len(prior)
            loss1 = prior_loss - series_loss
            loss2 = series_loss - prior_loss
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()


# === Representation extraction ===

def extract_representations(model, x_full, win_size, batch_size, device):
    """Extract per-position representations from DCdetector's embedding_window_size
    layer (= the post-RevIN, pre-attention representation).

    x_full: (T, 1) raw signal (will be RevIN-normed inside model.forward)
    Returns: (T, d_model) per-position representations
    """
    model.eval()
    n = len(x_full)
    pad = (win_size - (n % win_size)) % win_size
    if pad > 0:
        x_padded = np.concatenate([x_full, np.tile(x_full[-1:], (pad, 1))], axis=0)
    else:
        x_padded = x_full
    n_padded = len(x_padded)

    rep_chunks = []
    with torch.no_grad():
        for w_start in range(0, n_padded, win_size * batch_size):
            w_end = min(w_start + win_size * batch_size, n_padded)
            x_chunk = x_padded[w_start:w_end].reshape(-1, win_size, x_full.shape[1])
            x = torch.from_numpy(x_chunk.astype(np.float32)).to(device)
            # Replicate the relevant part of model.forward():
            from model.RevIN import RevIN
            revin_layer = RevIN(num_features=x.shape[2]).to(device)
            x_norm = revin_layer(x, 'norm')
            x_ori = model.embedding_window_size(x_norm)  # (B, win_size, d_model)
            rep_chunks.append(x_ori.cpu().numpy())
    rep = np.concatenate([r.reshape(-1, r.shape[-1]) for r in rep_chunks], axis=0)
    return rep[:n]  # Trim padding


def run_one(filename, dataset_dir, score_dir, win_size=100, patch_size=(5,), n_epochs=3,
            batch_size=64, seed=2024, log_handle=None):
    fid = filename.replace('.csv', '')
    out_eu = Path(score_dir) / f"{fid}_eu.npy"
    out_magG = Path(score_dir) / f"{fid}_magG.npy"
    out_T2 = Path(score_dir) / f"{fid}_T2.npy"
    out_stats = Path(score_dir) / f"{fid}_train_stats.npz"
    if all(p.exists() for p in [out_eu, out_magG, out_T2, out_stats]):
        return ('SKIP_DONE', fid, 0.0)
    try:
        from model.DCdetector import DCdetector
        df = pd.read_csv(Path(dataset_dir) / filename).dropna()
        data = df.iloc[:, 0:-1].values.astype(np.float64)
        if data.shape[1] > 1:
            data = data[:, 0:1]
        train_index = int(filename.split('.')[0].split('_')[-3])
        train_data = data[:train_index]
        test_data = data[train_index:]

        if len(train_data) < win_size + 10:
            np.save(out_eu, np.zeros(len(test_data), dtype=np.float32))
            np.save(out_magG, _magG(train_data, test_data))
            np.save(out_T2, _T2_meanshift(train_data, test_data, T2_WINDOW))
            np.savez(out_stats, eu_mean=0.0, eu_std=1.0, magG_mean=0.0, magG_std=1.0,
                     T2_mean=0.0, T2_std=1.0, bank_size=0, n_train=len(train_data), n_test=len(test_data))
            return ('SKIP_SHORT_TRAIN', fid, 0.0)

        scaler = StandardScaler()
        scaler.fit(train_data)
        train_norm = scaler.transform(train_data).astype(np.float32)
        test_norm = scaler.transform(test_data).astype(np.float32)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        model = DCdetector(
            win_size=win_size, enc_in=1, c_out=1, n_heads=1, d_model=256,
            e_layers=3, patch_size=list(patch_size), channel=1,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train
        train_ds = WindowDataset(train_norm, win_size=win_size, step=1)
        if len(train_ds) == 0:
            np.save(out_eu, np.zeros(len(test_data), dtype=np.float32))
            np.save(out_magG, _magG(train_data, test_data))
            np.save(out_T2, _T2_meanshift(train_data, test_data, T2_WINDOW))
            np.savez(out_stats, eu_mean=0.0, eu_std=1.0, magG_mean=0.0, magG_std=1.0,
                     T2_mean=0.0, T2_std=1.0, bank_size=0, n_train=len(train_data), n_test=len(test_data))
            return ('SKIP_SHORT_TRAIN', fid, 0.0)
        max_iters_per_epoch = 100
        if len(train_ds) > max_iters_per_epoch * batch_size:
            indices = np.random.permutation(len(train_ds))[:max_iters_per_epoch * batch_size]
            train_ds = torch.utils.data.Subset(train_ds, indices.tolist())
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        t0 = time.time()
        train_dcdetector(model, optimizer, train_loader, win_size=win_size, n_epochs=n_epochs, device=device)

        # Extract representations
        train_rep = extract_representations(model, train_norm, win_size, 64, device)
        test_rep = extract_representations(model, test_norm, win_size, 64, device)
        train_rep_t = torch.from_numpy(train_rep)
        test_rep_t = torch.from_numpy(test_rep)

        # PaAno-style scoring
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
    ap.add_argument('--win_size', type=int, default=100)
    ap.add_argument('--patch_size', type=int, nargs='+', default=[5])
    ap.add_argument('--n_epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    args = ap.parse_args()

    Path(args.score_dir).mkdir(parents=True, exist_ok=True)
    file_list = pd.read_csv(args.file_list)['file_name'].tolist()
    end = len(file_list) if args.end == -1 else args.end
    file_list = file_list[args.start:end]
    print(f"[dcdetector_uniform] {len(file_list)} fids; gpu={torch.cuda.is_available()}")
    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    log_handle = open(args.log_path, 'a', buffering=1)
    log_handle.write(f"\n=== Run start: {time.strftime('%Y-%m-%d %H:%M:%S')} start={args.start} end={end} ===\n")
    for i, fname in enumerate(file_list):
        status, fid, info = run_one(
            fname, args.dataset_dir, args.score_dir,
            win_size=args.win_size, patch_size=args.patch_size, n_epochs=args.n_epochs,
            batch_size=args.batch_size, seed=args.seed, log_handle=log_handle,
        )
        if status == 'OK':
            n_ok += 1
        elif status.startswith('SKIP'):
            n_skip += 1
        else:
            n_fail += 1
        if (i + 1) % 5 == 0 or status == 'FAIL':
            print(f"  [{i+1}/{len(file_list)}] {status} {fid} info={info}", flush=True)
    log_handle.write(f"=== Run end: ok={n_ok} skip={n_skip} fail={n_fail} wall={time.time()-t_start:.0f}s ===\n")
    log_handle.close()
    print(f"[dcdetector_uniform DONE] ok={n_ok} skip={n_skip} fail={n_fail} in {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
