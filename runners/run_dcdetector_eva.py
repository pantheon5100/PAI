"""DCdetector adapter for TSB-AD-U-Eva evaluation.

Standalone runner; doesn't modify the DCdetector repo. For each fid:
  1. Read CSV, split into train (first tr_* rows) and test (rest).
  2. StandardScaler fit on train, transform both.
  3. Train DCdetector for n_epochs on train sliding windows (win_size=100, step=1).
  4. Score test using non-overlapping windows; for each window the model produces
     a per-position energy (softmax of -series_loss - prior_loss across win_size).
  5. Concatenate to per-point score over the test portion. Save as .npy.

Score = per-position energy from DCdetector's contrastive scoring (the official
        method in solver.py:test). NOTE: higher = MORE anomalous, but the official
        score is a softmax which is bounded — the rank order is what matters for
        AUC-PR / VUS-PR.
"""
from __future__ import annotations
import argparse, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv

DCDETECTOR_REPO = os.environ.get(
    "PAIAD_DCDETECTOR_REPO",
    str(Path(__file__).resolve().parents[1] / "third_party" / "KDD2023-DCdetector"),
)
sys.path.insert(0, DCDETECTOR_REPO)


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, win_size: int, step: int = 1):
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
            # Min-max optimization: prior tries to minimize, series tries to maximize the contrastive distance
            loss1 = prior_loss - series_loss
            loss2 = series_loss - prior_loss
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()


def score_test_windows(model, test_x: np.ndarray, win_size: int, batch_size: int, device, temperature: float = 50.0):
    """Run model over non-overlapping windows of test, return per-point score."""
    model.eval()
    n = len(test_x)
    # We'll use sliding windows with step=win_size for non-overlapping coverage
    # Pad the end so we cover the full series
    pad = (win_size - (n % win_size)) % win_size
    if pad > 0:
        test_padded = np.concatenate([test_x, np.tile(test_x[-1:], (pad, 1))], axis=0)
    else:
        test_padded = test_x
    n_windows = len(test_padded) // win_size

    score_chunks = []
    with torch.no_grad():
        for w_start in range(0, len(test_padded), win_size * batch_size):
            w_end = min(w_start + win_size * batch_size, len(test_padded))
            x_chunk = test_padded[w_start:w_end].reshape(-1, win_size, test_x.shape[1])
            x = torch.from_numpy(x_chunk.astype(np.float32)).to(device)
            series, prior = model(x)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                norm = torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / norm).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / norm), series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / norm).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / norm), series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)  # (B, win_size)
            score_chunks.append(metric.detach().cpu().numpy())
    score_arr = np.concatenate([s.reshape(-1) for s in score_chunks], axis=0)[:n]
    return score_arr.astype(np.float32)


def run_one(filename, dataset_dir, score_dir, win_size=100, patch_size=(5,), n_epochs=3,
            batch_size=64, lr=1e-4, log_handle=None):
    fid = filename.replace('.csv', '')
    out_path = Path(score_dir) / f"{fid}.npy"
    if out_path.exists():
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
            # Skip — too short to train a meaningful model
            np.save(out_path, np.zeros(len(test_data), dtype=np.float32))
            return ('SKIP_SHORT_TRAIN', fid, 0.0)

        # Standardize on train
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_norm = scaler.transform(train_data).astype(np.float32)
        test_norm = scaler.transform(test_data).astype(np.float32)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DCdetector(
            win_size=win_size,
            enc_in=1,
            c_out=1,
            n_heads=1,
            d_model=256,
            e_layers=3,
            patch_size=list(patch_size),
            channel=1,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_ds = WindowDataset(train_norm, win_size=win_size, step=1)
        if len(train_ds) == 0:
            np.save(out_path, np.zeros(len(test_data), dtype=np.float32))
            return ('SKIP_SHORT_TRAIN', fid, 0.0)
        # Cap train batches per epoch to keep wall time manageable
        max_iters_per_epoch = 100
        if len(train_ds) > max_iters_per_epoch * batch_size:
            indices = np.random.permutation(len(train_ds))[:max_iters_per_epoch * batch_size]
            train_ds = torch.utils.data.Subset(train_ds, indices.tolist())
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        t0 = time.time()
        train_dcdetector(model, optimizer, train_loader, win_size=win_size, n_epochs=n_epochs, device=device)
        score = score_test_windows(model, test_norm, win_size=win_size, batch_size=64, device=device)
        # Truncate/pad to test length
        if len(score) != len(test_data):
            if len(score) < len(test_data):
                score = np.concatenate([score, np.full(len(test_data) - len(score), score[-1] if len(score) > 0 else 0.0, dtype=np.float32)])
            else:
                score = score[:len(test_data)]
        np.save(out_path, score)
        del model
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        if log_handle is not None:
            log_handle.write(f"OK {fid} elapsed={elapsed:.1f}s n_test={len(test_data)} train_n={len(train_norm)}\n")
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
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    args = ap.parse_args()

    Path(args.score_dir).mkdir(parents=True, exist_ok=True)
    file_list = pd.read_csv(args.file_list)['file_name'].tolist()
    end = len(file_list) if args.end == -1 else args.end
    file_list = file_list[args.start:end]
    print(f"[dcdetector] {len(file_list)} fids; gpu={torch.cuda.is_available()}")
    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    log_handle = open(args.log_path, 'a', buffering=1)
    log_handle.write(f"\n=== Run start: {time.strftime('%Y-%m-%d %H:%M:%S')} win_size={args.win_size} patch_size={args.patch_size} n_epochs={args.n_epochs} start={args.start} end={args.end} ===\n")
    for i, fname in enumerate(file_list):
        status, fid, info = run_one(
            fname, args.dataset_dir, args.score_dir,
            win_size=args.win_size, patch_size=args.patch_size, n_epochs=args.n_epochs,
            batch_size=args.batch_size, log_handle=log_handle,
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
    print(f"[dcdetector DONE] ok={n_ok} skip={n_skip} fail={n_fail} in {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
