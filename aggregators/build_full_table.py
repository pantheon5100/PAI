"""Build the TSB-AD-U Eva method comparison table.

The table compares original scores and PAI scores for:

  1. DCdetector
  2. TS2Vec
  3. TSPulse
  4. PaAno

6 metrics (TSB-AD's get_metrics names in parens):
  - VUS-PR
  - VUS-ROC
  - Range-F1     (= Event-based-F1)
  - AUC-PR
  - AUC-ROC
  - Point-F1     (= Standard-F1)
"""
from __future__ import annotations
import argparse, csv, os, time
from pathlib import Path
from multiprocessing import Pool
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import numpy as np
import pandas as pd
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv
warnings.filterwarnings('ignore')

SCORE_ROOT = os.environ.get(
    "PAIAD_SCORE_ROOT",
    "outputs/score",
)
DATASET_DIR = os.environ.get(
    "PAIAD_TSB_U_DATASET_DIR",
    str(tsbad_dataset_dir()),
)
EVA_CSV = os.environ.get(
    "PAIAD_TSB_U_EVA_CSV",
    str(tsbad_eva_csv()),
)
# Cache locations
TSPULSE_ZS_DIR    = os.environ.get("PAIAD_TSPULSE_ZS_DIR", str(Path(SCORE_ROOT) / "TSPulse_ZS"))
TSPULSE_FT_DIR    = os.environ.get("PAIAD_TSPULSE_FT_DIR", str(Path(SCORE_ROOT) / "TSPulse_FT"))
TS2VEC_NATIVE_DIR = os.environ.get("PAIAD_TS2VEC_NATIVE_DIR", str(Path(SCORE_ROOT) / "TS2Vec"))
DCDET_NATIVE_DIR  = os.environ.get("PAIAD_DCDET_NATIVE_DIR", str(Path(SCORE_ROOT) / "DCdetector"))
UNIFORM_TS2VEC    = os.environ.get("PAIAD_UNIFORM_TS2VEC_DIR", str(Path(SCORE_ROOT) / "UNIFORM_TS2Vec"))
UNIFORM_DCDET     = os.environ.get("PAIAD_UNIFORM_DCDET_DIR", str(Path(SCORE_ROOT) / "UNIFORM_DCdetector"))
PAANO_ORIGINAL_DIR = os.environ.get("PAIAD_PAANO_ORIGINAL_DIR", str(Path(SCORE_ROOT) / "PaAno_baseline"))
PAANO_PAI_DIR = os.environ.get("PAIAD_PAANO_PAI_DIR", str(Path(SCORE_ROOT) / "PaAno_PAI"))

EPS = 1e-9


def _zscore(x):
    m = float(np.nanmean(x)); s = float(np.nanstd(x)) + EPS
    return (x - m) / s


def _eval_metrics_full_or_test(score, label, train_index, n_full, data_for_window):
    """TSB-AD eval; handles full-series or test-only score lengths."""
    from TSB_AD.utils.slidingWindows import find_length_rank
    from TSB_AD.evaluation.metrics import get_metrics
    s_len = len(score)
    n_test = n_full - train_index
    if abs(s_len - n_full) <= 2:
        n = min(s_len, n_full)
        s = score[:n]; l = label[:n]
        d = data_for_window[:n, 0].reshape(-1, 1)
    elif abs(s_len - n_test) <= 2:
        n = min(s_len, n_test)
        s = score[:n]; l = label[train_index:train_index + n]
        d = data_for_window[train_index:train_index + n, 0].reshape(-1, 1)
    else:
        return None
    sw = find_length_rank(d, rank=1)
    return get_metrics(s, l, slidingWindow=sw)


def _load_score(method, fid):
    """Return per-fid score depending on method-variant. None if unavailable."""
    p = None
    if method == 'DCdetector_original':
        p = Path(DCDET_NATIVE_DIR) / f"{fid}.npy"
        return np.load(p) if p.exists() else None
    if method == 'TS2Vec_original':
        p = Path(TS2VEC_NATIVE_DIR) / f"{fid}.npy"
        return np.load(p) if p.exists() else None
    if method == 'TSPulse_ZS_original':
        p = Path(TSPULSE_ZS_DIR) / f"{fid}.npy"
        return np.load(p) if p.exists() else None
    if method == 'TSPulse_FT_original':
        p = Path(TSPULSE_FT_DIR) / f"{fid}.npy"
        return np.load(p) if p.exists() else None
    if method == 'PaAno_original':
        p = Path(PAANO_ORIGINAL_DIR) / fid / 'cos_score.npy'  # PaAno's published cosine baseline
        return np.load(p) if p.exists() else None
    if method == 'PaAno_ours':
        p = Path(PAANO_PAI_DIR) / fid / 'eucl_score.npy'
        return np.load(p) if p.exists() else None
    if method == 'TS2Vec_ours':
        eu_p = Path(UNIFORM_TS2VEC) / f"{fid}_eu.npy"
        mg_p = Path(UNIFORM_TS2VEC) / f"{fid}_magG.npy"
        t2_p = Path(UNIFORM_TS2VEC) / f"{fid}_T2.npy"
        st_p = Path(UNIFORM_TS2VEC) / f"{fid}_train_stats.npz"
        if not all(x.exists() for x in [eu_p, mg_p, t2_p, st_p]):
            return None
        eu = np.load(eu_p).astype(np.float64)
        mg = np.load(mg_p).astype(np.float64)
        t2 = np.load(t2_p).astype(np.float64)
        st = np.load(st_p)
        z_eu = (eu - float(st['eu_mean'])) / float(st['eu_std'])
        z_mg = (mg - float(st['magG_mean'])) / float(st['magG_std'])
        z_t2 = (t2 - float(st['T2_mean'])) / float(st['T2_std'])
        s = 0.6 * z_eu + 0.4 * z_mg + 0.2 * z_t2
        return np.nan_to_num(s.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if method == 'DCdetector_ours':
        eu_p = Path(UNIFORM_DCDET) / f"{fid}_eu.npy"
        mg_p = Path(UNIFORM_DCDET) / f"{fid}_magG.npy"
        t2_p = Path(UNIFORM_DCDET) / f"{fid}_T2.npy"
        st_p = Path(UNIFORM_DCDET) / f"{fid}_train_stats.npz"
        if not all(x.exists() for x in [eu_p, mg_p, t2_p, st_p]):
            return None
        eu = np.load(eu_p).astype(np.float64)
        mg = np.load(mg_p).astype(np.float64)
        t2 = np.load(t2_p).astype(np.float64)
        st = np.load(st_p)
        z_eu = (eu - float(st['eu_mean'])) / float(st['eu_std'])
        z_mg = (mg - float(st['magG_mean'])) / float(st['magG_std'])
        z_t2 = (t2 - float(st['T2_mean'])) / float(st['T2_std'])
        s = 0.6 * z_eu + 0.4 * z_mg + 0.2 * z_t2
        return np.nan_to_num(s.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if method == 'TSPulse_ZS_ours':  # native + magG + T2 fusion-on-top
        # Use TSPulse_ZS native score as eu; pull magG/T2 from UNIFORM_TS2Vec outputs.
        native_p = Path(TSPULSE_ZS_DIR) / f"{fid}.npy"
        mg_p = Path(UNIFORM_TS2VEC) / f"{fid}_magG.npy"
        t2_p = Path(UNIFORM_TS2VEC) / f"{fid}_T2.npy"
        st_p = Path(UNIFORM_TS2VEC) / f"{fid}_train_stats.npz"
        if not all(x.exists() for x in [native_p, mg_p, t2_p, st_p]):
            return None
        native = np.load(native_p).astype(np.float64)
        mg = np.load(mg_p).astype(np.float64)
        t2 = np.load(t2_p).astype(np.float64)
        # Important: TSPulse native covers full series; magG/T2 cover test only.
        # Slice native to test portion to align all three:
        train_index = int(fid.split('_')[-3])
        if len(native) > len(mg) + 100:
            # native is full-series; align to test
            native_test = native[train_index:train_index + len(mg)]
        else:
            native_test = native[:len(mg)]
        n = min(len(native_test), len(mg), len(t2))
        native_test = native_test[:n]; mg = mg[:n]; t2 = t2[:n]
        # Train-calibrated z-norm for magG/T2; test-pool z for native (since we don't have train stats for it)
        st = np.load(st_p)
        z_native = _zscore(native_test)
        z_mg = (mg - float(st['magG_mean'])) / float(st['magG_std'])
        z_t2 = (t2 - float(st['T2_mean'])) / float(st['T2_std'])
        s = 0.6 * z_native + 0.4 * z_mg + 0.2 * z_t2
        return np.nan_to_num(s.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return None


_METHOD = None


def _init(method):
    global _METHOD
    _METHOD = method


def _eval_one(fname):
    fid = fname.replace('.csv', '')
    score = _load_score(_METHOD, fid)
    if score is None:
        return ('SKIP_MISSING', fid, None)
    try:
        df = pd.read_csv(Path(DATASET_DIR) / fname).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        train_index = int(fid.split('_')[-3])
        result = _eval_metrics_full_or_test(score, label, train_index, len(label), data)
        if result is None:
            return ('SKIP_LEN', fid, None)
        return ('OK', fid, {k: float(v) for k, v in result.items()})
    except Exception as e:
        return ('FAIL', fid, str(e)[:200])


def aggregate_method(method, num_procs=16):
    eva = pd.read_csv(EVA_CSV)['file_name'].tolist()
    print(f"\n[{method}] {len(eva)} fids, {num_procs} procs", flush=True)
    t0 = time.time()
    rows = []
    n_ok = n_skip = n_fail = 0
    with Pool(processes=num_procs, initializer=_init, initargs=(method,)) as pool:
        for i, (status, fid, payload) in enumerate(pool.imap_unordered(_eval_one, eva), start=1):
            if status == 'OK':
                row = {'fid': fid}
                row.update(payload)
                rows.append(row)
                n_ok += 1
            elif status.startswith('SKIP'):
                n_skip += 1
            else:
                n_fail += 1
                print(f"  FAIL {fid}: {payload}", flush=True)
    print(f"[{method}] DONE ok={n_ok} skip={n_skip} fail={n_fail} in {time.time()-t0:.0f}s", flush=True)
    if not rows:
        return method, 0, {}
    df = pd.DataFrame(rows)
    pool_means = {}
    for c in ['AUC-PR', 'VUS-PR', 'AUC-ROC', 'VUS-ROC', 'Standard-F1', 'PA-F1', 'Event-based-F1', 'R-based-F1', 'Affiliation-F']:
        if c in df.columns:
            v = df[c].dropna().values
            pool_means[c] = float(v.mean()) if len(v) > 0 else float('nan')
    return method, len(rows), pool_means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--num_procs', type=int, default=16)
    args = ap.parse_args()

    methods = [
        'DCdetector_original', 'DCdetector_ours',
        'TS2Vec_original', 'TS2Vec_ours',
        'TSPulse_ZS_original', 'TSPulse_ZS_ours',
        'TSPulse_FT_original',  # for completeness
        'PaAno_original', 'PaAno_ours',
    ]
    results = []
    for m in methods:
        method, n, pool = aggregate_method(m, num_procs=args.num_procs)
        rec = {'method': method, 'n_fids': n}
        rec.update(pool)
        results.append(rec)

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)

    # Print compact table
    print("\n" + "=" * 130)
    print("FULL COMPARISON TABLE — TSB-AD-U-Eva pool means")
    print("=" * 130)
    print(f"  {'method':<24s} | {'n':>4s} | {'AUC-PR':>7s} {'AUC-ROC':>7s} {'VUS-PR':>7s} {'VUS-ROC':>7s} {'Point-F1':>8s} {'Range-F1':>8s}")
    print("  " + "-" * 100)
    for r in results:
        line = f"  {r['method']:<24s} | {r['n_fids']:>4d} | "
        line += f"{r.get('AUC-PR', float('nan')):>7.4f} "
        line += f"{r.get('AUC-ROC', float('nan')):>7.4f} "
        line += f"{r.get('VUS-PR', float('nan')):>7.4f} "
        line += f"{r.get('VUS-ROC', float('nan')):>7.4f} "
        line += f"{r.get('Standard-F1', float('nan')):>8.4f} "
        line += f"{r.get('Event-based-F1', float('nan')):>8.4f}"
        print(line)


if __name__ == '__main__':
    main()
