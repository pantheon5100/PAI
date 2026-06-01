"""Build the fusion-weight ablation table from model anomaly-score files.

This is the release-facing ablation for the PAI score fusion. It reuses
per-file anomaly-score components already produced by the runners:

  - encoder/native score component
  - raw magnitude component (magG)
  - local mean-shift component (T2)

The output is a compact sweep over fixed fusion weights for TS2Vec,
DCdetector, and TSPulse_ZS.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv
warnings.filterwarnings("ignore")

SCORE_ROOT = os.environ.get("PAIAD_SCORE_ROOT", "outputs/score")
DATASET_DIR = os.environ.get("PAIAD_TSB_U_DATASET_DIR", str(tsbad_dataset_dir()))
EVA_CSV = os.environ.get("PAIAD_TSB_U_EVA_CSV", str(tsbad_eva_csv()))

TSPULSE_ZS_DIR = os.environ.get("PAIAD_TSPULSE_ZS_DIR", str(Path(SCORE_ROOT) / "TSPulse_ZS"))
UNIFORM_TS2VEC = os.environ.get("PAIAD_UNIFORM_TS2VEC_DIR", str(Path(SCORE_ROOT) / "UNIFORM_TS2Vec"))
UNIFORM_DCDET = os.environ.get("PAIAD_UNIFORM_DCDET_DIR", str(Path(SCORE_ROOT) / "UNIFORM_DCdetector"))

METHODS = ("TS2Vec", "DCdetector", "TSPulse_ZS")

WEIGHT_GRID = (
    (1.00, 0.00, 0.00),
    (0.00, 1.00, 0.00),
    (0.00, 0.00, 1.00),
    (0.00, 1.00, 0.50),
    (1.00, 0.40, 0.20),
    (0.40, 0.40, 0.20),
    (0.50, 0.30, 0.15),
    (0.60, 0.20, 0.20),
    (0.60, 0.40, 0.00),
    (0.60, 0.40, 0.20),
    (0.60, 0.40, 0.40),
    (0.60, 0.40, 0.60),
    (0.60, 0.50, 0.30),
    (0.60, 0.60, 0.20),
    (0.70, 0.50, 0.25),
    (0.80, 0.40, 0.20),
)

METRIC_COLUMNS = (
    "AUC-PR",
    "VUS-PR",
    "AUC-ROC",
    "VUS-ROC",
    "Standard-F1",
    "Event-based-F1",
)

EPS = 1e-9
_JOB = None


def _zscore(x):
    m = float(np.nanmean(x))
    s = float(np.nanstd(x)) + EPS
    return (x - m) / s


def _weight_tag(method, weights):
    eu_w, mag_w, t2_w = weights
    return f"eva350_{method}_eu{eu_w:.2f}_g{mag_w:.2f}_t{t2_w:.2f}"


def _eval_metrics_full_or_test(score, label, train_index, n_full, data_for_window):
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank

    s_len = len(score)
    n_test = n_full - train_index
    if abs(s_len - n_full) <= 2:
        n = min(s_len, n_full)
        s = score[:n]
        l = label[:n]
        d = data_for_window[:n, 0].reshape(-1, 1)
    elif abs(s_len - n_test) <= 2:
        n = min(s_len, n_test)
        s = score[:n]
        l = label[train_index : train_index + n]
        d = data_for_window[train_index : train_index + n, 0].reshape(-1, 1)
    else:
        return None

    sw = find_length_rank(d, rank=1)
    return get_metrics(s, l, slidingWindow=sw)


def _uniform_components(root, fid):
    eu_p = Path(root) / f"{fid}_eu.npy"
    mag_p = Path(root) / f"{fid}_magG.npy"
    t2_p = Path(root) / f"{fid}_T2.npy"
    st_p = Path(root) / f"{fid}_train_stats.npz"
    if not all(p.exists() for p in (eu_p, mag_p, t2_p, st_p)):
        return None

    eu = np.load(eu_p).astype(np.float64)
    mag = np.load(mag_p).astype(np.float64)
    t2 = np.load(t2_p).astype(np.float64)
    st = np.load(st_p)
    n = min(len(eu), len(mag), len(t2))
    return {
        "eu": (eu[:n] - float(st["eu_mean"])) / float(st["eu_std"]),
        "mag": (mag[:n] - float(st["magG_mean"])) / float(st["magG_std"]),
        "t2": (t2[:n] - float(st["T2_mean"])) / float(st["T2_std"]),
    }


def _tspulse_components(fid):
    native_p = Path(TSPULSE_ZS_DIR) / f"{fid}.npy"
    base = _uniform_components(UNIFORM_TS2VEC, fid)
    if base is None or not native_p.exists():
        return None

    native = np.load(native_p).astype(np.float64)
    train_index = int(fid.split("_")[-3])
    n = min(len(base["mag"]), len(base["t2"]))
    if len(native) > n + 100:
        native = native[train_index : train_index + n]
    else:
        native = native[:n]
    n = min(len(native), n)
    return {
        "eu": _zscore(native[:n]),
        "mag": base["mag"][:n],
        "t2": base["t2"][:n],
    }


def _load_components(method, fid):
    if method == "TS2Vec":
        return _uniform_components(UNIFORM_TS2VEC, fid)
    if method == "DCdetector":
        return _uniform_components(UNIFORM_DCDET, fid)
    if method == "TSPulse_ZS":
        return _tspulse_components(fid)
    raise ValueError(f"unknown method: {method}")


def _init(job):
    global _JOB
    _JOB = job


def _eval_one(fname):
    method, weights = _JOB
    fid = fname.replace(".csv", "")
    comps = _load_components(method, fid)
    if comps is None:
        return ("SKIP_MISSING", fid, None)

    eu_w, mag_w, t2_w = weights
    n = min(len(comps["eu"]), len(comps["mag"]), len(comps["t2"]))
    score = eu_w * comps["eu"][:n] + mag_w * comps["mag"][:n] + t2_w * comps["t2"][:n]
    score = np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    try:
        df = pd.read_csv(Path(DATASET_DIR) / fname).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df["Label"].astype(int).to_numpy()
        train_index = int(fid.split("_")[-3])
        result = _eval_metrics_full_or_test(score, label, train_index, len(label), data)
        if result is None:
            return ("SKIP_LEN", fid, None)
        return ("OK", fid, {k: float(v) for k, v in result.items()})
    except Exception as e:
        return ("FAIL", fid, str(e)[:200])


def _aggregate_part(method, weights, file_names, out_dir, num_procs, reuse_parts):
    part_dir = out_dir / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_csv = part_dir / f"{_weight_tag(method, weights)}.csv"

    if reuse_parts and part_csv.exists():
        df = pd.read_csv(part_csv)
        return _summarize_existing_part(method, weights, df)

    print(f"\n[{method} weights={weights}] {len(file_names)} fids", flush=True)
    t0 = time.time()
    rows = []
    n_ok = n_skip = n_fail = 0
    with Pool(processes=num_procs, initializer=_init, initargs=((method, weights),)) as pool:
        for status, fid, payload in pool.imap_unordered(_eval_one, file_names):
            if status == "OK":
                row = {"fid": fid}
                row.update(payload)
                rows.append(row)
                n_ok += 1
            elif status.startswith("SKIP"):
                n_skip += 1
            else:
                n_fail += 1
                print(f"  FAIL {fid}: {payload}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(part_csv, index=False)
    print(
        f"[{method} weights={weights}] ok={n_ok} skip={n_skip} fail={n_fail} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )
    return _summarize(method, weights, df, reused=False)


def _summarize(method, weights, df, reused):
    eu_w, mag_w, t2_w = weights
    rec = {
        "method": method,
        "eu_weight": eu_w,
        "magG_weight": mag_w,
        "T2_weight": t2_w,
        "n_fids": int(len(df)),
        "reused_part": bool(reused),
    }
    for col in METRIC_COLUMNS:
        rec[col] = float(df[col].dropna().mean()) if col in df else float("nan")
    return rec


def _summarize_existing_part(method, weights, df):
    if "fid" in df.columns:
        return _summarize(method, weights, df, reused=True)

    if "n" in df.columns:
        rec = _summarize(method, weights, pd.DataFrame(), reused=True)
        rec["n_fids"] = int(df["n"].iloc[0])
        for col in METRIC_COLUMNS:
            rec[col] = float(df[col].iloc[0]) if col in df else float("nan")
        return rec

    raise ValueError(
        "existing part CSV is neither per-file nor summary format: "
        f"columns={list(df.columns)}"
    )


def _write_markdown(df, out_md):
    cols = [
        "method",
        "eu_weight",
        "magG_weight",
        "T2_weight",
        "n_fids",
        "AUC-PR",
        "VUS-PR",
        "AUC-ROC",
        "VUS-ROC",
        "Standard-F1",
        "Event-based-F1",
    ]
    show = df[cols].copy()
    for c in cols[5:]:
        show[c] = show[c].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "nan")
    for c in ("eu_weight", "magG_weight", "T2_weight"):
        show[c] = show[c].map(lambda x: f"{x:.2f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in show.iterrows()]
    text = [
        "# Weight Ablation Table\n\n",
        "Fusion sweep over representation, pointwise amplitude, and local mean-shift score components.\n\n",
        header,
        "\n",
        sep,
        "\n",
        "\n".join(body),
        "\n",
    ]
    Path(out_md).write_text("".join(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md")
    ap.add_argument("--num_procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug limit on file count")
    ap.add_argument("--recompute", action="store_true", help="ignore existing part CSVs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_names = pd.read_csv(EVA_CSV)["file_name"].tolist()
    if args.limit:
        file_names = file_names[: args.limit]

    rows = []
    for method in METHODS:
        for weights in WEIGHT_GRID:
            rows.append(
                _aggregate_part(
                    method=method,
                    weights=weights,
                    file_names=file_names,
                    out_dir=out_dir,
                    num_procs=args.num_procs,
                    reuse_parts=not args.recompute and not args.limit,
                )
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["method", "VUS-PR", "AUC-PR"], ascending=[True, False, False])
    df.to_csv(args.out_csv, index=False)
    if args.out_md:
        _write_markdown(df, args.out_md)

    print("\nWEIGHT ABLATION SUMMARY")
    print(df[["method", "eu_weight", "magG_weight", "T2_weight", "n_fids", "AUC-PR", "VUS-PR"]].to_string(index=False))


if __name__ == "__main__":
    main()
