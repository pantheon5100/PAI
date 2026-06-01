"""Finalize the comparison table once build_full_table.py has produced full_comparison_table.csv.

Reads full_comparison_table.csv and writes:
  - FULL_COMPARISON_TABLE.md (markdown table for paper)
  - Console-printed compact table

Computes:
  - Δ vs original for each "+ ours" row
  - Best per-metric column highlighted

Run after build_full_table.py finishes (CSV must exist).
"""
import argparse, csv
from pathlib import Path
import numpy as np
import pandas as pd

METRIC_DISPLAY = [
    ('VUS-PR',         'VUS-PR'),
    ('VUS-ROC',        'VUS-ROC'),
    ('Event-based-F1', 'Range-F1'),
    ('AUC-PR',         'AUC-PR'),
    ('AUC-ROC',        'AUC-ROC'),
    ('Standard-F1',    'Point-F1'),
]

PAIR_ORDER = [
    ('DCdetector_original',   'DCdetector_ours',     'DCdetector'),
    ('TS2Vec_original',       'TS2Vec_ours',         'TS2Vec'),
    ('TSPulse_ZS_original',   'TSPulse_ZS_ours',     'TSPulse_ZS'),
    ('TSPulse_FT_original',   None,                   'TSPulse_FT'),
    ('PaAno_original',        'PaAno_ours',          'PaAno'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out_md', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} method rows")
    methods = {r['method']: r for _, r in df.iterrows()}

    lines = []
    lines.append("# Full method comparison on TSB-AD-U-Eva (350 fids)\n")
    lines.append("\nPool means across the eligible fids (1 NEK fid skipped on score formats that fail get_metrics edge cases; PaAno scores contain 349/350 files).\n")

    # Markdown table
    lines.append("\n## Pool comparison table\n")
    hdr = "| Method | n | " + " | ".join(d for _, d in METRIC_DISPLAY) + " |"
    sep = "|---|---:|" + "---:|" * len(METRIC_DISPLAY)
    lines.append(hdr)
    lines.append(sep)
    for orig_key, ours_key, label in PAIR_ORDER:
        if orig_key in methods:
            row = methods[orig_key]
            cells = [f"{label} (original)", f"{int(row['n_fids'])}"]
            for col, _ in METRIC_DISPLAY:
                v = row.get(col, np.nan)
                cells.append(f"{v:.4f}" if pd.notna(v) else "n/a")
            lines.append("| " + " | ".join(cells) + " |")
        if ours_key and ours_key in methods:
            row = methods[ours_key]
            cells = [f"**{label} + ours**", f"{int(row['n_fids'])}"]
            for col, _ in METRIC_DISPLAY:
                v = row.get(col, np.nan)
                cells.append(f"**{v:.4f}**" if pd.notna(v) else "n/a")
            lines.append("| " + " | ".join(cells) + " |")

    # Δ table
    lines.append("\n## Δ ours − original\n")
    hdr = "| Method | " + " | ".join(d for _, d in METRIC_DISPLAY) + " |"
    sep = "|---|" + "---:|" * len(METRIC_DISPLAY)
    lines.append(hdr)
    lines.append(sep)
    for orig_key, ours_key, label in PAIR_ORDER:
        if not ours_key or orig_key not in methods or ours_key not in methods:
            continue
        orig = methods[orig_key]; ours = methods[ours_key]
        cells = [label]
        for col, _ in METRIC_DISPLAY:
            o = orig.get(col, np.nan); u = ours.get(col, np.nan)
            if pd.notna(o) and pd.notna(u):
                d = u - o
                cells.append(f"{d:+.4f}")
            else:
                cells.append("n/a")
        lines.append("| " + " | ".join(cells) + " |")

    Path(args.out_md).write_text("\n".join(lines))
    print(f"\nWrote {args.out_md}")

    # Console compact view
    print("\n" + "=" * 100)
    print("FULL TABLE (compact)")
    print("=" * 100)
    metric_cols = ['VUS-PR', 'VUS-ROC', 'Event-based-F1', 'AUC-PR', 'AUC-ROC', 'Standard-F1']
    print(f"  {'method':<26s} | {'n':>4s} | " + " ".join(f"{m[:8]:>8s}" for m in metric_cols))
    print("  " + "-" * 90)
    for orig_key, ours_key, label in PAIR_ORDER:
        for k, suffix in [(orig_key, '(orig)'), (ours_key, '(+ours)')]:
            if k and k in methods:
                row = methods[k]
                line = f"  {label + ' ' + suffix:<26s} | {int(row['n_fids']):>4d} | "
                line += " ".join(f"{row.get(m, float('nan')):>8.4f}" if pd.notna(row.get(m, np.nan)) else "     n/a" for m in metric_cols)
                print(line)


if __name__ == '__main__':
    main()
