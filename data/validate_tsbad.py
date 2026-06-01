"""Validate the TSB-AD-U Eva file list and raw CSV directory.

This is an optional sanity check. By default the release looks for data under
the repository's local `data/` directory.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv


def env_path(name: str, default: str | None = None) -> Path | None:
    value = os.environ.get(name, default)
    return Path(value).expanduser() if value else None


def parse_train_index(file_name: str) -> int:
    stem = Path(file_name).stem
    try:
        return int(stem.split("_")[-3])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse train index from {file_name!r}") from exc


def validate_one(file_name: str, dataset_dir: Path) -> dict:
    src = dataset_dir / file_name
    if not src.exists():
        raise FileNotFoundError(f"missing raw CSV: {src}")

    df_raw = pd.read_csv(src)
    if "Label" not in df_raw.columns:
        raise ValueError(f"{file_name}: missing required Label column")
    if len(df_raw.columns) < 2:
        raise ValueError(f"{file_name}: expected at least one value column plus Label")

    df = df_raw.dropna()
    train_index = parse_train_index(file_name)
    if train_index <= 0 or train_index >= len(df):
        raise ValueError(
            f"{file_name}: train_index={train_index} outside valid range 1..{len(df) - 1}"
        )

    label = df["Label"].astype(int)
    return {
        "file_name": file_name,
        "path": str(src.resolve()),
        "n_rows_raw": int(len(df_raw)),
        "n_rows": int(len(df)),
        "n_features": int(len(df.columns) - 1),
        "train_index": int(train_index),
        "test_length": int(len(df) - train_index),
        "n_anomaly_points": int(label.sum()),
        "label_values": ",".join(str(v) for v in sorted(label.unique().tolist())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TSB-AD-U Eva inputs.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=tsbad_dataset_dir(),
        help="Directory containing raw TSB-AD-U CSV files.",
    )
    parser.add_argument(
        "--file_list",
        type=Path,
        default=tsbad_eva_csv(),
        help="CSV with a file_name column for the Eva split.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=env_path("PAIAD_TSB_U_MANIFEST_CSV"),
        help="Optional output manifest CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional first-N limit for quick validation. 0 means all files.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    file_list = args.file_list.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset_dir does not exist: {dataset_dir}")
    if not file_list.is_file():
        raise SystemExit(f"file_list does not exist: {file_list}")

    files_df = pd.read_csv(file_list)
    if "file_name" not in files_df.columns:
        raise SystemExit(f"file_list must contain a file_name column: {file_list}")

    file_names = files_df["file_name"].astype(str).tolist()
    if args.limit:
        file_names = file_names[: args.limit]

    rows = [validate_one(file_name, dataset_dir) for file_name in file_names]
    manifest = pd.DataFrame(rows)

    if args.out_csv is not None:
        out_csv = args.out_csv.expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_csv, index=False)
        print(f"manifest={out_csv}")

    print(f"Validated {len(rows)} TSB-AD-U Eva files")
    print(f"file_list={file_list}")
    print(f"dataset_dir={dataset_dir}")
    print(f"features_minmax={int(manifest.n_features.min())},{int(manifest.n_features.max())}")
    print(f"test_len_minmax={int(manifest.test_length.min())},{int(manifest.test_length.max())}")
    print(f"anomaly_files={int((manifest.n_anomaly_points > 0).sum())}")


if __name__ == "__main__":
    main()
