"""Run vendored PaAno on TSB-AD-U Eva and export table-compatible scores.

PaAno is executed in its own process with `third_party/PaAno` as the working
directory to avoid module-name collisions with TS2Vec/DCdetector (`model`,
`utils`). The exported layout matches `aggregators/build_full_table.py`:

- original: `<score_dir>/<fid>/cos_score.npy`
- pai:      `<score_dir>/<fid>/eucl_score.npy`
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pai_paths import tsbad_dataset_dir, tsbad_eva_csv


VARIANT_CONFIG = {
    "original": {
        "track": "paano_native",
        "export_name": "cos_score.npy",
    },
    "pai": {
        "track": "strict_external_zscore",
        "export_name": "eucl_score.npy",
    },
}


def default_paano_repo() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "PaAno"


def read_file_names(file_list: Path, start: int, end: int) -> list[str]:
    df = pd.read_csv(file_list)
    if "file_name" not in df.columns:
        raise ValueError(f"file_list must contain a file_name column: {file_list}")
    files = df["file_name"].astype(str).tolist()
    stop = len(files) if end == -1 else end
    return files[start:stop]


def write_selected_file(path: Path, file_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(file_names) + "\n", encoding="utf-8")


def run_paano(args: argparse.Namespace, file_names: list[str], config: dict[str, str]) -> Path:
    artifact_root = args.artifact_root or (Path(args.score_dir) / "_paano_runs" / args.variant)
    artifact_root = artifact_root.expanduser().resolve()
    selected_path = artifact_root / "selected_files.txt"
    write_selected_file(selected_path, file_names)

    runner = args.paano_repo / "phase2_score_runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"missing PaAno runner: {runner}")

    cmd = [
        sys.executable,
        str(runner),
        "--method",
        "paano",
        "--track",
        config["track"],
        "--data_dir",
        str(args.dataset_dir),
        "--file_list",
        str(selected_path),
        "--artifact_root",
        str(artifact_root),
        "--patch_size",
        str(args.patch_size),
        "--seed",
        str(args.seed),
        "--num_iters",
        str(args.num_iters),
        "--batch_size",
        str(args.batch_size),
        "--cpu_threads",
        str(args.cpu_threads),
        "--device",
        args.device,
        "--metric_version",
        args.metric_version,
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.paano_repo)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = str(args.cpu_threads)

    subprocess.run(cmd, cwd=str(args.paano_repo), env=env, check=True)
    return artifact_root


def export_scores(artifact_root: Path, score_dir: Path, file_names: list[str], config: dict[str, str]) -> int:
    run_root = artifact_root / config["track"] / "paano"
    export_name = config["export_name"]
    exported = 0
    for file_name in file_names:
        fid = Path(file_name).stem
        point_csv = run_root / "files" / fid / "point_scores.csv"
        if not point_csv.is_file():
            raise FileNotFoundError(f"PaAno score not found: {point_csv}")
        scores = pd.read_csv(point_csv)["Anomaly scores"].to_numpy(dtype=np.float32)
        out_dir = score_dir / fid
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / export_name, scores)
        exported += 1
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaAno and export PAI table scores.")
    parser.add_argument("--file_list", type=Path, default=tsbad_eva_csv(), required=False)
    parser.add_argument("--dataset_dir", type=Path, default=tsbad_dataset_dir(), required=False)
    parser.add_argument("--score_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANT_CONFIG), required=True)
    parser.add_argument("--paano_repo", type=Path, default=Path(os.environ.get("PAIAD_PAANO_REPO", default_paano_repo())))
    parser.add_argument("--artifact_root", type=Path, default=None)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--metric_version", default="opt_mem", choices=["opt", "opt_mem"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--export_only", action="store_true")
    args = parser.parse_args()

    args.file_list = args.file_list.expanduser().resolve()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.score_dir = args.score_dir.expanduser().resolve()
    args.paano_repo = args.paano_repo.expanduser().resolve()

    config = VARIANT_CONFIG[args.variant]
    file_names = read_file_names(args.file_list, args.start, args.end)
    if not file_names:
        raise SystemExit("selected file slice is empty")

    artifact_root = args.artifact_root
    if artifact_root is None:
        artifact_root = args.score_dir / "_paano_runs" / args.variant
    artifact_root = artifact_root.expanduser().resolve()

    if not args.export_only:
        artifact_root = run_paano(args, file_names, config)

    exported = export_scores(artifact_root, args.score_dir, file_names, config)
    print(
        f"[paano {args.variant}] exported={exported} score_dir={args.score_dir} "
        f"artifact_root={artifact_root}"
    )


if __name__ == "__main__":
    main()
