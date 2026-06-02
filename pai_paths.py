"""Path bootstrap for the self-contained PAI-AnomalyDetection workflow."""
from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
THIRD_PARTY = REPO_ROOT / "third_party"
DATA_ROOT = REPO_ROOT / "data"


def third_party_path(name: str) -> str:
    return str(THIRD_PARTY / name)


def add_third_party_paths() -> None:
    """Prefer vendored baseline implementations, while allowing env overrides."""
    paths = [
        os.environ.get("PAIAD_TSB_AD_REPO", third_party_path("TSB-AD")),
        os.environ.get("PAIAD_TS2VEC_REPO", third_party_path("ts2vec")),
        os.environ.get("PAIAD_DCDETECTOR_REPO", third_party_path("KDD2023-DCdetector")),
    ]
    for path in reversed(paths):
        if path and path not in sys.path:
            sys.path.insert(0, path)


add_third_party_paths()


def first_existing_path(candidates: list[Path], fallback: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return fallback


def tsbad_dataset_dir() -> Path:
    env = os.environ.get("PAIAD_TSB_U_DATASET_DIR")
    if env:
        return Path(env).expanduser()
    return first_existing_path(
        [
            DATA_ROOT / "TSB-AD-U" / "TSB-AD-U",
            DATA_ROOT / "TSB-AD" / "Datasets" / "TSB-AD-U" / "TSB-AD-U",
            DATA_ROOT / "TSB-AD" / "TSB-AD-U" / "TSB-AD-U",
        ],
        DATA_ROOT / "TSB-AD-U" / "TSB-AD-U",
    )


def tsbad_eva_csv() -> Path:
    env = os.environ.get("PAIAD_TSB_U_EVA_CSV")
    if env:
        return Path(env).expanduser()
    return first_existing_path(
        [
            DATA_ROOT / "TSB-AD" / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv",
            DATA_ROOT / "TSB-AD" / "File_List" / "TSB-AD-U-Eva.csv",
            DATA_ROOT / "File_List" / "TSB-AD-U-Eva.csv",
            DATA_ROOT / "TSB-AD-U-Eva.csv",
        ],
        DATA_ROOT / "TSB-AD" / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv",
    )
