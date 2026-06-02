import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


METRIC_COLUMNS = [
    "AUC-ROC",
    "AUC-PR",
    "VUS-PR",
    "VUS-ROC",
    "Standard-F1",
    "R-based-F1",
]
MERGED_COLUMNS = ["file", "method", "track", "comparison_group", *METRIC_COLUMNS]
SCORE_MANIFEST_COLUMNS = [
    "file",
    "method",
    "track",
    "status",
    "num_points",
    "sliding_window",
    "window_mode",
    "source_file",
    "point_score_path",
    "metadata_path",
]
METHOD_COMPARISON_GROUP = {
    "paano": "strict_matched",
    "paano_aug_amp": "strict_matched",
    "paano_aug_multi": "strict_matched",
    "paano_aug_mask": "strict_matched",
    "paano_wider_radius": "strict_matched",
    "paano_shape_pos": "strict_matched",
    "paano_time_warp_neg": "strict_matched",
    "rescnn_cnrv": "strict_matched",
    "sbd_knn": "strict_matched",
    "sbd_knn_revin": "strict_matched",
    "kshape_proto": "strict_matched",
    "kshape_proto_revin": "strict_matched",
    "matrix_profile": "auxiliary",
}
WINDOW_MODE_NATIVE = "native_full_data"
WINDOW_MODE_TRAIN_ZSCORE = "train_stat_zscore_full_data"


def is_paano_method(method: str) -> bool:
    return method == "paano" or method.startswith("paano_")


def read_file_list(file_list_arg: str) -> list[str]:
    candidate_path = Path(file_list_arg)
    if candidate_path.exists():
        raw_text = candidate_path.read_text()
    else:
        raw_text = file_list_arg
    items = [token.strip() for token in re.split(r"[\n,]+", raw_text) if token.strip()]
    if not items:
        raise ValueError("No input files were parsed")
    return items


def parse_train_end(file_name: str) -> int:
    match = re.search(r"_tr_(\d+)_", file_name)
    if not match:
        raise ValueError(f"Could not parse train split from file name: {file_name}")
    return int(match.group(1))


def load_series(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Input file is empty: {file_path}")
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1].to_numpy(dtype=np.float32)
    if features.shape[1] == 1:
        data = features.iloc[:, 0].to_numpy(dtype=np.float32)
    else:
        data = features.to_numpy(dtype=np.float32)
    return data, labels


def load_split(
    file_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data, labels = load_series(file_path)
    train_end = parse_train_end(file_path.name)
    train_data = np.asarray(data[:train_end], dtype=np.float32)
    test_data = np.asarray(data[train_end:], dtype=np.float32)
    train_labels = np.asarray(labels[:train_end], dtype=np.float32)
    test_labels = np.asarray(labels[train_end:], dtype=np.float32)
    full_data = np.concatenate([train_data, test_data], axis=0)
    full_labels = np.concatenate([train_labels, test_labels], axis=0)
    return train_data, train_labels, test_data, test_labels, full_data, full_labels


def apply_train_zscore(train_data: np.ndarray, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.std(train_data, axis=0, keepdims=True).astype(np.float32)
    train_std = np.where(train_std == 0.0, 1e-8, train_std)
    train_z = (train_data - train_mean) / train_std
    test_z = (test_data - train_mean) / train_std
    return train_z.astype(np.float32), test_z.astype(np.float32)


def _acf_compatible(data: np.ndarray, nlags: int) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(nlags + 1, dtype=np.float64)
    values = values - np.mean(values)
    fft_size = 1 << (2 * len(values) - 1).bit_length()
    spectrum = np.fft.fft(values, n=fft_size)
    acov = np.fft.ifft(spectrum * np.conjugate(spectrum)).real[: nlags + 1]
    acov /= len(values)
    if acov[0] == 0.0:
        result = np.zeros(nlags + 1, dtype=np.float64)
        result[0] = 1.0
        return result
    return acov / acov[0]


def find_length_rank_compatible(data, rank: int = 1) -> int:
    values = np.asarray(data).squeeze()
    if len(values.shape) > 1:
        return 100
    if rank == 0:
        return 1
    values = values[: min(20000, len(values))]
    base = 3
    auto_corr = _acf_compatible(values, nlags=400)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]
        max_local_max = sorted_local_max[0]
        if rank == 1:
            max_local_max = sorted_local_max[0]
        if rank == 2:
            for idx in sorted_local_max[1:]:
                if idx > sorted_local_max[0]:
                    max_local_max = idx
                    break
        if rank == 3:
            id_tmp = sorted_local_max[0]
            for idx in sorted_local_max[1:]:
                if idx > sorted_local_max[0]:
                    id_tmp = idx
                    break
            for idx in sorted_local_max[id_tmp:]:
                if idx > sorted_local_max[id_tmp]:
                    max_local_max = idx
                    break
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return int(local_max[max_local_max] + base)
    except Exception:
        return 125


def compute_sliding_window(full_data: np.ndarray) -> int:
    if full_data.ndim == 1:
        sliding_input = full_data.reshape(-1, 1)
    else:
        sliding_input = full_data[:, 0].reshape(-1, 1)
    return int(find_length_rank_compatible(sliding_input, rank=1))


def metric_window_mode(method: str, track: str) -> str:
    if is_paano_method(method) and track == "paano_native":
        return WINDOW_MODE_NATIVE
    return WINDOW_MODE_TRAIN_ZSCORE


def compute_metric_context(file_path: Path, method: str, track: str) -> tuple[int, np.ndarray, str]:
    train_data, _, test_data, _, full_data, full_labels = load_split(file_path)
    window_mode = metric_window_mode(method, track)
    if window_mode == WINDOW_MODE_TRAIN_ZSCORE:
        train_data, test_data = apply_train_zscore(train_data, test_data)
        full_data = np.concatenate([train_data, test_data], axis=0)
    sliding_window = compute_sliding_window(full_data)
    return sliding_window, full_labels, window_mode


def make_file_output_dir(run_root: Path, file_name: str) -> Path:
    file_dir = run_root / "files" / Path(file_name).stem
    file_dir.mkdir(parents=True, exist_ok=True)
    return file_dir


def point_score_path(run_root: Path, file_name: str) -> Path:
    return make_file_output_dir(run_root, file_name) / "point_scores.csv"


def summary_path(run_root: Path, file_name: str) -> Path:
    return make_file_output_dir(run_root, file_name) / "summary.csv"


def metadata_path(run_root: Path, file_name: str) -> Path:
    return make_file_output_dir(run_root, file_name) / "metadata.json"


def score_manifest_path(run_root: Path) -> Path:
    return run_root / "score_manifest.csv"


def build_summary_row(file_name: str, method: str, track: str, metrics: dict[str, float]) -> dict[str, float | str]:
    row = {
        "file": file_name,
        "method": method,
        "track": track,
        "comparison_group": METHOD_COMPARISON_GROUP[method],
    }
    for metric_name in METRIC_COLUMNS:
        row[metric_name] = float(metrics[metric_name])
    return row


def write_method_summary(run_root: Path, summary_rows: list[dict[str, float | str]]) -> None:
    pd.DataFrame(summary_rows, columns=MERGED_COLUMNS).to_csv(run_root / "summary.csv", index=False)


def build_metric_metadata(
    file_name: str,
    method: str,
    track: str,
    source_file: Path,
    sliding_window: int,
    num_points: int,
    window_mode: str,
) -> dict[str, object]:
    return {
        "file": file_name,
        "method": method,
        "track": track,
        "sliding_window": int(sliding_window),
        "window_mode": window_mode,
        "num_points": int(num_points),
        "source_file": str(source_file.resolve()),
    }


def save_point_scores_csv(path: Path, labels: np.ndarray, scores: np.ndarray) -> None:
    pd.DataFrame(
        {
            "True Labels": np.asarray(labels, dtype=np.float32),
            "Anomaly scores": np.asarray(scores, dtype=np.float32),
        }
    ).to_csv(path, index=False)


def write_metadata_json(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, indent=2))


def build_score_manifest_rows(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for meta_path in sorted(run_root.glob("files/*/metadata.json")):
        metadata = json.loads(meta_path.read_text())
        file_name = str(metadata.get("file") or f"{meta_path.parent.name}.csv")
        score_path = point_score_path(run_root, file_name)
        if not score_path.exists():
            continue
        rows.append(
            {
                "file": file_name,
                "method": metadata.get("method"),
                "track": metadata.get("track"),
                "status": "completed",
                "num_points": metadata.get("num_points"),
                "sliding_window": metadata.get("sliding_window"),
                "window_mode": metadata.get("window_mode"),
                "source_file": metadata.get("source_file"),
                "point_score_path": str(score_path.resolve()),
                "metadata_path": str(meta_path.resolve()),
            }
        )
    return rows


def write_score_manifest(run_root: Path) -> list[dict[str, object]]:
    rows = build_score_manifest_rows(run_root)
    pd.DataFrame(rows, columns=SCORE_MANIFEST_COLUMNS).to_csv(score_manifest_path(run_root), index=False)
    return rows


def _iter_run_roots(artifact_root: Path) -> list[tuple[str, str, Path]]:
    run_roots: list[tuple[str, str, Path]] = []
    if not artifact_root.exists():
        return run_roots
    for track_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        track = track_dir.name
        for method_dir in sorted(path for path in track_dir.iterdir() if path.is_dir()):
            run_roots.append((track, method_dir.name, method_dir))
    return run_roots


def mirror_score_outputs(artifact_root: Path, mirror_root: Path) -> None:
    mirror_root.mkdir(parents=True, exist_ok=True)
    for name in ["score_coverage.json", "selected_file_list.txt"]:
        source = artifact_root / name
        if source.exists():
            (mirror_root / name).write_bytes(source.read_bytes())
    for track, method, run_root in _iter_run_roots(artifact_root):
        manifest = score_manifest_path(run_root)
        if not manifest.exists():
            continue
        target_dir = mirror_root / track / method
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "score_manifest.csv").write_bytes(manifest.read_bytes())


def write_score_coverage(
    artifact_root: Path,
    mirror_root: Path | None,
    selected_files: list[str],
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    methods_present: set[str] = set()
    tracks_present: set[str] = set()
    total_completed = 0

    for track, method, run_root in _iter_run_roots(artifact_root):
        completed_files = sum(1 for _ in run_root.glob("files/*/point_scores.csv"))
        manifest_rows = 0
        manifest = score_manifest_path(run_root)
        if manifest.exists():
            try:
                manifest_rows = max(0, sum(1 for _ in manifest.open()) - 1)
            except Exception:
                manifest_rows = -1
        runs.append(
            {
                "track": track,
                "method": method,
                "completed_files": completed_files,
                "expected_files": len(selected_files),
                "manifest_rows": manifest_rows,
            }
        )
        total_completed += completed_files
        methods_present.add(method)
        tracks_present.add(track)

    payload = {
        "num_files": len(selected_files),
        "total_completed_score_files": total_completed,
        "methods_present": sorted(methods_present),
        "tracks_present": sorted(tracks_present),
        "all_runs_complete": bool(runs) and all(
            int(run["completed_files"]) >= len(selected_files) for run in runs
        ),
        "runs": runs,
    }
    (artifact_root / "score_coverage.json").write_text(json.dumps(payload, indent=2))
    (artifact_root / "selected_file_list.txt").write_text("\n".join(selected_files) + "\n")

    if mirror_root is not None:
        mirror_root.mkdir(parents=True, exist_ok=True)
        mirror_score_outputs(artifact_root, mirror_root)

    return payload


def update_merged_outputs(artifact_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    for path in sorted(artifact_root.glob("*/*/files/*/summary.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            rows.append(df.iloc[0][MERGED_COLUMNS].to_dict())
    merged_df = pd.DataFrame(rows, columns=MERGED_COLUMNS)
    if not merged_df.empty:
        merged_df = merged_df.sort_values(["file", "method", "track"]).reset_index(drop=True)
    merged_df.to_csv(artifact_root / "merged_comparison.csv", index=False)

    mean_df = pd.DataFrame(columns=["method", "track", "comparison_group", *METRIC_COLUMNS])
    if not merged_df.empty:
        mean_df = (
            merged_df.groupby(["method", "track", "comparison_group"], as_index=False)[METRIC_COLUMNS]
            .mean()
            .sort_values(["comparison_group", "method", "track"])
            .reset_index(drop=True)
        )
    mean_df.to_csv(artifact_root / "mean_by_method_track.csv", index=False)
    return merged_df, mean_df


def mirror_small_outputs(artifact_root: Path, mirror_root: Path) -> None:
    mirror_root.mkdir(parents=True, exist_ok=True)
    for name in [
        "merged_comparison.csv",
        "mean_by_method_track.csv",
        "source_summary.csv",
        "category_summary.csv",
        "kshape_proto_metadata.csv",
        "coverage_summary.json",
        "selected_file_list.txt",
        "run_manifest.json",
    ]:
        source = artifact_root / name
        if source.exists():
            (mirror_root / name).write_bytes(source.read_bytes())


def write_global_summaries(
    artifact_root: Path,
    mirror_root: Path | None,
    selected_files: list[str],
    fallback_reason: str | None = None,
) -> None:
    merged_path = artifact_root / "merged_comparison.csv"
    if not merged_path.exists():
        return
    merged = pd.read_csv(merged_path)
    if merged.empty:
        return

    source_summary = (
        merged.assign(source=merged["file"].str.split("_").str[1], category=merged["file"].str.split("_").str[4])
        .groupby(["source", "method", "track", "comparison_group"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(["source", "comparison_group", "method", "track"])
        .reset_index(drop=True)
    )
    category_summary = (
        merged.assign(source=merged["file"].str.split("_").str[1], category=merged["file"].str.split("_").str[4])
        .groupby(["category", "method", "track", "comparison_group"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(["category", "comparison_group", "method", "track"])
        .reset_index(drop=True)
    )
    source_summary.to_csv(artifact_root / "source_summary.csv", index=False)
    category_summary.to_csv(artifact_root / "category_summary.csv", index=False)

    metadata_rows = []
    for path in sorted(artifact_root.glob("*/*/files/*/metadata.json")):
        metadata_rows.append(json.loads(path.read_text()))
    pd.DataFrame(metadata_rows).to_csv(artifact_root / "kshape_proto_metadata.csv", index=False)

    coverage_summary = {
        "num_files": len(selected_files),
        "methods_present": sorted(merged["method"].unique().tolist()),
        "tracks_present": sorted(merged["track"].unique().tolist()),
        "comparison_groups_present": sorted(merged["comparison_group"].unique().tolist()),
        "fallback_reason": fallback_reason,
    }
    (artifact_root / "coverage_summary.json").write_text(json.dumps(coverage_summary, indent=2))

    if mirror_root is not None:
        mirror_root.mkdir(parents=True, exist_ok=True)
        (mirror_root / "selected_file_list.txt").write_text("\n".join(selected_files) + "\n")
        (mirror_root / "run_manifest.json").write_text(json.dumps(coverage_summary, indent=2))
        mirror_small_outputs(artifact_root, mirror_root)
