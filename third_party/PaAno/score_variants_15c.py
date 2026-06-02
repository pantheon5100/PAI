from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import MiniBatchKMeans

from utils.evaluation import distribute_patch_scores_to_points

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


VARIANTS = {
    "c1_euclidean",
    "c2_e2a_euclidean",
    "c2_cosine_orig",
    "c4_euclidean_clean",
    "c5_euclidean_median",
    "c6_point_euclidean",
    "c7_point_euclid_s32",
    "c8_point_euclid_s128",
    "c9_point_lof_mag",
    "c10_temporal_lof",
}

PATCH_LEVEL_VARIANTS = {
    "c1_euclidean",
    "c2_e2a_euclidean",
    "c2_cosine_orig",
    "c4_euclidean_clean",
    "c5_euclidean_median",
}

POINT_LEVEL_VARIANTS = {
    "c6_point_euclidean",
    "c7_point_euclid_s32",
    "c8_point_euclid_s128",
    "c9_point_lof_mag",
    "c10_temporal_lof",
}


class VariantTimeoutError(TimeoutError):
    pass


@dataclass
class ScoreOutput:
    point_scores: np.ndarray
    native_scores: np.ndarray
    native_level: str  # "patch" or "point"
    info: dict[str, Any]


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise VariantTimeoutError("variant exceeded per-file timeout")


def _as_tensor(x: Any) -> torch.Tensor:
    if x is None:
        raise ValueError("required tensor input is missing")
    if torch.is_tensor(x):
        return x.detach().cpu().float()
    return torch.as_tensor(x, dtype=torch.float32).cpu()


def _as_1d_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr


def _time_length(x: Any) -> int:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError("Expected 1D/2D time series, got scalar")
    return int(arr.shape[0])


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float32)
    return ranks / float(x.size)


def _baseline_bank_size(num_samples: int) -> int:
    if num_samples <= 1:
        return num_samples
    k = int(round(0.1 * num_samples))
    min_cores_eff = min(500, max(1, num_samples - 1))
    return max(min_cores_eff, min(k, num_samples - 1))


def _kmeans_centers(features: torch.Tensor, random_state: int, bank_size: int | None = None) -> torch.Tensor:
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


def _euclidean_topk(
    query: torch.Tensor,
    bank: torch.Tensor,
    top_k: int,
    batch_size: int,
    device: torch.device,
    agg: str = "mean",
) -> np.ndarray:
    q = _as_tensor(query)
    b = _as_tensor(bank)
    if int(b.shape[0]) == 0:
        raise ValueError("Empty bank")

    k = max(1, min(int(top_k), int(b.shape[0])))
    b = b.to(device=device, dtype=torch.float32)

    out: list[torch.Tensor] = []
    step = max(1, int(batch_size))
    for start in range(0, int(q.shape[0]), step):
        end = min(int(q.shape[0]), start + step)
        qb = q[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        d = torch.cdist(qb, b, p=2)
        top = d.topk(k=k, dim=1, largest=False).values
        if agg == "mean":
            score = top.mean(dim=1)
        elif agg == "median":
            score = top.median(dim=1).values
        else:
            raise ValueError(f"Unsupported agg={agg}")
        out.append(score.cpu())
    scores = torch.cat(out, dim=0).numpy().astype(np.float32, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite Euclidean scores")
    return scores


def _cosine_topk(
    query: torch.Tensor,
    bank: torch.Tensor,
    top_k: int,
    batch_size: int,
    device: torch.device,
    agg: str = "mean",
) -> np.ndarray:
    """Cosine-distance top-k aggregator. Both inputs are L2-normalized before
    the inner product so this is true cosine — independent of the caller's
    normalization. Mirrors the original PaAno (Xu et al. 2024) scoring.
    """
    q = _as_tensor(query)
    b = _as_tensor(bank)
    if int(b.shape[0]) == 0:
        raise ValueError("Empty bank")

    q = F.normalize(q, dim=1, eps=1e-12)
    b = F.normalize(b, dim=1, eps=1e-12)

    k = max(1, min(int(top_k), int(b.shape[0])))
    b = b.to(device=device, dtype=torch.float32)

    out: list[torch.Tensor] = []
    step = max(1, int(batch_size))
    for start in range(0, int(q.shape[0]), step):
        end = min(int(q.shape[0]), start + step)
        qb = q[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        sims = qb @ b.T                       # cosine similarity (unit vectors)
        dists = 1.0 - sims                    # cosine distance
        top = dists.topk(k=k, dim=1, largest=False).values
        if agg == "mean":
            score = top.mean(dim=1)
        elif agg == "median":
            score = top.median(dim=1).values
        else:
            raise ValueError(f"Unsupported agg={agg}")
        out.append(score.cpu())
    scores = torch.cat(out, dim=0).numpy().astype(np.float32, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite cosine scores")
    return scores


def _drop_self_neighbors(dists: np.ndarray, idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    t = int(dists.shape[0])
    out_d = np.empty((t, k), dtype=np.float32)
    out_i = np.empty((t, k), dtype=np.int64)

    for r in range(t):
        mask = idx[r] != r
        rd = dists[r][mask]
        ri = idx[r][mask]
        if ri.size == 0:
            out_d[r, :] = 1.0
            out_i[r, :] = r
            continue
        take = min(k, ri.size)
        out_d[r, :take] = rd[:take]
        out_i[r, :take] = ri[:take]
        if take < k:
            out_d[r, take:] = out_d[r, take - 1]
            out_i[r, take:] = out_i[r, take - 1]
    return out_d, out_i


def _lof_from_knn(dists: np.ndarray, idx: np.ndarray) -> np.ndarray:
    d = np.asarray(dists, dtype=np.float32)
    i = np.asarray(idx, dtype=np.int64)
    if d.ndim != 2 or i.ndim != 2:
        raise ValueError("dists/idx must be 2D")
    k = int(d.shape[1])
    if k <= 0:
        return np.ones(int(d.shape[0]), dtype=np.float32)

    k_dist = d[:, -1]
    reach = np.maximum(k_dist[i], d)
    lrd = float(k) / np.clip(reach.sum(axis=1), 1e-10, None)
    lof = np.mean(lrd[i], axis=1) / np.clip(lrd, 1e-10, None)
    lof = lof.astype(np.float32, copy=False)
    if not np.isfinite(lof).all():
        raise ValueError("LOF produced non-finite values")
    return lof


def _self_lof_prune_l2(
    train_feats: torch.Tensor,
    lof_k: int,
    prune_pct: float,
    deadline: float | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if faiss is None:
        raise ImportError("faiss is required for c4_euclidean_clean")
    _check_deadline(deadline)

    z = _as_tensor(train_feats)
    n, d = int(z.shape[0]), int(z.shape[1])
    if n <= 2:
        return z, {"pruned_count": 0, "pruned_fraction": 0.0, "train_count": n}

    k_eff = min(max(1, int(lof_k)), n - 1)
    x = np.ascontiguousarray(z.numpy().astype(np.float32, copy=False))

    index = faiss.IndexFlatL2(d)
    index.add(x)
    d_full, i_full = index.search(x, min(n, k_eff + 1))
    d_knn, i_knn = _drop_self_neighbors(d_full, i_full, k=k_eff)

    lof = _lof_from_knn(d_knn, i_knn)
    threshold = float(np.quantile(lof, 1.0 - float(prune_pct)))
    keep = lof <= threshold
    keep_count = int(np.sum(keep))

    if keep_count < 2:
        keep_idx = np.argsort(lof)[:2]
        clean = z[torch.as_tensor(keep_idx, dtype=torch.long)]
    else:
        clean = z[torch.as_tensor(keep, dtype=torch.bool)]

    pruned = int(n - int(clean.shape[0]))
    info = {
        "pruned_count": pruned,
        "pruned_fraction": float(pruned / max(1, n)),
        "train_count": int(n),
    }
    return clean, info


def _global_lof_cosine(test_point_embeds: torch.Tensor, k: int, deadline: float | None) -> np.ndarray:
    if faiss is None:
        raise ImportError("faiss is required for c9_point_lof_mag")
    _check_deadline(deadline)

    z = F.normalize(_as_tensor(test_point_embeds), dim=1, eps=1e-12).numpy().astype(np.float32, copy=False)
    t, d = int(z.shape[0]), int(z.shape[1])
    if t <= 1:
        return np.ones(t, dtype=np.float32)

    k_eff = min(max(1, int(k)), t - 1)
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(z))
    sims, idx = index.search(z, min(t, k_eff + 1))

    dists = 1.0 - sims
    d_knn, i_knn = _drop_self_neighbors(dists, idx, k=k_eff)
    return _lof_from_knn(d_knn, i_knn)


def _magnitude_zscore(train_raw: Any, test_raw: Any) -> np.ndarray:
    train = np.asarray(train_raw, dtype=np.float32)
    test = np.asarray(test_raw, dtype=np.float32)
    if train.ndim == 2 and test.ndim == 2:
        mu = np.mean(np.abs(train), axis=0, keepdims=True).astype(np.float32)
        sigma = np.std(np.abs(train), axis=0, keepdims=True).astype(np.float32)
        sigma = np.maximum(sigma, 1e-6)
        out = ((np.abs(test) - mu) / sigma).max(axis=1).astype(np.float32, copy=False)
    else:
        train_1d = np.abs(train.reshape(-1))
        test_1d = np.abs(test.reshape(-1))
        mu = float(train_1d.mean()) if train_1d.size else 0.0
        sigma = float(train_1d.std()) if train_1d.size else 1.0
        sigma = max(sigma, 1e-6)
        out = ((test_1d - mu) / sigma).astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise ValueError("Magnitude z-score produced non-finite values")
    return out


def _temporal_lof_faiss_chunked(
    test_point_embeds: torch.Tensor,
    k: int,
    window: int,
    chunk_size: int,
    deadline: float | None,
) -> np.ndarray:
    if faiss is None:
        raise ImportError("faiss is required for c10_temporal_lof")

    z = F.normalize(_as_tensor(test_point_embeds), dim=1, eps=1e-12).numpy().astype(np.float32, copy=False)
    t, d = int(z.shape[0]), int(z.shape[1])
    scores = np.ones(t, dtype=np.float32)

    if t <= 2:
        return scores

    k = max(1, int(k))
    window = max(1, int(window))
    chunk_size = max(128, int(chunk_size))

    for chunk_start in range(0, t, chunk_size):
        _check_deadline(deadline)
        chunk_end = min(t, chunk_start + chunk_size)

        lo = max(0, chunk_start - window)
        hi = min(t, chunk_end + window)
        local = np.ascontiguousarray(z[lo:hi])
        n_local = int(local.shape[0])
        if n_local <= 2:
            continue

        k_eff_global = min(k, n_local - 1)
        search_k = min(n_local, max(k_eff_global + 1, min(n_local, 2 * window + 32)))

        index = faiss.IndexFlatIP(d)
        index.add(local)
        sims, idx = index.search(local, search_k)
        dists = 1.0 - sims

        nbr_idx: list[np.ndarray] = []
        nbr_dist: list[np.ndarray] = []
        k_eff = np.zeros(n_local, dtype=np.int32)

        for i in range(n_local):
            if (i % 256) == 0:
                _check_deadline(deadline)

            mask = (idx[i] != i) & (np.abs(idx[i] - i) <= window)
            cand_i = idx[i][mask]
            cand_d = dists[i][mask]

            if cand_i.size < k:
                wlo = max(0, i - window)
                whi = min(n_local, i + window + 1)
                cand = np.arange(wlo, whi, dtype=np.int64)
                cand = cand[cand != i]
                if cand.size == 0:
                    nbr_idx.append(np.empty((0,), dtype=np.int64))
                    nbr_dist.append(np.empty((0,), dtype=np.float32))
                    continue
                sims_local = (local[cand] @ local[i]).astype(np.float32, copy=False)
                order = np.argsort(-sims_local)
                cand_i = cand[order]
                cand_d = (1.0 - sims_local[order]).astype(np.float32, copy=False)

            take = min(k, int(cand_i.size))
            nbr_idx.append(cand_i[:take].astype(np.int64, copy=False))
            nbr_dist.append(cand_d[:take].astype(np.float32, copy=False))
            k_eff[i] = int(take)

        kdist = np.ones(n_local, dtype=np.float32)
        for i in range(n_local):
            if k_eff[i] > 0:
                kdist[i] = float(nbr_dist[i][k_eff[i] - 1])

        lrd = np.ones(n_local, dtype=np.float32)
        for i in range(n_local):
            ke = int(k_eff[i])
            if ke <= 0:
                continue
            ni = nbr_idx[i][:ke]
            reach = np.maximum(kdist[ni], nbr_dist[i][:ke])
            lrd[i] = float(ke) / max(float(np.sum(reach)), 1e-10)

        for tt in range(chunk_start, chunk_end):
            i = tt - lo
            ke = int(k_eff[i])
            if ke <= 0:
                scores[tt] = 1.0
                continue
            ni = nbr_idx[i][:ke]
            scores[tt] = float(np.mean(lrd[ni]) / max(lrd[i], 1e-10))

    scores = scores.astype(np.float32, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("Temporal LOF produced non-finite values")
    return scores


def score_with_variant_15c(
    variant: str,
    payload: dict[str, Any],
    seed: int,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    timeout_seconds: float | None = None,
) -> ScoreOutput:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported Task-15c variant: {variant}")

    deadline = None
    if timeout_seconds is not None and float(timeout_seconds) > 0:
        deadline = time.monotonic() + float(timeout_seconds)

    train_raw = payload["train_raw"]
    test_raw = payload["test_raw"]
    t_test = _time_length(test_raw)

    train_patch = _as_tensor(payload.get("train_patch_embeds_unnorm", payload.get("train_patch_embeds")))
    test_patch = _as_tensor(payload.get("test_patch_embeds_unnorm", payload.get("test_patch_embeds")))
    patch_bank = payload.get("patch_bank_unnorm")
    if patch_bank is None:
        patch_bank = payload.get("patch_bank")
    patch_bank = _as_tensor(patch_bank) if patch_bank is not None else None

    train_point_norm = None
    test_point_norm = None
    train_point_unnorm = None
    test_point_unnorm = None
    point_bank_unnorm = None

    if variant in POINT_LEVEL_VARIANTS:
        train_point_norm = _as_tensor(payload["train_point_embeds"])
        test_point_norm = _as_tensor(payload["test_point_embeds"])

        train_point_unnorm = payload.get("train_point_embeds_unnorm")
        if train_point_unnorm is None:
            train_point_unnorm = train_point_norm
        train_point_unnorm = _as_tensor(train_point_unnorm)

        test_point_unnorm = payload.get("test_point_embeds_unnorm")
        if test_point_unnorm is None:
            test_point_unnorm = test_point_norm
        test_point_unnorm = _as_tensor(test_point_unnorm)

        point_bank_unnorm = payload.get("point_bank_unnorm")
        if point_bank_unnorm is None:
            point_bank_unnorm = payload.get("point_bank")
        point_bank_unnorm = _as_tensor(point_bank_unnorm) if point_bank_unnorm is not None else None

    native_level = "patch"
    info: dict[str, Any] = {
        "variant": variant,
        "faiss_available": bool(faiss is not None),
    }

    if variant in {"c1_euclidean", "c2_e2a_euclidean"}:
        if patch_bank is None or int(patch_bank.shape[0]) == 0:
            patch_bank = _kmeans_centers(train_patch, random_state=int(seed))
            info["bank_rebuilt"] = True
        native_scores = _euclidean_topk(test_patch, patch_bank, top_k=3, batch_size=batch_size, device=device, agg="mean")

    elif variant == "c2_cosine_orig":
        # Original PaAno (Xu et al. 2024): cosine distance on L2-normalized
        # patch embeds, kmeans bank seeded from normalized train embeds, top-k=3
        # mean aggregation. We use the normalized payload ("train_patch_embeds"
        # / "test_patch_embeds") rather than the *_unnorm versions that feed
        # the Euclidean family. If a prebuilt normalized bank is in the payload
        # we use it; otherwise we rebuild via kmeans on normalized train feats.
        train_patch_norm = _as_tensor(payload["train_patch_embeds"])
        test_patch_norm = _as_tensor(payload["test_patch_embeds"])
        patch_bank_norm = payload.get("patch_bank")
        have_bank = (
            patch_bank_norm is not None
            and hasattr(patch_bank_norm, "shape")
            and int(np.asarray(patch_bank_norm).shape[0]) > 0
        )
        if have_bank:
            patch_bank_norm = _as_tensor(patch_bank_norm)
        else:
            patch_bank_norm = _kmeans_centers(train_patch_norm, random_state=int(seed))
            info["bank_rebuilt"] = True
        native_scores = _cosine_topk(
            test_patch_norm,
            patch_bank_norm,
            top_k=3,
            batch_size=batch_size,
            device=device,
            agg="mean",
        )

    elif variant == "c4_euclidean_clean":
        _check_deadline(deadline)
        clean, prune_info = _self_lof_prune_l2(train_patch, lof_k=10, prune_pct=0.05, deadline=deadline)
        bank = _kmeans_centers(clean, random_state=int(seed))
        native_scores = _euclidean_topk(test_patch, bank, top_k=3, batch_size=batch_size, device=device, agg="mean")
        info.update(prune_info)

    elif variant == "c5_euclidean_median":
        if patch_bank is None or int(patch_bank.shape[0]) == 0:
            patch_bank = _kmeans_centers(train_patch, random_state=int(seed))
            info["bank_rebuilt"] = True
        native_scores = _euclidean_topk(test_patch, patch_bank, top_k=5, batch_size=batch_size, device=device, agg="median")

    elif variant == "c6_point_euclidean":
        if point_bank_unnorm is None or int(point_bank_unnorm.shape[0]) == 0:
            point_bank_unnorm = _kmeans_centers(train_point_unnorm, random_state=int(seed))
            info["point_bank_rebuilt"] = True
        native_scores = _euclidean_topk(
            test_point_unnorm,
            point_bank_unnorm,
            top_k=3,
            batch_size=batch_size,
            device=device,
            agg="mean",
        )
        native_level = "point"

    elif variant == "c7_point_euclid_s32":
        if point_bank_unnorm is None or int(point_bank_unnorm.shape[0]) == 0:
            point_bank_unnorm = _kmeans_centers(train_point_unnorm, random_state=int(seed))
            info["point_bank_rebuilt"] = True
        raw = _euclidean_topk(test_point_unnorm, point_bank_unnorm, top_k=3, batch_size=batch_size, device=device, agg="mean")
        native_scores = gaussian_filter1d(raw, sigma=32).astype(np.float32, copy=False)
        native_level = "point"
        info["raw_std"] = float(np.std(raw))
        info["smoothed_std"] = float(np.std(native_scores))

    elif variant == "c8_point_euclid_s128":
        if point_bank_unnorm is None or int(point_bank_unnorm.shape[0]) == 0:
            point_bank_unnorm = _kmeans_centers(train_point_unnorm, random_state=int(seed))
            info["point_bank_rebuilt"] = True
        raw = _euclidean_topk(test_point_unnorm, point_bank_unnorm, top_k=3, batch_size=batch_size, device=device, agg="mean")
        native_scores = gaussian_filter1d(raw, sigma=128).astype(np.float32, copy=False)
        native_level = "point"
        info["raw_std"] = float(np.std(raw))
        info["smoothed_std"] = float(np.std(native_scores))

    elif variant == "c9_point_lof_mag":
        _check_deadline(deadline)
        lof = _global_lof_cosine(test_point_norm, k=20, deadline=deadline)
        mag = _magnitude_zscore(train_raw, test_raw)
        native_scores = (0.5 * _rank_normalize(lof) + 0.5 * _rank_normalize(mag)).astype(np.float32, copy=False)
        native_level = "point"

    elif variant == "c10_temporal_lof":
        _check_deadline(deadline)
        native_scores = _temporal_lof_faiss_chunked(
            test_point_norm,
            k=10,
            window=200,
            chunk_size=2000,
            deadline=deadline,
        )
        native_level = "point"

    else:  # pragma: no cover
        raise ValueError(f"Unhandled variant: {variant}")

    native_scores = np.asarray(native_scores, dtype=np.float32).reshape(-1)
    if not np.isfinite(native_scores).all():
        raise ValueError(f"{variant} produced non-finite native scores")

    if native_level == "patch":
        point_scores = distribute_patch_scores_to_points(native_scores, int(patch_size), int(t_test))
    else:
        point_scores = native_scores

    point_scores = np.asarray(point_scores, dtype=np.float32).reshape(-1)
    if int(point_scores.shape[0]) != int(t_test):
        raise ValueError(f"{variant} produced point length {point_scores.shape[0]}, expected {t_test}")
    if not np.isfinite(point_scores).all():
        raise ValueError(f"{variant} produced non-finite point scores")

    info.update(
        {
            "native_level": native_level,
            "native_len": int(native_scores.shape[0]),
            "point_len": int(point_scores.shape[0]),
        }
    )

    return ScoreOutput(
        point_scores=point_scores,
        native_scores=native_scores,
        native_level=native_level,
        info=info,
    )
