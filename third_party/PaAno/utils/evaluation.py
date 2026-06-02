import torch
import torch.nn.functional as F
import numpy as np


def _iter_score_batches(data_source, batch_size=None):
    if torch.is_tensor(data_source):
        if batch_size is None or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer when data_source is a tensor")
        total = int(data_source.shape[0])
        for start in range(0, total, int(batch_size)):
            end = min(total, start + int(batch_size))
            yield data_source[start:end]
        return
    for data, _ in data_source:
        yield data


# Distance-based anomaly scoring 
@torch.inference_mode()
def calculate_anomaly_scores(model, data_source, memory_bank, device, top_k=3, batch_size=None):
    model.eval()
    all_scores = []
    memory_bank = F.normalize(memory_bank.to(device, dtype=torch.float32), dim=1, eps=1e-12)
    inferred_batch_size = batch_size or getattr(data_source, "batch_size", None)

    for data in _iter_score_batches(data_source, batch_size=inferred_batch_size):
        data = data.to(device, non_blocking=True, dtype=torch.float32)
        feats = model.embedding(data)  # (B, D)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        feats = F.normalize(feats, dim=1, eps=1e-12)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Cosine similarity & distance
        sims = feats @ memory_bank.T                    # (B, M)
        sims = torch.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
        topk_sim, _ = torch.topk(sims, k=top_k, dim=1, largest=True)
        dists = 1.0 - topk_sim
        scores = dists.mean(dim=1)

        scores = torch.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)
        all_scores.extend(scores.cpu().tolist())

    return all_scores


# Patch-to-point score distribution 
def distribute_patch_scores_to_points(patch_scores, patch_size, num_points): 

    patch_scores = np.nan_to_num(np.asarray(patch_scores, dtype=np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)

    kernel = np.ones(patch_size, dtype=np.float32)
    sums   = np.convolve(patch_scores, kernel, mode='full')[:num_points]
    counts = np.convolve(np.ones_like(patch_scores), kernel, mode='full')[:num_points]

    point_scores = np.divide(
        sums, counts,
        out=np.zeros(num_points, dtype=np.float32),
        where=counts != 0
    )
    return np.nan_to_num(point_scores, nan=0.0, posinf=0.0, neginf=0.0)
