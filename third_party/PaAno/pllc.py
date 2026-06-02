from __future__ import annotations

import torch


def _patch_mad(signal: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    median = signal.median(dim=1, keepdim=True).values
    mad = (signal - median).abs().median(dim=1, keepdim=True).values
    return mad.clamp_min(eps)


def pllc_sample(
    anchors_batch: torch.Tensor,
    W: int,
    D_n: int,
    A: int,
    K: int,
    decile: float,
):
    if anchors_batch.ndim != 3:
        raise ValueError(f"Expected anchors_batch [N, C, W], got {tuple(anchors_batch.shape)}")
    if int(W) != int(anchors_batch.shape[-1]):
        raise ValueError(f"W mismatch: expected {W}, got {anchors_batch.shape[-1]}")
    if int(D_n) < 1:
        raise ValueError(f"D_n must be >= 1, got {D_n}")
    if not (0.0 < float(decile) < 0.5):
        raise ValueError(f"decile must be in (0, 0.5), got {decile}")

    device = anchors_batch.device
    batch_size = int(anchors_batch.shape[0])
    anchor_budget = max(1, int(A))
    easy_neg = max(0, int(K))

    if batch_size == 0:
        empty_a = torch.empty((0, anchor_budget), dtype=torch.long, device=device)
        empty_hit = torch.empty((0, anchor_budget), dtype=torch.bool, device=device)
        empty_neg = torch.empty((0, anchor_budget, easy_neg), dtype=torch.long, device=device)
        return empty_a, empty_a.clone(), empty_a.clone(), empty_neg, empty_hit

    signal = anchors_batch[:, 0, :].to(dtype=torch.float32)
    mad = _patch_mad(signal)
    diffs = (signal.unsqueeze(2) - signal.unsqueeze(1)).abs() / mad.unsqueeze(-1)

    time_idx = torch.arange(W, device=device)
    rel = (time_idx[:, None] - time_idx[None, :]).abs()
    anchor_mask = (time_idx >= 1) & (time_idx <= (W - 2))
    neigh_mask = (rel > 0) & (rel <= int(D_n))
    pair_mask = neigh_mask & anchor_mask[:, None]

    pair_values = diffs[:, pair_mask]
    low_q = torch.quantile(pair_values, float(decile), dim=1)
    high_q = torch.quantile(pair_values, 1.0 - float(decile), dim=1)

    low_mask = pair_mask.unsqueeze(0) & (diffs <= low_q[:, None, None])
    high_mask = pair_mask.unsqueeze(0) & (diffs >= high_q[:, None, None])
    eligible = low_mask.any(dim=-1) & high_mask.any(dim=-1)
    eligible_count = eligible.sum(dim=1)
    patch_hit = eligible_count >= 2
    eligible = eligible & patch_hit.unsqueeze(1)

    anchor_scores = torch.rand((batch_size, W), device=device)
    anchor_scores = anchor_scores.masked_fill(~eligible, -1.0)
    _, anchors_t = anchor_scores.topk(k=min(anchor_budget, W), dim=1)
    if anchors_t.shape[1] < anchor_budget:
        pad = anchor_budget - anchors_t.shape[1]
        anchors_t = torch.cat([anchors_t, anchors_t.new_zeros((batch_size, pad))], dim=1)
    hit_mask = torch.gather(eligible, 1, anchors_t)

    gather_idx = anchors_t.unsqueeze(-1).expand(-1, -1, W)
    low_anchor = torch.gather(low_mask, 1, gather_idx)
    high_anchor = torch.gather(high_mask, 1, gather_idx)

    pos_scores = torch.rand((batch_size, anchor_budget, W), device=device)
    pos_scores = pos_scores.masked_fill(~low_anchor, -1.0)
    pos_t = pos_scores.argmax(dim=-1)

    neg_scores = torch.rand((batch_size, anchor_budget, W), device=device)
    neg_scores = neg_scores.masked_fill(~high_anchor, -1.0)
    hard_neg_t = neg_scores.argmax(dim=-1)

    if easy_neg > 0 and batch_size > 1:
        patch_base = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        patch_offsets = torch.randint(1, batch_size, (batch_size, anchor_budget, easy_neg), device=device)
        easy_patch = (patch_base + patch_offsets) % batch_size
        easy_time = torch.randint(0, W, (batch_size, anchor_budget, easy_neg), device=device)
        easy_neg_idx = easy_patch * W + easy_time
    else:
        easy_neg_idx = torch.empty((batch_size, anchor_budget, 0), dtype=torch.long, device=device)

    return anchors_t, pos_t, hard_neg_t, easy_neg_idx, hit_mask


def pllc_infonce(
    proj_n: torch.Tensor,
    anchors_t: torch.Tensor,
    pos_t: torch.Tensor,
    hard_neg_t: torch.Tensor,
    easy_neg_idx: torch.Tensor,
    tau: float,
    hit_mask: torch.Tensor,
) -> torch.Tensor:
    if proj_n.ndim != 3:
        raise ValueError(f"Expected proj_n [N, W, d], got {tuple(proj_n.shape)}")

    batch_size, width, dim = proj_n.shape
    gather_anchor = anchors_t.unsqueeze(-1).expand(-1, -1, dim)
    gather_pos = pos_t.unsqueeze(-1).expand(-1, -1, dim)
    gather_neg = hard_neg_t.unsqueeze(-1).expand(-1, -1, dim)

    anchor_vec = torch.gather(proj_n, 1, gather_anchor)
    pos_vec = torch.gather(proj_n, 1, gather_pos)
    hard_vec = torch.gather(proj_n, 1, gather_neg)

    logits = [
        (anchor_vec * pos_vec).sum(dim=-1, keepdim=True) / float(tau),
        (anchor_vec * hard_vec).sum(dim=-1, keepdim=True) / float(tau),
    ]

    if easy_neg_idx.numel() > 0:
        flat_proj = proj_n.reshape(batch_size * width, dim)
        easy_vec = flat_proj.index_select(0, easy_neg_idx.reshape(-1)).view(*easy_neg_idx.shape, dim)
        easy_logits = (anchor_vec.unsqueeze(-2) * easy_vec).sum(dim=-1) / float(tau)
        logits.append(easy_logits)

    all_logits = torch.cat(logits, dim=-1)
    pos_logits = logits[0].squeeze(-1)
    valid = hit_mask.to(dtype=torch.bool)
    if not valid.any():
        return proj_n.new_zeros(())
    loss = torch.logsumexp(all_logits, dim=-1) - pos_logits
    return loss.masked_select(valid).mean()
