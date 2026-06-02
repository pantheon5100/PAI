import copy
import math
import time
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ema_utils import EMATeacher
from pllc import pllc_infonce, pllc_sample
from utils.utils import *


SHAPE_POSITIVE_WINDOW = 128
SHAPE_POSITIVE_BATCH_SIZE = 128
SHAPE_POSITIVE_FULL_SCAN_THRESHOLD = 4096
SHAPE_POSITIVE_SUBSAMPLE_STRIDE = 4


def _amplitude_augment(patches):
    patch_std = patches.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    scale = torch.empty(
        (patches.shape[0], 1, 1),
        device=patches.device,
        dtype=patches.dtype,
    ).uniform_(0.5, 2.0)
    offset = torch.randn(
        (patches.shape[0], 1, 1),
        device=patches.device,
        dtype=patches.dtype,
    ) * (0.3 * patch_std)
    return patches * scale + offset


def _gaussian_noise_augment(patches):
    patch_std = patches.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    noise = torch.randn_like(patches) * (0.1 * patch_std)
    return patches + noise


def _temporal_shift_augment(patches):
    batch_size, channels, patch_length = patches.shape
    shifted = patches.clone()
    shifts = torch.randint(1, 4, (batch_size,), device=patches.device)
    directions = torch.where(
        torch.rand(batch_size, device=patches.device) < 0.5,
        -torch.ones(batch_size, device=patches.device, dtype=torch.long),
        torch.ones(batch_size, device=patches.device, dtype=torch.long),
    )
    signed_shifts = shifts * directions
    for idx in range(batch_size):
        shift = int(signed_shifts[idx].item())
        if shift > 0:
            shifted[idx, :, shift:] = patches[idx, :, : patch_length - shift]
            shifted[idx, :, :shift] = patches[idx, :, :1]
        else:
            amount = -shift
            shifted[idx, :, : patch_length - amount] = patches[idx, :, amount:]
            shifted[idx, :, patch_length - amount :] = patches[idx, :, -1:]
    return shifted


def _channel_dropout_augment(patches):
    if patches.shape[1] <= 1:
        return patches
    dropped = patches.clone()
    channel_ids = torch.randint(0, patches.shape[1], (patches.shape[0],), device=patches.device)
    for idx in range(patches.shape[0]):
        dropped[idx, channel_ids[idx]] = 0
    return dropped


def _patch_mask_augment(patches, mask_ratio_min=0.2, mask_ratio_max=0.4):
    batch_size, channels, patch_length = patches.shape
    mask_ratio_min = float(mask_ratio_min)
    mask_ratio_max = float(mask_ratio_max)
    if not (0.0 < mask_ratio_min < 1.0):
        raise ValueError(f"mask_ratio_min must be in (0, 1), got {mask_ratio_min}")
    if not (0.0 < mask_ratio_max < 1.0):
        raise ValueError(f"mask_ratio_max must be in (0, 1), got {mask_ratio_max}")
    if mask_ratio_max < mask_ratio_min:
        raise ValueError(
            f"mask_ratio_max must be >= mask_ratio_min, got {mask_ratio_max} < {mask_ratio_min}"
        )
    mask_scores = torch.rand((batch_size, patch_length), device=patches.device, dtype=patches.dtype)
    if math.isclose(mask_ratio_min, mask_ratio_max):
        mask_ratio = torch.full(
            (batch_size, 1),
            fill_value=mask_ratio_min,
            device=patches.device,
            dtype=patches.dtype,
        )
    else:
        mask_ratio = torch.empty((batch_size, 1), device=patches.device, dtype=patches.dtype).uniform_(
            mask_ratio_min,
            mask_ratio_max,
        )
    mask_counts = (mask_ratio * patch_length).round().clamp(1, patch_length - 1).to(torch.long)
    cutoff = mask_scores.argsort(dim=1)
    time_mask = torch.zeros((batch_size, patch_length), device=patches.device, dtype=torch.bool)
    for idx in range(batch_size):
        time_mask[idx, cutoff[idx, : mask_counts[idx].item()]] = True
    fill_value = patches.mean(dim=-1, keepdim=True)
    expanded_mask = time_mask.unsqueeze(1).expand(batch_size, channels, patch_length)
    return torch.where(expanded_mask, fill_value.expand_as(patches), patches)


def _time_warp_patches(patches):
    batch_size, channels, patch_length = patches.shape
    if patch_length <= 2:
        return patches.clone()

    steps = torch.empty(
        (batch_size, patch_length),
        device=patches.device,
        dtype=patches.dtype,
    ).uniform_(0.8, 1.25)
    warped_pos = torch.cumsum(steps, dim=1)
    warped_pos = warped_pos - warped_pos[:, :1]
    scale = (patch_length - 1) / warped_pos[:, -1:].clamp_min(1e-6)
    warped_pos = warped_pos * scale
    grid_x = (warped_pos / max(patch_length - 1, 1)) * 2.0 - 1.0
    grid_y = torch.zeros_like(grid_x)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)
    warped = F.grid_sample(
        patches.unsqueeze(2),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.squeeze(2)


def apply_patch_augmentation(patches, mode="none"):
    if mode == "none":
        return patches
    if mode == "amplitude":
        return _amplitude_augment(patches)
    if mode == "multi":
        augmented = patches
        if torch.rand((), device=patches.device) < 0.75:
            augmented = _amplitude_augment(augmented)
        if torch.rand((), device=patches.device) < 0.75:
            augmented = _gaussian_noise_augment(augmented)
        if torch.rand((), device=patches.device) < 0.5:
            augmented = _temporal_shift_augment(augmented)
        if patches.shape[1] > 1 and torch.rand((), device=patches.device) < 0.25:
            augmented = _channel_dropout_augment(augmented)
        return augmented
    if mode == "mask":
        return _patch_mask_augment(patches)
    raise ValueError(f"Unsupported anchor augmentation mode: {mode}")


def _temporal_positive_offsets(radius: int, device: torch.device) -> torch.Tensor:
    if radius < 1:
        raise ValueError(f"positive_radius must be >= 1, got {radius}")
    return torch.tensor(
        [*range(-radius, 0), *range(1, radius + 1)],
        dtype=torch.long,
        device=device,
    )


def sample_temporal_positive_indices(
    batch_indexes: torch.Tensor,
    total_len: int,
    device: torch.device,
    radius: int,
) -> torch.Tensor:
    offsets = _temporal_positive_offsets(radius=radius, device=device)
    cand = batch_indexes.unsqueeze(1) + offsets.unsqueeze(0)
    valid = (cand >= 0) & (cand < total_len)

    if radius > 2:
        weights = (radius + 1 - offsets.abs()).to(dtype=torch.float32)
        weights = valid.to(dtype=torch.float32) * weights.unsqueeze(0)
        none_valid = weights.sum(dim=1) <= 0
        if none_valid.any():
            weights[none_valid, 0] = 1.0
        choice = torch.multinomial(weights, num_samples=1).squeeze(1)
    else:
        noise = torch.rand(cand.shape, device=device)
        score = torch.where(valid, noise, torch.full_like(noise, -1.0))
        choice = score.argmax(dim=1)
        none_valid = valid.sum(dim=1) == 0

    pos_idx = cand.gather(1, choice.unsqueeze(1)).squeeze(1)
    if none_valid.any():
        pos_idx[none_valid] = batch_indexes[none_valid]
    return pos_idx


def _fft_sbd_distance_batched(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    patch_length = query.shape[-1]
    fft_size = 1 << (2 * patch_length - 1).bit_length()

    query_norm = query.norm(dim=1).clamp_min(1e-12)
    reference_norm = reference.norm(dim=2).clamp_min(1e-12)
    query_fft = torch.fft.rfft(query, n=fft_size, dim=-1)
    reference_fft = torch.fft.rfft(reference, n=fft_size, dim=-1).conj()

    cross_corr = torch.fft.irfft(query_fft[:, None, :] * reference_fft, n=fft_size, dim=-1)
    cross_corr = torch.cat([cross_corr[..., -(patch_length - 1):], cross_corr[..., :patch_length]], dim=-1)
    denom = (query_norm[:, None, None] * reference_norm[:, :, None]).clamp_min(1e-12)
    ncc = cross_corr / denom
    return 1.0 - ncc.amax(dim=-1)


def _fallback_shape_positive_indices(batch_indexes: torch.Tensor, total_len: int) -> torch.Tensor:
    fallback = batch_indexes.clone()
    minus_two = batch_indexes - 2
    plus_two = batch_indexes + 2

    use_minus_two = minus_two >= 0
    use_plus_two = (~use_minus_two) & (plus_two < total_len)
    fallback[use_minus_two] = minus_two[use_minus_two]
    fallback[use_plus_two] = plus_two[use_plus_two]
    return fallback


def build_shape_positive_index(
    patches,
    window=128,
    subsample_stride=SHAPE_POSITIVE_SUBSAMPLE_STRIDE,
    large_file_threshold=SHAPE_POSITIVE_FULL_SCAN_THRESHOLD,
    batch_size=SHAPE_POSITIVE_BATCH_SIZE,
):
    if patches.ndim != 3:
        raise ValueError(f"Expected patches with shape [N, C, L], got {tuple(patches.shape)}")
    if patches.shape[1] != 1:
        raise NotImplementedError("Shape-aware positive selection currently supports univariate patches only")

    total_len = int(patches.shape[0])
    if total_len == 0:
        return torch.empty(0, dtype=torch.long, device=patches.device)

    candidate_stride = 1 if total_len <= large_file_threshold else max(1, int(subsample_stride))
    offsets = torch.arange(-window, window + 1, candidate_stride, device=patches.device, dtype=torch.long)
    offsets = offsets[offsets.abs() > 1]
    if offsets.numel() == 0:
        raise ValueError("Shape-positive offset set is empty; increase window or reduce stride")

    signal = patches[:, 0, :].to(dtype=torch.float32)
    patch_length = signal.shape[-1]
    positive_index = torch.empty(total_len, dtype=torch.long, device=patches.device)

    for start in range(0, total_len, batch_size):
        end = min(total_len, start + batch_size)
        query_index = torch.arange(start, end, device=patches.device, dtype=torch.long)
        cand_index = query_index.unsqueeze(1) + offsets.unsqueeze(0)
        valid = (cand_index >= 0) & (cand_index < total_len)
        cand_clamped = cand_index.clamp(0, total_len - 1)

        query = signal[start:end]
        reference = signal.index_select(0, cand_clamped.reshape(-1)).view(end - start, offsets.numel(), patch_length)
        dist = _fft_sbd_distance_batched(query, reference)
        dist = dist.masked_fill(~valid, float("inf"))
        choice = dist.argmin(dim=1)
        selected = cand_clamped.gather(1, choice.unsqueeze(1)).squeeze(1)

        no_valid = ~valid.any(dim=1)
        if no_valid.any():
            selected[no_valid] = _fallback_shape_positive_indices(query_index[no_valid], total_len)

        positive_index[start:end] = selected

    return positive_index


def koleo_regularizer(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {tuple(features.shape)}")
    if features.shape[0] < 2:
        return features.new_zeros(())

    normalized = F.normalize(features, dim=1)
    pairwise_dist = torch.cdist(normalized, normalized, p=2)
    diagonal_mask = torch.eye(pairwise_dist.shape[0], device=pairwise_dist.device, dtype=torch.bool)
    pairwise_dist = pairwise_dist.masked_fill(diagonal_mask, float("inf"))
    nearest_dist = pairwise_dist.amin(dim=1).clamp_min(eps)
    return -torch.log(nearest_dist).mean()


def _compute_branch_triplet_pretext_losses(
    model,
    branch,
    h_anchors,
    h_pos,
    h_pretext,
    pre_mask,
    batch_size,
    temperature,
    mu,
    num_rand_patches,
    criterion_pretext,
    device,
):
    z_anchor = F.normalize(model.projection(h_anchors, branch=branch), dim=1)
    z_pos = F.normalize(model.projection(h_pos, branch=branch), dim=1)

    sim_ap = (z_anchor @ z_pos.T) / temperature
    pos_sims = sim_ap.diag()
    sim_ap_fill = sim_ap.clone()
    sim_ap_fill.diagonal().fill_(float("inf"))
    neg_dists = 1 - sim_ap_fill
    hard_neg_dists, _ = torch.max(neg_dists, dim=1)

    pos_dists = 1 - pos_sims
    triplet_loss = F.relu(pos_dists - hard_neg_dists + 0.5).mean() / mu
    triplet_loss = triplet_grad(triplet_loss)

    if h_pretext is None or pre_mask is None:
        pretext_loss = torch.tensor(0.0, device=device)
        return triplet_loss, pretext_loss

    if int(pre_mask.sum().item()) == 0:
        pretext_loss = torch.tensor(0.0, device=device)
        return triplet_loss, pretext_loss

    h_pre = h_pretext[pre_mask]
    h_anchor_pre = h_anchors[pre_mask]
    h_concat_pre = torch.cat([h_anchor_pre, h_pre], dim=1)
    all_indices = torch.arange(batch_size, device=device)
    anchor_indices = all_indices.repeat_interleave(num_rand_patches)
    rand_offsets = torch.randint(1, batch_size, (batch_size * num_rand_patches,), device=device)
    unadj_indices = (anchor_indices + rand_offsets) % batch_size
    h_unadj = h_anchors[unadj_indices]
    h_anchor_unadj = h_anchors.repeat_interleave(num_rand_patches, dim=0)
    h_concat_unadj = torch.cat([h_anchor_unadj, h_unadj], dim=1)
    all_pretext_features = torch.cat([h_concat_pre, h_concat_unadj], dim=0)
    all_pretext_labels = torch.cat([
        torch.ones(h_concat_pre.size(0), device=device),
        torch.zeros(h_concat_unadj.size(0), device=device),
    ])
    pretext_outputs = model.classification_logits(all_pretext_features, branch=branch).squeeze(1)
    pretext_loss_all = criterion_pretext(pretext_outputs, all_pretext_labels)
    loss_pre = pretext_loss_all[:h_concat_pre.size(0)].mean()
    loss_unadj = pretext_loss_all[h_concat_pre.size(0):].mean()
    pretext_loss = loss_pre + loss_unadj
    return triplet_loss, pretext_loss


def _train_dual_branch_model(
    model,
    train_loader,
    train_patches,
    device,
    num_iter=200,
    pretext_step=64,
    lr=1e-4,
    see_loss=None,
    wandb_run=None,
    wandb_prefix="train",
    log_every=1,
    best_checkpoint_path=None,
    final_checkpoint_path=None,
    anchor_augmentation="none",
    positive_radius=2,
    positive_mode="temporal",
    time_warp_negatives=False,
    koleo_weight=0.0,
    mask_prediction=False,
    mask_prediction_weight=0.5,
    mask_ratio_min=0.2,
    mask_ratio_max=0.4,
    ema_teacher=False,
    ema_teacher_weight=0.5,
    ema_teacher_positive=False,
    ema_tau_start=0.996,
    ema_tau_end=0.999,
    checkpoint_interval=0,
    checkpoint_callback=None,
    agree_weight=0.0,
    sharp_weight=0.0,
    sharp_mode="off",
    sharp_dim=64,
    sharp_neigh=8,
    sharp_tau=0.1,
    sharp_anchors=32,
    sharp_easy_neg=16,
    sharp_decile=0.1,
    sharp_beta=1.0,
):
    unsupported = []
    if time_warp_negatives:
        unsupported.append("time_warp_negatives")
    if float(koleo_weight) > 0.0:
        unsupported.append("koleo_weight")
    if mask_prediction:
        unsupported.append("mask_prediction")
    if ema_teacher:
        unsupported.append("ema_teacher")
    if float(agree_weight) > 0.0:
        unsupported.append("agree_weight")
    if checkpoint_callback is not None:
        unsupported.append("checkpoint_callback")
    if unsupported:
        raise NotImplementedError(
            "DualBranchPatchEncoder does not support these training options yet: "
            + ", ".join(unsupported)
        )

    lambda_weight = 1
    temperature = 1.0
    num_rand_patches = 5
    initial_lr = float(lr)

    def cosine_annealed_lr(iteration, start_lr, end_lr):
        t = min(iteration, num_iter)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / num_iter))
        return end_lr + (start_lr - end_lr) * cosine_factor

    optimizer = torch.optim.AdamW(
        [{
            "params": list(model.parameters()),
            "lr": initial_lr,
            "base_lr": initial_lr,
            "final_lr": initial_lr / 10.0,
        }],
        weight_decay=1e-4,
    )
    pos_weight = torch.tensor([1.0], device=device)
    criterion_pretext = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    iteration_count = 0
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    last_loss = float("inf")
    cumulative_train_time_s = 0.0
    display_losses = bool(see_loss)
    last_triplet_loss = 0.0
    last_pllc_loss = 0.0
    last_llb_loss = 0.0
    last_anchor_hit_rate = 0.0
    last_global_triplet_loss = 0.0
    last_local_triplet_loss = 0.0
    last_global_pretext_loss = 0.0
    last_local_pretext_loss = 0.0
    sum_triplet_loss = 0.0
    sum_pllc_loss = 0.0
    sum_llb_loss = 0.0
    sum_anchor_hit_rate = 0.0
    sum_global_triplet_loss = 0.0
    sum_local_triplet_loss = 0.0
    sum_global_pretext_loss = 0.0
    sum_local_pretext_loss = 0.0
    shape_positive_index = None
    shape_positive_stride = None
    peak_gpu_memory_mb = 0.0

    train_patches = train_patches.to(device=device, non_blocking=True)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if positive_mode == "shape":
        shape_positive_stride = 1 if train_patches.shape[0] <= SHAPE_POSITIVE_FULL_SCAN_THRESHOLD else SHAPE_POSITIVE_SUBSAMPLE_STRIDE
        print(
            "    >> Precomputing shape-positive index "
            f"(window={SHAPE_POSITIVE_WINDOW}, stride={shape_positive_stride}, patches={train_patches.shape[0]})"
        )
        shape_positive_index = build_shape_positive_index(train_patches, window=SHAPE_POSITIVE_WINDOW)

    print("    [Training Info]")
    pbar = tqdm(total=num_iter, desc="    >> Training", ncols=80)
    model.train()

    while iteration_count < num_iter:
        for batch_data, batch_indexes in train_loader:
            if iteration_count >= num_iter:
                break

            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            iter_start_time = time.perf_counter()
            iteration_count += 1
            for param_group in optimizer.param_groups:
                param_group["lr"] = cosine_annealed_lr(
                    iteration_count,
                    start_lr=param_group["base_lr"],
                    end_lr=param_group["final_lr"],
                )

            batch_data = batch_data.to(device, non_blocking=True)
            batch_indexes = batch_indexes.squeeze(-1).to(device=device, non_blocking=True).long()
            anchors = apply_patch_augmentation(batch_data, mode=anchor_augmentation)
            batch_size = batch_data.shape[0]
            mu = 1 if batch_data.shape[1] != 1 else 10
            total_len = train_patches.shape[0]

            if positive_mode == "shape":
                pos_idx = shape_positive_index.index_select(0, batch_indexes)
            else:
                pos_idx = sample_temporal_positive_indices(
                    batch_indexes=batch_indexes,
                    total_len=total_len,
                    device=device,
                    radius=positive_radius,
                )
            positives = train_patches.index_select(0, pos_idx)
            if anchor_augmentation == "multi":
                positives = apply_patch_augmentation(positives, mode=anchor_augmentation)

            if iteration_count < (num_iter / 10):
                current_lambda_pretext = lambda_weight * (1 - (iteration_count / (num_iter / 10)))
            else:
                current_lambda_pretext = 0.0

            if current_lambda_pretext > 0.0:
                tgt = batch_indexes - pretext_step
                pre_mask = (tgt >= 0) & (tgt < total_len)
                tgt_clamped = tgt.clamp(0, total_len - 1)
                pretext_patches = train_patches.index_select(0, tgt_clamped).clone()
                if (~pre_mask).any():
                    pretext_patches[~pre_mask] = 0
                all_patches = torch.cat([anchors, positives, pretext_patches], dim=0)
            else:
                pre_mask = None
                all_patches = torch.cat([anchors, positives], dim=0)

            global_embeddings = model.embedding(all_patches, branch="global")
            local_embeddings = model.embedding(all_patches, branch="local")

            cursor = 0
            g_anchor = global_embeddings[cursor:cursor + batch_size]
            l_anchor = local_embeddings[cursor:cursor + batch_size]
            cursor += batch_size
            g_pos = global_embeddings[cursor:cursor + batch_size]
            l_pos = local_embeddings[cursor:cursor + batch_size]
            cursor += batch_size
            if current_lambda_pretext > 0.0:
                g_pretext = global_embeddings[cursor:cursor + batch_size]
                l_pretext = local_embeddings[cursor:cursor + batch_size]
            else:
                g_pretext = None
                l_pretext = None

            global_triplet_loss, global_pretext_loss = _compute_branch_triplet_pretext_losses(
                model=model,
                branch="global",
                h_anchors=g_anchor,
                h_pos=g_pos,
                h_pretext=g_pretext,
                pre_mask=pre_mask,
                batch_size=batch_size,
                temperature=temperature,
                mu=mu,
                num_rand_patches=num_rand_patches,
                criterion_pretext=criterion_pretext,
                device=device,
            )
            local_triplet_loss, local_pretext_loss = _compute_branch_triplet_pretext_losses(
                model=model,
                branch="local",
                h_anchors=l_anchor,
                h_pos=l_pos,
                h_pretext=l_pretext,
                pre_mask=pre_mask,
                batch_size=batch_size,
                temperature=temperature,
                mu=mu,
                num_rand_patches=num_rand_patches,
                criterion_pretext=criterion_pretext,
                device=device,
            )

            triplet_loss = global_triplet_loss + local_triplet_loss
            pretext_loss = global_pretext_loss + local_pretext_loss

            llb_loss = torch.tensor(0.0, device=device)
            anchor_hit_rate = torch.tensor(0.0, device=device)
            if float(sharp_weight) > 0.0 and str(sharp_mode) == "pllc":
                proj = model.forward_sharp(anchors, branch="local")
                proj_n = F.normalize(proj, dim=-1, eps=1e-12)
                anchors_t, pos_t, hard_neg_t, easy_neg_idx, hit_mask = pllc_sample(
                    anchors_batch=anchors,
                    W=int(anchors.shape[-1]),
                    D_n=int(sharp_neigh),
                    A=int(sharp_anchors),
                    K=int(sharp_easy_neg),
                    decile=float(sharp_decile),
                )
                pllc_loss = pllc_infonce(
                    proj_n=proj_n,
                    anchors_t=anchors_t,
                    pos_t=pos_t,
                    hard_neg_t=hard_neg_t,
                    easy_neg_idx=easy_neg_idx,
                    tau=float(sharp_tau),
                    hit_mask=hit_mask,
                )
                anchor_hit_rate = hit_mask.any(dim=1).float().mean()
            elif float(sharp_weight) > 0.0 and str(sharp_mode) == "llb":
                proj = model.forward_sharp(anchors, branch="local")
                delta_feat = (proj[:, 1:, :] - proj[:, :-1, :]).norm(dim=-1)
                raw_signal = anchors[:, 0, :]
                median = raw_signal.median(dim=1, keepdim=True).values
                mad = (raw_signal - median).abs().median(dim=1, keepdim=True).values.clamp_min(1e-6)
                delta_sig = (raw_signal[:, 1:] - raw_signal[:, :-1]).abs() / mad
                llb_loss = F.relu(float(sharp_beta) * delta_sig - delta_feat).pow(2).mean()
                pllc_loss = llb_loss
            else:
                pllc_loss = torch.tensor(0.0, device=device)

            final_loss = triplet_loss + current_lambda_pretext * pretext_loss + (float(sharp_weight) * pllc_loss)
            last_loss = float(final_loss.item())
            last_triplet_loss = float(triplet_loss.item())
            last_pllc_loss = float(pllc_loss.item())
            last_llb_loss = float(llb_loss.item())
            last_anchor_hit_rate = float(anchor_hit_rate.item())
            last_global_triplet_loss = float(global_triplet_loss.item())
            last_local_triplet_loss = float(local_triplet_loss.item())
            last_global_pretext_loss = float(global_pretext_loss.item())
            last_local_pretext_loss = float(local_pretext_loss.item())

            sum_triplet_loss += last_triplet_loss
            sum_pllc_loss += last_pllc_loss
            sum_llb_loss += last_llb_loss
            sum_anchor_hit_rate += last_anchor_hit_rate
            sum_global_triplet_loss += last_global_triplet_loss
            sum_local_triplet_loss += last_local_triplet_loss
            sum_global_pretext_loss += last_global_pretext_loss
            sum_local_pretext_loss += last_local_pretext_loss

            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            optimizer.step()

            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            cumulative_train_time_s += time.perf_counter() - iter_start_time
            pbar.update(1)

            if final_loss.item() < best_loss:
                best_loss = final_loss.item()
                best_model_wts = copy.deepcopy(model.state_dict())

            should_log = wandb_run is not None and (
                iteration_count == 1 or iteration_count % log_every == 0 or iteration_count == num_iter
            )
            if should_log:
                wandb_run.log({
                    f"{wandb_prefix}/final_loss": float(final_loss.item()),
                    f"{wandb_prefix}/triplet_loss": float(triplet_loss.item()),
                    f"{wandb_prefix}/pretext_loss": float(pretext_loss.item()),
                    f"{wandb_prefix}/global_triplet_loss": float(global_triplet_loss.item()),
                    f"{wandb_prefix}/local_triplet_loss": float(local_triplet_loss.item()),
                    f"{wandb_prefix}/global_pretext_loss": float(global_pretext_loss.item()),
                    f"{wandb_prefix}/local_pretext_loss": float(local_pretext_loss.item()),
                    f"{wandb_prefix}/pllc_loss": float(pllc_loss.item()),
                    f"{wandb_prefix}/llb_loss": float(llb_loss.item()),
                    f"{wandb_prefix}/anchor_hit_rate": float(anchor_hit_rate.item()),
                    f"{wandb_prefix}/sharp_weight": float(sharp_weight),
                    f"{wandb_prefix}/lambda_pretext": float(current_lambda_pretext),
                    f"{wandb_prefix}/lr": float(optimizer.param_groups[0]["lr"]),
                    f"{wandb_prefix}/best_loss": float(best_loss),
                    f"{wandb_prefix}/iteration": iteration_count,
                }, step=iteration_count)

            if display_losses:
                pbar.set_postfix({
                    "loss": f"{final_loss.item():.4f}",
                    "g_tri": f"{global_triplet_loss.item():.4f}",
                    "l_tri": f"{local_triplet_loss.item():.4f}",
                    "g_pre": f"{global_pretext_loss.item():.4f}",
                    "l_pre": f"{local_pretext_loss.item():.4f}",
                    "pllc": f"{pllc_loss.item():.4f}",
                    "best": f"{best_loss:.4f}",
                })

    pbar.close()
    model.load_state_dict(best_model_wts)

    if best_checkpoint_path is not None:
        best_checkpoint_path = Path(best_checkpoint_path)
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_wts, best_checkpoint_path)

    if final_checkpoint_path is not None:
        final_checkpoint_path = Path(final_checkpoint_path)
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_checkpoint_path)

    if device.type == "cuda" and torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    return {
        "best_loss": float(best_loss),
        "last_loss": float(last_loss),
        "iterations": int(iteration_count),
        "train_time_cumulative_s": float(cumulative_train_time_s),
        "mean_train_iter_ms": float((cumulative_train_time_s / max(iteration_count, 1)) * 1000.0),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "positive_mode": positive_mode,
        "positive_radius": int(positive_radius),
        "shape_positive_window": int(SHAPE_POSITIVE_WINDOW) if positive_mode == "shape" else None,
        "shape_positive_stride": int(shape_positive_stride) if shape_positive_stride is not None else None,
        "time_warp_negatives": False,
        "koleo_weight": 0.0,
        "mask_prediction": False,
        "mask_prediction_weight": 0.0,
        "mask_ratio_min": float(mask_ratio_min),
        "mask_ratio_max": float(mask_ratio_max),
        "ema_teacher": False,
        "ema_teacher_weight": 0.0,
        "ema_teacher_positive": False,
        "ema_tau_start": float(ema_tau_start),
        "ema_tau_end": float(ema_tau_end),
        "agree_weight": 0.0,
        "sharp_weight": float(sharp_weight),
        "sharp_mode": str(sharp_mode),
        "sharp_dim": int(sharp_dim),
        "sharp_neigh": int(sharp_neigh),
        "sharp_tau": float(sharp_tau),
        "sharp_anchors": int(sharp_anchors),
        "sharp_easy_neg": int(sharp_easy_neg),
        "sharp_decile": float(sharp_decile),
        "sharp_beta": float(sharp_beta),
        "triplet_loss_last": float(last_triplet_loss),
        "triplet_loss_mean": float(sum_triplet_loss / max(iteration_count, 1)),
        "pllc_loss_last": float(last_pllc_loss),
        "pllc_loss_mean": float(sum_pllc_loss / max(iteration_count, 1)),
        "llb_loss_last": float(last_llb_loss),
        "llb_loss_mean": float(sum_llb_loss / max(iteration_count, 1)),
        "anchor_hit_rate_last": float(last_anchor_hit_rate),
        "anchor_hit_rate_mean": float(sum_anchor_hit_rate / max(iteration_count, 1)),
        "global_triplet_loss_last": float(last_global_triplet_loss),
        "global_triplet_loss_mean": float(sum_global_triplet_loss / max(iteration_count, 1)),
        "local_triplet_loss_last": float(last_local_triplet_loss),
        "local_triplet_loss_mean": float(sum_local_triplet_loss / max(iteration_count, 1)),
        "global_pretext_loss_last": float(last_global_pretext_loss),
        "global_pretext_loss_mean": float(sum_global_pretext_loss / max(iteration_count, 1)),
        "local_pretext_loss_last": float(last_local_pretext_loss),
        "local_pretext_loss_mean": float(sum_local_pretext_loss / max(iteration_count, 1)),
        "branch_type": "dual",
        "shared_block_count": int(getattr(model, "share_blocks", 0)),
        "detach_local_stem": bool(getattr(model, "detach_local_stem", False)),
        "teacher_model": None,
    }


def train_model(model, train_loader, train_patches, device, num_iter=200, pretext_step=64,
                lr=1e-4, see_loss=None, wandb_run=None, wandb_prefix="train",
                log_every=1, best_checkpoint_path=None, final_checkpoint_path=None,
                anchor_augmentation="none", positive_radius=2, positive_mode="temporal",
                time_warp_negatives=False, koleo_weight=0.0,
                mask_prediction=False, mask_prediction_weight=0.5,
                mask_ratio_min=0.2, mask_ratio_max=0.4,
                ema_teacher=False, ema_teacher_weight=0.5,
                ema_teacher_positive=False, ema_tau_start=0.996, ema_tau_end=0.999,
                checkpoint_interval=0, checkpoint_callback=None,
                # Task 26k: point-patch agreement (dense-SSL). See PatchEncoder.forward_agree.
                # Set agree_weight>0 and the model's agree_mode to "projector" or "raw".
                agree_weight=0.0,
                sharp_weight=0.0, sharp_mode="off", sharp_dim=64, sharp_neigh=8,
                sharp_tau=0.1, sharp_anchors=32, sharp_easy_neg=16,
                sharp_decile=0.1, sharp_beta=1.0):

    if positive_mode not in {"temporal", "shape"}:
        raise ValueError(f"Unsupported positive_mode: {positive_mode}")
    if positive_radius < 1:
        raise ValueError(f"positive_radius must be >= 1, got {positive_radius}")
    if float(koleo_weight) < 0:
        raise ValueError(f"koleo_weight must be >= 0, got {koleo_weight}")
    if float(mask_prediction_weight) < 0:
        raise ValueError(f"mask_prediction_weight must be >= 0, got {mask_prediction_weight}")
    if not (0.0 < float(mask_ratio_min) < 1.0):
        raise ValueError(f"mask_ratio_min must be in (0, 1), got {mask_ratio_min}")
    if not (0.0 < float(mask_ratio_max) < 1.0):
        raise ValueError(f"mask_ratio_max must be in (0, 1), got {mask_ratio_max}")
    if float(mask_ratio_max) < float(mask_ratio_min):
        raise ValueError(
            f"mask_ratio_max must be >= mask_ratio_min, got {mask_ratio_max} < {mask_ratio_min}"
        )
    if float(ema_teacher_weight) < 0:
        raise ValueError(f"ema_teacher_weight must be >= 0, got {ema_teacher_weight}")
    if not (0.0 < float(ema_tau_start) < 1.0):
        raise ValueError(f"ema_tau_start must be in (0, 1), got {ema_tau_start}")
    if not (0.0 < float(ema_tau_end) < 1.0):
        raise ValueError(f"ema_tau_end must be in (0, 1), got {ema_tau_end}")
    if float(ema_tau_end) < float(ema_tau_start):
        raise ValueError(
            f"ema_tau_end must be >= ema_tau_start, got {ema_tau_end} < {ema_tau_start}"
        )
    if float(agree_weight) < 0:
        raise ValueError(f"agree_weight must be >= 0, got {agree_weight}")
    if float(agree_weight) > 0.0 and getattr(model, "agree_mode", "off") == "off":
        raise ValueError(
            "agree_weight>0 requires the PatchEncoder to be built with "
            "agree_mode='projector' or 'raw'."
        )
    if float(sharp_weight) < 0:
        raise ValueError(f"sharp_weight must be >= 0, got {sharp_weight}")
    if str(sharp_mode) not in {"off", "pllc", "llb"}:
        raise ValueError(f"sharp_mode must be one of 'off'/'pllc'/'llb', got {sharp_mode}")
    if float(sharp_tau) <= 0.0:
        raise ValueError(f"sharp_tau must be > 0, got {sharp_tau}")
    if int(sharp_neigh) < 1:
        raise ValueError(f"sharp_neigh must be >= 1, got {sharp_neigh}")
    if int(sharp_anchors) < 1:
        raise ValueError(f"sharp_anchors must be >= 1, got {sharp_anchors}")
    if int(sharp_easy_neg) < 0:
        raise ValueError(f"sharp_easy_neg must be >= 0, got {sharp_easy_neg}")
    if not (0.0 < float(sharp_decile) < 0.5):
        raise ValueError(f"sharp_decile must be in (0, 0.5), got {sharp_decile}")
    if float(sharp_weight) > 0.0 and getattr(model, "sharp_mode", "off") == "off":
        raise ValueError(
            "sharp_weight>0 requires the PatchEncoder to be built with "
            "sharp_mode='pllc' or 'llb'."
        )
    if float(sharp_weight) > 0.0 and str(sharp_mode) != getattr(model, "sharp_mode", "off"):
        raise ValueError(
            f"sharp_mode mismatch between train_model ({sharp_mode}) and model ({getattr(model, 'sharp_mode', 'off')})"
        )
    checkpoint_interval = int(checkpoint_interval)
    if checkpoint_interval < 0:
        raise ValueError(f"checkpoint_interval must be >= 0, got {checkpoint_interval}")

    if getattr(model, "is_dual_branch", False):
        return _train_dual_branch_model(
            model=model,
            train_loader=train_loader,
            train_patches=train_patches,
            device=device,
            num_iter=num_iter,
            pretext_step=pretext_step,
            lr=lr,
            see_loss=see_loss,
            wandb_run=wandb_run,
            wandb_prefix=wandb_prefix,
            log_every=log_every,
            best_checkpoint_path=best_checkpoint_path,
            final_checkpoint_path=final_checkpoint_path,
            anchor_augmentation=anchor_augmentation,
            positive_radius=positive_radius,
            positive_mode=positive_mode,
            time_warp_negatives=time_warp_negatives,
            koleo_weight=koleo_weight,
            mask_prediction=mask_prediction,
            mask_prediction_weight=mask_prediction_weight,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            ema_teacher=ema_teacher,
            ema_teacher_weight=ema_teacher_weight,
            ema_teacher_positive=ema_teacher_positive,
            ema_tau_start=ema_tau_start,
            ema_tau_end=ema_tau_end,
            checkpoint_interval=checkpoint_interval,
            checkpoint_callback=checkpoint_callback,
            agree_weight=agree_weight,
            sharp_weight=sharp_weight,
            sharp_mode=sharp_mode,
            sharp_dim=sharp_dim,
            sharp_neigh=sharp_neigh,
            sharp_tau=sharp_tau,
            sharp_anchors=sharp_anchors,
            sharp_easy_neg=sharp_easy_neg,
            sharp_decile=sharp_decile,
            sharp_beta=sharp_beta,
        )

    lambda_weight = 1
    temperature = 1.0
    num_rand_patches = 5
    initial_lr = float(lr)

    def cosine_annealed_lr(iteration, start_lr, end_lr):
        t = min(iteration, num_iter)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / num_iter))
        return end_lr + (start_lr - end_lr) * cosine_factor

    mask_head_params = list(model.mask_prediction_head.parameters())
    mask_head_param_ids = {id(param) for param in mask_head_params}
    base_params = [param for param in model.parameters() if id(param) not in mask_head_param_ids]
    optimizer_param_groups = [{
        "params": base_params,
        "lr": initial_lr,
        "base_lr": initial_lr,
        "final_lr": initial_lr / 10.0,
    }]
    if mask_prediction:
        optimizer_param_groups.append({
            "params": mask_head_params,
            "lr": initial_lr * 5.0,
            "base_lr": initial_lr * 5.0,
            "final_lr": initial_lr * 0.5,
        })

    optimizer = torch.optim.AdamW(optimizer_param_groups, weight_decay=1e-4)
    pos_weight = torch.tensor([1.0], device=device)
    criterion_pretext = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    teacher_wrapper = (
        EMATeacher(
            model,
            tau_start=float(ema_tau_start),
            tau_end=float(ema_tau_end),
            total_steps=num_iter,
        )
        if ema_teacher
        else None
    )

    iteration_count = 0
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_teacher_wts = copy.deepcopy(teacher_wrapper.teacher.state_dict()) if teacher_wrapper is not None else None
    last_loss = float('inf')
    cumulative_train_time_s = 0.0
    display_losses = bool(see_loss)
    last_triplet_loss = 0.0
    last_pllc_loss = 0.0
    last_llb_loss = 0.0
    last_anchor_hit_rate = 0.0
    sum_triplet_loss = 0.0
    sum_pllc_loss = 0.0
    sum_llb_loss = 0.0
    sum_anchor_hit_rate = 0.0
    train_patches = train_patches.to(device=device, non_blocking=True)
    shape_positive_index = None
    shape_positive_stride = None
    peak_gpu_memory_mb = 0.0

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if positive_mode == "shape":
        shape_positive_stride = 1 if train_patches.shape[0] <= SHAPE_POSITIVE_FULL_SCAN_THRESHOLD else SHAPE_POSITIVE_SUBSAMPLE_STRIDE
        print(
            "    >> Precomputing shape-positive index "
            f"(window={SHAPE_POSITIVE_WINDOW}, stride={shape_positive_stride}, patches={train_patches.shape[0]})"
        )
        shape_positive_index = build_shape_positive_index(train_patches, window=SHAPE_POSITIVE_WINDOW)

    print("    [Training Info]")
    pbar = tqdm(total=num_iter, desc="    >> Training", ncols=80)

    while iteration_count < num_iter:
        for batch_data, batch_indexes in train_loader:
            if iteration_count >= num_iter:
                break

            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            iter_start_time = time.perf_counter()
            iteration_count += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = cosine_annealed_lr(
                    iteration_count,
                    start_lr=param_group["base_lr"],
                    end_lr=param_group["final_lr"],
                )

            batch_data = batch_data.to(device, non_blocking=True)
            batch_indexes = batch_indexes.squeeze(-1).to(device=device, non_blocking=True).long()
            anchors = apply_patch_augmentation(batch_data, mode=anchor_augmentation)
            masked_anchors = (
                _patch_mask_augment(
                    batch_data,
                    mask_ratio_min=mask_ratio_min,
                    mask_ratio_max=mask_ratio_max,
                )
                if mask_prediction
                else None
            )
            warped = _time_warp_patches(batch_data) if time_warp_negatives else None
            batch_size = batch_data.shape[0]
            mu = 1 if batch_data.shape[1] != 1 else 10
            total_len = train_patches.shape[0]

            if positive_mode == "shape":
                pos_idx = shape_positive_index.index_select(0, batch_indexes)
            else:
                pos_idx = sample_temporal_positive_indices(
                    batch_indexes=batch_indexes,
                    total_len=total_len,
                    device=device,
                    radius=positive_radius,
                )
            positives = train_patches.index_select(0, pos_idx)
            if anchor_augmentation == "multi":
                positives = apply_patch_augmentation(positives, mode=anchor_augmentation)

            if iteration_count < (num_iter / 10):
                current_lambda_pretext = lambda_weight * (1 - (iteration_count / (num_iter / 10)))
            else:
                current_lambda_pretext = 0.0

            if current_lambda_pretext > 0.0:
                tgt = batch_indexes - pretext_step
                pre_mask = (tgt >= 0) & (tgt < total_len)
                tgt_clamped = tgt.clamp(0, total_len - 1)
                pretext_patches = train_patches.index_select(0, tgt_clamped).clone()
                if (~pre_mask).any():
                    pretext_patches[~pre_mask] = 0
                patch_groups = [anchors, positives, pretext_patches]
                if warped is not None:
                    patch_groups.append(warped)
                all_patches = torch.cat(patch_groups, dim=0)
                all_embeddings = model.embedding(all_patches)
            else:
                pre_mask = None
                patch_groups = [anchors, positives]
                if warped is not None:
                    patch_groups.append(warped)
                all_patches = torch.cat(patch_groups, dim=0)
                all_embeddings = model.embedding(all_patches)

            cursor = 0
            h_anchors = all_embeddings[cursor:cursor + batch_size]
            cursor += batch_size
            h_pos = all_embeddings[cursor:cursor + batch_size]
            cursor += batch_size
            if current_lambda_pretext > 0.0:
                h_pretext = all_embeddings[cursor:cursor + batch_size]
                cursor += batch_size
            else:
                h_pretext = None
            if warped is not None:
                h_warp = all_embeddings[cursor:cursor + batch_size]
                cursor += batch_size
            else:
                h_warp = None
            h_clean_student = h_anchors

            if masked_anchors is not None:
                bn_training_states = []
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        bn_training_states.append((module, module.training))
                        module.eval()
                try:
                    h_masked = model.embedding(masked_anchors)
                finally:
                    for module, was_training in bn_training_states:
                        module.train(was_training)
            else:
                h_masked = None

            z_anchor = F.normalize(model.projection(h_anchors), dim=1)
            if ema_teacher_positive and teacher_wrapper is not None:
                with torch.no_grad():
                    h_teacher_pos = teacher_wrapper.embedding(positives)
                    z_pos = F.normalize(teacher_wrapper.projection(h_teacher_pos), dim=1)
            else:
                z_pos = F.normalize(model.projection(h_pos), dim=1)

            sim_ap = (z_anchor @ z_pos.T) / temperature
            pos_sims = sim_ap.diag()
            sim_ap_fill = sim_ap.clone()
            sim_ap_fill.diagonal().fill_(float('inf'))
            neg_dists = 1 - sim_ap_fill
            hard_neg_dists, _ = torch.max(neg_dists, dim=1)

            pos_dists = 1 - pos_sims
            triplet_loss = F.relu(pos_dists - hard_neg_dists + 0.5).mean() / mu
            triplet_loss = triplet_grad(triplet_loss)

            if current_lambda_pretext > 0.0:
                h_pre = h_pretext[pre_mask]
                h_anchor_pre = h_anchors[pre_mask]
                h_concat_pre = torch.cat([h_anchor_pre, h_pre], dim=1)
                all_indices = torch.arange(batch_size, device=device)
                anchor_indices = all_indices.repeat_interleave(num_rand_patches)
                rand_offsets = torch.randint(1, batch_size, (batch_size * num_rand_patches,), device=device)
                unadj_indices = (anchor_indices + rand_offsets) % batch_size
                h_unadj = h_anchors[unadj_indices]
                h_anchor_unadj = h_anchors.repeat_interleave(num_rand_patches, dim=0)
                h_concat_unadj = torch.cat([h_anchor_unadj, h_unadj], dim=1)
                all_pretext_features = torch.cat([h_concat_pre, h_concat_unadj], dim=0)
                all_pretext_labels = torch.cat([
                    torch.ones(h_concat_pre.size(0), device=device),
                    torch.zeros(h_concat_unadj.size(0), device=device),
                ])
                pretext_outputs = model.classification_head(all_pretext_features).squeeze(1)
                pretext_loss_all = criterion_pretext(pretext_outputs, all_pretext_labels)
                loss_pre = pretext_loss_all[:h_concat_pre.size(0)].mean()
                loss_unadj = pretext_loss_all[h_concat_pre.size(0):].mean()
                pretext_loss = loss_pre + loss_unadj
            else:
                pretext_loss = torch.tensor(0.0, device=device)

            if teacher_wrapper is not None:
                with torch.no_grad():
                    h_teacher_clean = teacher_wrapper.embedding(batch_data)
                    if not ema_teacher_positive:
                        z_teacher = F.normalize(teacher_wrapper.projection(h_teacher_clean), dim=1)
                if ema_teacher_positive:
                    ema_loss = torch.tensor(0.0, device=device)
                else:
                    ema_loss = (1.0 - (z_anchor * z_teacher.detach()).sum(dim=1)).mean()
                mask_target = h_teacher_clean.detach()
            else:
                ema_loss = torch.tensor(0.0, device=device)
                mask_target = h_clean_student.detach()

            if mask_prediction:
                mask_pred = model.mask_prediction(h_masked)
                mask_loss = (1.0 - F.cosine_similarity(mask_pred, mask_target, dim=1)).mean()
            else:
                mask_loss = torch.tensor(0.0, device=device)

            if time_warp_negatives:
                z_warp = F.normalize(model.projection(h_warp), dim=1)
                warp_sim = (z_anchor * z_warp).sum(dim=1)
                warp_loss = F.relu(warp_sim - 0.2).mean()
            else:
                warp_loss = torch.tensor(0.0, device=device)

            if koleo_weight > 0.0:
                koleo_loss = koleo_regularizer(h_anchors)
            else:
                koleo_loss = torch.tensor(0.0, device=device)

            # Task 26k: point-patch agreement loss (positive-only, no negatives).
            # For each timestep t inside a patch of length W, pull the per-timestep
            # projection toward the patch-pooled projection from the SAME patch.
            # Implicitly this forces the conv mid-feature to be internally
            # consistent across scales, closing the c2/c6 readout asymmetry that
            # Task 26j diagnosed.
            if float(agree_weight) > 0.0:
                # Re-run forward for just the anchor batch through the agreement
                # pathway (separate from the pooled-h path used above).
                point_projs, patch_projs = model.forward_agree(anchors)
                # point_projs: [B, W, d], patch_projs: [B, d]
                point_n = F.normalize(point_projs, dim=-1, eps=1e-12)
                patch_n = F.normalize(patch_projs, dim=-1, eps=1e-12)
                cos_t = (point_n * patch_n.unsqueeze(1)).sum(dim=-1)  # [B, W]
                agree_loss = (1.0 - cos_t).mean()
            else:
                agree_loss = torch.tensor(0.0, device=device)

            llb_loss = torch.tensor(0.0, device=device)
            anchor_hit_rate = torch.tensor(0.0, device=device)
            if float(sharp_weight) > 0.0 and str(sharp_mode) == "pllc":
                proj = model.forward_sharp(anchors)
                proj_n = F.normalize(proj, dim=-1, eps=1e-12)
                anchors_t, pos_t, hard_neg_t, easy_neg_idx, hit_mask = pllc_sample(
                    anchors_batch=anchors,
                    W=int(anchors.shape[-1]),
                    D_n=int(sharp_neigh),
                    A=int(sharp_anchors),
                    K=int(sharp_easy_neg),
                    decile=float(sharp_decile),
                )
                pllc_loss = pllc_infonce(
                    proj_n=proj_n,
                    anchors_t=anchors_t,
                    pos_t=pos_t,
                    hard_neg_t=hard_neg_t,
                    easy_neg_idx=easy_neg_idx,
                    tau=float(sharp_tau),
                    hit_mask=hit_mask,
                )
                anchor_hit_rate = hit_mask.any(dim=1).float().mean()
            elif float(sharp_weight) > 0.0 and str(sharp_mode) == "llb":
                proj = model.forward_sharp(anchors)
                delta_feat = (proj[:, 1:, :] - proj[:, :-1, :]).norm(dim=-1)
                raw_signal = anchors[:, 0, :]
                median = raw_signal.median(dim=1, keepdim=True).values
                mad = (raw_signal - median).abs().median(dim=1, keepdim=True).values.clamp_min(1e-6)
                delta_sig = (raw_signal[:, 1:] - raw_signal[:, :-1]).abs() / mad
                llb_loss = F.relu(float(sharp_beta) * delta_sig - delta_feat).pow(2).mean()
                pllc_loss = llb_loss
            else:
                pllc_loss = torch.tensor(0.0, device=device)

            final_loss = (
                triplet_loss
                + current_lambda_pretext * pretext_loss
                + warp_loss
                + (koleo_weight * koleo_loss)
                + (float(mask_prediction_weight) * mask_loss)
                + (float(ema_teacher_weight) * ema_loss)
                + (float(agree_weight) * agree_loss)
                + (float(sharp_weight) * pllc_loss)
            )
            last_loss = final_loss.item()
            last_triplet_loss = float(triplet_loss.item())
            last_pllc_loss = float(pllc_loss.item())
            last_llb_loss = float(llb_loss.item())
            last_anchor_hit_rate = float(anchor_hit_rate.item())
            sum_triplet_loss += last_triplet_loss
            sum_pllc_loss += last_pllc_loss
            sum_llb_loss += last_llb_loss
            sum_anchor_hit_rate += last_anchor_hit_rate

            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            optimizer.step()
            if teacher_wrapper is not None:
                teacher_wrapper.update(model)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            cumulative_train_time_s += time.perf_counter() - iter_start_time
            pbar.update(1)

            if final_loss.item() < best_loss:
                best_loss = final_loss.item()
                best_model_wts = copy.deepcopy(model.state_dict())
                if teacher_wrapper is not None:
                    best_teacher_wts = copy.deepcopy(teacher_wrapper.teacher.state_dict())

            should_log = wandb_run is not None and (
                iteration_count == 1 or iteration_count % log_every == 0 or iteration_count == num_iter
            )
            if should_log:
                wandb_run.log({
                    f"{wandb_prefix}/final_loss": float(final_loss.item()),
                    f"{wandb_prefix}/triplet_loss": float(triplet_loss.item()),
                    f"{wandb_prefix}/pretext_loss": float(pretext_loss.item()),
                    f"{wandb_prefix}/mask_loss": float(mask_loss.item()),
                    f"{wandb_prefix}/ema_loss": float(ema_loss.item()),
                    f"{wandb_prefix}/warp_loss": float(warp_loss.item()),
                    f"{wandb_prefix}/koleo_loss": float(koleo_loss.item()),
                    f"{wandb_prefix}/agree_loss": float(agree_loss.item()),
                    f"{wandb_prefix}/pllc_loss": float(pllc_loss.item()),
                    f"{wandb_prefix}/llb_loss": float(llb_loss.item()),
                    f"{wandb_prefix}/anchor_hit_rate": float(anchor_hit_rate.item()),
                    f"{wandb_prefix}/koleo_weight": float(koleo_weight),
                    f"{wandb_prefix}/mask_prediction_weight": float(mask_prediction_weight),
                    f"{wandb_prefix}/ema_teacher_weight": float(ema_teacher_weight),
                    f"{wandb_prefix}/agree_weight": float(agree_weight),
                    f"{wandb_prefix}/sharp_weight": float(sharp_weight),
                    f"{wandb_prefix}/lambda_pretext": float(current_lambda_pretext),
                    f"{wandb_prefix}/lr": float(optimizer.param_groups[0]["lr"]),
                    f"{wandb_prefix}/best_loss": float(best_loss),
                    f"{wandb_prefix}/iteration": iteration_count,
                }, step=iteration_count)

            if display_losses:
                pbar.set_postfix({
                    "loss": f"{final_loss.item():.4f}",
                    "triplet": f"{triplet_loss.item():.4f}",
                    "pretext": f"{pretext_loss.item():.4f}",
                    "mask": f"{mask_loss.item():.4f}",
                    "ema": f"{ema_loss.item():.4f}",
                    "warp": f"{warp_loss.item():.4f}",
                    "koleo": f"{koleo_loss.item():.4f}",
                    "agree": f"{agree_loss.item():.4f}",
                    "pllc_loss": f"{pllc_loss.item():.4f}",
                    "anchor_hit_rate": f"{anchor_hit_rate.item():.3f}",
                    "best": f"{best_loss:.4f}",
                })

            if (
                checkpoint_callback is not None
                and checkpoint_interval > 0
                and iteration_count % checkpoint_interval == 0
                and iteration_count < num_iter
            ):
                checkpoint_callback(
                    iteration=int(iteration_count),
                    model=model,
                    teacher_model=teacher_wrapper.teacher if teacher_wrapper is not None else None,
                    cumulative_train_time_s=float(cumulative_train_time_s),
                )
                model.train()

    pbar.close()
    model.load_state_dict(best_model_wts)
    teacher_model = None
    if teacher_wrapper is not None:
        teacher_wrapper.teacher.load_state_dict(best_teacher_wts)
        teacher_model = teacher_wrapper.teacher

    if best_checkpoint_path is not None:
        best_checkpoint_path = Path(best_checkpoint_path)
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_wts, best_checkpoint_path)

    if final_checkpoint_path is not None:
        final_checkpoint_path = Path(final_checkpoint_path)
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_checkpoint_path)

    if device.type == "cuda" and torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    return {
        "best_loss": float(best_loss),
        "last_loss": float(last_loss),
        "iterations": int(iteration_count),
        "train_time_cumulative_s": float(cumulative_train_time_s),
        "mean_train_iter_ms": float((cumulative_train_time_s / max(iteration_count, 1)) * 1000.0),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "positive_mode": positive_mode,
        "positive_radius": int(positive_radius),
        "shape_positive_window": int(SHAPE_POSITIVE_WINDOW) if positive_mode == "shape" else None,
        "shape_positive_stride": int(shape_positive_stride) if shape_positive_stride is not None else None,
        "time_warp_negatives": bool(time_warp_negatives),
        "koleo_weight": float(koleo_weight),
        "mask_prediction": bool(mask_prediction),
        "mask_prediction_weight": float(mask_prediction_weight),
        "mask_ratio_min": float(mask_ratio_min),
        "mask_ratio_max": float(mask_ratio_max),
        "ema_teacher": bool(ema_teacher),
        "ema_teacher_weight": float(ema_teacher_weight),
        "ema_teacher_positive": bool(ema_teacher_positive),
        "ema_tau_start": float(ema_tau_start),
        "ema_tau_end": float(ema_tau_end),
        "agree_weight": float(agree_weight),
        "sharp_weight": float(sharp_weight),
        "sharp_mode": str(sharp_mode),
        "sharp_dim": int(sharp_dim),
        "sharp_neigh": int(sharp_neigh),
        "sharp_tau": float(sharp_tau),
        "sharp_anchors": int(sharp_anchors),
        "sharp_easy_neg": int(sharp_easy_neg),
        "sharp_decile": float(sharp_decile),
        "sharp_beta": float(sharp_beta),
        "triplet_loss_last": float(last_triplet_loss),
        "triplet_loss_mean": float(sum_triplet_loss / max(iteration_count, 1)),
        "pllc_loss_last": float(last_pllc_loss),
        "pllc_loss_mean": float(sum_pllc_loss / max(iteration_count, 1)),
        "llb_loss_last": float(last_llb_loss),
        "llb_loss_mean": float(sum_llb_loss / max(iteration_count, 1)),
        "anchor_hit_rate_last": float(last_anchor_hit_rate),
        "anchor_hit_rate_mean": float(sum_anchor_hit_rate / max(iteration_count, 1)),
        "teacher_model": teacher_model,
    }
