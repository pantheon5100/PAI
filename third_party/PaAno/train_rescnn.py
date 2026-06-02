import copy
import math
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn.functional as F


NN_POSITIVE_CHUNK_SIZE = 2048


def build_nn_positive_index(train_patches: torch.Tensor, chunk_size: int = NN_POSITIVE_CHUNK_SIZE) -> torch.Tensor:
    if train_patches.ndim != 3:
        raise ValueError(f"Expected train_patches with shape [N, C, L], got {tuple(train_patches.shape)}")
    num_patches = int(train_patches.shape[0])
    if num_patches == 0:
        return torch.empty(0, dtype=torch.long, device=train_patches.device)
    if num_patches == 1:
        return torch.zeros(1, dtype=torch.long, device=train_patches.device)

    flat = train_patches.reshape(num_patches, -1).to(dtype=torch.float32)
    flat = F.normalize(flat, dim=1, eps=1e-12)
    nn_index = torch.empty(num_patches, dtype=torch.long, device=train_patches.device)
    for start in range(0, num_patches, chunk_size):
        end = min(num_patches, start + chunk_size)
        sims = flat[start:end] @ flat.T
        row_idx = torch.arange(end - start, device=train_patches.device)
        sims[row_idx, start + row_idx] = -1.0
        nn_index[start:end] = sims.argmax(dim=1)
    return nn_index


def nt_xent_loss(z_anchor: torch.Tensor, z_pos: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (z_anchor @ z_pos.T) / temperature
    labels = torch.arange(z_anchor.shape[0], device=z_anchor.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a + loss_b)


def koleo_regularizer(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {tuple(features.shape)}")
    if features.shape[0] < 2:
        return features.new_zeros(())

    normalized = F.normalize(features, dim=1, eps=1e-12)
    pairwise_dist = torch.cdist(normalized, normalized, p=2)
    diagonal_mask = torch.eye(pairwise_dist.shape[0], device=pairwise_dist.device, dtype=torch.bool)
    pairwise_dist = pairwise_dist.masked_fill(diagonal_mask, float("inf"))
    nearest_dist = pairwise_dist.amin(dim=1).clamp_min(eps)
    return -torch.log(nearest_dist).mean()


def train_model(
    model,
    train_loader,
    train_patches,
    device,
    num_iter=200,
    lr=1e-4,
    see_loss=None,
    wandb_run=None,
    wandb_prefix="train",
    log_every=1,
    best_checkpoint_path=None,
    final_checkpoint_path=None,
    koleo_weight=0.0,
):
    if float(koleo_weight) < 0:
        raise ValueError(f"koleo_weight must be >= 0, got {koleo_weight}")
    temperature = 0.2
    initial_lr = lr
    final_lr = lr / 10

    def cosine_annealed_lr(iteration):
        t = min(iteration, num_iter)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / num_iter))
        return final_lr + (initial_lr - final_lr) * cosine_factor

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    iteration_count = 0
    best_loss = float("inf")
    last_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    train_patches = train_patches.to(device=device, non_blocking=True)
    nn_positive_index = build_nn_positive_index(train_patches)

    print("    [Training Info]")
    print(f"    >> Precomputed NN positive index for {train_patches.shape[0]} train patches")
    pbar = tqdm(total=num_iter, desc="    >> Training", ncols=80)

    while iteration_count < num_iter:
        for batch_data, batch_indexes in train_loader:
            if iteration_count >= num_iter:
                break

            iteration_count += 1
            current_lr = cosine_annealed_lr(iteration_count)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            batch_data = batch_data.to(device=device, non_blocking=True)
            batch_indexes = batch_indexes.squeeze(-1).to(device=device, non_blocking=True).long()
            pos_idx = nn_positive_index.index_select(0, batch_indexes)
            positives = train_patches.index_select(0, pos_idx)

            all_patches = torch.cat([batch_data, positives], dim=0)
            all_embeddings = model.embedding(all_patches)
            h_anchor = all_embeddings[: batch_data.shape[0]]
            h_pos = all_embeddings[batch_data.shape[0] :]

            z_anchor = F.normalize(model.projection(h_anchor), dim=1, eps=1e-12)
            z_pos = F.normalize(model.projection(h_pos), dim=1, eps=1e-12)
            contrastive_loss = nt_xent_loss(z_anchor, z_pos, temperature=temperature)
            if koleo_weight > 0.0:
                koleo_loss = koleo_regularizer(h_anchor)
            else:
                koleo_loss = torch.tensor(0.0, device=device)
            final_loss = contrastive_loss + (float(koleo_weight) * koleo_loss)
            last_loss = float(final_loss.item())

            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            optimizer.step()
            pbar.update(1)

            if final_loss.item() < best_loss:
                best_loss = float(final_loss.item())
                best_model_wts = copy.deepcopy(model.state_dict())

            should_log = wandb_run is not None and (
                iteration_count == 1 or iteration_count % log_every == 0 or iteration_count == num_iter
            )
            if should_log:
                wandb_run.log(
                    {
                        f"{wandb_prefix}/final_loss": float(final_loss.item()),
                        f"{wandb_prefix}/contrastive_loss": float(contrastive_loss.item()),
                        f"{wandb_prefix}/koleo_loss": float(koleo_loss.item()),
                        f"{wandb_prefix}/koleo_weight": float(koleo_weight),
                        f"{wandb_prefix}/lr": float(current_lr),
                        f"{wandb_prefix}/best_loss": float(best_loss),
                        f"{wandb_prefix}/iteration": iteration_count,
                    },
                    step=iteration_count,
                )

            if see_loss:
                pbar.set_postfix(
                    {
                        "loss": f"{final_loss.item():.4f}",
                        "ntxent": f"{contrastive_loss.item():.4f}",
                        "koleo": f"{koleo_loss.item():.4f}",
                        "best": f"{best_loss:.4f}",
                    }
                )

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

    return {
        "best_loss": float(best_loss),
        "last_loss": float(last_loss),
        "iterations": int(iteration_count),
        "positive_mode": "nn_positive",
        "koleo_weight": float(koleo_weight),
    }
