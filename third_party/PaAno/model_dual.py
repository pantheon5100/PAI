import torch
import torch.nn as nn

from utils.utils import RevIN1d


DUAL_ENCODER_VARIANTS = {"db_s2_base", "db_s2_pllc", "db_s0_pllc"}


def _make_conv_blocks(in_channels, layers, kss):
    blocks = nn.ModuleList()
    current_in = int(in_channels)
    for out_channels, kernel_size in zip(layers, kss, strict=True):
        blocks.append(
            nn.Sequential(
                nn.Conv1d(
                    current_in,
                    int(out_channels),
                    kernel_size=int(kernel_size),
                    stride=1,
                    padding=int(kernel_size) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(int(out_channels)),
                nn.ReLU(inplace=True),
            )
        )
        current_in = int(out_channels)
    return blocks


def _make_projector(input_dim, projection_dim):
    return nn.Sequential(
        nn.Linear(int(input_dim), int(projection_dim)),
        nn.ReLU(),
        nn.Linear(int(projection_dim), int(projection_dim)),
    )


def _make_sharp_head(input_dim, sharp_dim):
    hidden = max(int(input_dim), int(sharp_dim)) * 2
    return nn.Sequential(
        nn.Linear(int(input_dim), hidden),
        nn.GELU(),
        nn.Linear(hidden, int(sharp_dim)),
    )


class DualBranchPatchEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        projection_dim=256,
        layers=[128, 256, 128, 64],
        kss=[7, 5, 3, 3],
        use_revin=True,
        revin_affine=False,
        revin_eps=1e-5,
        revin_min_sigma=1e-5,
        share_blocks=2,
        detach_local_stem=True,
        sharp_mode="off",
        sharp_dim=64,
    ):
        super().__init__()
        if int(share_blocks) not in {0, 2}:
            raise ValueError(f"share_blocks must be 0 or 2, got {share_blocks}")
        if sharp_mode not in {"off", "pllc", "llb"}:
            raise ValueError(f"sharp_mode must be one of 'off'/'pllc'/'llb', got {sharp_mode}")

        self.is_dual_branch = True
        self.layers = list(layers)
        self.kss = list(kss)
        self.projection_dim = int(projection_dim)
        self.share_blocks = int(share_blocks)
        self.detach_local_stem = bool(detach_local_stem)
        self.sharp_mode = str(sharp_mode)
        self.sharp_dim = int(sharp_dim)
        self.agree_mode = "off"

        self.revin = None
        if use_revin:
            self.revin = RevIN1d(
                num_channels=in_channels,
                eps=revin_eps,
                min_sigma=revin_min_sigma,
                affine=revin_affine,
            )

        shared_layers = self.layers[: self.share_blocks]
        shared_kss = self.kss[: self.share_blocks]
        tail_layers = self.layers[self.share_blocks :]
        tail_kss = self.kss[self.share_blocks :]

        self.shared_blocks = _make_conv_blocks(in_channels, shared_layers, shared_kss)
        tail_in = in_channels if self.share_blocks == 0 else self.layers[self.share_blocks - 1]
        self.global_tail = _make_conv_blocks(tail_in, tail_layers, tail_kss)
        self.local_tail = _make_conv_blocks(tail_in, tail_layers, tail_kss)

        self.global_gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.local_gap = nn.AdaptiveAvgPool1d(output_size=1)

        last_dim = int(self.layers[-1])
        self.global_projection_head = _make_projector(last_dim, self.projection_dim)
        self.local_projection_head = _make_projector(last_dim, self.projection_dim)
        self.global_classification_head = nn.Linear(last_dim * 2, 1)
        self.local_classification_head = nn.Linear(last_dim * 2, 1)

        self.local_sharp_head = None
        if self.sharp_mode != "off":
            self.local_sharp_head = _make_sharp_head(last_dim, self.sharp_dim)

    def _normalize_input(self, x):
        if self.revin is not None:
            x = self.revin.norm(x)
        return x

    def _apply_blocks(self, x, blocks):
        for block in blocks:
            x = block(x)
        return x

    def feature_map(self, x, branch="global"):
        branch = str(branch)
        if branch not in {"global", "local"}:
            raise ValueError(f"branch must be 'global' or 'local', got {branch}")

        x = self._normalize_input(x)
        stem = self._apply_blocks(x, self.shared_blocks)
        if branch == "global":
            return self._apply_blocks(stem, self.global_tail)
        local_input = stem.detach() if (self.share_blocks > 0 and self.detach_local_stem) else stem
        return self._apply_blocks(local_input, self.local_tail)

    def embedding(self, x, branch="global"):
        fmap = self.feature_map(x, branch=branch)
        pool = self.global_gap if str(branch) == "global" else self.local_gap
        return pool(fmap).flatten(start_dim=1)

    def projection(self, h, branch="global"):
        if str(branch) == "global":
            return self.global_projection_head(h)
        if str(branch) == "local":
            return self.local_projection_head(h)
        raise ValueError(f"branch must be 'global' or 'local', got {branch}")

    def classification_logits(self, features, branch="global"):
        if str(branch) == "global":
            return self.global_classification_head(features)
        if str(branch) == "local":
            return self.local_classification_head(features)
        raise ValueError(f"branch must be 'global' or 'local', got {branch}")

    def forward(self, x, return_embedding=False, return_projection=False, branch="global"):
        h = self.embedding(x, branch=branch)
        if return_embedding:
            return h
        if return_projection:
            return self.projection(h, branch=branch)
        raise ValueError("The forward method is not designed to handle classification directly.")

    def forward_sharp(self, x, branch="local"):
        if self.sharp_mode == "off":
            raise RuntimeError("forward_sharp called with sharp_mode='off'")
        if str(branch) != "local":
            raise ValueError(f"DualBranchPatchEncoder.forward_sharp only supports branch='local', got {branch}")
        fmap = self.feature_map(x, branch="local")
        m_bt = fmap.transpose(1, 2).contiguous()
        return self.local_sharp_head(m_bt)


def build_encoder(
    in_channels,
    use_revin,
    encoder_variant="apure",
    agree_mode="off",
    agree_dim=64,
    sharp_mode="off",
    sharp_dim=64,
):
    variant = str(encoder_variant)
    if variant == "apure":
        from model import PatchEncoder

        return PatchEncoder(
            in_channels=int(in_channels),
            use_revin=bool(use_revin),
            agree_mode=str(agree_mode),
            agree_dim=int(agree_dim),
            sharp_mode=str(sharp_mode),
            sharp_dim=int(sharp_dim),
        )

    if variant == "db_s2_base":
        if float(sharp_dim) < 1:
            raise ValueError(f"sharp_dim must be >= 1, got {sharp_dim}")
        return DualBranchPatchEncoder(
            in_channels=int(in_channels),
            use_revin=bool(use_revin),
            share_blocks=2,
            detach_local_stem=True,
            sharp_mode="off",
            sharp_dim=int(sharp_dim),
        )

    if variant == "db_s2_pllc":
        return DualBranchPatchEncoder(
            in_channels=int(in_channels),
            use_revin=bool(use_revin),
            share_blocks=2,
            detach_local_stem=True,
            sharp_mode=str(sharp_mode),
            sharp_dim=int(sharp_dim),
        )

    if variant == "db_s0_pllc":
        return DualBranchPatchEncoder(
            in_channels=int(in_channels),
            use_revin=bool(use_revin),
            share_blocks=0,
            detach_local_stem=False,
            sharp_mode=str(sharp_mode),
            sharp_dim=int(sharp_dim),
        )

    raise ValueError(f"Unsupported encoder_variant: {variant}")
