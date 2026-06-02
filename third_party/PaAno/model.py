import torch
import torch.nn as nn
from utils.utils import RevIN1d


class MaskPredictionHead(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, h):
        return self.head(h)


class PatchEncoder(nn.Module): #Simple 1D CNN with RevIN
    def __init__(self, in_channels=1, projection_dim=256, layers=[128, 256, 128, 64],
                 kss=[7, 5, 3, 3],
                 use_revin: bool = True,
                 revin_affine: bool = False,
                 revin_eps: float = 1e-5,
                 revin_min_sigma: float = 1e-5,
                 # Task 26k: Point-patch agreement heads for dense-SSL objective.
                 # When `agree_mode == "projector"`, two small MLP heads project the
                 # per-timestep mid-feature and the pooled patch feature into a
                 # common d-dim space for a cosine-agreement loss. When
                 # `agree_mode == "raw"`, no heads are created and the agreement
                 # is computed directly on the raw mid-feature. Default `"off"`
                 # preserves A_pure behaviour (no extra parameters, no extra loss).
                 agree_mode: str = "off",
                 agree_dim: int = 64,
                 sharp_mode: str = "off",
                 sharp_dim: int = 64,
                 ):
        super(PatchEncoder, self).__init__()
        self.layers = layers
        self.kss = kss
        self.projection_dim = projection_dim

        # RevIN
        self.revin = None
        if use_revin:
            self.revin = RevIN1d(num_channels=in_channels,
                                 eps=revin_eps,
                                 min_sigma=revin_min_sigma,
                                 affine=revin_affine)

        #  Conv blocks
        self.convblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(layers[i - 1] if i > 0 else in_channels, self.layers[i],
                          kernel_size=self.kss[i], stride=1, padding=self.kss[i] // 2, bias=False),
                nn.BatchNorm1d(self.layers[i]),
                nn.ReLU(inplace=True)
            ) for i in range(len(self.layers))
        ])

        # Heads
        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(self.layers[-1], self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )
        self.mask_prediction_head = MaskPredictionHead(embed_dim=self.layers[-1], hidden_dim=128)
        self.classification_head = nn.Linear(self.layers[-1]*2, 1)

        # Task 26k agreement heads (optional; only built when agree_mode == "projector")
        if agree_mode not in {"off", "projector", "raw"}:
            raise ValueError(f"agree_mode must be one of 'off'/'projector'/'raw', got {agree_mode}")
        self.agree_mode = str(agree_mode)
        self.agree_dim = int(agree_dim)
        self.agree_point_head = None
        self.agree_patch_head = None
        if self.agree_mode == "projector":
            hidden = max(self.layers[-1], self.agree_dim) * 2
            self.agree_point_head = nn.Sequential(
                nn.Linear(self.layers[-1], hidden),
                nn.GELU(),
                nn.Linear(hidden, self.agree_dim),
            )
            self.agree_patch_head = nn.Sequential(
                nn.Linear(self.layers[-1], hidden),
                nn.GELU(),
                nn.Linear(hidden, self.agree_dim),
            )

        if sharp_mode not in {"off", "pllc", "llb"}:
            raise ValueError(f"sharp_mode must be one of 'off'/'pllc'/'llb', got {sharp_mode}")
        self.sharp_mode = str(sharp_mode)
        self.sharp_dim = int(sharp_dim)
        self.sharp_head = None
        if self.sharp_mode != "off":
            hidden = max(self.layers[-1], self.sharp_dim) * 2
            self.sharp_head = nn.Sequential(
                nn.Linear(self.layers[-1], hidden),
                nn.GELU(),
                nn.Linear(hidden, self.sharp_dim),
            )

    def feature_map(self, x, branch="global"):
        if str(branch) != "global":
            raise ValueError(f"PatchEncoder only supports branch='global', got {branch}")
        if self.revin is not None:
            x = self.revin.norm(x)

        for block in self.convblocks:
            x = block(x)
        return x

    def forward(self, x, return_embedding=False, return_projection=False, branch="global"):
        x = self.feature_map(x, branch=branch)

        h = self.fc_embedding(x).flatten(start_dim=1)  # (N, D)

        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)

        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x, branch="global"):
        return self.forward(x, return_embedding=True, branch=branch)

    def projection(self, h, branch="global"):
        if str(branch) != "global":
            raise ValueError(f"PatchEncoder only supports branch='global', got {branch}")
        return self.projection_head(h)

    def mask_prediction(self, h):
        return self.mask_prediction_head(h)

    def classification_logits(self, features, branch="global"):
        if str(branch) != "global":
            raise ValueError(f"PatchEncoder only supports branch='global', got {branch}")
        return self.classification_head(features)

    def forward_agree(self, x):
        """Compute point-scale and patch-scale projections for L_agree (Task 26k).

        Does an independent forward pass through the conv stack (so it is safe
        to call alongside ``embedding()`` for the same batch; they do not share
        activations but BatchNorm uses training-mode stats either way).

        Returns
        -------
        point_projs : Tensor of shape ``[N, W, d]``
            Per-timestep projection of the mid-feature map.
        patch_projs : Tensor of shape ``[N, d]``
            Projection of the globally-average-pooled mid-feature.

        If ``agree_mode == "raw"``, the projections are the raw mid-feature and
        its pooled version (``d = D``). If ``agree_mode == "projector"``, both
        go through an MLP into a common ``agree_dim``-dim space.
        """
        if self.agree_mode == "off":
            raise RuntimeError("forward_agree called with agree_mode='off'")

        m = self.feature_map(x)
        h_pool = self.gap(m).flatten(start_dim=1)  # [N, D]
        m_bt = m.transpose(1, 2).contiguous()      # [N, W, D]

        if self.agree_mode == "projector":
            point_projs = self.agree_point_head(m_bt)   # [N, W, d]
            patch_projs = self.agree_patch_head(h_pool) # [N, d]
        else:
            # "raw": agreement is enforced directly on the mid-feature space.
            point_projs = m_bt    # [N, W, D]
            patch_projs = h_pool  # [N, D]
        return point_projs, patch_projs

    def forward_sharp(self, x, branch="global"):
        """Return per-timestep projections for sharpness-aware auxiliary losses."""
        if self.sharp_mode == "off":
            raise RuntimeError("forward_sharp called with sharp_mode='off'")
        if str(branch) != "global":
            raise ValueError(f"PatchEncoder only supports branch='global', got {branch}")

        x = self.feature_map(x)
        m_bt = x.transpose(1, 2).contiguous()  # [N, W, D]
        return self.sharp_head(m_bt)
