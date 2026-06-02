from typing import Optional

import torch
import torch.nn as nn

from model import MaskPredictionHead
from utils.utils import RevIN1d


def sinusoidal_position_embedding(embed_dim: int, length: int) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"sinusoidal_position_embedding requires even embed_dim, got {embed_dim}")
    positions = torch.arange(length, dtype=torch.float32)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    freqs = torch.outer(positions, omega)
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"rotate_half requires an even hidden size, got {x.shape[-1]}")
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryPositionEmbedding requires even dim, got {dim}")
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, freqs)
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[2]
        cos = self.freqs_cos[:seq_len].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.freqs_sin[:seq_len].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        return apply_rotary_pos_emb(x, cos, sin)


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: RotaryPositionEmbedding):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {self.head_dim}")
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if n_tokens > 1:
            q_cls, q_patches = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patches = k[:, :, :1, :], k[:, :, 1:, :]
            q_patches = self.rope(q_patches)
            k_patches = self.rope(k_patches)
            q = torch.cat([q_cls, q_patches], dim=2)
            k = torch.cat([k_cls, k_patches], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, channels)
        return self.out_proj(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        rope: Optional[RotaryPositionEmbedding] = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")
        mlp_dim = int(d_model * mlp_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.use_rope = bool(use_rope or rope is not None)
        if self.use_rope:
            if rope is None:
                raise ValueError("TransformerBlock with use_rope=True requires a RotaryPositionEmbedding instance")
            self.attn = RoPEMultiheadAttention(d_model, n_heads, rope)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        if self.use_rope:
            x = x + self.attn(attn_input)
        else:
            x = x + self.attn(attn_input, attn_input, attn_input, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PatchTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        projection_dim: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        sub_patch_size: int = 8,
        patch_size: int = 64,
        embed_dim: int = 64,
        pooling: str = "cls",
        pos_encoding: str = "learned",
        use_revin: bool = True,
        revin_affine: bool = False,
        revin_eps: float = 1e-5,
        revin_min_sigma: float = 1e-5,
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")
        if sub_patch_size <= 0:
            raise ValueError(f"sub_patch_size must be > 0, got {sub_patch_size}")
        if patch_size % sub_patch_size != 0:
            raise ValueError(
                f"patch_size must be divisible by sub_patch_size, got {patch_size} and {sub_patch_size}"
            )
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if pos_encoding not in {"learned", "sinusoidal", "rope", "sinusoidal_rope"}:
            raise ValueError(f"Unsupported pos_encoding: {pos_encoding}")

        self.patch_size = int(patch_size)
        self.sub_patch_size = int(sub_patch_size)
        self.n_tokens = self.patch_size // self.sub_patch_size
        self.pooling = pooling
        self.projection_dim = projection_dim
        self.embed_dim = embed_dim
        self.pos_encoding = pos_encoding

        self.revin = None
        if use_revin:
            self.revin = RevIN1d(
                num_channels=in_channels,
                eps=revin_eps,
                min_sigma=revin_min_sigma,
                affine=revin_affine,
            )

        token_dim = in_channels * self.sub_patch_size
        self.token_proj = nn.Linear(token_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if self.pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens + 1, d_model))
        elif self.pos_encoding in {"sinusoidal", "sinusoidal_rope"}:
            pos_embed = sinusoidal_position_embedding(d_model, self.n_tokens + 1).unsqueeze(0)
            self.register_buffer("pos_embed", pos_embed, persistent=False)
        else:
            self.pos_embed = None
        use_rope = self.pos_encoding in {"rope", "sinusoidal_rope"}
        rope = None
        if use_rope:
            rope = RotaryPositionEmbedding(dim=d_model // n_heads, max_seq_len=self.n_tokens + 1)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=4.0,
                    use_rope=use_rope,
                    rope=rope,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.embed_proj = nn.Identity() if d_model == embed_dim else nn.Linear(d_model, embed_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.mask_prediction_head = MaskPredictionHead(embed_dim=embed_dim, hidden_dim=128)
        self.classification_head = nn.Linear(embed_dim * 2, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.token_proj.weight, std=0.02)
        if self.token_proj.bias is not None:
            nn.init.zeros_(self.token_proj.bias)
        if isinstance(self.embed_proj, nn.Linear):
            nn.init.trunc_normal_(self.embed_proj.weight, std=0.02)
            if self.embed_proj.bias is not None:
                nn.init.zeros_(self.embed_proj.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.token_proj or module is self.embed_proj:
                    continue
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, patch_length = x.shape
        if patch_length != self.patch_size:
            raise ValueError(f"Expected patch length {self.patch_size}, got {patch_length}")
        x = x.reshape(batch_size, channels, self.n_tokens, self.sub_patch_size)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.n_tokens, channels * self.sub_patch_size)
        return x

    def forward(self, x: torch.Tensor, return_embedding: bool = False, return_projection: bool = False) -> torch.Tensor:
        h = self.embedding(x)
        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)
        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin.norm(x)

        tokens = self.token_proj(self._tokenize(x))
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        if self.pooling == "cls":
            h = tokens[:, 0]
        else:
            h = tokens[:, 1:].mean(dim=1)

        return self.embed_proj(h)

    def projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.projection_head(h)

    def mask_prediction(self, h: torch.Tensor) -> torch.Tensor:
        return self.mask_prediction_head(h)


class PatchHybridTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        projection_dim: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        sub_patch_size: int = 1,
        patch_size: int = 64,
        embed_dim: int = 64,
        pooling: str = "cls",
        pos_encoding: str = "learned",
        stem_norm: str = "bn",
        stem_depth: int = 2,
        use_revin: bool = True,
        revin_affine: bool = False,
        revin_eps: float = 1e-5,
        revin_min_sigma: float = 1e-5,
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")
        if sub_patch_size <= 0:
            raise ValueError(f"sub_patch_size must be > 0, got {sub_patch_size}")
        if patch_size % sub_patch_size != 0:
            raise ValueError(
                f"patch_size must be divisible by sub_patch_size, got {patch_size} and {sub_patch_size}"
            )
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if stem_norm not in {"bn", "gn"}:
            raise ValueError(f"Unsupported stem_norm: {stem_norm}")
        if stem_depth not in {2, 3}:
            raise ValueError(f"stem_depth must be 2 or 3, got {stem_depth}")
        if pos_encoding not in {"learned", "sinusoidal", "rope", "sinusoidal_rope"}:
            raise ValueError(f"Unsupported pos_encoding: {pos_encoding}")

        self.patch_size = int(patch_size)
        self.sub_patch_size = int(sub_patch_size)
        self.n_tokens = self.patch_size // self.sub_patch_size
        self.pooling = pooling
        self.projection_dim = projection_dim
        self.embed_dim = embed_dim
        self.pos_encoding = pos_encoding
        self.stem_norm = stem_norm
        self.stem_depth = int(stem_depth)
        self.stem_channels = 128 if self.stem_depth == 3 else 256

        self.revin = None
        if use_revin:
            self.revin = RevIN1d(
                num_channels=in_channels,
                eps=revin_eps,
                min_sigma=revin_min_sigma,
                affine=revin_affine,
            )

        stem_layers = [
            self._make_stem_block(in_channels, 128, kernel_size=7, padding=3),
            self._make_stem_block(128, 256, kernel_size=5, padding=2),
        ]
        if self.stem_depth == 3:
            stem_layers.append(self._make_stem_block(256, 128, kernel_size=3, padding=1))
        self.stem = nn.Sequential(*stem_layers)

        token_dim = self.stem_channels * self.sub_patch_size
        self.token_proj = nn.Linear(token_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if self.pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens + 1, d_model))
        elif self.pos_encoding in {"sinusoidal", "sinusoidal_rope"}:
            pos_embed = sinusoidal_position_embedding(d_model, self.n_tokens + 1).unsqueeze(0)
            self.register_buffer("pos_embed", pos_embed, persistent=False)
        else:
            self.pos_embed = None
        use_rope = self.pos_encoding in {"rope", "sinusoidal_rope"}
        rope = None
        if use_rope:
            rope = RotaryPositionEmbedding(dim=d_model // n_heads, max_seq_len=self.n_tokens + 1)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=4.0,
                    use_rope=use_rope,
                    rope=rope,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.embed_proj = nn.Identity() if d_model == embed_dim else nn.Linear(d_model, embed_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.mask_prediction_head = MaskPredictionHead(embed_dim=embed_dim, hidden_dim=128)
        self.classification_head = nn.Linear(embed_dim * 2, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.token_proj.weight, std=0.02)
        if self.token_proj.bias is not None:
            nn.init.zeros_(self.token_proj.bias)
        if isinstance(self.embed_proj, nn.Linear):
            nn.init.trunc_normal_(self.embed_proj.weight, std=0.02)
            if self.embed_proj.bias is not None:
                nn.init.zeros_(self.embed_proj.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.token_proj or module is self.embed_proj:
                    continue
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _make_stem_norm(self, num_channels: int) -> nn.Module:
        if self.stem_norm == "gn":
            num_groups = 16 if num_channels % 16 == 0 and num_channels >= 256 else 8
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        return nn.BatchNorm1d(num_channels)

    def _make_stem_block(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            self._make_stem_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, patch_length = x.shape
        if patch_length != self.patch_size:
            raise ValueError(f"Expected patch length {self.patch_size}, got {patch_length}")
        x = x.reshape(batch_size, channels, self.n_tokens, self.sub_patch_size)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.n_tokens, channels * self.sub_patch_size)
        return x

    def forward(self, x: torch.Tensor, return_embedding: bool = False, return_projection: bool = False) -> torch.Tensor:
        h = self.embedding(x)
        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)
        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin.norm(x)

        stem_features = self.stem(x)
        tokens = self.token_proj(self._tokenize(stem_features))
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        if self.pooling == "cls":
            h = tokens[:, 0]
        else:
            h = tokens[:, 1:].mean(dim=1)

        return self.embed_proj(h)

    def projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.projection_head(h)

    def mask_prediction(self, h: torch.Tensor) -> torch.Tensor:
        return self.mask_prediction_head(h)
