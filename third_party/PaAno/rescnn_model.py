import torch
import torch.nn as nn

from utils.utils import RevIN1d


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 64,
        projection_dim: int = 128,
        use_revin: bool = False,
        revin_affine: bool = False,
        revin_eps: float = 1e-5,
        revin_min_sigma: float = 1e-5,
    ):
        super().__init__()
        self.revin = None
        if use_revin:
            self.revin = RevIN1d(
                num_channels=in_channels,
                eps=revin_eps,
                min_sigma=revin_min_sigma,
                affine=revin_affine,
            )

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 64, kernel_size=7, stride=1),
            ResidualBlock1D(64, 128, kernel_size=5, stride=2),
            ResidualBlock1D(128, 128, kernel_size=3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_head = nn.Linear(128, embedding_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False, return_projection: bool = False) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin.norm(x)
        x = self.stem(x)
        x = self.blocks(x)
        h = self.pool(x).flatten(start_dim=1)
        h = self.embedding_head(h)
        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)
        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)

    def projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.projection_head(h)
