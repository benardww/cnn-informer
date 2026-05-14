import torch
import torch.nn as nn
from models.prob_attention import ProbAttention


class DistillationLayer(nn.Module):
    """
    自注意力蒸馏层：Conv1d → ELU → MaxPool1d(k=3,s=2,p=1)
    将序列长度减半：L → floor((L-1)/2) + 1
      63 → 32, 32 → 16
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model] → [B, ~L/2, d_model]"""
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    """单个 Informer 编码器层：ProbSparse 注意力 + 前馈网络 + LayerNorm。"""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 8,
        d_ff: int = 256,
        factor: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = ProbAttention(d_model, n_heads, factor, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model] → [B, L, d_model]"""
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class InformerEncoder(nn.Module):
    """
    3 层编码器 + 2 层蒸馏。
    序列长度变化: 63 → 32 → 16
    输出: [B, 16, 64]
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        factor: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(n_layers)
        ])
        self.distill_layers = nn.ModuleList([
            DistillationLayer(d_model)
            for _ in range(n_layers - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 63, 64]
        → 编码层1 → 蒸馏1 → [B, 32, 64]
        → 编码层2 → 蒸馏2 → [B, 16, 64]
        → 编码层3          → [B, 16, 64]
        """
        for i, enc in enumerate(self.enc_layers):
            x = enc(x)
            if i < len(self.distill_layers):
                x = self.distill_layers[i](x)
        return x
