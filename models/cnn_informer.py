import torch
import torch.nn as nn
from models.cnn_module import CNNModule
from models.informer_encoder import InformerEncoder


class CNNInformer(nn.Module):
    """
    完整的 CNN-Informer EEG 癫痫检测模型。
    输入:  [B, 18, 1024]
    输出:  [B, num_classes] 原始 logits
    """

    def __init__(
        self,
        n_channels: int = 18,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        factor: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn     = CNNModule(n_channels, d_model, dropout)
        self.encoder = InformerEncoder(d_model, n_heads, n_layers, d_ff, factor, dropout)

        # 蒸馏后序列长度: 16 (由 63→32→16 确定)
        flat_dim = 16 * d_model  # 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)           # [B, 63, 64]
        x = self.encoder(x)       # [B, 16, 64]
        return self.classifier(x) # [B, 2]
