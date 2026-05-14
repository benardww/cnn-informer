import torch
import torch.nn as nn


class CNNModule(nn.Module):
    """
    三层一维卷积特征提取器。
    输入: [B, 18, 1024]
    输出: [B, 63, 64]  (转置后作为 Informer 的序列输入)

    尺寸验证:
      Layer1: Conv(k=4,s=2) → (1024-4)/2+1=511 → MaxPool(2,2) → 255
      Layer2: Conv(k=4,s=2) → (255-4)/2+1=126  → MaxPool(2,2) → 63
      Layer3: Conv(k=3,s=1,p=1) → 63
    """

    def __init__(self, n_channels: int = 18, d_model: int = 64, dropout: float = 0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=4, stride=2),
            nn.BatchNorm1d(n_channels),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=4, stride=2),
            nn.BatchNorm1d(n_channels),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)         # [B, 18, 255]
        x = self.layer2(x)         # [B, 18, 63]
        x = self.layer3(x)         # [B, 64, 63]
        return x.transpose(1, 2)   # [B, 63, 64]
