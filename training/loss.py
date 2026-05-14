import numpy as np
import torch
import torch.nn as nn


def make_weighted_loss(labels: list, device: str = "cpu") -> nn.CrossEntropyLoss:
    """根据训练集标签分布计算逆频率类别权重的交叉熵损失。"""
    counts = np.bincount(labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(labels) / (2.0 * counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)
