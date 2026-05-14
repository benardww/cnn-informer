import copy
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.dataset import make_patient_dataloaders
from models.cnn_informer import CNNInformer
from training.loss import make_weighted_loss


def _train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / max(len(loader.dataset), 1)


def _eval_epoch(model, loader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray]:
    """返回 (mean_loss, labels[N], probs[N])，probs 为发作类概率。"""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    probs  = np.concatenate(all_probs)  if all_probs  else np.array([])
    return total_loss / max(len(loader.dataset), 1), labels, probs


def train_patient(
    patient: str,
    data_root: Path,
    device: str = config.DEVICE,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    train_ratio: float = config.TRAIN_RATIO,
) -> Tuple[torch.nn.Module, Dict]:
    """
    对单个患者进行完整训练流程。
    返回 (best_model, val_info)。
    val_info 包含: probs, labels, true_events, window_times, total_hours
    """
    print(f"[{patient}] 加载数据集...")
    train_loader, val_loader, val_meta = make_patient_dataloaders(
        patient, data_root, train_ratio=train_ratio, batch_size=batch_size
    )

    if len(train_loader.dataset) == 0:
        raise ValueError(f"[{patient}] 训练集为空")

    train_labels = train_loader.dataset.get_labels()
    criterion = make_weighted_loss(train_labels, device=device)

    model = CNNInformer(
        n_channels=len(config.CHANNELS),
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.INFORMER_LAYERS,
        d_ff=config.D_FF,
        factor=config.PROB_FACTOR,
        dropout=config.DROPOUT,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=False)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    patience_limit = 10

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_labels, val_probs = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"  Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  早停触发（epoch {epoch}）")
                break

    model.load_state_dict(best_state)

    # 用最优模型重新推理验证集，获取最终概率序列
    _, final_labels, final_probs = _eval_epoch(model, val_loader, criterion, device)

    val_info = {
        "probs":        final_probs,
        "labels":       final_labels.astype(int),
        "true_events":  val_meta["true_events"],
        "window_times": val_meta["window_times"],
        "total_hours":  val_meta["total_hours"],
    }
    return model, val_info
