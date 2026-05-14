from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.summary_parser import parse_summary
from data.edf_loader import load_edf_header, get_edf_slice, ChannelMismatchError
from preprocessing.dwt_filter import apply_dwt_to_segment


class CHBMITPatientDataset(Dataset):
    """
    针对单个 CHB-MIT 患者的在线窗口化数据集。
    DWT 滤波在 __getitem__ 时即时计算，不缓存到磁盘。

    windows 列表存储 (edf_path, start_sample, label, window_start_sec)。
    """

    def __init__(
        self,
        patient_dir: Path,
        edf_files: List[str],
        seizure_map: Dict[str, List[Tuple[int, int]]],
        channels: List[str] = None,
        window_samples: int = config.WINDOW_SAMPLES,
        seizure_stride: int = config.SEIZURE_STRIDE,
        non_seizure_stride: int = config.NON_SEIZURE_STRIDE,
        apply_dwt: bool = True,
    ):
        self.patient_dir = patient_dir
        self.channels = channels or config.CHANNELS
        self.window_samples = window_samples
        self.apply_dwt = apply_dwt

        # windows: List[(edf_path, start_sample, label, window_start_sec)]
        self.windows: List[Tuple[Path, int, int, float]] = []
        self._build_index(edf_files, seizure_map, seizure_stride, non_seizure_stride)

    def _build_index(
        self,
        edf_files: List[str],
        seizure_map: Dict[str, List[Tuple[int, int]]],
        seizure_stride: int,
        non_seizure_stride: int,
    ) -> None:
        W = self.window_samples
        sfreq = config.SFREQ

        for fname in edf_files:
            edf_path = self.patient_dir / fname
            if not edf_path.exists():
                continue

            try:
                n_samples = load_edf_header(edf_path)
            except Exception:
                continue

            seizures = seizure_map.get(fname, [])
            # 转换为采样点区间
            sz_intervals = [(int(s * sfreq), int(e * sfreq)) for s, e in seizures]

            def _overlaps(t: int) -> bool:
                for s, e in sz_intervals:
                    if t < e and t + W > s:
                        return True
                return False

            # Pass 1: 发作窗口（stride=512）
            for sz_start, sz_end in sz_intervals:
                t = max(0, sz_start)
                while t + W <= min(n_samples, sz_end + W):
                    if t + W > n_samples:
                        break
                    if _overlaps(t):
                        self.windows.append((edf_path, t, 1, t / sfreq))
                    t += seizure_stride

            # Pass 2: 非发作窗口（stride=1024）
            t = 0
            while t + W <= n_samples:
                if not _overlaps(t):
                    self.windows.append((edf_path, t, 0, t / sfreq))
                t += non_seizure_stride

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        edf_path, start, label, _ = self.windows[idx]
        segment = get_edf_slice(edf_path, self.channels, start, start + self.window_samples)
        if self.apply_dwt:
            segment = apply_dwt_to_segment(segment)
        return torch.from_numpy(segment).float(), label

    def get_labels(self) -> List[int]:
        return [w[2] for w in self.windows]

    def get_window_times(self) -> np.ndarray:
        return np.array([w[3] for w in self.windows], dtype=np.float64)


def make_patient_dataloaders(
    patient: str,
    data_root: Path,
    train_ratio: float = config.TRAIN_RATIO,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    按时间顺序切分 EDF 文件（前 train_ratio → 训练，其余 → 验证）。
    返回 (train_loader, val_loader, val_meta)。
    val_meta 包含 true_events 和 total_hours 用于事件级评估。
    """
    patient_dir = data_root / patient
    seizure_map = parse_summary(patient_dir)

    # EDF 文件按 summary 中的顺序排列（即时间顺序）
    all_files = [f for f in seizure_map.keys() if (patient_dir / f).exists()]
    if not all_files:
        raise FileNotFoundError(f"在 {patient_dir} 中找不到任何 EDF 文件")

    split_idx = max(1, int(len(all_files) * train_ratio))
    train_files = all_files[:split_idx]
    val_files   = all_files[split_idx:] if split_idx < len(all_files) else all_files[-1:]

    train_ds = CHBMITPatientDataset(patient_dir, train_files, seizure_map)
    val_ds   = CHBMITPatientDataset(patient_dir, val_files,   seizure_map)

    # 使用 WeightedRandomSampler 缓解类别不平衡
    train_labels = train_ds.get_labels()
    counts = np.bincount(train_labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)  # 避免除零
    class_weights = 1.0 / counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # 验证集元信息（用于事件级评估）
    val_seizure_events: List[Tuple[float, float]] = []
    val_total_seconds = 0.0
    for fname in val_files:
        val_seizure_events.extend(seizure_map.get(fname, []))
        try:
            n_samples = load_edf_header(patient_dir / fname)
            val_total_seconds += n_samples / config.SFREQ
        except Exception:
            pass

    val_meta = {
        "true_events": val_seizure_events,
        "total_hours": val_total_seconds / 3600.0,
        "window_times": val_ds.get_window_times(),
    }

    return train_loader, val_loader, val_meta
