from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.summary_parser import parse_summary
from data.edf_loader import load_edf, load_edf_header, ChannelMismatchError
from preprocessing.dwt_filter import dwt_bandpass


def _apply_dwt_to_file(data: np.ndarray) -> np.ndarray:
    """对完整 EDF 文件数据 [18, N] 逐通道做 DWT，返回 [18, N] float32。"""
    return np.stack([dwt_bandpass(data[i]) for i in range(data.shape[0])]).astype(np.float32)


class CHBMITPatientDataset(Dataset):
    """
    针对单个 CHB-MIT 患者的在线窗口化数据集。
    EDF 文件首次访问时加载并做 DWT，缓存到内存；后续窗口直接切片，不重复计算。
    不写磁盘。

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
        # EDF 文件级缓存：{path: [18, N] float32}（已做 DWT）
        self._cache: Dict[Path, np.ndarray] = {}

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

    def _get_cached(self, edf_path: Path) -> np.ndarray:
        """
        首次访问时加载整个 EDF 文件并（可选）做全文件 DWT，存入缓存。
        后续直接返回缓存数组，避免重复 I/O 和重复 DWT 计算。
        """
        if edf_path not in self._cache:
            data = load_edf(edf_path, self.channels)  # [18, N] float32
            if self.apply_dwt:
                data = _apply_dwt_to_file(data)       # 每文件只做一次 DWT
            self._cache[edf_path] = data
        return self._cache[edf_path]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        edf_path, start, label, _ = self.windows[idx]
        data = self._get_cached(edf_path)                            # [18, N]
        segment = data[:, start:start + self.window_samples].copy()  # [18, 1024]
        return torch.from_numpy(segment).float(), label

    def get_labels(self) -> List[int]:
        return [w[2] for w in self.windows]

    def get_window_times(self) -> np.ndarray:
        return np.array([w[3] for w in self.windows], dtype=np.float64)


def make_patient_dataloaders(
    patient: str,
    data_root: Path,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    按发作事件时序切分 EDF 文件为 train / val / test 三个集合，
    保证 val 和 test 中各含至少一个发作文件（如发作数允许）。

    返回 (train_loader, val_loader, test_loader, test_meta)。
    test_meta 包含 true_events / total_hours / window_times。
    """
    patient_dir = data_root / patient
    seizure_map = parse_summary(patient_dir)

    # 按 summary 顺序排列所有存在的 EDF 文件
    all_files = [f for f in seizure_map.keys() if (patient_dir / f).exists()]
    if not all_files:
        raise FileNotFoundError(f"在 {patient_dir} 中找不到任何 EDF 文件")

    # 将 EDF 文件分为"含发作"和"不含发作"两组，各自保持时间顺序
    sz_files     = [f for f in all_files if seizure_map.get(f)]
    non_sz_files = [f for f in all_files if not seizure_map.get(f)]
    n_sz = len(sz_files)

    # Step 1: 按比例分配含发作文件到 train / val / test
    if n_sz >= 3:
        n_train, n_val = n_sz - 2, 1   # 前 N-2 → train，倒数第2 → val，最后1 → test
    elif n_sz == 2:
        n_train, n_val = 1, 0           # 1→train，1→test；val 后续用 train 末尾
    elif n_sz == 1:
        n_train, n_val = 1, 0
    else:
        # 无发作：70/15/15 时序切分
        n = len(all_files)
        train_files = all_files[: int(n * 0.70)]
        val_files   = all_files[int(n * 0.70) : int(n * 0.85)]
        test_files  = all_files[int(n * 0.85):]
        if not val_files:
            val_files = train_files[-1:]
        if not test_files:
            test_files = all_files[-1:]
        train_ds = CHBMITPatientDataset(patient_dir, train_files, seizure_map)
        val_ds   = CHBMITPatientDataset(patient_dir, val_files,   seizure_map)
        test_ds  = CHBMITPatientDataset(patient_dir, test_files,  seizure_map)
        # 直接跳到 DataLoader 构建
        n_sz = -1   # 标记已处理

    if n_sz >= 0:
        train_sz = list(sz_files[:n_train])
        val_sz   = list(sz_files[n_train : n_train + n_val])
        test_sz  = list(sz_files[n_train + n_val :])

        # Step 2: 以时间位置边界将非发作文件填充到对应子集
        order = {f: i for i, f in enumerate(all_files)}
        train_boundary = order[train_sz[-1]]
        val_boundary   = order[val_sz[-1]] if val_sz else train_boundary

        for f in non_sz_files:
            pos = order[f]
            if pos <= train_boundary:
                train_sz.append(f)
            elif pos <= val_boundary:
                val_sz.append(f)
            else:
                test_sz.append(f)

        # Step 3: 各子集内按时间顺序排序
        train_files = sorted(train_sz, key=lambda f: order[f])
        val_files   = sorted(val_sz,   key=lambda f: order[f])
        test_files  = sorted(test_sz,  key=lambda f: order[f])

        # Step 4: 保底处理
        if not val_files:
            val_files = train_files[-1:]
        if not test_files:
            test_files = all_files[-1:]

        print(f"  [{patient}] 切分 — train:{len(train_files)}文件"
              f"(发作{sum(bool(seizure_map.get(f)) for f in train_files)}次) | "
              f"val:{len(val_files)}文件"
              f"(发作{sum(bool(seizure_map.get(f)) for f in val_files)}次) | "
              f"test:{len(test_files)}文件"
              f"(发作{sum(bool(seizure_map.get(f)) for f in test_files)}次)")

        train_ds = CHBMITPatientDataset(patient_dir, train_files, seizure_map)
        val_ds   = CHBMITPatientDataset(patient_dir, val_files,   seizure_map)
        test_ds  = CHBMITPatientDataset(patient_dir, test_files,  seizure_map)

    # WeightedRandomSampler 仅用于训练集
    train_labels = train_ds.get_labels()
    counts = np.bincount(train_labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
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
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # 测试集元信息（事件级评估用）
    test_seizure_events: List[Tuple[float, float]] = []
    test_total_seconds = 0.0
    for fname in test_files:
        test_seizure_events.extend(seizure_map.get(fname, []))
        try:
            n_samples = load_edf_header(patient_dir / fname)
            test_total_seconds += n_samples / config.SFREQ
        except Exception:
            pass

    test_meta = {
        "true_events": test_seizure_events,
        "total_hours": test_total_seconds / 3600.0,
        "window_times": test_ds.get_window_times(),
    }

    return train_loader, val_loader, test_loader, test_meta
