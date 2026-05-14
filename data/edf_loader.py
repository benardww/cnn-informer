import re
from pathlib import Path
from typing import List

import mne
import numpy as np

mne.set_log_level("ERROR")


class ChannelMismatchError(Exception):
    pass


def _normalize_name(raw_name: str) -> str:
    """将 MNE 通道名标准化，去除前缀/后缀，便于匹配。"""
    name = raw_name.strip().upper()
    name = re.sub(r"^EEG\s+", "", name)
    name = re.sub(r"[-_](REF|LE|AVG)$", "", name)
    return name


def _match_channels(available_raw: List[str], targets: List[str]) -> List[str]:
    """
    将目标通道列表映射到 available_raw 中的真实通道名。
    如果任何目标找不到，抛出 ChannelMismatchError。
    """
    norm_to_raw = {_normalize_name(ch): ch for ch in available_raw}
    matched = []
    missing = []
    for t in targets:
        norm_t = _normalize_name(t)
        # 精确匹配
        if norm_t in norm_to_raw:
            matched.append(norm_to_raw[norm_t])
        else:
            # 模糊匹配：去掉连字符后匹配（e.g. P3O1 vs P3-O1）
            norm_no_dash = norm_t.replace("-", "")
            found = None
            for k, v in norm_to_raw.items():
                if k.replace("-", "") == norm_no_dash:
                    found = v
                    break
            if found:
                matched.append(found)
            else:
                missing.append(t)
    if missing:
        raise ChannelMismatchError(f"以下通道未找到: {missing}")
    return matched


def load_edf(edf_path: Path, target_channels: List[str]) -> np.ndarray:
    """
    读取 EDF 文件并返回 shape [len(target_channels), n_samples] 的 float32 数组（单位：µV）。
    如果任意目标通道缺失，抛出 ChannelMismatchError。
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    available = raw.ch_names
    matched_names = _match_channels(available, target_channels)
    data, _ = raw[matched_names]  # shape [n_ch, n_samples], 单位 V
    return (data * 1e6).astype(np.float32)  # 转换为 µV


def load_edf_header(edf_path: Path) -> int:
    """仅读取头部，返回采样点数（不加载数据到内存）。"""
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    return raw.n_times


def get_edf_slice(
    edf_path: Path,
    target_channels: List[str],
    start: int,
    stop: int,
) -> np.ndarray:
    """
    高效读取 EDF 中 [start, stop) 采样点的数据。
    返回 shape [len(target_channels), stop-start] float32（µV）。
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    available = raw.ch_names
    matched_names = _match_channels(available, target_channels)
    raw.load_data()
    picks = mne.pick_channels(raw.ch_names, include=matched_names, ordered=True)
    data = raw.get_data(picks=picks, start=start, stop=stop)
    return (data * 1e6).astype(np.float32)
