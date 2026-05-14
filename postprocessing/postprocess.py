from typing import List, Tuple

import numpy as np


def moving_average_filter(probs: np.ndarray, window: int = 10) -> np.ndarray:
    """
    因果移动平均滤波：smoothed[t] = mean(probs[max(0,t-W+1) : t+1])。
    返回与输入等长的数组。
    """
    if len(probs) == 0:
        return probs
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, probs[0]), probs])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(probs)]


def apply_threshold(smoothed_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """返回二值标签数组（int），形状 [N]。"""
    return (smoothed_probs >= threshold).astype(int)


def binary_to_events(
    binary: np.ndarray,
    window_times: np.ndarray,
    window_duration: float = 4.0,
) -> List[Tuple[float, float]]:
    """
    将二值预测序列（及对应的窗口起始时间）转换为事件列表。
    连续正类窗口合并为单个事件 (start_sec, end_sec)。
    """
    if len(binary) == 0:
        return []

    events: List[Tuple[float, float]] = []
    in_event = False
    event_start = 0.0

    for i, label in enumerate(binary):
        t_start = float(window_times[i])
        t_end   = t_start + window_duration
        if label == 1 and not in_event:
            in_event = True
            event_start = t_start
            event_end = t_end
        elif label == 1 and in_event:
            event_end = t_end
        elif label == 0 and in_event:
            events.append((event_start, event_end))
            in_event = False

    if in_event:
        events.append((event_start, event_end))

    return events


def apply_collar(
    detected_events: List[Tuple[float, float]],
    true_events: List[Tuple[float, float]],
    collar_k: float = 5.0,
) -> Tuple[List, List, List]:
    """
    用 collar 技术将检测事件与真实事件匹配。
    规则: 检测事件起始时间落在 [onset - K, offset + K] 内视为 TP。
    每个真实事件最多匹配一次（贪婪匹配）。

    返回 (tp_pairs, fp_detected, fn_true_events)。
    tp_pairs: List[(detected_event, true_event)]
    """
    unmatched_true = list(true_events)
    tp_pairs = []
    fp_detected = []

    for det in detected_events:
        det_start = det[0]
        matched = None
        for true in unmatched_true:
            onset, offset = true
            if onset - collar_k <= det_start <= offset + collar_k:
                matched = true
                break
        if matched is not None:
            tp_pairs.append((det, matched))
            unmatched_true.remove(matched)
        else:
            fp_detected.append(det)

    fn_true = unmatched_true
    return tp_pairs, fp_detected, fn_true
