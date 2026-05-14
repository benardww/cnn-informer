import numpy as np
import pywt

# 5 层 db4 DWT 在 256 Hz 采样率下的子带划分：
# coeffs[0] = cA5: 0–4 Hz   (保留)
# coeffs[1] = cD5: 4–8 Hz   (保留)
# coeffs[2] = cD4: 8–16 Hz  (保留)
# coeffs[3] = cD3: 16–32 Hz (保留，覆盖目标上限 29 Hz)
# coeffs[4] = cD2: 32–64 Hz (置零)
# coeffs[5] = cD1: 64–128 Hz(置零)
_ZERO_INDICES = (4, 5)


def dwt_bandpass(window: np.ndarray, wavelet: str = "db4", level: int = 5) -> np.ndarray:
    """
    对单通道 1D 信号（形状 [N]）进行 5 层 db4 DWT，
    置零 cD2 和 cD1，重建 0.5–29 Hz 频带，返回等长数组。
    """
    n = len(window)
    coeffs = pywt.wavedec(window, wavelet, level=level, mode="periodization")
    for i in _ZERO_INDICES:
        coeffs[i] = np.zeros_like(coeffs[i])
    reconstructed = pywt.waverec(coeffs, wavelet, mode="periodization")
    return reconstructed[:n].astype(np.float32)


def apply_dwt_to_segment(segment: np.ndarray) -> np.ndarray:
    """
    对 18 通道片段（形状 [18, N]）逐通道应用 dwt_bandpass。
    返回 float32 数组，形状 [18, N]。
    """
    return np.stack([dwt_bandpass(segment[i]) for i in range(segment.shape[0])]).astype(np.float32)
