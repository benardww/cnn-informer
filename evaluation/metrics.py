from typing import Dict, List

import numpy as np
import pandas as pd

from postprocessing.postprocess import apply_collar


def compute_segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """计算分段级别的灵敏度、特异度、准确率。"""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    accuracy    = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy}


def compute_event_metrics(
    detected_events: List,
    true_events: List,
    total_hours: float,
    collar_k: float = 5.0,
) -> Dict[str, float]:
    """计算事件级别的灵敏度、每小时误检率（FDR）、平均检测延迟（秒）。"""
    tp_pairs, fp_detected, fn_true = apply_collar(detected_events, true_events, collar_k)

    n_tp = len(tp_pairs)
    n_fp = len(fp_detected)
    n_fn = len(fn_true)

    sensitivity = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    fdr_per_hour = n_fp / total_hours if total_hours > 0 else 0.0

    latencies = []
    for det, true in tp_pairs:
        # 延迟 = 检测事件起始时间 - 真实发作起始时间
        latencies.append(det[0] - true[0])
    latency = float(np.mean(latencies)) if latencies else float("nan")

    return {"sensitivity": sensitivity, "fdr_per_hour": fdr_per_hour, "latency_sec": latency}


def build_segment_table(per_patient_results: List[Dict]) -> pd.DataFrame:
    """
    构建分段指标表格，最后一行为所有患者的均值。
    列: Patient | Sensitivity | Specificity | Accuracy
    """
    rows = []
    for r in per_patient_results:
        rows.append({
            "Patient":     r["patient"],
            "Sensitivity": f"{r['sensitivity']:.4f}",
            "Specificity": f"{r['specificity']:.4f}",
            "Accuracy":    f"{r['accuracy']:.4f}",
        })
    df = pd.DataFrame(rows)

    # 均值行
    mean_row = {
        "Patient":     "Mean",
        "Sensitivity": f"{np.mean([r['sensitivity'] for r in per_patient_results]):.4f}",
        "Specificity": f"{np.mean([r['specificity'] for r in per_patient_results]):.4f}",
        "Accuracy":    f"{np.mean([r['accuracy']    for r in per_patient_results]):.4f}",
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df


def build_event_table(per_patient_results: List[Dict]) -> pd.DataFrame:
    """
    构建事件指标表格，最后一行为均值（延迟排除 NaN）。
    列: Patient | Sensitivity | FDR/hr | Latency(s)
    """
    rows = []
    for r in per_patient_results:
        lat = r["latency_sec"]
        rows.append({
            "Patient":     r["patient"],
            "Sensitivity": f"{r['sensitivity']:.4f}",
            "FDR/hr":      f"{r['fdr_per_hour']:.4f}",
            "Latency(s)":  f"{lat:.2f}" if not (isinstance(lat, float) and np.isnan(lat)) else "N/A",
        })
    df = pd.DataFrame(rows)

    valid_lats = [r["latency_sec"] for r in per_patient_results
                  if not (isinstance(r["latency_sec"], float) and np.isnan(r["latency_sec"]))]
    mean_row = {
        "Patient":     "Mean",
        "Sensitivity": f"{np.mean([r['sensitivity']  for r in per_patient_results]):.4f}",
        "FDR/hr":      f"{np.mean([r['fdr_per_hour'] for r in per_patient_results]):.4f}",
        "Latency(s)":  f"{np.mean(valid_lats):.2f}" if valid_lats else "N/A",
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df
