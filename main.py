"""
CNN-Informer EEG 癫痫自动检测 — 主入口
用法示例:
    python main.py
    python main.py --patients chb01 chb02
    python main.py --maf_window 10 --threshold 0.5 --collar_k 5.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from training.trainer import train_patient
from postprocessing.postprocess import (
    moving_average_filter,
    apply_threshold,
    binary_to_events,
)
from evaluation.metrics import (
    compute_segment_metrics,
    compute_event_metrics,
    build_segment_table,
    build_event_table,
)


def run(args):
    data_root = Path(args.data_root)
    patients  = args.patients if args.patients else config.ALL_PATIENTS

    seg_results = []
    evt_results = []

    for patient in patients:
        print(f"\n{'='*50}")
        print(f"  患者: {patient}")
        print(f"{'='*50}")

        try:
            model, val_info = train_patient(
                patient,
                data_root,
                device=config.DEVICE,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
        except Exception as e:
            print(f"  [警告] {patient} 训练失败: {e}，跳过")
            continue

        probs        = val_info["probs"]
        labels       = val_info["labels"]
        window_times = val_info["window_times"]
        true_events  = val_info["true_events"]
        total_hours  = val_info["total_hours"]

        # 后处理
        smoothed = moving_average_filter(probs, window=args.maf_window)
        preds    = apply_threshold(smoothed, threshold=args.threshold)

        # 分段指标
        seg_m = compute_segment_metrics(labels, preds)
        seg_results.append({"patient": patient, **seg_m})
        print(f"  分段指标 → 灵敏度={seg_m['sensitivity']:.4f}  "
              f"特异度={seg_m['specificity']:.4f}  "
              f"准确率={seg_m['accuracy']:.4f}")

        # 事件指标
        detected_events = binary_to_events(preds, window_times, window_duration=4.0)
        evt_m = compute_event_metrics(
            detected_events, true_events, total_hours, collar_k=args.collar_k
        )
        evt_results.append({"patient": patient, **evt_m})
        lat_str = f"{evt_m['latency_sec']:.2f}s" if evt_m['latency_sec'] == evt_m['latency_sec'] else "N/A"
        print(f"  事件指标 → 灵敏度={evt_m['sensitivity']:.4f}  "
              f"FDR/hr={evt_m['fdr_per_hour']:.4f}  "
              f"延迟={lat_str}")

    if not seg_results:
        print("\n没有有效的评估结果。")
        return

    print("\n\n" + "="*60)
    print("表格 1 — 分段级别指标 (Segment-Based Metrics)")
    print("="*60)
    seg_table = build_segment_table(seg_results)
    print(seg_table.to_string(index=False))

    print("\n\n" + "="*60)
    print("表格 2 — 事件级别指标 (Event-Based Metrics)")
    print("="*60)
    evt_table = build_event_table(evt_results)
    print(evt_table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="CNN-Informer EEG 癫痫检测")
    parser.add_argument("--data_root",   type=str,   default=config.DATA_ROOT)
    parser.add_argument("--patients",    nargs="+",  default=None,
                        help="指定患者列表，默认使用 config.ALL_PATIENTS")
    parser.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LR)
    parser.add_argument("--maf_window",  type=int,   default=config.MAF_WINDOW)
    parser.add_argument("--threshold",   type=float, default=config.THRESHOLD)
    parser.add_argument("--collar_k",    type=float, default=config.COLLAR_K)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
