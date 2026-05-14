import re
from pathlib import Path
from typing import Dict, List, Tuple

SeizureMap = Dict[str, List[Tuple[int, int]]]


def parse_summary(patient_dir: Path) -> SeizureMap:
    """
    解析 <patient_dir>/chb##-summary.txt，返回 {edf文件名: [(start_sec, end_sec), ...]}。
    无发作的文件映射到空列表。
    """
    patient_id = patient_dir.name  # e.g. "chb01"
    summary_path = patient_dir / f"{patient_id}-summary.txt"
    text = summary_path.read_text(encoding="utf-8", errors="replace")

    # 按 "File Name:" 分割成块
    blocks = re.split(r"(?=File Name:)", text)
    result: SeizureMap = {}
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        filename_match = re.search(r"File Name:\s*(\S+\.edf)", block, re.IGNORECASE)
        if not filename_match:
            continue
        filename = filename_match.group(1).strip()

        num_match = re.search(r"Number of Seizures in File:\s*(\d+)", block, re.IGNORECASE)
        n_seizures = int(num_match.group(1)) if num_match else 0

        seizures: List[Tuple[int, int]] = []
        if n_seizures > 0:
            starts = re.findall(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s*(\d+)", block, re.IGNORECASE)
            ends   = re.findall(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s*(\d+)",   block, re.IGNORECASE)
            for s, e in zip(starts, ends):
                seizures.append((int(s), int(e)))

        result[filename] = seizures

    return result
