import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.binary_classification import sanitize_comment_tool

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input_1210.json"
EXPECTED_OUTPUT_PATH = BASE_DIR / "output_1210.json"
PREDICTED_OUTPUT_PATH = BASE_DIR / "predicted_1210.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def compute_metrics(
    inputs: List[Dict[str, Any]],
    expected_kept: List[Dict[str, Any]],
    predicted_kept: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[int]]]:
    input_ids = [item["id"] for item in inputs if "id" in item]
    expected_keep_ids = {item["id"] for item in expected_kept if "id" in item}
    predicted_keep_ids = {item["id"] for item in predicted_kept if "id" in item}

    tp_ids: List[int] = []
    fp_ids: List[int] = []
    fn_ids: List[int] = []
    tn_ids: List[int] = []

    for cid in input_ids:
        if cid is None:
            continue

        actual_positive = cid in expected_keep_ids
        predicted_positive = cid in predicted_keep_ids

        if actual_positive and predicted_positive:
            tp_ids.append(cid)
        elif actual_positive and not predicted_positive:
            fn_ids.append(cid)
        elif not actual_positive and predicted_positive:
            fp_ids.append(cid)
        else:
            tn_ids.append(cid)

    tp = len(tp_ids)
    fp = len(fp_ids)
    fn = len(fn_ids)
    tn = len(tn_ids)
    total = len(input_ids)

    accuracy = (tp + tn) / total if total else 0.0
    error_rate = 1.0 - accuracy
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall_rate = tp / (tp + fn) if (tp + fn) else 0.0

    metrics = {
        "Accuracy": accuracy,
        "ErrorRate": error_rate,
        "Precision": precision,
        "RecallRate": recall_rate,
    }
    confusion_counts = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Total": total}
    records = {
        "TruePositive": tp_ids,
        "FalsePositive": fp_ids,
        "FalseNegative": fn_ids,
        "TrueNegative": tn_ids,
    }
    return metrics, confusion_counts, records


def write_report(
    inputs: List[Dict[str, Any]],
    metrics: Dict[str, float],
    confusion_counts: Dict[str, int],
    expected_kept: List[Dict[str, Any]],
    predicted_kept: List[Dict[str, Any]],
    records: Dict[str, List[int]],
) -> Path:
    timestamp = datetime.now()
    report_name = timestamp.strftime("report_%Y%m%d_%H%M%S.md")
    report_path = BASE_DIR / report_name

    id2item = {item["id"]: item for item in inputs if "id" in item}
    precision = metrics["Precision"]
    recall_rate = metrics["RecallRate"]
    f1_score = 2 * precision * recall_rate / (precision + recall_rate) if (precision + recall_rate) else 0.0
    lines = [
        f"# Binary Classification Sanitize Report ({timestamp.isoformat(timespec='seconds')})",
        "",
        f"- Total samples: {confusion_counts['Total']}",
        f"- Expected kept count: {len(expected_kept)}",
        f"- Predicted kept count: {len(predicted_kept)}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Accuracy | {metrics['Accuracy'] * 100:.2f}% |",
        f"| Error Rate | {metrics['ErrorRate'] * 100:.2f}% |",
        f"| Precision | {precision * 100:.2f}% |",
        f"| Recall Rate | {recall_rate * 100:.2f}% |",
        f"| F1 Score | {f1_score:.2f} |",
        f"| True Positives | {confusion_counts['TP']} |",
        f"| True Negatives | {confusion_counts['TN']} |",
        f"| False Positives | {confusion_counts['FP']} |",
        f"| False Negatives | {confusion_counts['FN']} |",
        "",
        "## False Positives",
        "",
        "Should be dropped but predicted as kept:",
    ]

    if records["FalsePositive"]:
        for fp_id in records["FalsePositive"]:
            if fp_id not in id2item:
                continue
            item = id2item[fp_id]
            user = item.get("user", "")
            content = str(item.get("content", item.get("comment", ""))).replace("\n", " ")
            lines.append(f"- id={fp_id}, user={user}, content={content}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## False Negatives",
            "",
            "Should be kept but predicted as dropped:",
        ]
    )

    if records["FalseNegative"]:
        for fn_id in records["FalseNegative"]:
            if fn_id not in id2item:
                continue
            item = id2item[fn_id]
            user = item.get("user", "")
            content = str(item.get("content", item.get("comment", ""))).replace("\n", " ")
            lines.append(f"- id={fn_id}, user={user}, content={content}")
    else:
        lines.append("- None")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    inputs = load_json(INPUT_PATH)
    expected_kept = load_json(EXPECTED_OUTPUT_PATH)

    predicted_path_str = sanitize_comment_tool.invoke(
        {
            "input_file_path": str(INPUT_PATH),
            "output_file_path": str(PREDICTED_OUTPUT_PATH),
        }
    )
    predicted_kept = load_json(Path(predicted_path_str))

    metrics, confusion_counts, records = compute_metrics(
        inputs=inputs,
        expected_kept=expected_kept,
        predicted_kept=predicted_kept,
    )
    report_path = write_report(
        inputs=inputs,
        metrics=metrics,
        confusion_counts=confusion_counts,
        expected_kept=expected_kept,
        predicted_kept=predicted_kept,
        records=records,
    )

    print(f"Sanitized output written to: {predicted_path_str}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
