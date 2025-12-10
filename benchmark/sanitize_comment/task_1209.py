import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llms import volcano
from src.agents.sanitize_comment import tool

BATCH_SIZE = 50
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input_1209.json"
OUTPUT_PATH = BASE_DIR / "output_1209.json"


def chunked(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    """Yield list slices with a fixed chunk size."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def run_batches(comments: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
    """Run sanitize_comment.tool in batches and aggregate outputs."""
    aggregated: List[Dict[str, Any]] = []
    model = volcano.create_model('doubao-seed-1-6-lite-251015')
    for batch in chunked(comments, batch_size):
        aggregated.extend(tool(comments=batch, model=model))
    return aggregated


def compute_metrics(
    inputs: List[Dict[str, Any]],
    expected_filtered: List[Dict[str, Any]],
    predicted_filtered: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    input_ids = {item["id"] for item in inputs}
    # 这里注意，这个二分类问题的 true 是被 filter 的数据，false 是未被 filter 的数据
    expected_false = {item["id"] for item in expected_filtered}
    predicted_false = {item["id"] for item in predicted_filtered}

    expected_ids = {item_id for item_id in input_ids if item_id not in expected_false}
    predicted_ids = {item_id for item_id in input_ids if item_id not in predicted_false}

    tp_ids = []
    fp_ids = []
    fn_ids = []
    tn_ids = []
    for item in inputs:
        cid = item["id"]
        actual_positive = cid in expected_ids
        predicted_positive = cid in predicted_ids

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
    total = len(inputs)

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
    expected_filtered_count: int,
    predicted_filtered_count: int,
    records: Dict[str, List[int]],
) -> Path:
    timestamp = datetime.now()
    report_name = timestamp.strftime("report_%Y%m%d_%H%M%S.md")
    report_path = BASE_DIR / report_name

    id2item = {item["id"]: item for item in inputs}
    precision = metrics["Precision"]
    recall_rate = metrics["RecallRate"]
    f1_score = 2 * precision * recall_rate / (precision + recall_rate) if (precision + recall_rate) else 0.0
    lines = [
        f"# Sanitize Comment Report ({timestamp.isoformat(timespec='seconds')})",
        "",
        f"- Total samples: {confusion_counts['Total']}",
        f"- Expected filtered count: {expected_filtered_count}",
        f"- Predicted filtered count: {predicted_filtered_count}",
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
        "应该不被过滤但预测被过滤的记录：",
    ]

    if records["FalsePositive"]:
        for fp_id in records["FalsePositive"]:
            if fp_id not in id2item:
                continue
            item = id2item[fp_id]
            user = item.get("user", "")
            content = item.get("content", "").replace("\n", " ")
            lines.append(f"- id={item['id']}, user={user}, content={content}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## False Negatives",
            "",
            "应该被过滤但预测未过滤的记录：",
        ]
    )

    if records["FalseNegative"]:
        for fp_id in records["FalseNegative"]:
            if fp_id not in id2item:
                continue
            item = id2item[fp_id]
            user = item.get("user", "")
            content = item.get("content", "").replace("\n", " ")
            lines.append(f"- id={item['id']}, user={user}, content={content}")
    else:
        lines.append("- None")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    inputs = load_json(INPUT_PATH)
    expected_filtered = load_json(OUTPUT_PATH)
    predicted_filtered = run_batches(inputs, BATCH_SIZE)

    metrics, confusion_counts, records = compute_metrics(inputs, expected_filtered, predicted_filtered)
    report_path = write_report(
        inputs=inputs,
        metrics=metrics,
        confusion_counts=confusion_counts,
        expected_filtered_count=len(expected_filtered),
        predicted_filtered_count=len(predicted_filtered),
        records=records,
    )
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
