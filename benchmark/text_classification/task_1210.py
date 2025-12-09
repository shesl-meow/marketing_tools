import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.text_classification.tool import tool
from src.llms import volcano

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input_1210.json"
OUTPUT_PATH = BASE_DIR / "output_1210.json"
BATCH_SIZE = 50


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def normalize_label_list(item: Any) -> List[str]:
    if isinstance(item, list):
        return [str(label) for label in item]
    if item is None:
        return []
    return [str(item)]


def compute_accuracy(
    expected: List[List[str]],
    predicted: List[List[str]],
) -> Tuple[float, List[Dict[str, Any]]]:
    total = len(expected)
    if total == 0:
        return 0.0, []

    mismatches: List[Dict[str, Any]] = []
    correct = 0

    for idx in range(total):
        exp_labels = normalize_label_list(expected[idx])
        pred_labels = normalize_label_list(predicted[idx]) if idx < len(predicted) else []

        if set(exp_labels) == set(pred_labels):
            correct += 1
        else:
            mismatches.append(
                {
                    "index": idx,
                    "expected": exp_labels,
                    "predicted": pred_labels,
                }
            )

    return correct / total, mismatches


def chunked(items: List[Any], size: int):
    """Yield list slices with a fixed chunk size."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_batches(
    texts: List[str],
    categories: List[str],
    batch_size: int,
    model,
) -> List[List[str]]:
    """Run classification in batches and aggregate outputs."""
    aggregated: List[List[str]] = []
    for batch in chunked(texts, batch_size):
        aggregated.extend(tool(texts=batch, categories=categories, model=model))
    return aggregated


def write_report(
    texts: List[Dict[str, Any]],
    expected: List[List[str]],
    predicted: List[List[str]],
    accuracy: float,
    mismatches: List[Dict[str, Any]],
) -> Path:
    timestamp = datetime.now()
    report_name = timestamp.strftime("report_%Y%m%d_%H%M%S.md")
    report_path = BASE_DIR / report_name

    lines = [
        f"# Text Classification Report ({timestamp.isoformat(timespec='seconds')})",
        "",
        f"- Total samples: {len(expected)}",
        f"- Predicted samples: {len(predicted)}",
        f"- Accuracy: {accuracy * 100:.2f}%",
        "",
        "## Mismatches",
    ]

    if not mismatches:
        lines.append("- None")
    else:
        for item in mismatches:
            idx = item["index"]
            text_item = texts[idx] if idx < len(texts) else {}
            content = str(text_item.get("content", "")).replace("\n", " ")
            lines.extend(
                [
                    f"- Sample #{idx + 1}:",
                    f"  - Text: {content}",
                    f"  - Expected: {item['expected']}",
                    f"  - Predicted: {item['predicted']}",
                ]
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    input_payload = load_json(INPUT_PATH)
    texts = input_payload["texts"]
    categories = input_payload["categories"]
    expected_labels = load_json(OUTPUT_PATH)

    text_contents = [item["content"] for item in texts]
    model = volcano.create_model("doubao-seed-1-6-lite-251015")
    predicted_labels = run_batches(
        texts=text_contents,
        categories=categories,
        batch_size=BATCH_SIZE,
        model=model,
    )

    accuracy, mismatches = compute_accuracy(expected=expected_labels, predicted=predicted_labels)
    report_path = write_report(
        texts=texts,
        expected=expected_labels,
        predicted=predicted_labels,
        accuracy=accuracy,
        mismatches=mismatches,
    )
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
