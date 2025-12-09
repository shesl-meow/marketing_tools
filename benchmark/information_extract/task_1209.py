import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.information_extract.tool import tool
from src.llms import volcano

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input_1209.json"
OUTPUT_PATH = BASE_DIR / "output_1209.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_report(
    expected: List[str],
    predicted: List[str],
    information_type: str,
) -> Path:
    timestamp = datetime.now()
    report_name = timestamp.strftime("report_%Y%m%d_%H%M%S.md")
    report_path = BASE_DIR / report_name

    lines = [
        f"# Information Extract Report ({timestamp.isoformat(timespec='seconds')})",
        "",
        f"- Information type: {information_type}",
        f"- Expected item count: {len(expected)}",
        f"- Predicted item count: {len(predicted)}",
        "",
        "## Expected Items",
    ]

    lines.extend([f"- {item}" for item in expected] or ["- None"])

    lines.extend(["", "## Predicted Items"])
    lines.extend([f"- {item}" for item in predicted] or ["- None"])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    input_payload = load_json(INPUT_PATH)
    texts = input_payload["texts"]
    information_type = input_payload["information_type"]
    expected_items = load_json(OUTPUT_PATH)

    model = volcano.create_model("doubao-seed-1-6-lite-251015")
    predicted_items = tool(texts=texts, information_type=information_type, model=model)

    report_path = write_report(
        expected=expected_items,
        predicted=predicted_items,
        information_type=information_type,
    )
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
