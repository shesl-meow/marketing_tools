import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.information_extract import demand_extract_tool

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input_1209.json"
EXPECTED_OUTPUT_PATH = BASE_DIR / "output_1209.json"
PREDICTED_OUTPUT_PATH = BASE_DIR / "predicted_1209.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def normalize_demands(items: List[Any]) -> List[str]:
    """Normalize and deduplicate demand items while preserving order."""
    demands: List[str] = []
    for raw in items:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text or text in demands:
            continue
        demands.append(text)
    return demands

def write_report(
    expected: List[str],
    predicted: List[str],
) -> Path:
    timestamp = datetime.now()
    report_name = timestamp.strftime("report_%Y%m%d_%H%M%S.md")
    report_path = BASE_DIR / report_name

    lines = [
        f"# Demand Extract Report ({timestamp.isoformat(timespec='seconds')})",
        "",
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
    expected_demands = normalize_demands(load_json(EXPECTED_OUTPUT_PATH))
    predicted_path_str = demand_extract_tool.invoke(
        {
            "input_file_path": str(INPUT_PATH),
            "output_file_path": str(PREDICTED_OUTPUT_PATH),
        }
    )
    predicted_demands = load_json(Path(predicted_path_str))

    report_path = write_report(
        expected=expected_demands,
        predicted=predicted_demands,
    )

    print(f"Demand extraction output written to: {predicted_path_str}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
