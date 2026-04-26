"""Dataset EDA command placeholder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a minimal dataset EDA report placeholder")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--schema-path", default=None)
    parser.add_argument("--output", default="outputs/reports/dataset_eda.json")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": "dataset_eda",
        "dataset_path": args.dataset_path,
        "schema_path": args.schema_path,
        "status": "placeholder",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
