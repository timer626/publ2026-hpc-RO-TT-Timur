"""Build a fixed long-context pilot source with distinct retrieved distractors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=10)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if len(records) < args.questions + 1:
        raise ValueError("Not enough records to draw distinct distractors")
    rows = []
    for index, record in enumerate(records[: args.questions]):
        existing = {doc["title"] for doc in record["retrieved_documents"]}
        fillers = []
        for other in records[args.questions :] + records[:index] + records[index + 1 : args.questions]:
            for doc in other["retrieved_documents"]:
                if doc["title"] not in existing:
                    fillers.append(doc)
                    existing.add(doc["title"])
                    if len(fillers) >= 64:
                        break
            if len(fillers) >= 64:
                break
        rows.append({
            "id": record["id"],
            "question": record["question"],
            "reference_answer": record["reference_answer"],
            "retrieved_documents": record["retrieved_documents"],
            "filler_documents": fillers,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    main()
