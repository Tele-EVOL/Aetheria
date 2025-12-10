#!/usr/bin/env python3
"""Compute custom risk distribution across the merged datasets."""

from __future__ import annotations

import json
from collections import Counter
from math import floor
from pathlib import Path

DATASETS = ("only_img.json", "text_img.json")
TEXT_ONLY_FILE = "only_text.json"

# Map the original level-2 categories into a handful of consolidated buckets
# (≈7) so that尾部的小类被合并得更紧凑。
CATEGORY_MAP_L2 = {
    "Bias & Discrimination": "Harassment & Hate",
    "Insults & Condemnation": "Harassment & Hate",
    "Inappropriate values": "Moral & Content Safety",
    "Content Safety": "Moral & Content Safety",
    "Psychological Health": "Societal Influence & Wellbeing",
    "Personal Rights & Property": "Privacy & Personal Data",
    "Privacy Protection": "Privacy & Personal Data",
    "Business": "Institutional & Strategic Safety",
    "Intellectual Property": "Institutional & Strategic Safety",
    "Network Attacks": "Cybersecurity & Network",
    "Military": "Institutional & Strategic Safety",
    "Hazardous & Controlled Materials": "Hazardous & Biological Threats",
    "Biology & Environment": "Hazardous & Biological Threats",
    "Culture & History": "Societal Influence & Wellbeing",
    "Superstition": "Societal Influence & Wellbeing",
    "Other Public Safety": "Institutional & Strategic Safety",
}

# Split the text-only dataset according to the WildGuard prior, then merge
# them into the consolidated categories above.
TEXT_ONLY_SPLIT = {
    "Privacy": 0.23,
    "Misinformation": 0.15,
    "Harmful Language": 0.40,
    "Malicious": 0.22,
}
TEXT_ONLY_TO_CATEGORY = {
    "Privacy": "Privacy & Personal Data",
    "Misinformation": "Societal Influence & Wellbeing",
    "Harmful Language": "Harassment & Hate",
    "Malicious": "Cybersecurity & Network",
}


def load_entries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def distribute(total: int, split: dict[str, float]) -> dict[str, int]:
    """Return integer allocations that match the desired split."""

    if total == 0:
        return {key: 0 for key in split}

    scaled = {k: total * v for k, v in split.items()}
    base = {k: floor(val) for k, val in scaled.items()}
    remainder = total - sum(base.values())

    fractional_order = sorted(
        scaled.keys(),
        key=lambda key: scaled[key] - base[key],
        reverse=True,
    )
    for key in fractional_order:
        if remainder == 0:
            break
        base[key] += 1
        remainder -= 1

    return base


def main() -> None:
    counts = Counter()
    base_dir = Path(__file__).resolve().parent

    for dataset in DATASETS:
        for entry in load_entries(base_dir / dataset):
            level2 = entry.get("level2_category")
            category = CATEGORY_MAP_L2.get(level2)
            if not category:
                category = entry.get("level1_category", "Uncategorized")
            counts[category] += 1

    text_only_entries = load_entries(base_dir / TEXT_ONLY_FILE)
    allocations = distribute(len(text_only_entries), TEXT_ONLY_SPLIT)
    for raw_category, value in allocations.items():
        mapped_category = TEXT_ONLY_TO_CATEGORY[raw_category]
        counts[mapped_category] += value

    total = sum(counts.values())

    if not total:
        print("No samples found.")
        return

    print(f"Total samples: {total}")
    print("\nRisk type proportions (level1_category):")
    for category, count in counts.most_common():
        percentage = count / total * 100
        print(f"- {category}: {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    main()
