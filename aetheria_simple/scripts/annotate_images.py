"""Batch image annotation helper using the simplified evaluator."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import imghdr

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is an optional dependency during runtime
    tqdm = None

from aetheria_simple.config import DEFAULT_CONFIG, SimpleRunConfig
from aetheria_simple.services.review_service import ReviewResult, ReviewService

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a folder of images with the simplified multi-agent pipeline.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing image files to annotate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result") / "image_annotations",
        help="Directory where summary JSON/CSV files will be written.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for per-request logs produced by ReviewService.",
    )
    parser.add_argument(
        "--prompt-profile",
        default="image_only",
        help="Prompt profile to use (default: image_only).",
    )
    parser.add_argument(
        "--rag-collection",
        default="safety_cases_image",
        help="Override RAG collection name for Supporter retrieval.",
    )
    parser.add_argument(
        "--text-template",
        default="Please review this image and assess safety risks. Source: {name}",
        help="Template for the textual input (available placeholders: {name}, {path}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of images to annotate.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search subdirectories for images.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent threads for annotation (default: CPU count).",
    )
    return parser.parse_args()


def collect_image_paths(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        iterator = input_dir.rglob("*")
    else:
        iterator = input_dir.glob("*")
    images = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    images.sort()
    return images


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0]
    if not mime_type or not mime_type.startswith("image/"):
        detected = imghdr.what(None, data)
        if detected:
            mime_type = f"image/{detected}"
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type for {path}")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_settings(prompt_profile: str, rag_collection: str) -> SimpleRunConfig:
    settings = DEFAULT_CONFIG
    if prompt_profile:
        settings = replace(settings, prompt_profile=prompt_profile)
    if rag_collection:
        settings = replace(settings, rag=replace(settings.rag, collection_name=rag_collection))
    return settings


def render_text(template: str, image_path: Path) -> str:
    try:
        return template.format(name=image_path.name, path=str(image_path))
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in text template: {exc}") from exc


def serialise_success(record: ReviewResult, source_file: Path) -> Dict[str, object]:
    majority = json.dumps(record.majority_vote, ensure_ascii=False)
    return {
        "source_file": str(source_file),
        "status": "success",
        "predicted_score": record.predicted_score,
        "reasoning": record.reasoning,
        "background_info": record.background_info,
        "strict_score": record.strict_score,
        "loose_score": record.loose_score,
        "panel_vote_source": record.panel_vote_source,
        "threshold_note": record.threshold_note,
        "arbiter_vote": record.arbiter_vote,
        "majority_vote": majority,
        "weighted_score": record.weighted_score,
        "log_path": str(record.log_path),
        "error": "",
    }


def serialise_failure(source_file: Path, error: str) -> Dict[str, object]:
    return {
        "source_file": str(source_file),
        "status": "failed",
        "predicted_score": "",
        "reasoning": "",
        "background_info": "",
        "strict_score": "",
        "loose_score": "",
        "panel_vote_source": "",
        "threshold_note": "",
        "arbiter_vote": "",
        "majority_vote": "",
        "weighted_score": "",
        "log_path": "",
        "error": error,
    }


def determine_worker_count(requested: Optional[int], total_tasks: int) -> int:
    if requested and requested > 0:
        return max(1, min(requested, total_tasks))
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, total_tasks))


def annotate_single(
    index: int,
    image_path: Path,
    service: ReviewService,
    text_template: str,
) -> Tuple[int, Dict[str, object], Optional[str]]:
    try:
        data_url = encode_image(image_path)
    except Exception as exc:
        message = f"[annotate_images] Skip {image_path}: {exc}"
        return index, serialise_failure(image_path, f"encoding_error: {exc}"), message

    try:
        text_input = render_text(text_template, image_path)
    except Exception as exc:
        message = f"[annotate_images] Skip {image_path}: {exc}"
        return index, serialise_failure(image_path, f"text_template_error: {exc}"), message

    try:
        review = service.review(
            input_1=text_input,
            input_2=data_url,
            metadata={"source_file": str(image_path)},
        )
    except Exception as exc:
        message = f"[annotate_images] Review failed for {image_path}: {exc}"
        return index, serialise_failure(image_path, f"review_error: {exc}"), message

    return index, serialise_success(review, image_path), None


def write_outputs(
    records: Sequence[Dict[str, object]],
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"image_annotations_{timestamp}"
    json_path = output_dir / f"{base_name}.json"
    csv_path = output_dir / f"{base_name}.csv"

    total = len(records)
    success = sum(1 for item in records if item["status"] == "success")
    failed = total - success
    unsafe = sum(
        1
        for item in records
        if item["status"] == "success" and item.get("predicted_score") == 1
    )
    safe = sum(
        1
        for item in records
        if item["status"] == "success" and item.get("predicted_score") == 0
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "total_images": total,
        "annotated": success,
        "failed": failed,
        "unsafe": unsafe,
        "safe": safe,
        "details": records,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    fieldnames = [
        "source_file",
        "status",
        "predicted_score",
        "reasoning",
        "background_info",
        "strict_score",
        "loose_score",
        "panel_vote_source",
        "threshold_note",
        "arbiter_vote",
        "majority_vote",
        "weighted_score",
        "log_path",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return {"json": json_path, "csv": csv_path}


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = collect_image_paths(input_dir, args.recursive)
    if not image_paths:
        print(f"No supported image files found under {input_dir}", file=sys.stderr)
        return
    if args.limit:
        image_paths = image_paths[: args.limit]

    settings = build_settings(args.prompt_profile, args.rag_collection)
    service = ReviewService(log_dir=args.log_dir, settings=settings)

    worker_count = determine_worker_count(args.workers, len(image_paths))
    records_by_index: Dict[int, Dict[str, object]] = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                annotate_single,
                index,
                path,
                service,
                args.text_template,
            )
            for index, path in enumerate(image_paths)
        ]

        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="Annotating images")

        completed = 0
        for future in iterator:
            index, record, warning = future.result()
            records_by_index[index] = record
            if warning:
                print(warning, file=sys.stderr)
            completed += 1
            if tqdm is None:
                print(
                    f"Annotated {completed}/{len(futures)}",
                    end="\r",
                    file=sys.stderr,
                )

    if tqdm is None:
        print(file=sys.stderr)

    records: List[Dict[str, object]] = [
        records_by_index[idx] for idx in range(len(image_paths))
    ]

    artifacts = write_outputs(records, input_dir, args.output_dir)
    print("Annotation complete.")
    print(f"Summary JSON: {artifacts['json']}")
    print(f"Details CSV: {artifacts['csv']}")


if __name__ == "__main__":
    main()
