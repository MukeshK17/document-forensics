"""
Generate a field-focused OCR crop dataset and statistics report.

Output structure:
    data/field_crops/
        pan_number/real/
        pan_number/fake/
        dob/real/
        dob/fake/
        name/real/
        name/fake/
        father_name/real/
        father_name/fake/

Also writes:
    data/field_crops/metadata.csv
    data/field_crops/statistics.txt
    data/field_crops/review/<field_type>/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FIELD_CROPS_ROOT = ROOT / "data" / "field_crops"
FIELD_METADATA = FIELD_CROPS_ROOT / "metadata.csv"
FIELD_STATISTICS = FIELD_CROPS_ROOT / "statistics.txt"
REVIEW_ROOT = FIELD_CROPS_ROOT / "review"

sys.path.insert(0, str(ROOT))

FIELD_TYPES = ["pan_number", "dob", "name", "father_name"]
PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
DOB_REGEX = re.compile(r"^\d{2}[/-]\d{2}[/-]\d{4}$")
NAME_REGEX = re.compile(r"^[A-Za-z]{4,}$")
FATHER_HINT = re.compile(r"FATHER|S/O|C/O", re.IGNORECASE)
HINDI_REGEX = re.compile(r"[\u0900-\u097F]")
BLACKLIST_WORDS = {
    "INCOME",
    "TAX",
    "DEPARTMENT",
    "GOVT",
    "INDIA",
    "GOVERNMENT",
    "ACCOUNT",
    "NUMBER",
    "SIGNATURE",
    "PERMANENT",
    "PAN",
    "AADHAR",
    "DOB",
    "DATE",
    "BIRTH",
    "NAME",
    "OF",
    "THE",
    "AND",
    "MINISTRY",
    "DOCUMENT",
    "CARD",
    "YEAR",
    "OLD",
    "NEW",
    "FORM",
    "NO",
    "NO.",
    "ID",
    "UID",
    "OF",
    "S/O",
    "C/O",
    "FATHER",
}
LINE_BLACKLIST_WORDS = {
    "INCOME",
    "TAX",
    "DEPARTMENT",
    "GOVT",
    "INDIA",
    "GOVERNMENT",
    "ACCOUNT",
    "NUMBER",
    "SIGNATURE",
    "PERMANENT",
    "AADHAR",
    "MINISTRY",
    "DOCUMENT",
    "FORM",
    "UID",
}


def load_existing_ocr_cache(
    splits_dir: Path, splits: list[str]
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for split in splits:
        cache_path = splits_dir / f"{split}_ocr_cache.jsonl"
        if not cache_path.exists():
            continue
        with cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cache[entry["doc_id"]] = entry
    return cache


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def words_are_header_or_template(text: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z0-9/:-]", "", text).upper().strip()
    if not cleaned:
        return True
    if any(keyword in cleaned for keyword in BLACKLIST_WORDS):
        return True
    if HINDI_REGEX.search(text):
        return True
    if len(cleaned) <= 3 and not NAME_REGEX.match(cleaned):
        return True
    return False


def is_pan_token(text: str) -> bool:
    candidate = text.strip().replace(" ", "").upper()
    return bool(PAN_REGEX.fullmatch(candidate))


def is_dob_token(text: str) -> bool:
    candidate = text.strip()
    return bool(DOB_REGEX.fullmatch(candidate))


def is_name_token(text: str) -> bool:
    token = text.strip()
    if not NAME_REGEX.fullmatch(token):
        return False
    if words_are_header_or_template(token):
        return False
    return True


def is_template_line(text: str) -> bool:
    if HINDI_REGEX.search(text):
        return True
    cleaned = re.sub(r"[^A-Z0-9/:\-\s]", "", text.upper())
    if any(keyword in cleaned for keyword in LINE_BLACKLIST_WORDS):
        if any(is_pan_token(w) for w in text.split()):
            return False
        if any(is_dob_token(w) for w in text.split()):
            return False
        return True
    return False


def get_line_groups(
    words: list[str], boxes: list[list[int]], img_h: int
) -> list[list[int]]:
    if not words or not boxes:
        return []

    centers = [((b[1] + b[3]) / 2) for b in boxes]
    idxs = sorted(range(len(words)), key=lambda i: centers[i])
    groups: list[list[int]] = []
    current: list[int] = [idxs[0]]
    for idx in idxs[1:]:
        if abs(centers[idx] - centers[current[-1]]) <= max(0.05 * img_h, 15):
            current.append(idx)
        else:
            groups.append(sorted(current, key=lambda i: boxes[i][0]))
            current = [idx]
    if current:
        groups.append(sorted(current, key=lambda i: boxes[i][0]))
    return groups


def get_line_field_type(
    line_text: str,
    line_indices: list[int],
    words: list[str],
    boxes: list[list[int]],
    img_h: int,
) -> str | None:
    if not line_text.strip():
        return None
    if is_template_line(line_text):
        return None

    line_words = [words[i] for i in line_indices]
    valid_name_tokens = [w for w in line_words if is_name_token(w)]
    if any(is_pan_token(w) for w in line_words):
        return "pan_number"
    if any(is_dob_token(w) for w in line_words):
        return "dob"
    line_upper = line_text.upper()
    if FATHER_HINT.search(line_upper) and valid_name_tokens:
        return "father_name"
    if re.search(r"\bNAME\b", line_upper) and valid_name_tokens:
        return "name"
    if len(valid_name_tokens) >= 2:
        return "name"
    return None


def is_father_hint(text: str) -> bool:
    return bool(FATHER_HINT.search(text))


def is_in_same_band(box1: list[int], box2: list[int], img_h: int) -> bool:
    y1 = (box1[1] + box1[3]) / 2
    y2 = (box2[1] + box2[3]) / 2
    return abs(y1 - y2) <= max(0.12 * img_h, 40)


def get_field_type(
    text: str,
    index: int,
    boxes: list[list[int]],
    words: list[str],
    img_h: int,
) -> str | None:
    clean = text.strip()
    if not clean:
        return None

    if is_pan_token(clean):
        return "pan_number"
    if is_dob_token(clean):
        return "dob"

    if not is_name_token(clean):
        return None

    # Father name if the token is close to a father hint region
    token_box = boxes[index]
    for hint_index, hint_text in enumerate(words):
        if is_father_hint(hint_text):
            if is_in_same_band(token_box, boxes[hint_index], img_h):
                return "father_name"
    return "name"


def label_dir(label: str) -> str:
    return "fake" if "FAKE" in label.upper() else "real"


def clamp_box(
    x_min: int, y_min: int, x_max: int, y_max: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    return (
        max(0, min(x_min, img_w - 1)),
        max(0, min(y_min, img_h - 1)),
        max(1, min(x_max, img_w)),
        max(1, min(y_max, img_h)),
    )


def expand_box(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    pad_fraction: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    width = x_max - x_min
    height = y_max - y_min
    pad_x = int(round(width * pad_fraction))
    pad_y = int(round(height * pad_fraction))
    return clamp_box(
        x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y, img_w, img_h
    )


def write_statistics(records: list[dict[str, str]]) -> None:
    counts = Counter()
    label_counts = defaultdict(Counter)
    text_counts = defaultdict(Counter)
    label_dir_counts = defaultdict(Counter)
    for row in records:
        field = row["field_type"]
        counts[field] += 1
        label_counts[field][row["label"]] += 1
        label_dir_counts[field][label_dir(row["label"])] += 1
        text_counts[field][row["ocr_text"]] += 1

    lines: list[str] = []
    lines.append("Field crop statistics")
    lines.append("======================")
    lines.append("")
    for field in FIELD_TYPES:
        lines.append(f"Field: {field}")
        lines.append(f"  total crops: {counts[field]}")
        lines.append(f"  real crops : {label_dir_counts[field]['real']}")
        lines.append(f"  fake crops : {label_dir_counts[field]['fake']}")
        lines.append("")
        lines.append("  Top 100 OCR texts:")
        for text, count in text_counts[field].most_common(100):
            lines.append(f"    {count:5d}  {text}")
        lines.append("")
    FIELD_STATISTICS.parent.mkdir(parents=True, exist_ok=True)
    FIELD_STATISTICS.write_text("\n".join(lines), encoding="utf-8")


def save_review_samples(
    records: list[dict[str, str]], max_samples: int = 100, seed: int = 42
) -> None:
    random.seed(seed)
    by_field: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        by_field[row["field_type"]].append(row)

    if REVIEW_ROOT.exists():
        shutil.rmtree(REVIEW_ROOT)

    for field, rows in by_field.items():
        if not rows:
            continue
        review_dir = REVIEW_ROOT / field
        review_dir.mkdir(parents=True, exist_ok=True)
        sample_rows = random.sample(rows, min(max_samples, len(rows)))
        for row in sample_rows:
            src = ROOT / row["crop_path"]
            if not src.exists():
                continue
            dst_name = f"{row['label']}_{row['doc_id']}_{row['field_type']}_{src.name}"
            dst = review_dir / dst_name
            shutil.copy(src, dst)


def build_field_dataset(
    splits_dir: Path,
    config: dict[str, Any],
    splits: list[str],
    overwrite: bool,
    dry_run: bool,
) -> None:
    from src.data.ingestion import DocumentIngester
    from src.data.preprocessing import DocumentPreprocessor

    manifests: list[dict[str, str]] = []
    for split in splits:
        path = splits_dir / f"{split}.csv"
        manifests.extend(read_manifest(path))

    if not manifests:
        raise FileNotFoundError(f"No manifests found in {splits_dir}")

    ocr_cache = load_existing_ocr_cache(splits_dir, splits)
    if not ocr_cache:
        raise FileNotFoundError(
            f"No OCR cache files found in {splits_dir}. Run tools/build_ocr_cache.py first."
        )

    if dry_run:
        print("[dry-run] Would build field crops for", len(manifests), "documents")
        return

    if overwrite and FIELD_CROPS_ROOT.exists():
        shutil.rmtree(FIELD_CROPS_ROOT)

    DocumentIngesterClass = DocumentIngester
    DocumentPreprocessorClass = DocumentPreprocessor
    ingester = DocumentIngesterClass(config)
    preprocessor = DocumentPreprocessorClass(config)

    FIELD_CROPS_ROOT.mkdir(parents=True, exist_ok=True)
    with FIELD_METADATA.open("w", newline="", encoding="utf-8") as meta_fh:
        writer = csv.DictWriter(
            meta_fh,
            fieldnames=[
                "doc_id",
                "label",
                "field_type",
                "ocr_text",
                "ocr_score",
                "crop_path",
            ],
        )
        writer.writeheader()

        saved_records: list[dict[str, str]] = []
        processed = 0
        failed = 0
        for row in manifests:
            doc_id = row["doc_id"]
            label = row["label"]
            cache_entry = ocr_cache.get(doc_id)
            if cache_entry is None:
                print(f"Skipping missing OCR cache for {doc_id}")
                continue

            try:
                file_path = ROOT / row["file_path"]
                pages = ingester.load(file_path)
                image = preprocessor.process(pages[0], augment=False)
                img_w, img_h = image.size

                words = cache_entry["words"]
                boxes = cache_entry["boxes"]
                scores = cache_entry["scores"]

                line_groups = get_line_groups(words, boxes, img_h)
                for line_index, line_indices in enumerate(line_groups):
                    line_text = " ".join(words[i] for i in line_indices)
                    field_type = get_line_field_type(
                        line_text, line_indices, words, boxes, img_h
                    )
                    if field_type is None:
                        continue

                    x_min = min(boxes[i][0] for i in line_indices)
                    y_min = min(boxes[i][1] for i in line_indices)
                    x_max = max(boxes[i][2] for i in line_indices)
                    y_max = max(boxes[i][3] for i in line_indices)
                    x_min, y_min, x_max, y_max = clamp_box(
                        x_min, y_min, x_max, y_max, img_w, img_h
                    )
                    cropped_box = expand_box(
                        x_min, y_min, x_max, y_max, 0.15, img_w, img_h
                    )
                    field_dir = FIELD_CROPS_ROOT / field_type / label_dir(label)
                    field_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{doc_id}_{field_type}_{line_index:03d}.png"
                    crop_path = field_dir / crop_name
                    image.crop(cropped_box).save(crop_path, format="PNG")

                    line_scores = [scores[i] for i in line_indices]
                    avg_score = (
                        sum(line_scores) / len(line_scores) if line_scores else 0.0
                    )
                    record = {
                        "doc_id": doc_id,
                        "label": label,
                        "field_type": field_type,
                        "ocr_text": line_text,
                        "ocr_score": f"{avg_score:.4f}",
                        "crop_path": str(crop_path.relative_to(ROOT)),
                    }
                    writer.writerow(record)
                    saved_records.append(record)

                processed += 1
            except Exception as exc:
                print(f"Failed {doc_id}: {exc}")
                failed += 1

    write_statistics(saved_records)
    save_review_samples(saved_records)
    print(f"Built field crop dataset: {len(saved_records)} crops")
    print(f"Metadata: {FIELD_METADATA}")
    print(f"Statistics: {FIELD_STATISTICS}")
    print(f"Review samples: {REVIEW_ROOT}")
    if failed:
        print(f"Skipped {failed} documents due to errors.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build field-focused OCR crop dataset from existing OCR cache"
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline_prod.yaml",
        help="Pipeline config to load image ingestion/preprocessing settings",
    )
    parser.add_argument(
        "--splits-dir",
        default=str(ROOT / "data" / "splits"),
        help="Directory containing train/val/test manifests and OCR cache files",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split(s) to process  (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing field crop outputs before building",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without writing any crops",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as fh:
        import yaml

        cfg = yaml.safe_load(fh)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    build_field_dataset(
        Path(args.splits_dir), cfg, splits, args.overwrite, args.dry_run
    )


if __name__ == "__main__":
    main()
