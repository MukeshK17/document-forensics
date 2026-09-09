"""
VLM Complementarity Experiment
================================
Tests whether Qwen2-VL-7B-Instruct adds detection signal beyond the existing
4-stage fake-PAN-card detection ensemble.

Reproduces the experiment described in experiments/vlm_complementarity/README.md

Usage:
    # Step 1: build the balanced test split (run locally, not on Kaggle)
    python vlm_experiment.py --prepare-data \
        --fake-dir data/raw/fake_pan_card \
        --real-dir data/raw/pan_card \
        --out-dir vlm_test_split

    # Step 2: run VLM inference (run on Kaggle/Colab with GPU)
    python vlm_experiment.py --run-vlm \
        --test-split vlm_test_split \
        --out vlm_results.csv

    # Step 3: merge with existing system results and compute metrics
    python vlm_experiment.py --analyze \
        --vlm-results vlm_results.csv \
        --existing-results existing_results.csv \
        --out merged_results.csv
"""

import argparse
import csv
import json
import os
import random
import re
import shutil

import pandas as pd

# ---------------------------------------------------------------------------
# STEP 1: Build a balanced test split from the raw dataset folders
# ---------------------------------------------------------------------------

def prepare_data(fake_dir: str, real_dir: str, out_dir: str, seed: int = 42):
    """
    Copies all fake images + an equal-sized random sample of real images
    into a single folder, and writes labels.csv mapping filename -> REAL/FAKE.
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png")
    fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(valid_ext)]
    real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(valid_ext)]

    n_real = len(fake_files)  # balanced split
    sampled_real = random.sample(real_files, min(n_real, len(real_files)))

    rows = []
    for f in fake_files:
        shutil.copy(os.path.join(fake_dir, f), os.path.join(out_dir, f))
        rows.append({"image_path": f, "label": "FAKE"})

    for f in sampled_real:
        shutil.copy(os.path.join(real_dir, f), os.path.join(out_dir, f))
        rows.append({"image_path": f, "label": "REAL"})

    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total images: {len(rows)} ({len(fake_files)} fake, {len(sampled_real)} real)")
    print(f"Saved to: {out_dir}/")


# ---------------------------------------------------------------------------
# STEP 2: Run VLM inference (requires GPU )
# ---------------------------------------------------------------------------

VLM_PROMPT = """You are examining an Indian PAN card image for signs of forgery or tampering.

Look carefully at:
- Text alignment, font consistency, spacing irregularities
- Photo quality, edges, and blending with the background
- Any visible signs of digital editing, overlay, or splicing
- Ink/color consistency across text fields

Respond ONLY in this exact JSON format, nothing else:
{"decision": "FAKE" or "REAL" or "INSUFFICIENT_EVIDENCE", "confidence": <float 0 to 1>, "reason": "<one sentence, specific visual evidence you saw or why you could not tell>"}

Use INSUFFICIENT_EVIDENCE if the image quality, resolution, or lack of visible tampering marks makes it genuinely impossible to tell from pixels alone. Do not guess just to produce FAKE or REAL."""


def load_vlm_model(max_pixels_tokens: int = 1024):
    """
    Loads Qwen2-VL-7B-Instruct in 4-bit quantization.
    max_pixels_tokens controls vision-encoder memory usage — lower this if
    you hit CUDA OOM during the vision attention step.
    """
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels_tokens * 28 * 28,
    )

    return model, processor


def run_vlm_inference(image_path: str, model, processor) -> dict:
    """Runs a single image through the VLM and returns a parsed decision dict."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    try:
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else None
    except Exception:
        parsed = None

    return {
        "raw_output": output_text,
        "decision": parsed.get("decision") if parsed else "PARSE_ERROR",
        "confidence": parsed.get("confidence") if parsed else None,
        "reason": parsed.get("reason") if parsed else None,
    }


def run_vlm_on_split(test_split_dir: str, out_csv: str, checkpoint_every: int = 10):
    """Runs the VLM over every image in the test split and saves results to CSV."""
    import torch
    from tqdm import tqdm

    labels_path = os.path.join(test_split_dir, "labels.csv")
    labels_df = pd.read_csv(labels_path)
    labels_df["full_path"] = test_split_dir + "/" + labels_df["image_path"]

    missing = [p for p in labels_df["full_path"] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:3]}")

    print("Loading VLM (this may take a few minutes on first run)...")
    model, processor = load_vlm_model()

    results = []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        try:
            vlm_result = run_vlm_inference(row["full_path"], model, processor)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            vlm_result = {"raw_output": "OOM_ERROR", "decision": "OOM_ERROR", "confidence": None, "reason": None}
        except Exception as e:
            vlm_result = {"raw_output": str(e), "decision": "RUNTIME_ERROR", "confidence": None, "reason": None}

        results.append({
            "image_path": row["image_path"],
            "ground_truth": row["label"],
            "vlm_decision": vlm_result["decision"],
            "vlm_confidence": vlm_result["confidence"],
            "vlm_reason": vlm_result["reason"],
            "raw_output": vlm_result["raw_output"],
        })

        torch.cuda.empty_cache()

        if len(results) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(out_csv.replace(".csv", "_partial.csv"), index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv, index=False)

    print(results_df["vlm_decision"].value_counts())
    print(f"\nDistinct reason strings: {results_df['vlm_reason'].nunique()} out of {len(results_df)}")
    print(f"Saved to: {out_csv}")


# ---------------------------------------------------------------------------
# STEP 3: Merge VLM results with existing-system results and compute metrics
# ---------------------------------------------------------------------------

def analyze(vlm_results_csv: str, existing_results_csv: str, out_csv: str):
    from sklearn.metrics import f1_score, precision_score, recall_score

    existing_df = pd.read_csv(existing_results_csv)
    vlm_df = pd.read_csv(vlm_results_csv)

    existing_df["filename_lower"] = existing_df["filename"].str.lower()
    vlm_df["image_path_lower"] = vlm_df["image_path"].str.lower()

    merged = vlm_df.merge(
        existing_df, left_on="image_path_lower", right_on="filename_lower", how="inner"
    )
    print(f"Merged rows: {len(merged)}")

    merged["gt_match"] = (merged["ground_truth"] == "FAKE") == merged["expected_fake"]
    n_mismatch = (~merged["gt_match"]).sum()
    if n_mismatch:
        print(f"WARNING: {n_mismatch} ground truth mismatches between the two sources")

    y_true = merged["expected_fake"].astype(int)
    y_existing = merged["collective_fake"].astype(int)
    y_vlm = (merged["vlm_decision"] == "FAKE").astype(int)

    merged["existing_correct"] = (y_existing == y_true)
    merged["vlm_correct"] = (y_vlm == y_true)

    crosstab = pd.crosstab(
        merged["existing_correct"], merged["vlm_correct"],
        rownames=["Existing correct"], colnames=["VLM correct"],
    )
    print("\nComplementarity crosstab:")
    print(crosstab)

    existing_right_vlm_wrong = ((merged["existing_correct"]) & (~merged["vlm_correct"])).sum()
    existing_wrong_vlm_right = ((~merged["existing_correct"]) & (merged["vlm_correct"])).sum()
    both_wrong = ((~merged["existing_correct"]) & (~merged["vlm_correct"])).sum()
    both_right = ((merged["existing_correct"]) & (merged["vlm_correct"])).sum()

    print(f"\nBoth correct: {both_right}")
    print(f"Existing correct, VLM wrong: {existing_right_vlm_wrong}")
    print(f"Existing wrong, VLM correct: {existing_wrong_vlm_right}  <-- key complementarity number")
    print(f"Both wrong: {both_wrong}")

    y_or = ((y_vlm == 1) | (y_existing == 1)).astype(int)

    def report(name, y_pred):
        print(f"{name}: F1={f1_score(y_true, y_pred):.3f} "
              f"P={precision_score(y_true, y_pred):.3f} "
              f"R={recall_score(y_true, y_pred):.3f}")

    print()
    report("Existing alone", y_existing)
    report("VLM alone", y_vlm)
    report("Naive OR-ensemble", y_or)

    unique_catches = merged[(~merged["existing_correct"]) & (merged["vlm_correct"])]
    print(f"\nCases where VLM was right and existing was wrong: {len(unique_catches)}")

    merged.to_csv(out_csv, index=False)
    print(f"\nFull merged data saved to: {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VLM Complementarity Experiment")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--run-vlm", action="store_true")
    parser.add_argument("--analyze", action="store_true")

    parser.add_argument("--fake-dir", default="data/raw/fake_pan_card")
    parser.add_argument("--real-dir", default="data/raw/pan_card")
    parser.add_argument("--test-split", default="vlm_test_split")
    parser.add_argument("--out-dir", default="vlm_test_split")

    parser.add_argument("--vlm-results", default="vlm_results.csv")
    parser.add_argument("--existing-results", default="existing_results.csv")
    parser.add_argument("--out", default="merged_results.csv")

    args = parser.parse_args()

    if args.prepare_data:
        prepare_data(args.fake_dir, args.real_dir, args.out_dir)
    elif args.run_vlm:
        run_vlm_on_split(args.test_split, args.vlm_results)
    elif args.analyze:
        analyze(args.vlm_results, args.existing_results, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
