# import csv
# import json
# import sys
# from collections import Counter
# from pathlib import Path

# sys.path.insert(0, ".")
# from src.models.cv_analyzer import analyze_image
# from src.models.tamper_detector import detect_regional_tampering
# from src.validators.field_validator import assess
# from tools.visual_forensics import analyze_document as run_visual_forensics

# SPLITS_DIR = Path("data/splits")
# OUTPUT_PATH = Path("data/splits/baseline_results_all.csv")

# RAW_DIRS = {
#     "PAN_CARD": Path("data/raw/pan_card"),
#     "FAKE_PAN": Path("data/raw/fake_pan_card"),
# }

# results = []

# for split in ["train", "val", "test"]:
#     cache_path = SPLITS_DIR / f"{split}_ocr_cache.jsonl"
#     csv_path = SPLITS_DIR / f"{split}.csv"

#     doc_labels = {}
#     with open(csv_path) as f:
#         for row in csv.DictReader(f):
#             doc_labels[row["doc_id"]] = row["label"]

#     with open(cache_path) as f:
#         for line in f:
#             entry = json.loads(line)
#             doc_id = entry["doc_id"]
#             label = doc_labels.get(doc_id, "UNKNOWN")

#             if label not in ("PAN_CARD", "FAKE_PAN"):
#                 continue

#             # OCR rules score
#             result = assess(entry)
#             ocr_score = int(result["validation"]["risk_score"])
#             ocr_violations = result["validation"]["violations"]

#             # CV score
#             cv_score = 0
#             cv_issues = []
#             # if label in RAW_DIRS:
#             #     matches = list(RAW_DIRS[label].glob(f"{doc_id}.*"))
#             #     if matches:
#             #         cv_result = analyze_image(str(matches[0]))
#             #         cv_score = cv_result["cv_risk_score"]
#             #         cv_issues = cv_result.get("all_issues", [])

#             # if label in RAW_DIRS:
#             #     matches = list(RAW_DIRS[label].glob(f"{doc_id}.*"))
#             #     if matches:
#             #         cv_result = analyze_image(str(matches[0]))
#             #         cv_score = cv_result["cv_risk_score"]
#             #         cv_issues = cv_result.get("all_issues", [])

#             #         # Stage 3: Regional tampering detection
#             #         tamper_result = detect_regional_tampering(
#             #             str(matches[0]), entry.get("boxes", [])
#             #         )
#             #         tamper_score = tamper_result.get("tamper_score", 0)
#             #         tamper_regions = len(tamper_result.get("suspicious_regions", []))
#             #         # tamper_score = 0
#             #         # tamper_regions = 0
#             #         cv_score = min(cv_score + tamper_score, 100)
#             #         if tamper_regions > 0:
#             #             cv_issues.append(f"regional_tampering:{tamper_regions}_regions")
#             # # Combined
#             # combined_score = min(ocr_score + cv_score, 100)
#             # combined_verdict = "SUSPICIOUS" if combined_score >= 30 else "CLEAN"
#             if label in RAW_DIRS:
#                 matches = list(RAW_DIRS[label].glob(f"{doc_id}.*"))
#                 if matches:
#                     img_file_path = matches[0]

#                     # 2. Run Regional Tampering Sharpness Check (Median Metric Fix)
#                     tamper_result = detect_regional_tampering(str(img_file_path), entry.get("boxes", []))
#                     tamper_score = tamper_result.get("tamper_score", 0)
#                     tamper_regions = len(tamper_result.get("suspicious_regions", []))

#                     if tamper_regions > 0:
#                         cv_issues.append(f"regional_tampering:{tamper_regions}_regions")

#                     # 3. Run Visual Forensics Module (Face Counter & Pixel Color Deep Scan)

#                     forensic_results = run_visual_forensics(img_file_path, entry)

#                     forensic_score = forensic_results["forensic_risk_score"]
#                     if forensic_results["cv_issues"]:
#                         cv_issues.append(forensic_results["cv_issues"])

#                     # Aggregate complete Image Layer metrics
#                     cv_score = min(tamper_score + forensic_score, 100)

#             # 4. Enforcing Combined Verdict Configuration
#             combined_score = min(ocr_score + cv_score, 100)
#             combined_verdict = "SUSPICIOUS" if combined_score >= 30 else "CLEAN"

#             results.append(
#                 {
#                     "split": split,
#                     "doc_id": doc_id,
#                     "label": label,
#                     "ocr_risk_score": ocr_score,
#                     "cv_risk_score": cv_score,
#                     "combined_score": combined_score,
#                     "verdict": combined_verdict,
#                     "violation_count": result["validation"]["violation_count"],
#                     "violations": "|".join(result["validation"]["violations"]),
#                     "cv_issues": "|".join(cv_issues),
#                     "avg_confidence": result["fields"]["avg_confidence"],
#                     "pan_found": len(result["fields"]["pan_numbers"]) > 0,
#                     "date_found": len(result["fields"]["dates"]) > 0,
#                     "tamper_score": tamper_score,
#                     "tamper_regions": tamper_regions,
#                 }
#             )

# # Write CSV
# with open(OUTPUT_PATH, "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=results[0].keys())
#     writer.writeheader()
#     writer.writerows(results)

# # Split by label
# fake_docs = [r for r in results if r["label"] == "FAKE_PAN"]
# real_docs = [r for r in results if r["label"] == "PAN_CARD"]

# # Core counts
# tp = sum(1 for r in fake_docs if r["verdict"] == "SUSPICIOUS")
# fn = sum(1 for r in fake_docs if r["verdict"] == "CLEAN")
# fp = sum(1 for r in real_docs if r["verdict"] == "SUSPICIOUS")
# tn = sum(1 for r in real_docs if r["verdict"] == "CLEAN")

# precision = tp / (tp + fp) if (tp + fp) > 0 else 0
# recall = tp / (tp + fn) if (tp + fn) > 0 else 0
# f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
# fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# print(f"{'=' * 50}")
# print("STAGE 2 RESULTS — Rules + CV Combined")
# print(f"{'=' * 50}")
# print(f"Real PAN documents:   {len(real_docs)}")
# print(f"Fake PAN documents:   {len(fake_docs)}")
# print(f"{'=' * 50}")
# print(f"True Positives:       {tp}  (fakes caught)")
# print(f"False Negatives:      {fn}  (fakes missed)")
# print(f"False Positives:      {fp}  (real wrongly flagged)")
# print(f"True Negatives:       {tn}  (real correctly passed)")
# print(f"{'=' * 50}")
# print(f"Precision:            {precision:.3f}")
# print(f"Recall:               {recall:.3f}")
# print(f"F1 Score:             {f1:.3f}")
# print(f"False Positive Rate:  {fpr:.3f}")
# print(f"{'=' * 50}")
# print()

# # Score breakdown
# print("Score breakdown (fake docs):")
# print(
#     f"  OCR only caught:     {sum(1 for r in fake_docs if int(r['ocr_risk_score']) >= 30)}"
# )
# print(
#     f"  CV only caught:      {sum(1 for r in fake_docs if int(r['cv_risk_score']) >= 30)}"
# )
# print(f"  Combined caught:     {tp}")
# print()

# # Violation breakdown
# all_violations = []
# for r in results:
#     if r["violations"]:
#         all_violations.extend(r["violations"].split("|"))

# print("OCR Violation breakdown:")
# for v, count in Counter(all_violations).most_common():
#     print(f"  {v:<35} {count}")

# # CV issues breakdown
# all_cv_issues = []
# for r in results:
#     if r["cv_issues"]:
#         all_cv_issues.extend(r["cv_issues"].split("|"))

# print("\nCV Issues breakdown:")
# for v, count in Counter(all_cv_issues).most_common():
#     print(f"  {v:<35} {count}")

# print("\nFakes missed (CLEAN verdict):")
# for r in fake_docs:
#     if r["verdict"] == "CLEAN":
#         print(
#             f"  {r['doc_id']}  ocr={r['ocr_risk_score']}  cv={r['cv_risk_score']}  combined={r['combined_score']}  conf={r['avg_confidence']}"
#         )

# print("\nComparison:")
# print("  Stage 1 (Rules only): Recall=0.380  Precision=0.217  F1=0.277  FPR=0.073")
# print(
#     f"  Stage 2 (Rules + CV): Recall={recall:.3f}  Precision={precision:.3f}  F1={f1:.3f}  FPR={fpr:.3f}"
# )
# print(f"\nResults saved to {OUTPUT_PATH}")


# import csv
# import json
# import sys
# from collections import Counter
# from pathlib import Path

# sys.path.insert(0, ".")
# from src.models.cv_analyzer import analyze_image
# from src.models.tamper_detector import detect_regional_tampering
# from src.validators.field_validator import assess
# from tools.visual_forensics import analyze_document as run_visual_forensics

# SPLITS_DIR = Path("data/splits")
# OUTPUT_PATH = Path("data/splits/baseline_results_all.csv")

# RAW_DIRS = {
#     "PAN_CARD": Path("data/raw/pan_card"),
#     "FAKE_PAN": Path("data/raw/fake_pan_card"),
# }

# results = []

# for split in ["train", "val", "test"]:
#     cache_path = SPLITS_DIR / f"{split}_ocr_cache.jsonl"
#     csv_path = SPLITS_DIR / f"{split}.csv"

#     if not csv_path.exists() or not cache_path.exists():
#         continue

#     doc_labels = {}
#     with open(csv_path) as f:
#         for row in csv.DictReader(f):
#             doc_labels[row["doc_id"]] = row["label"]

#     with open(cache_path) as f:
#         for line in f:
#             entry = json.loads(line)
#             doc_id = entry["doc_id"]
#             label = doc_labels.get(doc_id, "UNKNOWN")

#             if label not in ("PAN_CARD", "FAKE_PAN"):
#                 continue

#             # --- STAGE 1: Standard OCR Validation Pass ---
#             ocr_result = assess(entry)
#             ocr_score = int(ocr_result["validation"]["risk_score"])
#             ocr_verdict = ocr_result["validation"]["verdict"]

#             # --- STAGE 2: Handcrafted Computer Vision Pass ---
#             cv_score = 0
#             cv_issues = []
#             tamper_score = 0
#             tamper_regions = 0
#             forensic_score = 0

#             if label in RAW_DIRS:
#                 matches = list(RAW_DIRS[label].glob(f"{doc_id}.*"))
#                 if matches:
#                     img_file_path = matches[0]

#                     # 1. Base CV image fingerprinting
#                     cv_base = analyze_image(str(img_file_path))
#                     cv_score = cv_base.get("cv_risk_score", 0)
#                     cv_issues = cv_base.get("all_issues", [])

#                     # 2. Regional Median sharpness metric
#                     tamper_result = detect_regional_tampering(str(img_file_path), entry.get("boxes", []))
#                     tamper_score = tamper_result.get("tamper_score", 0)
#                     tamper_regions = len(tamper_result.get("suspicious_regions", []))
#                     if tamper_regions > 0:
#                         cv_issues.append(f"regional_tampering:{tamper_regions}_regions")

#                     # 3. Clean Visual Forensics (White fills scan + Face verification)
#                     forensic_results = run_visual_forensics(img_file_path, entry)
#                     forensic_score = forensic_results["forensic_risk_score"]
#                     if forensic_results["cv_issues"]:
#                         cv_issues.append(forensic_results["cv_issues"])

#                     cv_score = min(cv_score + tamper_score + forensic_score, 100)

#             # Consolidated Evaluation Verdict
#             combined_score = min(ocr_score + cv_score, 100)
#             combined_verdict = "SUSPICIOUS" if combined_score >= 30 else "CLEAN"

#             results.append(
#                 {
#                     "split": split,
#                     "doc_id": doc_id,
#                     "label": label,
#                     "ocr_risk_score": ocr_score,
#                     "ocr_verdict": ocr_verdict,
#                     "cv_risk_score": cv_score,
#                     "combined_score": combined_score,
#                     "verdict": combined_verdict,
#                     "violation_count": ocr_result["validation"]["violation_count"],
#                     "violations": "|".join(ocr_result["validation"]["violations"]),
#                     "cv_issues": "|".join(cv_issues),
#                     "avg_confidence": ocr_result["fields"]["avg_confidence"],
#                 }
#             )

# # Metrics calculation logic
# fake_docs = [r for r in results if r["label"] == "FAKE_PAN"]
# real_docs = [r for r in results if r["label"] == "PAN_CARD"]

# st1_tp = sum(1 for r in fake_docs if r["ocr_verdict"] == "SUSPICIOUS")
# st1_fn = sum(1 for r in fake_docs if r["ocr_verdict"] == "CLEAN")
# st1_fp = sum(1 for r in real_docs if r["ocr_verdict"] == "SUSPICIOUS")
# st1_tn = sum(1 for r in real_docs if r["ocr_verdict"] == "CLEAN")

# st1_precision = st1_tp / (st1_tp + st1_fp) if (st1_tp + st1_fp) > 0 else 0
# st1_recall = st1_tp / (st1_tp + st1_fn) if (st1_tp + st1_fn) > 0 else 0
# st1_f1 = 2 * st1_precision * st1_recall / (st1_precision + st1_recall) if (st1_precision + st1_recall) > 0 else 0
# st1_fpr = st1_fp / (st1_fp + st1_tn) if (st1_fp + st1_tn) > 0 else 0

# st2_tp = sum(1 for r in fake_docs if r["verdict"] == "SUSPICIOUS")
# st2_fn = sum(1 for r in fake_docs if r["verdict"] == "CLEAN")
# st2_fp = sum(1 for r in real_docs if r["verdict"] == "SUSPICIOUS")
# st2_tn = sum(1 for r in real_docs if r["verdict"] == "CLEAN")

# st2_precision = st2_tp / (st2_tp + st2_fp) if (st2_tp + st2_fp) > 0 else 0
# st2_recall = st2_tp / (st2_tp + st2_fn) if (st2_tp + st2_fn) > 0 else 0
# st2_f1 = 2 * st2_precision * st2_recall / (st2_precision + st2_recall) if (st2_precision + st2_recall) > 0 else 0
# st2_fpr = st2_fp / (st2_fp + st2_tn) if (st2_fp + st2_tn) > 0 else 0

# print(f"\n{'=' * 50}")
# print("STAGE 1 RESULTS — OCR Rules Pass Only")
# print(f"{'=' * 50}")
# print(f"True Positives:       {st1_tp}  (fakes caught)")
# print(f"False Negatives:      {st1_fn}  (fakes missed)")
# print(f"False Positives:      {st1_fp}  (real wrongly flagged)")
# print(f"True Negatives:       {st1_tn}  (real correctly passed)")
# print(f"Recall: {st1_recall:.3f} | Precision: {st1_precision:.3f} | F1: {st1_f1:.3f} | FPR: {st1_fpr:.3f}")

# print(f"\n{'=' * 50}")
# print("STAGE 2 RESULTS — Rules + Handcrafted CV Combined")
# print(f"{'=' * 50}")
# print(f"Real PAN documents:   {len(real_docs)}")
# print(f"Fake PAN documents:   {len(fake_docs)}")
# print(f"{'=' * 50}")
# print(f"True Positives:       {st2_tp}  (fakes caught)")
# print(f"False Negatives:      {st2_fn}  (fakes missed)")
# print(f"False Positives:      {st2_fp}  (real wrongly flagged)")
# print(f"True Negatives:       {st2_tn}  (real correctly passed)")
# print(f"{'=' * 50}")
# print(f"Precision:            {st2_precision:.3f}")
# print(f"Recall:               {st2_recall:.3f}")
# print(f"F1 Score:             {st2_f1:.3f}")
# print(f"False Positive Rate:  {st2_fpr:.3f}")
# print(f"{'=' * 50}\n")

# all_violations = []
# for r in results:
#     if r["violations"]:
#         all_violations.extend(r["violations"].split("|"))
# print("OCR Violation Breakdown:")
# for v, count in Counter(all_violations).most_common(10):
#     print(f"  {v:<40} {count}")

# all_cv_issues = []
# for r in results:
#     if r["cv_issues"]:
#         all_cv_issues.extend(r["cv_issues"].split("|"))
# print("\nCV Issues Breakdown:")
# for v, count in Counter(all_cv_issues).most_common(15):
#     print(f"  {v:<40} {count}")


# 3 stages
# import csv, json, sys
# from pathlib import Path
# sys.path.insert(0, ".")
# from src.models.cv_analyzer import analyze_image
# from src.models.tamper_detector import detect_regional_tampering
# from src.validators.field_validator import assess
# from tools.visual_forensics import analyze_document as run_visual_forensics

# SPLITS_DIR = Path("data/splits")
# OUTPUT_PATH = Path("data/splits/baseline_results_v2.csv")
# RAW_DIRS = {
#     "PAN_CARD": Path("data/raw/pan_card"),
#     "FAKE_PAN": Path("data/raw/fake_pan_card"),
# }

# # ── High-signal violations only ───────────────────────────────────────────────
# # Scores based on fake%/real% ratio — only violations with ratio > 3x kept
# HIGH_SIGNAL_SCORES = {
#     "known_testing_pan_hash":              100,  # pure fake signal
#     "multiple_conflicting_pans_detected":   80,  # pure fake signal
#     "invalid_taxpayer_entity_code:E":       70,  # pure fake
#     "invalid_taxpayer_entity_code:K":       70,  # pure fake
#     "invalid_taxpayer_entity_code:U":       70,  # pure fake
#     "invalid_taxpayer_entity_code:W":       70,  # pure fake
#     "invalid_taxpayer_entity_code:R":       50,  # 42x more likely fake
#     "missing_pan_number_block":             40,  # 5x more likely fake
#     "future_birth_year":                    60,  # near-pure fake
#     "impossible_calendar_date_value":       30,  # 8x more likely fake
# }
# # Violations explicitly ignored (fire equally or more on reals):
# IGNORED_VIOLATIONS = {
#     "pan_5th_character_surname_mismatch",  # fires 79% on reals — useless
#     "missing_critical_issuer_keywords",   # identical rates
#     "pan_recovered_via_normalization",     # minor OCR noise
#     "missing_date_of_birth",              # near-identical rates
# }

# # Thresholds per stage
# OCR_THRESHOLD      = 40
# TAMPER_THRESHOLD   = 40
# FORENSIC_THRESHOLD = 80   # raised — forensics has too many false positives below this

# # Boilerplate words to skip in ink color check
# TEMPLATE_WORDS = {
#     "SIGNATURE", "DATE", "BIRTH", "INCOME", "TAX",
#     "FATHER", "NAME", "GOVT", "INDIA", "ACCOUNT",
#     "PERMANENT", "NUMBER", "CARD", "MINISTRY"
# }

# results = []

# for split in ["train", "val", "test"]:
#     cache_path = SPLITS_DIR / f"{split}_ocr_cache.jsonl"
#     csv_path   = SPLITS_DIR / f"{split}.csv"
#     if not csv_path.exists() or not cache_path.exists(): continue

#     doc_labels = {}
#     with open(csv_path) as f:
#         for row in csv.DictReader(f):
#             doc_labels[row["doc_id"]] = row["label"]

#     with open(cache_path) as f:
#         for line in f:
#             entry  = json.loads(line)
#             doc_id = entry["doc_id"]
#             label  = doc_labels.get(doc_id, "UNKNOWN")
#             if label not in ("PAN_CARD", "FAKE_PAN"): continue

#             # ── Stage 1: OCR — high signal violations only ────────────────────
#             ocr_result = assess(entry)
#             violations = ocr_result["validation"]["violations"]
#             ocr_score  = 0
#             for v in violations:
#                 if v in IGNORED_VIOLATIONS:
#                     continue
#                 # handle violations with dynamic suffix e.g. "invalid_taxpayer_entity_code:R"
#                 score = HIGH_SIGNAL_SCORES.get(v, 0)
#                 if score == 0:
#                     # try prefix match for dynamic violations
#                     for key, val in HIGH_SIGNAL_SCORES.items():
#                         if v.startswith(key.split(":")[0]) and ":" in key:
#                             score = val
#                             break
#                 ocr_score += score
#             ocr_score = min(ocr_score, 100)
#             ocr_fires = ocr_score >= OCR_THRESHOLD

#             # ── Stages 2 & 3: Visual ─────────────────────────────────────────
#             tamper_score   = 0
#             forensic_score = 0
#             cv_issues      = []
#             tamper_regions = 0

#             matches = list(RAW_DIRS.get(label, Path("nonexistent")).glob(f"{doc_id}.*"))
#             if matches:
#                 img_path = matches[0]

#                 # tamper_result  = detect_regional_tampering(str(img_path), entry.get("boxes", []))
#                 tamper_result = detect_regional_tampering(
#                     str(img_path),
#                     entry.get("boxes", []),
#                     entry.get("scores", [])   # ← pass OCR confidence scores
#                 )
#                 tamper_score   = tamper_result.get("tamper_score", 0)
#                 tamper_regions = len(tamper_result.get("suspicious_regions", []))

#                 forensics_result = run_visual_forensics(img_path, entry)
#                 forensic_score   = forensics_result.get("forensic_risk_score", 0)

#                 # Filter boilerplate from ink color flags
#                 for issue in (forensics_result.get("cv_issues", "") or "").split("|"):
#                     if not issue: continue
#                     if "invalid_ink_color_detected" in issue:
#                         flagged_text = issue.split(":")[-1].upper()
#                         if any(w in flagged_text for w in TEMPLATE_WORDS):
#                             continue
#                     cv_issues.append(issue)

#                 cv_base   = analyze_image(str(img_path))
#                 cv_issues += cv_base.get("all_issues", [])

#             tamper_fires   = tamper_score   >= TAMPER_THRESHOLD
#             # forensic_fires = forensic_score >= FORENSIC_THRESHOLD

#             forensic_fires = (
#                 forensic_score >= 80 and
#                 any("solid_digital_box_overlay" in i or
#                     "pasted_high_res_photo_detected" in i
#                     for i in cv_issues)
#             )

#             # ── 2-of-3 voting ─────────────────────────────────────────────────
#             stages_fired = sum([ocr_fires, tamper_fires, forensic_fires])
#             verdict      = "SUSPICIOUS" if stages_fired >= 2 else "CLEAN"

#             results.append({
#                 "split": split, "doc_id": doc_id, "label": label,
#                 "verdict": verdict,
#                 "ocr_score": ocr_score, "tamper_score": tamper_score,
#                 "forensic_score": forensic_score,
#                 "ocr_fires": int(ocr_fires), "tamper_fires": int(tamper_fires),
#                 "forensic_fires": int(forensic_fires),
#                 "stages_fired": stages_fired,
#                 "violations": "|".join(v for v in violations if v not in IGNORED_VIOLATIONS),
#                 "cv_issues": "|".join(cv_issues),
#                 "tamper_regions": tamper_regions,
#             })

# # ── Write CSV ─────────────────────────────────────────────────────────────────
# OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# with open(OUTPUT_PATH, "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=results[0].keys())
#     writer.writeheader()
#     writer.writerows(results)

# # ── Metrics ───────────────────────────────────────────────────────────────────
# fake_docs = [r for r in results if r["label"] == "FAKE_PAN"]
# real_docs = [r for r in results if r["label"] == "PAN_CARD"]

# tp = sum(1 for r in fake_docs if r["verdict"] == "SUSPICIOUS")
# fn = sum(1 for r in fake_docs if r["verdict"] == "CLEAN")
# fp = sum(1 for r in real_docs if r["verdict"] == "SUSPICIOUS")
# tn = sum(1 for r in real_docs if r["verdict"] == "CLEAN")

# precision = tp / (tp + fp + 1e-8)
# recall    = tp / (tp + fn + 1e-8)
# f1        = 2 * precision * recall / (precision + recall + 1e-8)
# fpr       = fp / (fp + tn + 1e-8)

# print(f"\n{'='*55}")
# print(f"  BASELINE v2  (2-of-3, high-signal OCR only)")
# print(f"{'='*55}")
# print(f"  Real: {len(real_docs)}   Fake: {len(fake_docs)}")
# print(f"{'='*55}")
# print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
# print(f"  Precision : {precision:.3f}")
# print(f"  Recall    : {recall:.3f}")
# print(f"  F1        : {f1:.3f}")
# print(f"  FPR       : {fpr:.3f}")
# print(f"{'='*55}")

# print(f"\nPer-stage on fakes:")
# print(f"  OCR:      {sum(r['ocr_fires'] for r in fake_docs)}/{len(fake_docs)}")
# print(f"  Tamper:   {sum(r['tamper_fires'] for r in fake_docs)}/{len(fake_docs)}")
# print(f"  Forensic: {sum(r['forensic_fires'] for r in fake_docs)}/{len(fake_docs)}")

# print(f"\nPer-stage on reals (FP sources):")
# print(f"  OCR:      {sum(r['ocr_fires'] for r in real_docs)}/{len(real_docs)}")
# print(f"  Tamper:   {sum(r['tamper_fires'] for r in real_docs)}/{len(real_docs)}")
# print(f"  Forensic: {sum(r['forensic_fires'] for r in real_docs)}/{len(real_docs)}")

# print(f"\nResults saved to {OUTPUT_PATH}")


# all four stages


"""
tools/run_baseline.py

Four-stage PAN card fake detection pipeline.

Stage 1 — OCR Rules        (field_validator)
Stage 2 — Tamper Detector  (sharpness + confidence signals)
Stage 3 — Visual Forensics (solid overlay + face analysis)
Stage 4 — DL Patch Model   (EfficientNet-B0, AUC=0.732)

Voting: SUSPICIOUS if >= 2 stages fire.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.models.cv_analyzer import analyze_image
from src.models.dl_scorer import PatchTamperScorer
from src.models.tamper_detector import detect_regional_tampering
from src.validators.field_validator import assess
from tools.visual_forensics import analyze_document as run_visual_forensics

# ── paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR = Path("data/splits")
OUTPUT_PATH = Path("data/splits/baseline_results_final.csv")
CHECKPOINT = Path("colab/best_patch_model.pt")
RAW_DIRS = {
    "PAN_CARD": Path("data/raw/pan_card"),
    "FAKE_PAN": Path("data/raw/fake_pan_card"),
}

# ── stage thresholds ──────────────────────────────────────────────────────────
OCR_THRESHOLD = 40
TAMPER_THRESHOLD = 45
FORENSIC_THRESHOLD = 80
# DL threshold comes from checkpoint (0.952)

# ── high-signal OCR violations only ──────────────────────────────────────────
HIGH_SIGNAL_SCORES = {
    "known_testing_pan_hash": 100,
    "multiple_conflicting_pans_detected": 80,
    "invalid_taxpayer_entity_code:E": 70,
    "invalid_taxpayer_entity_code:K": 70,
    "invalid_taxpayer_entity_code:U": 70,
    "invalid_taxpayer_entity_code:W": 70,
    "invalid_taxpayer_entity_code:R": 50,
    "missing_pan_number_block": 40,
    "future_birth_year": 60,
    "impossible_calendar_date_value": 30,
}
IGNORED_VIOLATIONS = {
    "pan_5th_character_surname_mismatch",
    "missing_critical_issuer_keywords",
    "pan_recovered_via_normalization",
    "missing_date_of_birth",
}
TEMPLATE_WORDS = {
    "SIGNATURE",
    "DATE",
    "BIRTH",
    "INCOME",
    "TAX",
    "FATHER",
    "NAME",
    "GOVT",
    "INDIA",
    "ACCOUNT",
    "PERMANENT",
    "NUMBER",
    "CARD",
    "MINISTRY",
}

# ── load DL model once ────────────────────────────────────────────────────────
dl_scorer = None
if CHECKPOINT.exists():
    try:
        dl_scorer = PatchTamperScorer(str(CHECKPOINT))
    except Exception as e:
        print(f"[WARN] Could not load DL model: {e}")
        print("[WARN] Stage 4 (DL) will be skipped.")
else:
    print(f"[WARN] Checkpoint not found at {CHECKPOINT} — Stage 4 will be skipped.")

# ── process all splits ────────────────────────────────────────────────────────
results = []

for split in ["train", "val", "test"]:
    cache_path = SPLITS_DIR / f"{split}_ocr_cache.jsonl"
    csv_path = SPLITS_DIR / f"{split}.csv"
    if not csv_path.exists() or not cache_path.exists():
        continue

    doc_labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            doc_labels[row["doc_id"]] = row["label"]

    with open(cache_path) as f:
        for line in f:
            entry = json.loads(line)
            doc_id = entry["doc_id"]
            label = doc_labels.get(doc_id, "UNKNOWN")
            if label not in ("PAN_CARD", "FAKE_PAN"):
                continue

            # ── Stage 1: OCR rules ────────────────────────────────────────────
            ocr_result = assess(entry)
            violations = ocr_result["validation"]["violations"]
            ocr_score = 0
            for v in violations:
                if v in IGNORED_VIOLATIONS:
                    continue
                score = HIGH_SIGNAL_SCORES.get(v, 0)
                if score == 0:
                    prefix = v.split(":")[0] + ":"
                    for key, val in HIGH_SIGNAL_SCORES.items():
                        if key.startswith(prefix):
                            score = val
                            break
                ocr_score += score
            ocr_score = min(ocr_score, 100)
            ocr_fires = ocr_score >= OCR_THRESHOLD

            # ── Stages 2-4: visual + DL (need image file) ────────────────────
            tamper_score = 0
            forensic_score = 0
            dl_score = 0.0
            dl_fires = False
            cv_issues = []
            tamper_regions = 0

            matches = list(RAW_DIRS.get(label, Path("__none__")).glob(f"{doc_id}.*"))
            if matches:
                img_path = matches[0]

                # Stage 2: tamper detector
                tamper_result = detect_regional_tampering(
                    str(img_path),
                    entry.get("boxes", []),
                    entry.get("scores", []),
                )
                tamper_score = tamper_result.get("tamper_score", 0)
                tamper_regions = len(tamper_result.get("suspicious_regions", []))

                # Stage 3: visual forensics
                forensics_result = run_visual_forensics(img_path, entry)
                forensic_score = forensics_result.get("forensic_risk_score", 0)

                # filter boilerplate from ink color flags
                for issue in (forensics_result.get("cv_issues") or "").split("|"):
                    if not issue:
                        continue
                    if "invalid_ink_color_detected" in issue:
                        flagged_text = issue.split(":")[-1].upper()
                        if any(w in flagged_text for w in TEMPLATE_WORDS):
                            continue
                    cv_issues.append(issue)

                cv_base = analyze_image(str(img_path))
                cv_issues += cv_base.get("all_issues", [])

                # Stage 4: DL patch model
                if dl_scorer is not None:
                    dl_result = dl_scorer.score(str(img_path))
                    dl_score = dl_result.get("dl_score", 0.0)
                    dl_fires = dl_result.get("dl_fires", False)

            tamper_fires = tamper_score >= TAMPER_THRESHOLD
            forensic_fires = forensic_score >= FORENSIC_THRESHOLD

            # ── voting: 2 of 4 stages must fire ──────────────────────────────
            stages_fired = sum([ocr_fires, tamper_fires, forensic_fires, dl_fires])
            verdict = "SUSPICIOUS" if stages_fired >= 2 else "CLEAN"

            results.append(
                {
                    "split": split,
                    "doc_id": doc_id,
                    "label": label,
                    "verdict": verdict,
                    "ocr_score": ocr_score,
                    "tamper_score": tamper_score,
                    "forensic_score": forensic_score,
                    "dl_score": round(dl_score, 4),
                    "stages_fired": stages_fired,
                    "ocr_fires": int(ocr_fires),
                    "tamper_fires": int(tamper_fires),
                    "forensic_fires": int(forensic_fires),
                    "dl_fires": int(dl_fires),
                    "violations": "|".join(
                        v for v in violations if v not in IGNORED_VIOLATIONS
                    ),
                    "cv_issues": "|".join(cv_issues),
                    "tamper_regions": tamper_regions,
                }
            )

# ── write CSV ─────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# ── metrics ───────────────────────────────────────────────────────────────────
fake_docs = [r for r in results if r["label"] == "FAKE_PAN"]
real_docs = [r for r in results if r["label"] == "PAN_CARD"]

tp = sum(1 for r in fake_docs if r["verdict"] == "SUSPICIOUS")
fn = sum(1 for r in fake_docs if r["verdict"] == "CLEAN")
fp = sum(1 for r in real_docs if r["verdict"] == "SUSPICIOUS")
tn = sum(1 for r in real_docs if r["verdict"] == "CLEAN")

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
fpr = fp / (fp + tn + 1e-8)

print(f"\n{'=' * 55}")
print("  FINAL PIPELINE  (4 stages, 2-of-4 voting)")
print(f"{'=' * 55}")
print(f"  Real: {len(real_docs)}   Fake: {len(fake_docs)}")
print(f"{'=' * 55}")
print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
print(f"  Precision : {precision:.3f}")
print(f"  Recall    : {recall:.3f}")
print(f"  F1        : {f1:.3f}")
print(f"  FPR       : {fpr:.3f}")
print(f"{'=' * 55}")

print(f"\nPer-stage firing on fakes ({len(fake_docs)}):")
print(f"  OCR:      {sum(r['ocr_fires'] for r in fake_docs)}/{len(fake_docs)}")
print(f"  Tamper:   {sum(r['tamper_fires'] for r in fake_docs)}/{len(fake_docs)}")
print(f"  Forensic: {sum(r['forensic_fires'] for r in fake_docs)}/{len(fake_docs)}")
print(f"  DL:       {sum(r['dl_fires'] for r in fake_docs)}/{len(fake_docs)}")

print(f"\nPer-stage firing on reals ({len(real_docs)}) — FP sources:")
print(f"  OCR:      {sum(r['ocr_fires'] for r in real_docs)}/{len(real_docs)}")
print(f"  Tamper:   {sum(r['tamper_fires'] for r in real_docs)}/{len(real_docs)}")
print(f"  Forensic: {sum(r['forensic_fires'] for r in real_docs)}/{len(real_docs)}")
print(f"  DL:       {sum(r['dl_fires'] for r in real_docs)}/{len(real_docs)}")

print(f"\nResults saved to {OUTPUT_PATH}")
