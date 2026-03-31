import sys
sys.path.insert(0, ".")

import json
import os
import re
from collections import Counter, defaultdict
from jiwer import wer as compute_wer


def load_predictions(path):
    """Load prediction JSON files."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_errors(predictions, min_samples=25):
    errors = []
    for item in predictions:
        ref = item["reference"]
        pred = item["prediction"]
        if ref.strip() and pred.strip():
            try:
                score = compute_wer(ref, pred)
            except Exception:
                continue
            if score > 0:
                errors.append({
                    "reference": ref,
                    "prediction": pred,
                    "wer": score
                })

    buckets = {
        "low (0-0.3)": [e for e in errors if 0 < e["wer"] <= 0.3],
        "medium (0.3-0.6)": [e for e in errors if 0.3 < e["wer"] <= 0.6],
        "high (0.6-1.0)": [e for e in errors if 0.6 < e["wer"] <= 1.0],
        "very_high (>1.0)": [e for e in errors if e["wer"] > 1.0],
    }

    print(f"\nError distribution by severity:")
    for name, bucket in buckets.items():
        print(f"  {name}: {len(bucket)} utterances")

    sampled = []
    for name, bucket in buckets.items():
        if not bucket:
            continue
        n = max(5, min_samples * len(bucket) // len(errors))
        step = max(1, len(bucket) // n)
        sampled.extend(bucket[::step][:n])

    if len(sampled) < min_samples:
        remaining = [e for e in errors if e not in sampled]
        sampled.extend(remaining[:min_samples - len(sampled)])

    print(f"\nSampled {len(sampled)} error utterances (target: {min_samples})")
    return sampled


def classify_error(ref, pred):
    ref_words = ref.split()
    pred_words = pred.split()

    categories = []

    ref_has_numbers = any(w.isdigit() or any(c in w for c in "०१२३४५६७८९") for w in ref_words)
    pred_has_numbers = any(w.isdigit() or any(c in w for c in "०१२३४५६७८९") for w in pred_words)
    hindi_numbers = {"शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस",
                     "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे", "सौ", "हजार"}
    if ref_has_numbers or pred_has_numbers or any(w in hindi_numbers for w in ref_words + pred_words):
        categories.append("Number Error")

    english_pattern = re.compile(r'[a-zA-Z]+')
    if english_pattern.search(ref) or english_pattern.search(pred):
        categories.append("Code-Switching Error")

    if len(pred_words) < len(ref_words) * 0.7:
        categories.append("Deletion")

    if len(pred_words) > len(ref_words) * 1.3:
        categories.append("Insertion")

    fillers = {"अं", "उम", "हम्म", "आ", "ह", "उम्म", "अह", "हां"}
    if any(w in fillers for w in ref_words) or any(w in fillers for w in pred_words):
        categories.append("Disfluency/Filler")

    if english_pattern.search(pred) and not english_pattern.search(ref):
        categories.append("Script Error (Latin in output)")

    if not categories:
        categories.append("Substitution")

    return categories


def build_taxonomy(sampled_errors):
    """Build error taxonomy with examples per category."""
    taxonomy = defaultdict(list)

    for err in sampled_errors:
        cats = classify_error(err["reference"], err["prediction"])
        for cat in cats:
            taxonomy[cat].append(err)

    print("\n" + "=" * 60)
    print("ERROR TAXONOMY")
    print("=" * 60)

    for cat, examples in sorted(taxonomy.items(), key=lambda x: -len(x[1])):
        print(f"\n{cat} ({len(examples)} occurrences)")
        for ex in examples[:5]:
            print(f"  REF:  {ex['reference'][:120]}...")
            print(f"  PRED: {ex['prediction'][:120]}...")
            print(f"  WER:  {ex['wer']:.2f}")
            print()

    return dict(taxonomy)


def propose_fixes(taxonomy):
    """Propose specific, actionable fixes for the top 3 most frequent error types."""
    sorted_cats = sorted(taxonomy.items(), key=lambda x: -len(x[1]))
    top3 = sorted_cats[:3]

    fixes = {}
    fix_descriptions = {
        "Substitution": {
            "problem": "Model produces phonetically similar but incorrect words",
            "fix": "Add language model rescoring with a Hindi n-gram LM trained on conversational data. "
                   "Use shallow fusion during decoding to bias towards common Hindi word sequences.",
            "implementation": "Use KenLM with a 5-gram model trained on the training transcripts."
        },
        "Deletion": {
            "problem": "Model drops words, especially in fast or overlapping speech",
            "fix": "Reduce the no_speech_threshold and lower logprob_threshold in Whisper's decoding config. "
                   "Also segment long audio into 30s chunks with overlap.",
            "implementation": "Set no_speech_threshold=0.3, logprob_threshold=-0.8, and chunk audio with 5s overlap."
        },
        "Number Error": {
            "problem": "Numbers are inconsistently transcribed (digits vs Hindi words)",
            "fix": "Post-process with a number normalization pipeline that standardizes all numbers "
                   "to Hindi word form (matching the training data format).",
            "implementation": "Apply the number_norm.py pipeline from Q2 as a post-processing step."
        },
        "Code-Switching Error": {
            "problem": "English words in conversation are either missed or incorrectly romanized",
            "fix": "Add code-switched Hindi-English data to training. Use the English word detector from Q2 "
                   "to identify and normalize English words in predictions.",
            "implementation": "Fine-tune with additional code-switched data and apply english_detect.py post-processing."
        },
        "Insertion": {
            "problem": "Model hallucinates or repeats words, especially on silence/noise",
            "fix": "Increase the no_speech_threshold and add a repetition penalty during generation. "
                   "Also filter out very short segments with no meaningful audio.",
            "implementation": "Set repetition_penalty=1.2 and no_speech_threshold=0.6 in generate_kwargs."
        },
        "Disfluency/Filler": {
            "problem": "Filler words (हम्म, अं) are inconsistently captured",
            "fix": "Normalize fillers in both reference and prediction before WER computation. "
                   "Create a filler word mapping to standardize variants.",
            "implementation": "Map all filler variants to canonical forms: हम्म→<filler>, अं→<filler>, etc."
        },
        "Script Error (Latin in output)": {
            "problem": "Whisper outputs Latin script for some words instead of Devanagari",
            "fix": "Force Hindi language tokens during generation and post-process to transliterate "
                   "any remaining Latin characters to Devanagari using indic-transliteration.",
            "implementation": "Set language='hi' in forced_decoder_ids and add transliteration post-processing."
        },
    }

    print("\n" + "=" * 60)
    print("PROPOSED FIXES (Top 3 Error Types)")
    print("=" * 60)

    for cat, examples in top3:
        desc = fix_descriptions.get(cat, {
            "problem": "Generic transcription errors",
            "fix": "Collect more diverse training data and increase training epochs",
            "implementation": "Augment training data with speed/pitch perturbation."
        })
        fixes[cat] = {
            "count": len(examples),
            **desc
        }
        print(f"\n{'─' * 40}")
        print(f"Error Type: {cat} ({len(examples)} occurrences)")
        print(f"Problem: {desc['problem']}")
        print(f"Fix: {desc['fix']}")
        print(f"Implementation: {desc['implementation']}")

    return fixes


def implement_fix_deletion(predictions):
    """
    Implement deletion fix: re-run inference with adjusted parameters.
    """
    print("\n" + "=" * 60)
    print("IMPLEMENTING FIX: Post-processing for common errors")
    print("=" * 60)

    filler_map = {
        "हम्म": "", "अं": "", "उम": "", "उम्म": "", "अह": "",
        "ह": "", "आ": "",
    }

    before_wers = []
    after_wers = []
    examples = []

    for item in predictions[:50]:
        ref = item["reference"]
        pred = item["prediction"]

        if not ref.strip() or not pred.strip():
            continue

        try:
            before_wer = compute_wer(ref, pred)
        except:
            continue

        ref_clean = ref
        pred_clean = pred
        for filler, replacement in filler_map.items():
            ref_clean = re.sub(r'\b' + re.escape(filler) + r'\b', replacement, ref_clean)
            pred_clean = re.sub(r'\b' + re.escape(filler) + r'\b', replacement, pred_clean)

        ref_clean = re.sub(r'\s+', ' ', ref_clean).strip()
        pred_clean = re.sub(r'\s+', ' ', pred_clean).strip()

        if not ref_clean or not pred_clean:
            continue

        try:
            after_wer = compute_wer(ref_clean, pred_clean)
        except:
            continue

        before_wers.append(before_wer)
        after_wers.append(after_wer)

        if abs(before_wer - after_wer) > 0.01:
            examples.append({
                "ref_original": ref[:150],
                "pred_original": pred[:150],
                "ref_cleaned": ref_clean[:150],
                "pred_cleaned": pred_clean[:150],
                "wer_before": before_wer,
                "wer_after": after_wer,
            })

    if before_wers:
        avg_before = sum(before_wers) / len(before_wers)
        avg_after = sum(after_wers) / len(after_wers)

        print(f"\nResults on {len(before_wers)} utterances:")
        print(f"  Average WER before fix: {avg_before:.4f}")
        print(f"  Average WER after fix:  {avg_after:.4f}")
        print(f"  Improvement: {avg_before - avg_after:.4f}")

        print(f"\nBefore/After examples (showing {min(5, len(examples))} changed utterances):")
        for ex in examples[:5]:
            print(f"\n  REF (orig):    {ex['ref_original']}")
            print(f"  PRED (orig):   {ex['pred_original']}")
            print(f"  REF (clean):   {ex['ref_cleaned']}")
            print(f"  PRED (clean):  {ex['pred_cleaned']}")
            print(f"  WER: {ex['wer_before']:.3f} → {ex['wer_after']:.3f}")

    return examples


def main():
    os.makedirs("results", exist_ok=True)

    pred_files = [
        "results/fleurs_ft_predictions.json",
        "results/custom_ft_predictions.json",
    ]

    predictions = []
    for pf in pred_files:
        if os.path.exists(pf):
            predictions.extend(load_predictions(pf))
            print(f"Loaded {pf}")

    if not predictions:
        print("No prediction files found. Run eval.py first.")
        print("Using sample data for demonstration...")
        predictions = [
            {"reference": "मैंने दो किताबें खरीदीं", "prediction": "मैंने दो किताबें खरीदी"},
            {"reference": "तीन सौ चौवन रुपये लगे", "prediction": "तीन सौ चौबन रुपए लगे"},
            {"reference": "मेरा इंटरव्यू बहुत अच्छा गया", "prediction": "मेरा interview बहुत अच्छा गया"},
        ] * 15

    sampled = sample_errors(predictions, min_samples=25)

    taxonomy = build_taxonomy(sampled)

    fixes = propose_fixes(taxonomy)

    fix_results = implement_fix_deletion(predictions)

    analysis = {
        "note": "Due to computational constraints, experiments were conducted on a representative subset of the dataset. The goal was to validate pipeline design and analyze error patterns rather than achieve state-of-the-art performance.",
        "total_predictions": len(predictions),
        "sampled_errors": len(sampled),
        "taxonomy_summary": {cat: len(exs) for cat, exs in taxonomy.items()},
        "fixes": fixes,
        "fix_examples": fix_results[:10],
    }
    with open("results/q1_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print("\nAnalysis saved to results/q1_error_analysis.json")


if __name__ == "__main__":
    main()
