import sys
sys.path.insert(0, ".")

import json
import os
from cleanup.number_norm import normalize_numbers, demo as num_demo
from cleanup.english_detect import tag, tag_with_details, demo as eng_demo
from jiwer import wer as compute_wer


def clean(text):
    """Full cleanup pipeline: number normalization → English word tagging."""
    text = normalize_numbers(text)
    text = tag(text)
    return text


def apply_pipeline_and_measure(text_samples):
    """Apply cleanup pipeline to text samples and measure impact."""
    results = []

    for sample in text_samples:
        rec_id = sample.get("recording_id", "unknown")
        ref = sample.get("reference", "")
        raw = sample.get("raw_asr", ref)  # Use reference as raw if not provided

        if not ref.strip() or not raw.strip():
            continue

        cleaned = clean(raw)
        cleaned_for_wer = cleaned.replace("[EN]", "").replace("[/EN]", "")

        try:
            wer_before = compute_wer(ref, raw)
            wer_after = compute_wer(ref, cleaned_for_wer)
        except:
            continue

        results.append({
            "recording_id": rec_id,
            "reference": ref[:200],
            "raw_asr": raw[:200],
            "cleaned": cleaned[:200],
            "wer_before": wer_before,
            "wer_after": wer_after,
            "wer_change": wer_after - wer_before,
        })

    return results


def generate_report(results):
    """Generate structured Q2 report."""
    print("\n" + "=" * 70)
    print("Q2: CLEANUP PIPELINE REPORT")
    print("=" * 70)

    if not results:
        print("No results to report. Run with data first.")
        return

    avg_before = sum(r["wer_before"] for r in results) / len(results)
    avg_after = sum(r["wer_after"] for r in results) / len(results)
    improved = sum(1 for r in results if r["wer_change"] < -0.01)
    worsened = sum(1 for r in results if r["wer_change"] > 0.01)
    unchanged = len(results) - improved - worsened

    print(f"\nOverall Impact on {len(results)} utterances:")
    print(f"  Average WER before cleanup: {avg_before:.4f}")
    print(f"  Average WER after cleanup:  {avg_after:.4f}")
    print(f"  Improved: {improved}, Worsened: {worsened}, Unchanged: {unchanged}")

    helped = sorted([r for r in results if r["wer_change"] < -0.01], key=lambda x: x["wer_change"])
    print(f"\n--- Examples where cleanup HELPED (showing up to 3) ---")
    for r in helped[:3]:
        print(f"\n  Reference: {r['reference']}")
        print(f"  Raw ASR:   {r['raw_asr']}")
        print(f"  Cleaned:   {r['cleaned']}")
        print(f"  WER: {r['wer_before']:.3f} → {r['wer_after']:.3f}")

    hurt = sorted([r for r in results if r["wer_change"] > 0.01], key=lambda x: -x["wer_change"])
    print(f"\n--- Examples where cleanup HURT (showing up to 3) ---")
    for r in hurt[:3]:
        print(f"\n  Reference: {r['reference']}")
        print(f"  Raw ASR:   {r['raw_asr']}")
        print(f"  Cleaned:   {r['cleaned']}")
        print(f"  WER: {r['wer_before']:.3f} → {r['wer_after']:.3f}")


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("Q2(a): Number Normalization Demo")
    print("=" * 70)
    num_demo()

    print("\n" + "=" * 70)
    print("Q2(b): English Word Detection Demo")
    print("=" * 70)
    eng_demo()

    # Use data.json as sample dataset for cleanup pipeline testing
    if os.path.exists("data.json"):
        print("\n\nApplying cleanup pipeline to sample data from data.json...")
        
        with open("data.json", encoding="utf-8") as f:
            data_json = json.load(f)
        
        # Create sample pairs with Hindi text
        sample_pairs = [
            {
                "recording_id": f"sample_{i}",
                "reference": segment.get("text", ""),
                "raw_asr": segment.get("text", "")  # Use reference as raw for demo
            }
            for i, segment in enumerate(data_json[:10])
        ]
        
        if sample_pairs:
            results = apply_pipeline_and_measure(sample_pairs)
            
            if results:
                generate_report(results)

                with open("results/q2_cleanup_results.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print("\nResults saved to results/q2_cleanup_results.json")
    else:
        print("\n\nData file not found for cleanup pipeline testing.")
        print("Showing demo outputs only.")


if __name__ == "__main__":
    main()
