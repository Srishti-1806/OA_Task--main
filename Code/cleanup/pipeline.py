import sys
sys.path.insert(0, ".")

# 🔥 Load .env for HuggingFace token (faster downloads)
import os
from pathlib import Path
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

import json
import torch
import warnings

# 🔥 FIX: Suppress max_length/max_new_tokens conflict warning
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")

from cleanup.number_norm import normalize_numbers, demo as num_demo
from cleanup.english_detect import tag, tag_with_details, demo as eng_demo
from jiwer import wer as compute_wer


def clean(text):
    """Full cleanup pipeline: number normalization → English word tagging."""
    text = normalize_numbers(text)
    text = tag(text)
    return text


def generate_raw_asr_transcripts(audio_dir="data/audio", text_dir="data/text"):
    """
    Generate raw ASR transcripts using pretrained Whisper-small.
    Pairs each raw output with the human reference.
    """
    from transformers import pipeline as hf_pipeline, GenerationConfig

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
        generate_kwargs={"max_new_tokens": 256},
    )

    # Update generation config to avoid outdated language/task incompatibility
    pipe.model.generation_config = GenerationConfig.from_model_config(pipe.model.config)
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

    import librosa
    pairs = []
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    print(f"  Found {len(audio_files)} audio files")

    for i, af in enumerate(audio_files):
        rec_id = af.replace(".wav", "")
        json_path = os.path.join(text_dir, f"{rec_id}.json")

        if not os.path.exists(json_path):
            continue

        with open(json_path, encoding="utf-8") as f:
            segments = json.load(f)

        if isinstance(segments, list):
            ref = " ".join([s["text"] for s in segments if s.get("text", "").strip()])
        else:
            continue

        audio_path = os.path.join(audio_dir, af)
        try:
            audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)

            # 🔥 trim to 30 sec
            MAX_SAMPLES = 16000 * 30
            if len(audio_array) > MAX_SAMPLES:
                audio_array = audio_array[:MAX_SAMPLES]
            result = pipe({"array": audio_array, "sampling_rate": 16000}, return_timestamps=False)
            raw_asr = result["text"]
        except Exception as e:
            print(f"    Error processing {af}: {e}")
            continue

        pairs.append({
            "recording_id": rec_id,
            "reference": ref,
            "raw_asr": raw_asr,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(audio_files)}")

    return pairs


def apply_pipeline_and_measure(pairs):
    """Apply cleanup pipeline and measure WER impact."""
    results = []

    for pair in pairs:
        ref = pair["reference"]
        raw = pair["raw_asr"]

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
            "recording_id": pair["recording_id"],
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
    print(f"\n--- Examples where cleanup HELPED (showing top 3) ---")
    for r in helped[:3]:
        print(f"\n  Raw ASR:   {r['raw_asr']}")
        print(f"  Cleaned:   {r['cleaned']}")
        print(f"  Reference: {r['reference']}")
        print(f"  WER: {r['wer_before']:.3f} → {r['wer_after']:.3f}")

    hurt = sorted([r for r in results if r["wer_change"] > 0.01], key=lambda x: -x["wer_change"])
    print(f"\n--- Examples where cleanup HURT (showing top 3) ---")
    for r in hurt[:3]:
        print(f"\n  Raw ASR:   {r['raw_asr']}")
        print(f"  Cleaned:   {r['cleaned']}")
        print(f"  Reference: {r['reference']}")
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

    if os.path.exists("data/audio") and os.listdir("data/audio"):
        print("\n\nGenerating raw ASR transcripts...")
        pairs = generate_raw_asr_transcripts()

        if pairs:
            print(f"\nApplying cleanup pipeline to {len(pairs)} pairs...")
            results = apply_pipeline_and_measure(pairs)
            generate_report(results)

            with open("results/q2_cleanup_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print("\nResults saved to results/q2_cleanup_results.json")
    else:
        print("\n\nNo audio data found. Run download_data.py first for full pipeline.")
        print("Showing demo outputs only.")


if __name__ == "__main__":
    main()
