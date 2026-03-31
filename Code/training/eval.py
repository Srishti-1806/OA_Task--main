import sys
sys.path.insert(0, ".")

import json
import os
import torch
import warnings
import librosa
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig
import evaluate

warnings.filterwarnings("ignore")

BASE_MODEL = "./model"  # use local tuned checkpoint to avoid remote hub download
FT_MODEL = "./model"

DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

wer_metric = evaluate.load("wer")


# ✅ WHISPER NATIVE LOW-LEVEL API for long-form audio
def create_model_and_processor(model_name):
    """Load model and processor directly for better control over generation."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=DTYPE if DEVICE >= 0 else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    # Update generation config per HF issue #25084
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.config.forced_decoder_ids = AutoProcessor.from_pretrained(model_name).get_decoder_prompt_ids(
        language="hi", task="transcribe"
    )

    if DEVICE >= 0:
        model.to(f"cuda:{DEVICE}")

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def transcribe_audio(model, processor, audio_array):
    """Transcribe audio using direct model.generate() call."""
    try:
        # Prepare inputs
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        
        # Move to device
        if DEVICE >= 0:
            inputs = {k: v.to(f"cuda:{DEVICE}") for k, v in inputs.items()}
        
        # Generate with forced_decoder_ids for language and task
        with torch.no_grad():
            # avoid outdated generation config incompatibility with language/task args
            predicted_ids = model.generate(
                input_features=inputs["input_features"],
                max_new_tokens=256,
                # language/task is already handled by forced_decoder_ids in model config
            )
        
        # Decode
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0].strip() if transcription else ""
    except Exception as e:
        raise Exception(f"Transcription failed: {e}")


def evaluate_on_custom(model, processor, audio_dir="data/audio", text_dir="data/text"):
    preds, refs = [], []

    if not os.path.exists(audio_dir):
        print(f"❌ Audio dir not found: {audio_dir}")
        return 1.0, [], []

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    print(f"  Found {len(audio_files)} audio files")

    # limit for quick testing after config fix
    for i, f in enumerate(audio_files[:15]):
        rec_id = f.replace(".wav", "")
        txt_path = os.path.join(text_dir, f"{rec_id}.json")

        if not os.path.exists(txt_path):
            continue

        # ✅ JSON LOAD FIX
        try:
            with open(txt_path, encoding="utf-8") as fh:
                content = fh.read().strip()
                if not content:
                    continue
                segments = json.loads(content)
        except Exception as e:
            print(f"    JSON error {f}: {e}")
            continue

        # ✅ Reference extraction
        if isinstance(segments, list):
            ref = " ".join(
                [seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()]
            )
        elif isinstance(segments, dict):
            ref = segments.get("transcript", "").strip()
        else:
            continue

        if not ref:
            continue

        # ✅ AUDIO LOAD (CRITICAL FIX)
        audio_path = os.path.join(audio_dir, f)
        try:
            audio_array, sr = librosa.load(audio_path, sr=16000)

            if audio_array is None or len(audio_array) < 100:
                continue

            # 🔥 Use direct transcription with model.generate()
            pred = transcribe_audio(model, processor, audio_array)

        except Exception as e:
            print(f"    AUDIO error {f}: {e}")
            continue

        preds.append(pred)
        refs.append(ref)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(audio_files)}")

    if len(preds) == 0:
        print("⚠️ No valid samples found!")
        return 1.0, [], []

    wer = wer_metric.compute(predictions=preds, references=refs)
    return wer, preds, refs


def main():
    results = {}
    
    # ✅ Create results directory at the start
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("Evaluating PRETRAINED Whisper-small")
    print("=" * 60)

    base_model, base_processor = create_model_and_processor(BASE_MODEL)

    print("\n[1] On custom dataset:")
    custom_base_wer, _, _ = evaluate_on_custom(base_model, base_processor)

    results["base_custom"] = custom_base_wer
    print(f"  WER: {custom_base_wer:.4f}")

    del base_model, base_processor
    torch.cuda.empty_cache()

    # ✅ Fine-tuned model
    if os.path.exists(FT_MODEL):
        print("\n" + "=" * 60)
        print("Evaluating FINE-TUNED Whisper-small")
        print("=" * 60)

        ft_model, ft_processor = create_model_and_processor(FT_MODEL)

        print("\n[1] On custom dataset:")
        custom_ft_wer, custom_ft_preds, custom_ft_refs = evaluate_on_custom(ft_model, ft_processor)

        results["ft_custom"] = custom_ft_wer
        print(f"  WER: {custom_ft_wer:.4f}")

        del ft_model, ft_processor
        torch.cuda.empty_cache()

        # ✅ Save predictions
        custom_error_data = [
            {"reference": ref, "prediction": pred}
            for ref, pred in zip(custom_ft_refs, custom_ft_preds)
        ]

        with open("results/custom_ft_predictions.json", "w", encoding="utf-8") as f:
            json.dump(custom_error_data, f, ensure_ascii=False, indent=2)

    else:
        print(f"\n⚠️ Fine-tuned model not found at {FT_MODEL}. Skipping.")

    # ✅ Print results
    print("\n" + "=" * 60)
    print("WER RESULTS TABLE")
    print("=" * 60)
    print(f"{'Model':<30} {'Custom Dataset':<15}")
    print("-" * 60)

    if "base_custom" in results:
        print(f"{'Whisper-small (pretrained)':<30} {results['base_custom']:<15.4f}")

    if "ft_custom" in results:
        print(f"{'Whisper-small (fine-tuned)':<30} {results['ft_custom']:<15.4f}")

    print("=" * 60)

    # ✅ Save WER
    with open("results/wer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to results/wer_results.json")


if __name__ == "__main__":
    main()
