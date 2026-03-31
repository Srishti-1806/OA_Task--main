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

import glob
import json
import unicodedata
import re
from datasets import Dataset


def normalize_hindi(text):
    """Normalize Hindi text: NFKC unicode, strip extra spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_transcription(json_path):
    with open(json_path, encoding="utf-8") as f:
        segments = json.load(f)

    if isinstance(segments, list):
        texts = [seg["text"] for seg in segments if seg.get("text", "").strip()]
        return " ".join(texts)
    elif isinstance(segments, dict) and "transcript" in segments:
        return segments["transcript"]
    return ""


def build_segments_dataset(audio_dir="data/audio", text_dir="data/text", max_segment_duration=30):
    data = []
    json_files = sorted(glob.glob(f"{text_dir}/*.json"))

    for jf in json_files:
        rec_id = jf.replace("\\", "/").split("/")[-1].replace(".json", "")
        audio_path = f"{audio_dir}/{rec_id}.wav"

        if not __import__("os").path.exists(audio_path):
            continue

        try:
            with open(jf, encoding="utf-8") as f:
                segments = json.load(f)

            if not isinstance(segments, list):
                continue

            full_text = " ".join(
                [normalize_hindi(seg["text"]) for seg in segments if seg.get("text", "").strip()]
            )

            if full_text:
                data.append({
                    "audio": {"path": audio_path},  # 🔥 PROPER FORMAT for HuggingFace Audio
                    "sentence": full_text,
                    "recording_id": rec_id,
                })

        except Exception as e:
            print(f"Error processing {jf}: {e}")
            continue

    print(f"Built dataset with {len(data)} recordings")

    if len(data) == 0:
        print("⚠️  WARNING: No data was loaded!")
    
    ds = Dataset.from_list(data)
    
    # No need to cast to Audio feature - we load audio with librosa in the training script
    return ds


def build_refined_dataset():
    """Backward-compatible wrapper."""
    return build_segments_dataset()


if __name__ == "__main__":
    ds = build_segments_dataset()
    print(f"Dataset: {ds}")
    if len(ds) > 0:
        print(f"Sample sentence: {ds[0]['sentence'][:200]}...")