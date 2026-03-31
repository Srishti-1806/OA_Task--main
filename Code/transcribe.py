import os
import json
import torch
import warnings
from tqdm import tqdm
from transformers import pipeline
import evaluate
import logging

# -------------------- SETTINGS --------------------
BASE_MODEL = "openai/whisper-small"
AUDIO_FOLDER = "./audio"     # folder with .wav files
OUTPUT_FILE = "predictions.json"

DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    filename="asr.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# WER metric
wer_metric = evaluate.load("wer")


# -------------------- PIPELINE --------------------
def create_pipeline(model_name):
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=DEVICE,
        dtype=DTYPE,
        generate_kwargs={
            "max_new_tokens": 256,
            "max_length": None,
            "temperature": 0.0,
            "do_sample": False
        }
    )

    # Fix attention mask issue
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

    return pipe


# -------------------- LOAD DATA --------------------
def load_audio_files(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            files.append(os.path.join(folder, file))
    return files


# -------------------- TRANSCRIBE --------------------
def transcribe(pipe, files):
    results = []

    for file in tqdm(files, desc="Transcribing"):
        try:
            output = pipe(file)
            text = output["text"]

            results.append({
                "file": file,
                "prediction": text
            })

        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")
            print(f"❌ Error: {file}")

    return results


# -------------------- SAVE --------------------
def save_results(results, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


# -------------------- OPTIONAL WER --------------------
def compute_wer(predictions, references):
    return wer_metric.compute(predictions=predictions, references=references)


# -------------------- MAIN --------------------
def main():
    print("🚀 Loading pipeline...")
    pipe = create_pipeline(BASE_MODEL)

    print("📂 Loading audio files...")
    audio_files = load_audio_files(AUDIO_FOLDER)
    print(f"Found {len(audio_files)} audio files")

    print("🎙️ Transcribing...")
    results = transcribe(pipe, audio_files)

    print("💾 Saving results...")
    save_results(results, OUTPUT_FILE)

    print("✅ Done!")


if __name__ == "__main__":
    main()