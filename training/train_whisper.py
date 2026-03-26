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

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

from training.dataset import build_dataset

MODEL_NAME = "openai/whisper-tiny"
OUTPUT_DIR = "model"
LANGUAGE = "hi"
TASK = "transcribe"


print("Loading model and processor...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK
)
model.generation_config.suppress_tokens = []

print("Loading dataset...")
ds = build_dataset()
print(f"Dataset size: {len(ds)}")
print(f"Dataset columns: {ds.column_names}")

if len(ds) == 0:
    print("❌ ERROR: Dataset is empty!")
    sys.exit(1)


import librosa

def preprocess(batch):
    # 🔥 LOAD AUDIO: Extract path from audio dict format (avoid torchcodec)
    audio_data = batch["audio"]
    
    # Get path - dict format from build_dataset
    if isinstance(audio_data, dict) and "path" in audio_data:
        audio_path = audio_data["path"]
    elif isinstance(audio_data, str):
        audio_path = audio_data
    else:
        print(f"Warning: Unexpected audio format: {type(audio_data)}")
        return None
    
    try:
        # Load audio using librosa (no torchcodec required)
        audio_array, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Warning: Failed to load audio: {e}")
        return None
    
    try:
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        batch["input_features"] = inputs.input_features[0]
    except Exception as e:
        print(f"Warning: Failed to process audio: {e}")
        return None

    batch["labels"] = processor.tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=256,
    ).input_ids

    return batch


print("Preprocessing dataset...")
# 🔥 SAFE VERSION: Get column names BEFORE mapping
original_columns = ds.column_names
print(f"Removing columns: {original_columns}")

# 🔥 FIX 1: Parallel processing (4 workers for faster preprocessing)
num_workers = 4 if torch.cuda.is_available() else 2
print(f"Using {num_workers} workers for parallel preprocessing...")
ds = ds.map(preprocess, remove_columns=original_columns, num_proc=num_workers)
print(f"✅ Preprocessing complete. Dataset size after mapping: {len(ds)}")
print(f"✅ Dataset columns after mapping: {ds.column_names}")

ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds = ds["train"]
eval_ds = ds["test"]
print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

# 🔥 DEBUG MODE: Use only 10 samples for quick test
train_ds = train_ds.select(range(10))
eval_ds = eval_ds.select(range(5))
print(f"DEBUG: Using {len(train_ds)} train, {len(eval_ds)} eval samples")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


use_fp16 = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
device_batch = 4 if torch.cuda.is_available() else 2

# 🔥 CLEAN TRAINING BLOCK WITH ALL FIXES
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    
    # FIX 4: Optimize for CPU (increase gradient accumulation)
    gradient_accumulation_steps=4,
    
    learning_rate=1e-5,
    warmup_steps=50,
    num_train_epochs=1,  # Start with 1 epoch for testing
    
    # FIX 5: CPU optimization
    fp16=use_fp16,
    bf16=False,
    
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    
    predict_with_generate=True,
    generation_max_length=256,  # correct arg for Seq2SeqTrainingArguments
    
    # FIX 2: Disable pin_memory warning
    dataloader_pin_memory=False,
    dataloader_num_workers=2,
    
    report_to="none",
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
print("Training started...")
trainer.train()

print("\n" + "="*60)
print("SAVING MODEL WITH PROPER CONFIGURATION...")
print("="*60)

# FIX 6: Proper model saving (prevent generation config issues on reload)
from transformers import GenerationConfig

# Clean up generation config before saving
model.generation_config = GenerationConfig.from_model_config(model.config)
model.config.suppress_tokens = None
model.config.begin_suppress_tokens = None

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}/")
print(f"✅ Training complete!")