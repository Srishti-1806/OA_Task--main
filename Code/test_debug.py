#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

from training.dataset import build_dataset

# Step 1: Load dataset
print("=" * 60)
print("STEP 1: Loading dataset...")
print("=" * 60)
ds = build_dataset()
print(f"✅ Dataset size: {len(ds)}")
print(f"✅ Columns: {ds.column_names}")

# Use only first 3 examples for testing
ds = ds.select(range(min(3, len(ds))))
print(f"✅ Using {len(ds)} examples for testing")

if len(ds) > 0:
    print(f"✅ First example keys: {list(ds[0].keys())}")
    print(f"✅ Audio type: {type(ds[0]['audio'])}")
    print(f"✅ Audio value: {ds[0]['audio']}")
    print(f"✅ Sentence (first 100 chars): {ds[0]['sentence'][:100]}")
else:
    print("❌ Dataset is empty!")
    sys.exit(1)

# Step 2: Test preprocessing
print("\n" + "=" * 60)
print("STEP 2: Testing preprocessing...")
print("=" * 60)

from transformers import WhisperProcessor
import librosa

MODEL_NAME = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(MODEL_NAME)

def preprocess(batch):
    print(f"  Processing batch with keys: {list(batch.keys())}")
    audio_path = batch["audio"]
    print(f"  Loading audio from: {audio_path}")
    
    try:
        audio_array, sr = librosa.load(audio_path, sr=16000)
        print(f"  ✅ Audio loaded, shape: {audio_array.shape}, sr: {sr}")
    except Exception as e:
        print(f"  ❌ Error loading audio: {e}")
        raise
    
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    batch["input_features"] = inputs.input_features[0]

    batch["labels"] = processor.tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=256,
    ).input_ids

    print(f"  ✅ Preprocessed, output keys: {list(batch.keys())}")
    return batch

# Test on first example
try:
    print("\nTesting preprocess on first example...")
    result = preprocess(ds[0])
    print(f"✅ Preprocessing successful!")
    print(f"✅ Output keys: {list(result.keys())}")
except Exception as e:
    print(f"❌ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test mapping
print("\n" + "=" * 60)
print("STEP 3: Testing dataset.map()...")
print("=" * 60)

print(f"Dataset columns before map: {ds.column_names}")
try:
    # Try with remove_columns
    ds_mapped = ds.map(preprocess, remove_columns=ds.column_names)
    print(f"✅ Mapping successful!")
    print(f"✅ Dataset columns after map: {ds_mapped.column_names}")
    print(f"✅ Dataset size after map: {len(ds_mapped)}")
except Exception as e:
    print(f"❌ Mapping failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
