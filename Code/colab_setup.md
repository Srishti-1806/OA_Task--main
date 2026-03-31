# 🚀 Running on Google Colab

To run this project on Google Colab with GPU acceleration, follow these steps:

### 1. Set up GPU Runtime
- Open [Google Colab](https://colab.research.google.com/).
- Go to **Runtime** > **Change runtime type**.
- Select **T4 GPU** (or A100/L4 if available).

### 2. Run the Setup & Execution Cell
Copy and paste the following into a single Colab cell and run it:

```python
# 1. Clone the repository
import os
repo_url = "https://github.com/Srishti-1806/OA_Task.git"
repo_name = "OA_Task"

if not os.path.exists(repo_name):
    !git clone {repo_url}
%cd {repo_name}

# 2. Install dependencies
!pip install -r requirements.txt
# Ensure audio libraries are present for Linux (Colab)
!apt-get install -y ffmpeg
!pip install torchaudio librosa soundfile

# 3. Create necessary directories
!mkdir -p data/audio data/text results model

# 4. Run the components (Subset mode)
# Step 1: Download subset of data (30 recordings)
!python download_data.py

# Step 2: Fine-tune Whisper (T4 GPU makes this fast ~3-5 mins)
!python training/train_whisper.py

# Step 3: Run Evaluation
!python training/eval.py

# Step 4: Run Analysis & Cleanup Scripts
!python analysis/q1_error_analysis.py
!python cleanup/pipeline.py
!python spelling/q3_spelling.py
!python lattice/lattice_wer.py
```

### 3. Benefits of using Colab
- **GPU Acceleration:** Fine-tuning Whisper takes ~3-5 minutes on a T4 GPU. Training on CPU may fail or be very slow due to memory constraints.
- **FFmpeg Support:** Colab (Linux) has built-in support for FFmpeg, which avoids audio loading issues.
- **Memory:** Sufficient RAM for processing the dataset and model fine-tuning.
- **Pre-installed Libraries:** Most dependencies are already available, reducing setup time.

### 4. Expected Output
After running the cell, you should see:
- Data download progress
- Training progress with WER metrics
- Evaluation results comparing base and fine-tuned models
- Results saved to `results/wer_results.json`

### 5. Troubleshooting
- If training fails, ensure GPU runtime is selected.
- If downloads fail, check internet connection.
- For any errors, refer to the terminal output in Colab.
