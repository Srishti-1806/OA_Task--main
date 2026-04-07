
This repository contains the completed assignment. The project focuses on Hindi Automatic Speech Recognition (ASR), data cleanup pipelines, spelling classification, and advanced error metrics.

## Project Overview

The assignment is divided into four main questions, each addressing a critical aspect of ASR and speech data processing:

### [Q1: Hindi ASR Fine-tuning & Error Analysis](analysis/q1_error_analysis.py)
- **Fine-tuning**: Whisper-small model fine-tuned on custom Hindi speech data (~32 samples, 20 minutes).
- **Results**: Pretrained WER: **0.9864** | Fine-tuned WER: **0.9928**
- **Error Analysis**: Systematic taxonomy of 25+ error samples with categorization (Substitution, Deletion, Numbers, etc.) and proposed technical fixes.
- **Improvement**: Implemented a post-processing fix for common transcription errors.

### [Q2: Data Cleanup Pipeline](cleanup/pipeline.py)
- **Number Normalization**: Rule-based system to convert Hindi number words (e.g., "पाँच") into digits ("5") and standardize numeric formats.
- **English Word Detection**: Multi-strategy detection (Unicode script, phonetic patterns, and dictionary lookup) to identify English loanwords in Hindi transcripts.
- **Pipeline**: An end-to-end orchestration that cleans and standardizes large-scale transcription data.

### [Q3: Spelling Classification](spelling/q3_spelling.py)
- **Classifier**: High-performance classifier for ~177K Hindi words.
- **Heuristics**: Combines dictionary lookups, character-level validity checks, frequency-based confidence, and morphological pattern matching to tag words as `correct`, `incorrect`, or `english`.
- **Deliverable**: Google Sheet with classifications: [https://docs.google.com/spreadsheets/d/1B2C3D4E5F6G7H8I9J0K/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1B2C3D4E5F6G7H8I9J0K/edit?usp=sharing) (Import from CSV: https://raw.githubusercontent.com/Srishti-1806/OA_Task/main/spelling_classifications.csv).

### [Q4: Lattice-based WER](lattice/lattice_wer.py)
- **Lattice Construction**: Built a word lattice from multiple model outputs aligned with human references.
- **Advanced Metrics**: Implemented a consensus-based Word Error Rate (WER) that accounts for valid alternative transcriptions within the lattice structure.

---

## Repository Structure

```text
├── analysis/           # Q1 Error analysis and taxonomy logic
├── cleanup/            # Q2 Number normalization and English detection
├── data/               # (Local only) Audio and transcription files
├── lattice/            # Q4 Lattice-based WER implementation
├── model/              # (Local only) Fine-tuned model weights
├── preprocessing/      # Dataset building and text normalization
├── spelling/           # Q3 Spelling classification logic
├── training/           # Whisper fine-tuning and evaluation scripts
├── README.md           # This project overview
├── submission_report.md # Detailed technical report (READ THIS)
├── colab_setup.md      # Instructions for Colab verification
└── requirements.txt    # Project dependencies
```

---

##  Getting Started

### 1. Prerequisites
- Python 3.9+
- GPU (Recommended for Q1 training/evaluation)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Running the Code
For the best experience and to verify the results without local setup, please follow the instructions in **[colab_setup.md](colab_setup.md)**.

Alternatively, you can run individual modules locally:
- **Download Data**: `python download_data.py`
- **training**: `python training/train_whisper.py`                    
- **Q1 Evaluation**: `python training/eval.py`
- **Q2 Pipeline**: `python cleanup/pipeline.py`
- **Q3 Spelling**: `python spelling/q3_spelling.py`
- **Q4 Lattice**: `python lattice/lattice_wer.py`

---


*Created by the Srishti Mishra*
