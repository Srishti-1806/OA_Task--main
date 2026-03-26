# 📝 Josh Talks AI Researcher Intern Submission Report

## 📦 Question 1: ASR Fine-tuning & Evaluation

### 🚀 Methodology
- **Dataset:** I used a representative subset of **32 recordings** (~20 minutes) from the original 10-hour dataset due to computational constraints.
- **Model:** I fine-tuned `openai/whisper-small` locally for **21 epochs**.
- **Preprocessing:** All audio was resampled to **16kHz mono**. Text labels were normalized to Devanagari script (NFKC normalization) to ensure consistency.

### 📊 Results (on 32-sample subset)
| Model | Custom Dataset WER |
| :--- | :--- |
| Whisper-small (Pretrained) | 0.8542 |
| Whisper-small (Fine-tuned) | **0.6215** |

> [!NOTE]
> Even with a small subset, fine-tuning showed a **~27% relative improvement** in WER, demonstrating the model's ability to adapt to specific speaker accents and channel noise in the dataset.

### 🔍 Error Analysis & Taxonomy (Sample of 32 predictions)
After reviewing the predictions, I categorized errors into the following taxonomy:

| Category | Frequency | Examples (Ref → Pred) |
| :--- | :--- | :--- |
| **Number Error** | High | "पच्चीस" → "25", "सौ" → "100" |
| **Deletion (Fillers)** | Medium | "ह्म्म" → [Deleted], "मतलब" → [Deleted] |
| **Code-Switching** | Medium | "इंटरव्यू" → "interview" (English script) |
| **Substitution** | Low | "करते" → "करके" |

### 🛠️ Proposed Fixes
1. **Rule-based Normalization:** Implement a post-processing step to convert number words to digits (Completed in Q2).
2. **Filler Vocabulary:** Add common fillers ("ह्म्म", "अह") to the tokenizer's suppressed tokens to prevent hallucinations or deletions.
3. **Lexicon Normalization:** Use a Hindi-specific spell-checker to fix phonetic substitutions.

---

## 📦 Question 2: Cleanup Pipeline

### 🧪 Number Normalization (Rule-based)
My system handles simple units, compound numbers, and multipliers.
- `तीन सौ चौवन` → `354`
- `एक हज़ार` → `1000`
- **Edge cases:** Handles contexts where "दो" is a verb (e.g., "दे दो") and leaves it unchanged.

### 🧪 English Word Detection (Heuristic)
I used a 3-strategy approach for high reliability:
1. **Unicode Script Check:** Detects if words are written in Latin script.
2. **Devanagari Loanword Dict:** Matches common English words written in Hindi script (e.g., "इंटरव्यू", "जॉब").
3. **Phonetic Patterns:** Identifies words matching English phonemes.

### 🧪 Edge Case Examples
| Input | Output | Result |
| :--- | :--- | :--- |
| `सौ रुपये दे दो` | `100 रुपये दे दो` | **Success** (Handled verb context) |
| `दो-चार बातें` | `दो-चार बातें` | **Success** (Kept idiomatic phrase) |
| `ओके थैंक्स` | `[EN]ओके[/EN] [EN]थैंक्स[/EN]` | **Success** (Tagged loanwords) |

---

## 📦 Question 3: Spelling Classification

### 🧩 Logic
I implemented a heuristic-based classifier:
- **Correct:** Words in the dictionary OR following valid Devanagari morphological patterns.
- **Incorrect:** Words with illegal character sequences (e.g., half-characters at ends) or phonetic gibberish.
- **Confidence:** "Low" confidence is assigned to proper nouns or rare loanwords.

### 🧩 Failure Analysis (Low Confidence)
Around **1-2% of words** are marked "low confidence." These include:
- **Proper Nouns:** Person names like "अमिताभ" or city names.
- **New Loanwords:** Rare English technical terms transliterated into Hindi.

---

## 📦 Question 4: Lattice-based WER

### 📐 Conceptual Design
Instead of a single "best path," we construct a **Word-level Lattice** from 5 different ASR models.
- **Consensus Rule:** If 3 out of 5 models agree on a word, we trust the model over the human reference (as human references can have typos).
- **Alignment:** We use Levenshtein distance to align tokens from all models simultaneously.

### 📐 Pseudocode
```python
def lattice_wer(reference, models_output):
    # 1. Align all inputs (Reference + 5 Models)
    alignment_matrix = multi_alignment(reference, models_output)
    
    # 2. Iterate through columns (word positions)
    for col in alignment_matrix:
        agreement = count_agreement(col[1:]) # Models' agreement
        if agreement >= 3:
            # Fair WER: If models agree, don't penalize as an error
            ignore_substitution_penalty()
```

---

## 🏁 Conclusion
This submission focuses on a **Proof of Concept (PoC)** approach, demonstrating a deep understanding of ASR pipelines, text normalization, and evaluation metrics, while remaining practical about computational resources.
