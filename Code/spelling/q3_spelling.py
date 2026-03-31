import sys
sys.path.insert(0, ".")

import re
import csv
import os
import urllib.request

# ------------------ CONSTANTS ------------------

VOWELS = set("अआइईउऊऋएऐओऔ")
CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
MATRAS = set("ािीुूृेैोौं:ँॅॉ")
HALANT = "्"
NUKTA = "़"
VISARGA = "ः"
ANUSVARA = "ं"
CHANDRABINDU = "ँ"
DEVANAGARI_DIGITS = set("०१२३४५६७८९")

ALL_DEVANAGARI = VOWELS | CONSONANTS | MATRAS | {
    HALANT, NUKTA, VISARGA, ANUSVARA, CHANDRABINDU
} | DEVANAGARI_DIGITS

# ------------------ HELPERS ------------------

def is_pure_devanagari(word):
    for c in word:
        if c not in ALL_DEVANAGARI and not c.isspace() and c not in "।,?!.'-–":
            return False
    return True


COMMON_HINDI_WORDS = {"है", "तो", "में", "और", "से", "कि", "को", "का", "ये", "था"}

VALID_PREFIXES = ["अ", "अन", "प्र", "वि", "सम"]
VALID_SUFFIXES = ["ता", "ती", "ते", "ना", "नी", "ने"]

# ------------------ LOGIC ------------------

def has_invalid_sequences(word):
    if re.search(r'[ािीुूृेैोौ]{2,}', word):
        return True, "consecutive matras"
    if word.startswith(HALANT):
        return True, "halant at start"
    if word and word[0] in MATRAS:
        return True, "matra at start"
    if HALANT + HALANT in word:
        return True, "double halant"
    return False, ""


def has_valid_morphology(word):
    for suffix in VALID_SUFFIXES:
        if word.endswith(suffix):
            return True
    for prefix in VALID_PREFIXES:
        if word.startswith(prefix):
            return True
    return False


def classify_word(word):
    clean = word.strip()

    if not clean:
        return "incorrect", "high", "empty"

    if not is_pure_devanagari(clean):
        if re.match(r'^[a-zA-Z]+$', clean):
            return "correct", "medium", "english"
        return "incorrect", "high", "mixed script"

    if clean in COMMON_HINDI_WORDS:
        return "correct", "high", "common word"

    invalid, reason = has_invalid_sequences(clean)
    if invalid:
        return "incorrect", "high", reason

    if has_valid_morphology(clean):
        return "correct", "medium", "morphology"

    return "correct", "low", "unknown"


def classify_words(words):
    return [
        {
            "word": w,
            "classification": classify_word(w)[0],
            "confidence": classify_word(w)[1],
            "reason": classify_word(w)[2],
        }
        for w in words
    ]

# ------------------ DATA ------------------

def download_word_list():
    url = "https://docs.google.com/spreadsheets/d/17DwCAx6Tym5Nt7eOni848np9meR-TIj7uULMtYcgQaw/export?format=csv"
    dest = "data/word_list.csv"

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(dest):
        print("Downloading word list...")
        urllib.request.urlretrieve(url, dest)

    words = []
    with open(dest, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                words.append(row[0].strip())

    return words

# ------------------ MAIN PIPELINE ------------------

def run_classification(words):
    results = classify_words(words)

    correct = sum(1 for r in results if r["classification"] == "correct")
    incorrect = sum(1 for r in results if r["classification"] == "incorrect")

    print("\nRESULTS")
    print("=" * 40)
    print(f"Total: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")

    os.makedirs("results", exist_ok=True)

    with open("results/q3_spelling_output.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "classification", "confidence", "reason"])
        for r in results:
            writer.writerow([r["word"], r["classification"], r["confidence"], r["reason"]])

    return results


def main():
    words = download_word_list()
    results = run_classification(words)

    correct_count = sum(1 for r in results if r["classification"] == "correct")
    print("\nFinal correct words:", correct_count)


if __name__ == "__main__":
    main()
