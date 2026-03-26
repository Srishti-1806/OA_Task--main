import sys
sys.path.insert(0, ".")

import csv
import os
import re
import urllib.request
from collections import Counter
import json


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[?!,.\-–;:"\'\(\)]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def simple_word_align(ref_words, hyp_words):
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    alignment = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            alignment.append(("match", ref_words[i-1], hyp_words[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append(("sub", ref_words[i-1], hyp_words[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append(("del", ref_words[i-1], ""))
            i -= 1
        else:
            alignment.append(("ins", "", hyp_words[j-1]))
            j -= 1

    return list(reversed(alignment))


def build_lattice(human_ref, model_outputs):
    ref_words = clean_text(human_ref).split()
    if not ref_words:
        return []

    lattice = [{word} for word in ref_words]

    for output in model_outputs:
        hyp_words = clean_text(output).split()
        alignment = simple_word_align(ref_words, hyp_words)

        ref_idx = 0
        for op, ref_w, hyp_w in alignment:
            if op == "match":
                ref_idx += 1
            elif op == "sub":
                if ref_idx < len(lattice):
                    lattice[ref_idx].add(hyp_w)
                ref_idx += 1
            elif op == "del":
                ref_idx += 1

    return [list(bin) for bin in lattice]


def apply_model_consensus(lattice, model_outputs, min_agreement=3):
    ref_words = [bin[0] for bin in lattice]
    enhanced = [set(bin) for bin in lattice]

    for i in range(len(ref_words)):
        counter = Counter()

        for output in model_outputs:
            hyp_words = clean_text(output).split()
            alignment = simple_word_align(ref_words, hyp_words)

            ref_idx = 0
            for op, ref_w, hyp_w in alignment:
                if ref_idx == i and op in ("match", "sub"):
                    counter[hyp_w] += 1
                if op in ("match", "sub", "del"):
                    ref_idx += 1

        for word, count in counter.items():
            if count >= min_agreement:
                enhanced[i].add(word)

    return [list(bin) for bin in enhanced]


def compute_rigid_wer(ref, hyp):
    ref_words = clean_text(ref).split()
    hyp_words = clean_text(hyp).split()

    if not ref_words:
        return 0.0

    alignment = simple_word_align(ref_words, hyp_words)
    errors = sum(1 for op, _, _ in alignment if op != "match")

    return errors / len(ref_words)


def compute_lattice_wer(lattice, hyp):
    hyp_words = clean_text(hyp).split()
    ref_len = len(lattice)

    if ref_len == 0:
        return 0.0

    errors = 0
    max_len = max(ref_len, len(hyp_words))

    for i in range(max_len):
        if i < ref_len and i < len(hyp_words):
            if hyp_words[i] not in lattice[i]:
                errors += 1
        else:
            errors += 1

    return errors / ref_len


def download_lattice_data():
    url = "https://docs.google.com/spreadsheets/d/1J_I0raoRNbe29HiAPD5FROTr0jC93YtSkjOrIglKEjU/export?format=csv"
    dest = "data/lattice_data.csv"

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(dest):
        print("Downloading lattice data...")
        urllib.request.urlretrieve(url, dest)

    rows = []
    with open(dest, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows")
    return rows


def main():
    os.makedirs("results", exist_ok=True)

    rows = download_lattice_data()
    if not rows:
        print("No data found")
        return

    model_names = [k for k in rows[0].keys() if k.startswith("Model")]
    print("Models:", model_names)

    all_results = []

    for row in rows:
        ref = row.get("Human", "")
        if not ref.strip():
            continue

        outputs = [row[m] for m in model_names if row.get(m, "").strip()]
        if not outputs:
            continue

        lattice = build_lattice(ref, outputs)
        lattice = apply_model_consensus(lattice, outputs)

        entry = {}

        for m in model_names:
            hyp = row.get(m, "")
            if not hyp.strip():
                continue

            entry[f"{m}_rigid"] = compute_rigid_wer(ref, hyp)
            entry[f"{m}_lattice"] = compute_lattice_wer(lattice, hyp)

        all_results.append(entry)

    print("\nRESULTS SUMMARY")
    print("=" * 50)

    for m in model_names:
        rigid_vals = [r[f"{m}_rigid"] for r in all_results if f"{m}_rigid" in r]
        lat_vals = [r[f"{m}_lattice"] for r in all_results if f"{m}_lattice" in r]

        if rigid_vals:
            avg_r = sum(rigid_vals) / len(rigid_vals)
            avg_l = sum(lat_vals) / len(lat_vals)
            print(f"{m}: {avg_r:.4f} → {avg_l:.4f}")

    with open("results/q4_lattice_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nSaved to results/q4_lattice_results.json")


if __name__ == "__main__":
    main()
