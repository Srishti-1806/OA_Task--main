from collections import Counter
from Levenshtein import distance

def classify(words):
    freq = Counter(words)
    vocab = list(set(words))

    results = []

    for w in vocab:
        if freq[w] > 10:
            results.append((w, "correct", "high"))
        else:
            # find closest match
            close = min(vocab, key=lambda x: distance(w, x))

            if distance(w, close) <= 2:
                results.append((w, "incorrect", "medium"))
            else:
                results.append((w, "unknown", "low"))

    return results