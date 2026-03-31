import re


def categorize(ref, pred):
    """
    Categorize an error based on the difference between reference and prediction.
    Returns a list of applicable categories.
    """
    ref_words = ref.split()
    pred_words = pred.split()
    categories = []

    hindi_digits = set("०१२३४५६७८९")
    num_words = {"शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात",
                 "आठ", "नौ", "दस", "बीस", "सौ", "हजार", "हज़ार", "लाख"}
    if (any(w.isdigit() or any(c in w for c in hindi_digits) for w in pred_words + ref_words) or
        any(w in num_words for w in pred_words + ref_words)):
        categories.append("Number Error")

    if len(pred_words) < len(ref_words) * 0.7:
        categories.append("Deletion")

    if len(pred_words) > len(ref_words) * 1.3:
        categories.append("Insertion")

    if re.search(r'[a-zA-Z]+', ref) or re.search(r'[a-zA-Z]+', pred):
        categories.append("Code-Switching")

    fillers = {"हम्म", "अं", "उम", "उम्म", "अह", "ह", "आ", "ह्म्म"}
    if any(w in fillers for w in ref_words + pred_words):
        categories.append("Disfluency/Filler")

    if not categories:
        categories.append("Substitution")

    return categories


def build_taxonomy(samples):
    """
    Build a taxonomy from error samples.
    """
    taxonomy = {}
    for ref, pred, score in samples:
        cats = categorize(ref, pred)
        for cat in cats:
            taxonomy.setdefault(cat, []).append((ref, pred, score))
    return taxonomy