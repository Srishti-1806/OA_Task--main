from jiwer import wer


def get_errors(preds, refs):
    """Get all errors with WER scores, sorted by severity."""
    errors = []
    for p, r in zip(preds, refs):
        if not r.strip() or not p.strip():
            continue
        try:
            score = wer(r, p)
        except Exception:
            continue
        if score > 0:
            errors.append((r, p, score))
    return sorted(errors, key=lambda x: x[2])


def stratified_sample(errors, n_samples=25):
    """
    Stratified sampling by WER severity.
    """
    if not errors:
        return []

    buckets = {
        "low": [e for e in errors if e[2] <= 0.3],
        "medium": [e for e in errors if 0.3 < e[2] <= 0.6],
        "high": [e for e in errors if 0.6 < e[2] <= 1.0],
        "very_high": [e for e in errors if e[2] > 1.0],
    }

    sampled = []
    for name, bucket in buckets.items():
        if not bucket:
            continue
        per_bucket = max(3, n_samples * len(bucket) // len(errors))
        step = max(1, len(bucket) // per_bucket)
        sampled.extend(bucket[::step][:per_bucket])

    if len(sampled) < n_samples:
        remaining = [e for e in errors if e not in sampled]
        sampled.extend(remaining[:n_samples - len(sampled)])

    return sampled