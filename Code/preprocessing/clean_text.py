import re, json

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\u0900-\u097F\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_text(path):
    with open(path, encoding="utf-8") as f:
        t = json.load(f)["transcript"]
    return normalize(t)