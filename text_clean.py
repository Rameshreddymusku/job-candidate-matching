import re

def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-\+\./]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_keywords(text: str):
    return [t for t in re.split(r"[^a-z0-9\+\-\.#]+", text.lower()) if t and len(t) > 1]
