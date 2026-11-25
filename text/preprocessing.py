import os, pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
STOPWORDS = set(ENGLISH_STOP_WORDS)

def load_bbc(root): 
    rows = []
    for label in sorted(os.listdir(root)):
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            path = os.path.join(folder, fname)
            if not fname.endswith(".txt"):
                continue
            with open(path, encoding="latin-1") as f:
                rows.append({"text": f.read().strip(), "label": label})
    df = pd.DataFrame(rows)
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)