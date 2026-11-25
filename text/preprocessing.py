import os, pandas as pd

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