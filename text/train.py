#%%
from preprocessing import load_bbc
from sklearn.model_selection import train_test_split

df = load_bbc(r"dataset\bbc")
X, y = df["text"], df["label"]
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42, stratify=y_valtest)


print(len(X_train), len(X_val), len(X_test))
print(y_train.value_counts(normalize=True))
print(y_val.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))




# %%
