#%%
from preprocessing import load_bbc, clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = load_bbc(r"dataset\bbc")
df = df.drop_duplicates(subset="text", ignore_index=True)
df["clean_text"] = df["text"].apply(clean_text)

X, y = df["clean_text"], df["label"]
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42, stratify=y_valtest)

tfidf = TfidfVectorizer()

X_train_vectorized = tfidf.fit_transform(X_train)
X_val_transformed = tfidf.transform(X_val)
X_test_transformed = tfidf.transform(X_test)

# lgr = LogisticRegression(C=1.0, random_state=42, max_iter=100)
# lgr.fit(X_train_vectorized, y_train)
# score = lgr.score(X_val_transformed, y_val)
# print(score)

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
scores = {} # i thought of a sedikian idea where i want to get the best C out of 
# the for loop based on its score so i thought of creating a dictionary where it
# holds the C values as keys, and the scores as values for these keys, and later on
# we fetch the best score out of the dictionary and automatically its key will be with it (the best C value) 

for c in C_values :
    lgr = LogisticRegression(C=c, random_state=42, max_iter=100)
    lgr.fit(X_train_vectorized, y_train)
    score = lgr.score(X_val_transformed, y_val)
    scores[c] = score 
    # print(f"score with C = {c} is {score}")

best_C = max(scores, key=scores.get)
# print(best_C)

best_lgr = LogisticRegression(C=best_C, random_state=42, max_iter=100)

best_lgr.fit(X_train_vectorized, y_train)
y_pred = best_lgr.predict(X_test_transformed)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# btw best C is 1.0, after output results (accuracy = 0.9780)

# print(len(tfidf.vocabulary_))
# print(X_train_vectorized.shape)
# print(len(X_train), len(X_val), len(X_test))
# print(y_train.value_counts(normalize=True))
# print(y_val.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

# %%
