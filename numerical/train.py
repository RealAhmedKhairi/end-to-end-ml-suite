import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("numerical/dataset/insurance.csv")
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop('charges', axis = 1)
y = df['charges']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
) 

X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

age_mean = X_train['age'].mean()
bmi_mean = X_train['bmi'].mean()
children_mean = X_train['children'].mean()

smoker_mode = X_train["smoker"].mode()[0]
sex_mode = X_train["sex"].mode()[0]
region_mode = X_train["region"].mode()[0]

for split in [X_train, X_val, X_test]:
    split['age'] = split['age'].fillna(age_mean)
    split['bmi'] = split['bmi'].fillna(bmi_mean)
    split['children'] = split['children'].fillna(children_mean)

    split['smoker'] = split['smoker'].fillna(smoker_mode)
    split['sex'] = split['sex'].fillna(sex_mode)
    split['region'] = split['region'].fillna(region_mode)
 
    split['smoker'] = split['smoker'].map({"yes" : 1, "no" : 0})
    split['sex'] = split['sex'].map({"male" : 1, "female": 0})

X_train = pd.get_dummies(X_train, columns = ['region'], drop_first = True, dtype = int)
X_val = pd.get_dummies(X_val, columns = ['region'], drop_first = True, dtype = int)
X_test = pd.get_dummies(X_test, columns = ['region'], drop_first = True, dtype = int)

# testing (Ahmed/Habiba) 3ady momkn t3mlo delete 
print(f"Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print("Missing values in Train:", X_train.isnull().sum().sum())
print("Missing values in Val:", X_val.isnull().sum().sum())
print("Missing values in Test:", X_test.isnull().sum().sum())
print("\nEncoded columns:", list(X_train.columns))