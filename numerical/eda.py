import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("numerical/dataset/insurance.csv")

# writing statistics to eda.txt
with open ("numerical/eda.txt", "w") as f:
    f.write("=== DATASET INFO ===\n")
    df.info(buf=f)
    
    f.write("\n=== MISSING VALUES ===\n")
    f.write(df.isnull().sum().to_string())

    f.write("\n\n=== DUPLICATE ROWS ===\n")
    f.write(str(df.duplicated().sum()))
    
    f.write("\n\n=== SUMMARY STATISTICS ===\n")
    f.write(df.describe().to_string())


# making a figure and plotting our histograms to each axis
figure, axes = plt.subplots(2, 2)

axes[0, 0].hist(df['age'].dropna(), bins=20, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Age Distribution')

axes[0, 1].hist(df['bmi'].dropna(), bins=20, color='seagreen', edgecolor='black')
axes[0, 1].set_title('BMI Distribution')

axes[1, 0].hist(df['children'].dropna(), bins=6, color='coral', edgecolor='black')
axes[1, 0].set_title('Children Distribution')

axes[1, 1].hist(df['charges'].dropna(), bins=25, color='mediumpurple', edgecolor='black')
axes[1, 1].set_title('Charges Distribution')

plt.tight_layout()
plt.savefig("numerical/plots/histograms.png")

plt.show()

df['smoker'] = df['smoker'].map({"yes" : 1, "no": 0})
df['sex'] = df['sex'].map({"male" : 1, "female": 0})

# corr_matrix returns a matrix with our metadata and pearson cooficient for each cell
# check documentation or ask me(khairi) if you don't understand
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig("numerical/plots/correlation_heatmap.png")
plt.show()
