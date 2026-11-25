from pathlib import Path
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv 
from processor import load_df


df = load_df ()

print ("num of img is ", len(df)) 
print (df["label"].unique()) 
print (df["label"].value_counts())

df["label"].value_counts().plot(kind="pie" , autopct="%1.1f%%")
plt.show()

sizes = []

for path in df["path"] : 
    img = cv.imread(path) 
    sizes.append(img.shape) 

df_sizes = pd.DataFrame(sizes , columns=["height", "width", "channel"]) 
print (df_sizes.value_counts().head(5))
print (df_sizes.describe())