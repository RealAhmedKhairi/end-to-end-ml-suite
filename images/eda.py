from pathlib import Path
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv 

data_dir = Path(__file__).parent / "dataset/seg"
row = [] 
for class_d in sorted(data_dir.iterdir()) :
    if class_d.is_dir():
        for img_p in class_d.glob("*") :
            row.append({"path":str(img_p),"label":class_d.name})

df = pd.DataFrame(row) 

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