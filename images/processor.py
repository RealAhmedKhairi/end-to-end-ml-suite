from pathlib import Path
import pandas as pd 
import numpy as np 
import cv2 as cv 
from sklearn.preprocessing import LabelEncoder 

data_dir = Path(__file__).parent / "dataset/seg"
row = [] 
for class_d in sorted(data_dir.iterdir()) :
    if class_d.is_dir():
        for img_p in class_d.glob("*") :
            row.append({"path":str(img_p),"label":class_d.name})

df = pd.DataFrame(row) 

df_sample = (df.groupby("label")
             .sample(n=1000, random_state=42)
             .reset_index(drop=True)
             )

print(df_sample)
print (df_sample["label"].value_counts())

im_size= (64 , 64)

images= []
or_size = set() 

for pth in df_sample["path"]:
    img = cv.imread(pth)
    or_size.add(img.shape)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img,im_size,interpolation=cv.INTER_AREA)
    img = img.astype(np.float32) / 255 

    images.append(img)

x = np.array(images)
le = LabelEncoder()
y = le.fit_transform(df_sample["label"])