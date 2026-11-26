from pathlib import Path
import pandas as pd 
import numpy as np 
import cv2 as cv 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_df():
    data_dir = Path(__file__).parent / "dataset/seg"
    row = [] 
    for class_d in sorted(data_dir.iterdir()) :
        if class_d.is_dir():
            for img_p in class_d.glob("*") :
                row.append({"path":str(img_p),"label":class_d.name})
    df = pd.DataFrame(row) 
    return df

def load_data():
    df = load_df()
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
    x_train, x_temp, y_train, y_temp = train_test_split(x, y,test_size=.30 , random_state= 42 , stratify=y)
    x_val, x_test , y_val, y_test = train_test_split(x_temp, y_temp,test_size=.50 , random_state= 42 , stratify=y_temp)
    
    return x_train, x_val, x_test, y_train, y_val, y_test, le


if __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test, le = load_data()
    

x_train_flat=x_train.reshape(len(x_train), -1)
x_test_flat=x_test.reshape(len(x_test), -1)
x_val_flat=x_val.reshape(len(x_val), -1)

scaler =StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)
x_val_scaled = scaler.transform(x_val_flat)
