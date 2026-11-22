import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from processor import load_data

# Loading the processed data from the file processor, this is the data you are going to use
X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()

# You can now start training the model