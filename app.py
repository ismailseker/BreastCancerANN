import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

# data = pd.read_csv("data.csv")

breast_cancer_datasets = sklearn.datasets.load_breast_cancer()

data = pd.DataFrame(breast_cancer_datasets.data, columns=breast_cancer_datasets.feature_names)
data['label'] = breast_cancer_datasets.target

print(data.shape)
print(data.info)
print(data.isnull().sum())
print(data['label'].value_counts())

# 1 = Benign
# 0 = Malignant

X = data.drop(columns='label',axis=1)
Y = data['label']

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.2,random_state=42)

print(X.shape,xTrain.shape,xTest.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

xTrain_std = scaler.fit_transform(xTrain)
xTest_std = scaler.transform(xTest)

import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
print(tf.random.uniform([3]))

