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

# print(X.shape,xTrain.shape,xTest.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

xTrain_std = scaler.fit_transform(xTrain)
xTest_std = scaler.transform(xTest)

import tensorflow as tf
tf.random.set_seed(3)
print(tf.random.uniform([3]))

from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Input(shape=(30,)), 
    layers.Flatten(),           
    layers.Dense(20, activation='relu'),
    layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(xTrain_std, yTrain, validation_split=0.1, epochs=10)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(accuracy, label='training data')
plt.plot(val_accuracy, label='validation data')

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'upper right')
plt.show()

loss, accuracy = model.evaluate(xTest_std, yTest)
print(accuracy)

yPred = model.predict(xTest_std)

my_list = [0.25, 0.56]
index_of_max_value = np.argmax(my_list)
print(my_list)
print(index_of_max_value)

yPred_labels = [np.argmax(i) for i in yPred]
print(yPred_labels)

input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0] == 0):
  print('The tumor is Malignant')

else:
  print('The tumor is Benign')