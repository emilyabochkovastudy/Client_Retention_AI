import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading csv file
from google.colab import drive
drive.mount('/content/drive')
dataset = pd.read_csv("/content/drive/MyDrive/Customers_information.csv")
dataset.head()

# Split the dataset into input (independent) and output (dependent) variables
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]
X.head(10)

# Feature Engineering
geography = pd.get_dummies(X['Geography'], drop_first = True)
gender = pd.get_dummies(X['Gender'], drop_first = True)

# удаляем старые категориальные признаки
X = X.drop(['Geography', 'Gender'], axis = 1)
X.head()

# объединяем новые признаки с X
X = pd.concat([X, geography, gender], axis = 1)
X.head()

print(y.value_counts(normalize = True))

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("X_train:", X_train)
print("X_test:", X_test)
print("Размер X_train:", X_train.shape)

print("y_train:", y_train)
print("y_test:", y_test)
print("Размер y_train:", y_test.shape)

## creating ANN
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

# initialize
classifier = Sequential()

# вводный слой
classifier.add(Dense(units = 12, activation='relu'))

# первый скрытый слой
classifier.add(Dense(units = 8, activation = 'relu'))
classifier.add(Dropout(0.2))
# второй скрытый слой
classifier.add(Dense(units = 5, activation = 'relu'))
classifier.add(Dropout(0.3))
# Optimizer
import tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate = 0.01)
# Compiling
classifier.compile(optimizer = opt,loss = 'binary_crossentropy', metrics=['accuracy'])

# adding early stop
import tensorflow as tf
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    min_delta = 0.0001,
    patience = 20,
    verbose = 1,
    mode = "auto",
    baseline = None,
    restore_best_weights = False,
    start_from_epoch = 0,
)

# Modelling
model_history = classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=1000,callbacks=early_stopping)

model_history.history.keys()

# Summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Making the predictions and evaluation the model
#predicting the Test set results
y_pred = classifier.predict(X_test)
#y_pred = (y_pred >= 0.5).astype(int).flatten()
y_pred = (y_pred >= 0.5)

print(X_test.shape)  # должно быть (2000, n_features)
print(y_test.shape)  # должно быть (2000,)
print(y_pred.shape)  # должно совпадать с y_test


score
