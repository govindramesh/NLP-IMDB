import os
from mlflow import log_metric
MODEL_DIR = "/opt/dkube/output"

import pandas as pd
import numpy as np

npzfile = np.load("/tmp/imdb_preprocessed.npz")
x_final = npzfile["x"]
y_final = npzfile["y"]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Bidirectional

import tensorflow.keras as keras
opt = keras.optimizers.Adam(learning_rate=0.01)

import tensorflow.keras as keras
vector_feature = 200
voc_size = 90000
set_length = 700
model = Sequential()
model.add(Embedding(voc_size,vector_feature,input_length=set_length))
Dropout(0.20)
model.add(LSTM(64,return_sequences=True))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33)

print(y_test.shape)

# Add mlflow metric calls
class loggingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        log_metric ("train_loss", logs["loss"], step=epoch)
        log_metric ("train_accuracy", logs["accuracy"], step=epoch)
        log_metric ("val_loss", logs["val_loss"], step=epoch)
        log_metric ("val_accuracy", logs["val_accuracy"], step=epoch)

# Replace training command for formal training
model.fit(X_train, y_train,validation_data=(X_test,y_test),epochs=4,batch_size=64,verbose=1,
        callbacks=[loggingCallback(), tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR)])
        
os.makedirs(f"{MODEL_DIR}/1", exist_ok=True)
tf.saved_model.save(model,f"{MODEL_DIR}")
