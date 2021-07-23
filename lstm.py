from mlflow import log_metric
MODEL_DIR = "/opt/dkube/output"

import pandas as pd
import numpy as np

data = pd.read_csv('./IMDB Dataset.csv', engine="python", error_bad_lines=False)
print(data.head())

import nltk
nltk.download("stopwords")
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()

data['review'] = data['review'].str.lower()
data['review'].replace('https?://\S+|www\.\S+'," ",regex=True,inplace=True)
data['review'].replace('<.*?>'," ",regex=True,inplace=True)
data['review'].replace('@\w+'," ",regex=True,inplace=True)
data['review'].replace('#\w+'," ",regex=True,inplace=True)
data['review'].replace("[^\w\s\d]"," ",regex=True,inplace=True)
data['review'].replace(r'( +)'," ",regex=True,inplace=True)
data['review'].replace("[^a-zA-Z]"," ",regex=True,inplace=True)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm
data_lem = []
for i in tqdm(range(0,len(data.index))):
  temp = data['review'][i].split()
  temp = [word.lower() for word in temp]
  temp = [word for word in temp if word not in stopwords.words("english")]
  temp = " ".join(temp)
  data_lem.append(temp)
  
voc_size = 90000
from keras.preprocessing.text import Tokenizer
t = Tokenizer(num_words=voc_size,oov_token='<OOV>')
t.fit_on_texts(data_lem)
word_index=t.word_index
total_vocab=len(word_index)

train = t.texts_to_sequences(data_lem)

set_length = 700
embedded_docs_train = pad_sequences(train,padding='pre',maxlen =set_length)

from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Bidirectional

import keras
opt = keras.optimizers.Adam(learning_rate=0.01)

import keras
vector_feature = 200
model = Sequential()
model.add(Embedding(voc_size,vector_feature,input_length=set_length))
Dropout(0.20)
model.add(LSTM(64,return_sequences=True))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
labels = pd.get_dummies(data['sentiment'],drop_first=True)

labels['positive'] = labels['positive'].astype(int)
x_final = embedded_docs_train
y_final = labels['positive']

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
tf.saved_model.save(model,f"{MODEL_DIR}/1")
