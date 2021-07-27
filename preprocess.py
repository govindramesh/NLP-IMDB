import os
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


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
labels = pd.get_dummies(data['sentiment'],drop_first=True)

labels['positive'] = labels['positive'].astype(int)
x_final = embedded_docs_train
y_final = labels['positive']

np.savez("/output/imdb_preprocessed.npz",x=x_final, y=y_final)
