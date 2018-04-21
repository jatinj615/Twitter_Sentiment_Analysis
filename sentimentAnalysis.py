# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

df = pd.read_csv('train_E6oV3lV.csv');
df.head()
X = df[['tweet']]
y = df['label']

from keras.preprocessing.text import Tokenizer
max_features = 10000
tokenzer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', split=' ', lower=True, )
tokenzer.fit_on_texts(X['tweet'].values)
X = tokenzer.texts_to_sequences(X['tweet'].values)

from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X)
