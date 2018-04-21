# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

df = pd.read_csv('train_E6oV3lV.csv');
X = df[['tweet']]
y = df['label']

from keras.preprocessing.text import Tokenizer
max_features = 8000
tokenzer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', split=' ', lower=True, )
tokenzer.fit_on_texts(X['tweet'].values)
X = tokenzer.texts_to_sequences(X['tweet'].values)

from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



#import Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialise ANN
clf = Sequential()

# Adding the input layer and the first hidden layer
clf.add(Dense(output_dim = 12, kernel_initializer = 'uniform', activation='relu', input_dim = 39))
clf.add(Dropout(rate=0.1))

# Adding Second Hidden layer
clf.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation='relu'))
clf.add(Dropout(rate=0.1))

# Adding Output layer
clf.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation='sigmoid')) #use suftmax function in more than two categories

# Compiling ANN
clf.compile(optimizer= 'adam', loss='binary_crossentropy', metrics = ['accuracy']) # In case of more than two categories loss function equals 'categorical_crossentropy'

import tensorflow as tf
with tf.device('/gpu:0'):
    clf.fit(X_train, y_train, batch_size=32, epochs=10)

y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


df = pd.read_csv('test_tweets_anuFYb8.csv')
