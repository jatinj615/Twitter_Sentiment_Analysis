# -*- coding: utf-8 -*-
#import numpy and pandas library for data processing
import numpy as np
import pandas as pd

# importing data for training
df = pd.read_csv('train_E6oV3lV.csv');

# dividing data into target and input variables 
X = df[['tweet']]
y = pd.get_dummies(df['label']).values

# preprocessing data Tokenize the words
from keras.preprocessing.text import Tokenizer
max_features = 10000
tokenzer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', split=' ', lower=True, char_level=False, oov_token=None)
tokenzer.fit_on_texts(X['tweet'].values)
X = tokenzer.texts_to_sequences(X['tweet'].values)

# add padding
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, maxlen=100)


# Dividing data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#import Keras and important libraries and layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers import LSTM


# Using RNN
# Initialising classifier
clf = Sequential()
# Adding First Embedded Layer
clf.add(Embedding(max_features, 128, input_length=X.shape[1]))
clf.add(SpatialDropout1D(0.2))

# Adding Lstm Layer
clf.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))

# Adding fully connected layer
clf.add(Dense(128, activation='relu'))

# Adding output layer
clf.add(Dense(2, activation='softmax'))

# Compiling classifier
clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# importing tensorflow for training on gpu
import tensorflow as tf
with tf.device('/gpu:0'):
    clf.fit(X_train, y_train, batch_size=32, epochs=10)



# predicting on test data
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred[:, 1]

#evaluating classifier
score, acc = clf.evaluate(X_test, y_test, batch_size=32)
y_test = y_test[:, 1]

# import libraries for evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report
clf_report = classification_report(y_test, y_pred)


# importing test tweets file
df_test = pd.read_csv('test_tweets_anuFYb8.csv')
tweets = df_test['tweet']

# tokenize the words
tweets = tokenzer.texts_to_sequences(tweets.values)
tweets = pad_sequences(tweets, maxlen=100)

predicted = clf.predict(tweets)
predicted = (predicted > 0.5)
predicted = predicted[:, 1]
predicted = predicted*1

# creating csv file for predicted tweets
df_pred = pd.DataFrame(data=predicted, columns=['label'])
df_pred.to_csv('test_predictions.csv', sep='\t', index='False')
