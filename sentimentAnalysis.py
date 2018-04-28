# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

df = pd.read_csv('train_E6oV3lV.csv');
X = df[['tweet']]
y = pd.get_dummies(df['label']).values

from keras.preprocessing.text import Tokenizer
max_features = 10000
tokenzer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', split=' ', lower=True, )
tokenzer.fit_on_texts(X['tweet'].values)
X = tokenzer.texts_to_sequences(X['tweet'].values)

from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, maxlen=100)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#import Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers import LSTM


# Using RNN
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

import tensorflow as tf
with tf.device('/gpu:0'):
    clf.fit(X_train, y_train, batch_size=32, epochs=10)


from keras.models import model_from_json
model_json = clf.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
clf.save_weights("model.h5")

# loading model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics = ['accuracy']) # In case of more than two categories loss function equals 'categorical_crossentropy'

y_pred = loaded_model.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred[:, 1]
score, acc = loaded_model.evaluate(X_test, y_test, batch_size=32)

y_test = y_test[:, 1]
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report
clf_report = classification_report(y_test, y_pred)


df_test = pd.read_csv('test_tweets_anuFYb8.csv')
tweets = df_test['tweet']
tweets = tokenzer.texts_to_sequences(tweets.values)
tweets = pad_sequences(tweets, maxlen=100)
predicted = loaded_model.predict(tweets)
predicted = (predicted > 0.5)
predicted = predicted[:, 1]
predicted = predicted*1
df_test['labels'] = predicted
df_test.to_csv('test_predictions_tweets.csv', index="False")

df_pred = pd.DataFrame(data=predicted, columns=['label'])
df_pred.to_csv('test_predictions.csv', sep='\t', index='False')
