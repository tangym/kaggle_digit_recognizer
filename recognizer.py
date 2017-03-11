import random
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.utils import np_utils
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
raw_features = list(test.columns)

model = Sequential([
            BatchNormalization(input_shape=(len(raw_features),)),
            Dense(128, activation='relu'),
            Dense(50, activation='relu'),
            Dense(10, activation='softmax')
        ])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(train[raw_features].as_matrix(), np_utils.to_categorical(train.label))

test['Label'] = model.predict_classes(test[raw_features].as_matrix())
test['ImageId'] = test.index + 1
test[['ImageId', 'Label']].to_csv('result.csv', index=False)
