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

train = pd.concat([train, extract_features(train)], axis=1)
test = pd.concat([test, extract_features(test)], axis=1)

class SimpleNet(BaseEstimator, TransformerMixin):
    def __init__(self, neuron_numbers, input_dim=1, activation='tanh', loss='mse', optimizer='sgd', metrics=['accuracy']):
        self.model = Sequential()
        self.model.add(Dense(neuron_numbers[0], input_dim=input_dim))
        for i in neuron_numbers[1:]:
            self.model.add(Activation(activation))
            self.model.add(Dense(i))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    #
    def fit(self, x, y=None):
        self.model.fit(x, y)
        return self
    #
    def transform(self, x):
        return self.model.predict(x)
    #
    def transform(self, x):
        return self.model.predict(x)

N = 3
nets = [('features', 
            FeatureUnion([('net_%d' % i, 
                           SimpleNet([random.randint(30, 200) for j in range(3)] + [10], 
                                     input_dim=784)) 
                          for i in range(N)])), 
        ('classifier', SimpleNet([30, 50, 30, 50, 10], input_dim=N*10))]
model = Pipeline(nets)
model.fit(train[raw_features].as_matrix(), np_utils.to_categorical(train.label))

# for i in models:
#     models[i].fit(train[raw_features].as_matrix(), np_utils.to_categorical(train.label))

# features = ['p%d%d' % (i, j) for j in range(10) for i in range(N)]
# predicts = list(map(lambda i: 
#                         pd.DataFrame(models[i].predict(train[raw_features].as_matrix()), 
#                                      columns=['p%d%d' % (i, j) for j in range(10)]), range(N)))
# train = pd.concat([train] + predicts, axis=1)

# model = simple_net([30, 30, 50, 30, 50, 10], input_dim=10*N)
# model.fit(train[features].as_matrix(), np_utils.to_categorical(train.label))
# train['Label'] = model.predict_classes(train[features].as_matrix())

# for i in train.index[train.label != train.Label]:
#     matrix = train.iloc[i][raw_features].as_matrix().reshape(28, -1)
#     plot(matrix)

# train[['label', 'Label']][train.label != train.Label]

# predicts = list(map(lambda i: 
#                         pd.DataFrame(models[i].predict(test[raw_features].as_matrix()), 
#                                      columns=['p%d%d' % (i, j) for j in range(10)]), range(N)))
# test = pd.concat([test] + predicts, axis=1)
# test['Label'] = model.predict_classes(test[features].as_matrix())

for i in models:
    probabilities = models[i].predict(train[raw_features].as_matrix())
    train = pd.concat([train, 
                      pd.DataFrame(probabilities, columns=['p%d' % i for i in range(10)])], axis=1)

        features = get_feature_names()
        models = {i: simple_net([200, 50, 1], input_dim=len(features)) for i in range(10)}
        for i in range(10):
            train['l%d' % i] = 0
            train['l%d' % i][train.label==i] = 1 

        for i in range(10):
            models[i].fit(train[features].as_matrix(), train['l%d' % i])

        # probability predict
        reg_model = simple_net([200, 50, 10], input_dim=len(features))
        reg_model.fit(train[features].as_matrix(), np_utils.to_categorical(train.label))


        if 'Label' in test.columns:
            del test['Label']

        if 'ImageId' in test.columns:
            del test['ImageId']

        for i in range(10):
            test['l%d' % i] = models[i].predict_classes(test[features].as_matrix())

        test['lsum'] = (test['l0'] + test['l1'] + test['l2'] + test['l3'] + test['l4'] + 
                        test['l5'] + test['l6'] + test['l7'] + test['l8'] + test['l9'])

        probabilities = reg_model.predict(test[features].as_matrix())
        test = pd.concat([test, 
                          pd.DataFrame(probabilities, columns=['p%d' % i for i in range(10)])], axis=1)

        test['Label'] = -1
        # label the samples which is recognized by only one classifier
        for i in range(10):
            test['Label'][(test['l%d' % i]==1) & (test.lsum==1)] = i

        # label the sample with the maximum probability
        test['pmax'] = test[['p%d' % i for i in range(10)]].max(axis=1)
        for i in range(10):
            test['Label'][(test.Label==-1) & (test['p%d' % i]==test.pmax)] = i

        test['ImageId'] = test.index + 1
        test[['ImageId', 'Label']].to_csv('result.csv', index=False)



    def extract_features( data):
        columns = list(filter(lambda e: e.startswith('pixel'), data.columns))
        x = list(map(lambda e: e.reshape((28, 28)), data[columns].as_matrix()))
        col_sum = list(map(lambda e: e.sum(0), x))
        row_sum = list(map(lambda e: e.sum(1), x))
        #
        diag_sum = []
        for image in x:
            diag = [0 for i in range(28)]
            for i in range(28):
                for j in range(28):
                    diag[(i - j) % 28] += image[i][j]
            diag_sum += [diag]
        #
        result = pd.concat([pd.DataFrame(col_sum, columns=['c%d' % (i+1) for i in range(28)]), 
                            pd.DataFrame(row_sum, columns=['r%d' % (i+1) for i in range(28)]),
                            pd.DataFrame(diag_sum, columns=['d%d' % (i+1) for i in range(28)])], axis=1)
        return result


def plot(matrix):
    plt.set_cmap('gray')
    plt.pcolor(matrix)
    plt.show()


def interactive_classifier(data):
    for i in range(len(data)):
        matrix = data.iloc[i].as_matrix().reshape(28, -1)
        plt.imshow(matrix)
        plt.show()
        label = input('Input 0-9: ')
        yield int(label)

