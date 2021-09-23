import csv
from statistics import mode
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from keras import Input, Model, Sequential, metrics
from keras.layers import Bidirectional, LSTM, Reshape, Conv1D, Flatten, Dense, MaxPooling1D, Dropout, LeakyReLU
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot
from sklearn.preprocessing import scale, MinMaxScaler, binarize
from sklearn.model_selection import KFold
from sklearn import metrics
from collections import Counter
"""GPU设置为按需增长"""
import os
import tensorflow as tf
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def get_onehot_CNN_LSTM_model(train_X):
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')
    #inputDrop = Dropout(0.3)(inputs)
    conv1 = Conv1D(filters=32, strides=10, kernel_size=10, padding='valid', kernel_initializer='he_normal')(inputs)
    maxpool_pm = MaxPooling1D(6)(conv1)

    maxpool_pm = Dropout(0.5)(maxpool_pm)
    lstm1 = LSTM(64)(maxpool_pm)
    #lstm1 = Dropout(0.5)(lstm1)
    dense1 = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(lstm1)
    model = Model(inputs=inputs, outputs=dense1)

    #opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, decay=0.0, nesterov=False, name="SGD")
    # opt=keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def get_aaindex_LSTM_CNN_model(train_X):
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')
    # lstminputs = Dropout(0.5)(inputs)
    Lstm1 = LSTM(64, return_sequences=True)(inputs)
    Lstm1 = Dropout(0.5)(Lstm1)

    cnninputs = Reshape((-1, 1))(Lstm1)

    conv1 = Conv1D(filters=32, strides=1, kernel_size=20, padding='valid', kernel_initializer='random_normal')(
        cnninputs)
    conv1 = LeakyReLU(alpha=0.01)(conv1)
    conv1 = Dropout(0.5)(conv1)
    maxpool_pm = MaxPooling1D(6)(conv1)
    conv_ot = Flatten()(maxpool_pm)

    conv_ot = Dropout(0.5)(conv_ot)
    dense1 = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(conv_ot)

    # model = Model(inputs=[inputs,cnninputs], outputs=conv1)
    model = Model(inputs=inputs, outputs=dense1)
    # opt = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, decay=0.0, nesterov=False, name="SGD")
    # opt=keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def get_ACF_LSTM_model(train_X):
    model = Sequential()

    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
    model.add(Dropout(0.8))
    model.add(Dense(1, activation='sigmoid'))
    # 编译模型
    #opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")
    opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def get_blosum62_CNN_LSTM_model(train_X):
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')
    conv1 = Conv1D(filters=32, strides=20, kernel_size=20, padding='valid', kernel_initializer='random_normal')(inputs)
    maxpool_pm = MaxPooling1D(6)(conv1)
    maxpool_pm = Dropout(0.5)(maxpool_pm)
    lstm1 = LSTM(32)(maxpool_pm)
    dense1 = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(lstm1)
    model = Model(inputs=inputs, outputs=dense1)

    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, name="SGD")
    # opt=keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def get_CKSAAP_LSTM_CNN_model(train_X):
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')
    #lstminputs = Dropout(0.5)(inputs)

    Lstm1 = LSTM(64, return_sequences=True)(inputs)
    Lstm1 = Dropout(0.5)(Lstm1)

    cnninputs = Reshape((-1, 1))(Lstm1)

    conv1 = Conv1D(filters=32, strides=1, kernel_size=20, padding='valid', kernel_initializer='random_normal')(
        cnninputs)
    conv1 = LeakyReLU(alpha=0.01)(conv1)
    conv1 = Dropout(0.5)(conv1)

    maxpool_pm = MaxPooling1D(6)(conv1)
    conv_ot = Flatten()(maxpool_pm)

    conv_ot = Dropout(0.2)(conv_ot)
    dense1 = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(conv_ot)

    # model = Model(inputs=[inputs,cnninputs], outputs=conv1)
    model = Model(inputs=inputs, outputs=dense1)
    #opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, name="SGD")
    # opt=keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

BLOSUM62_train = pd.read_csv(r'BLOSUM62_train.csv',header=None)
BLOSUM62_train = np.array(BLOSUM62_train)
ACF_train = pd.read_csv(r'ACF_train.csv',header=None)
ACF_train = np.array(ACF_train)
onehot_train = pd.read_csv(r'onehot_train.csv',header=None)
onehot_train = np.array(onehot_train)
aaindex_train = pd.read_csv(r'aaindex_train.csv',header=None)
aaindex_train = np.array(aaindex_train)
CKSAAP_train = pd.read_csv(r'CKSAAP_train.csv',header=None)
CKSAAP_train = np.array(CKSAAP_train)
train_Y = pd.read_csv(r'train_Y.csv',header=None)
train_Y = np.array(train_Y)

BLOSUM62_test = pd.read_csv(r'BLOSUM62_test.csv',header=None)
BLOSUM62_test = np.array(BLOSUM62_test)
ACF_test = pd.read_csv(r'ACF_test.csv',header=None)
ACF_test = np.array(ACF_test)
onehot_test = pd.read_csv(r'onehot_test.csv',header=None)
onehot_test = np.array(onehot_test)
aaindex_test = pd.read_csv(r'aaindex_test.csv',header=None)
aaindex_test = np.array(aaindex_test)
CKSAAP_test = pd.read_csv(r'CKSAAP_test.csv',header=None)
CKSAAP_test = np.array(CKSAAP_test)
test_Y = pd.read_csv(r'test_Y.csv',header=None)
test_Y = np.array(test_Y)

print(BLOSUM62_train.shape,BLOSUM62_test.shape)
print(ACF_train.shape,ACF_test.shape)
print(onehot_train.shape,onehot_test.shape)
print(aaindex_train.shape,aaindex_test.shape)
print(CKSAAP_train.shape,CKSAAP_test.shape)
print(train_Y.shape,test_Y.shape)

onehot_train = onehot_train.reshape((onehot_train.shape[0], onehot_train.shape[1], 1))
onehot_test = onehot_test.reshape((onehot_test.shape[0], onehot_test.shape[1], 1))
aaindex_train = aaindex_train.reshape((aaindex_train.shape[0], 1, aaindex_train.shape[1]))
aaindex_test = aaindex_test.reshape((aaindex_test.shape[0], 1, aaindex_test.shape[1]))
ACF_train = ACF_train.reshape((ACF_train.shape[0], 1, ACF_train.shape[1]))
ACF_test = ACF_test.reshape((ACF_test.shape[0], 1, ACF_test.shape[1]))
BLOSUM62_train = BLOSUM62_train.reshape((BLOSUM62_train.shape[0], BLOSUM62_train.shape[1], 1))
BLOSUM62_test = BLOSUM62_test.reshape((BLOSUM62_test.shape[0], BLOSUM62_test.shape[1], 1))
CKSAAP_train = CKSAAP_train.reshape((CKSAAP_train.shape[0], 1, CKSAAP_train.shape[1]))
CKSAAP_test = CKSAAP_test.reshape((CKSAAP_test.shape[0], 1, CKSAAP_test.shape[1]))

oneot_CNN_model = get_onehot_CNN_LSTM_model(onehot_train)

aaindex_lstm_model = get_aaindex_LSTM_CNN_model(aaindex_train)

ACF_lstm_model = get_ACF_LSTM_model(ACF_train)

Blosum62_cnn_lstm_model = get_blosum62_CNN_LSTM_model(BLOSUM62_train)

CKAAAP_lstm_model = get_CKSAAP_LSTM_CNN_model(CKSAAP_train)

clfs = [
        oneot_CNN_model,
        aaindex_lstm_model,
        ACF_lstm_model,
        Blosum62_cnn_lstm_model,
        CKAAAP_lstm_model
    ]

blend_train = np.zeros((onehot_train.shape[0], 5)) # Number of training data x Number of classifiers
blend_test = np.zeros((BLOSUM62_test.shape[0], 5)) # Number of testing data x Number of classifiers
for j, clf in enumerate(clfs):
    print(j, clf)
    n_folds = 10
    train_num, test_num = BLOSUM62_train.shape[0], BLOSUM62_test.shape[0]
    blend_test_nfolds = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i, (train_index, test_index) in enumerate(kf.split(onehot_train)):
        #print(test_index)
        if j == 0:
            x_tra, y_tra = onehot_train[train_index], train_Y[train_index]
            x_tst, y_tst = onehot_train[test_index], train_Y[test_index]

            clf.fit(x_tra, y_tra, batch_size=128, epochs=300, shuffle=False, verbose=0)
            blend_train_nfolds = clf.predict(x_tst)
            #print(blend_train_nfolds.shape)
            blend_train[test_index, j] = blend_train_nfolds[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            blend_test_nfolds[:, i] = clf.predict(onehot_test)[:,0]
        if j == 1:
            x_tra, y_tra = aaindex_train[train_index], train_Y[train_index]
            x_tst, y_tst = aaindex_train[test_index], train_Y[test_index]

            clf.fit(x_tra, y_tra, batch_size=512, epochs=600, shuffle=False, verbose=0)
            blend_train[test_index, j] = clf.predict(x_tst)[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            blend_test_nfolds[:, i] = clf.predict(aaindex_test)[:,0]
        if j == 2:
            x_tra, y_tra = ACF_train[train_index], train_Y[train_index]
            x_tst, y_tst = ACF_train[test_index], train_Y[test_index]

            clf.fit(x_tra, y_tra, batch_size=128, epochs=50, shuffle=False, verbose=0)

            blend_train[test_index, j] = clf.predict(x_tst)[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            blend_test_nfolds[:, i] = clf.predict(ACF_test)[:,0]
        if j == 3:
            x_tra, y_tra = BLOSUM62_train[train_index], train_Y[train_index]
            x_tst, y_tst = BLOSUM62_train[test_index], train_Y[test_index]

            clf.fit(x_tra, y_tra, batch_size=32, epochs=600, shuffle=False, verbose=0)

            blend_train[test_index, j] = clf.predict(x_tst)[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            blend_test_nfolds[:, i] = clf.predict(BLOSUM62_test)[:,0]
        if j == 4:
            x_tra, y_tra = CKSAAP_train[train_index], train_Y[train_index]
            x_tst, y_tst = CKSAAP_train[test_index], train_Y[test_index]

            clf.fit(x_tra, y_tra, batch_size=512, epochs=80, shuffle=False, verbose=0)

            blend_train[test_index, j] = clf.predict(x_tst)[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            blend_test_nfolds[:, i] = clf.predict(CKSAAP_test)[:,0]

    blend_test[:, j] = blend_test_nfolds.mean(1)
    #y_pred_prob = blend_test[:, j].flatten()
    # binarize方法根据指定的阈值将特征二值化，小于等于阈值的，将特征值赋予0，大于特征值的赋予1，其阈值threshold默认为0
    #blend_test[:, j] = binarize([y_pred_prob], threshold=0.5)[0]
# 使用决策树作为我们的次级分类器

with open('blend_train.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in blend_train:
		writer.writerow(row)

with open('blend_test.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in blend_test:
		writer.writerow(row)