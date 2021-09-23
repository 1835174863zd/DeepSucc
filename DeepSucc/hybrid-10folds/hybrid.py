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

BLOSUM62_X = pd.read_csv(r'BLOSUM62_X.csv',header=None)
BLOSUM62_X = np.array(BLOSUM62_X)
ACF_X = pd.read_csv(r'ACF_X.csv',header=None)
ACF_X = np.array(ACF_X)
onehot_X = pd.read_csv(r'onehot_X.csv',header=None)
onehot_X = np.array(onehot_X)
aaindex_X = pd.read_csv(r'aaindex_X.csv',header=None)
aaindex_X = np.array(aaindex_X)
CKSAAP_X = pd.read_csv(r'CKSAAP_X.csv',header=None)
CKSAAP_X = np.array(CKSAAP_X)
Y_label = pd.read_csv(r'train_Y.csv',header=None)
Y_label = np.array(Y_label)

print(BLOSUM62_X.shape)
print(ACF_X.shape)
print(onehot_X.shape)
print(aaindex_X.shape)
print(CKSAAP_X.shape)
print(Y_label.shape)

onehot_X = onehot_X.reshape((onehot_X.shape[0], onehot_X.shape[1], 1))
aaindex_X = aaindex_X.reshape((aaindex_X.shape[0], 1, aaindex_X.shape[1]))
ACF_X = ACF_X.reshape((ACF_X.shape[0], 1, ACF_X.shape[1]))
BLOSUM62_X = BLOSUM62_X.reshape((BLOSUM62_X.shape[0], BLOSUM62_X.shape[1], 1))
CKSAAP_X = CKSAAP_X.reshape((CKSAAP_X.shape[0], 1, CKSAAP_X.shape[1]))

oneot_CNN_model = get_onehot_CNN_LSTM_model(onehot_X)

aaindex_lstm_model = get_aaindex_LSTM_CNN_model(aaindex_X)

ACF_lstm_model = get_ACF_LSTM_model(ACF_X)

Blosum62_cnn_lstm_model = get_blosum62_CNN_LSTM_model(BLOSUM62_X)

CKAAAP_lstm_model = get_CKSAAP_LSTM_CNN_model(CKSAAP_X)

clfs = [
        oneot_CNN_model,
        aaindex_lstm_model,
        ACF_lstm_model,
        Blosum62_cnn_lstm_model,
        CKAAAP_lstm_model
    ]

blend_train = np.zeros((onehot_X.shape[0], 5)) # Number of training data x Number of classifiers
#blend_test = np.zeros((BLOSUM62_test_X.shape[0], 5)) # Number of testing data x Number of classifiers
for j, clf in enumerate(clfs):
    print(j, clf)
    n_folds = 10
    #train_num, test_num = BLOSUM62_train_X.shape[0], BLOSUM62_test_X.shape[0]
    train_num = onehot_X.shape[0]
    #blend_test_nfolds = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i, (train_index, test_index) in enumerate(kf.split(onehot_X)):
        #print(test_index)
        if j == 0:
            x_tra, y_tra = onehot_X[train_index], Y_label[train_index]
            x_tst, y_tst = onehot_X[test_index], Y_label[test_index]

            clf.fit(x_tra, y_tra, batch_size=128, epochs=300, shuffle=False, verbose=0)
            blend_train_nfolds = clf.predict(x_tst)
            #print(blend_train_nfolds.shape)
            blend_train[test_index, j] = blend_train_nfolds[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            #blend_test_nfolds[:, i] = clf.predict(onehot_test_X)[:,0]
        if j == 1:
            x_tra, y_tra = aaindex_X[train_index], Y_label[train_index]
            x_tst, y_tst = aaindex_X[test_index], Y_label[test_index]

            clf.fit(x_tra, y_tra, batch_size=512, epochs=600, shuffle=False, verbose=0)
            blend_train[test_index, j] = clf.predict(x_tst)[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            #blend_test_nfolds[:, i] = clf.predict(aaindex_test_X)[:,0]
        if j == 2:
            x_tra, y_tra = ACF_X[train_index], Y_label[train_index]
            x_tst, y_tst = ACF_X[test_index], Y_label[test_index]

            clf.fit(x_tra, y_tra, batch_size=128, epochs=50, shuffle=False, verbose=0)

            blend_train[test_index, j] = clf.predict(x_tst)[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            #blend_test_nfolds[:, i] = clf.predict(ACF_test_X)[:,0]
        if j == 3:
            x_tra, y_tra = BLOSUM62_X[train_index], Y_label[train_index]
            x_tst, y_tst = BLOSUM62_X[test_index], Y_label[test_index]

            clf.fit(x_tra, y_tra, batch_size=32, epochs=600, shuffle=False, verbose=0)

            blend_train[test_index, j] = clf.predict(x_tst)[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            #blend_test_nfolds[:, i] = clf.predict(BLOSUM62_test_X)[:,0]
        if j == 4:
            x_tra, y_tra = CKSAAP_X[train_index], Y_label[train_index]
            x_tst, y_tst = CKSAAP_X[test_index], Y_label[test_index]

            clf.fit(x_tra, y_tra, batch_size=512, epochs=80, shuffle=False, verbose=0)

            blend_train[test_index, j] = clf.predict(x_tst)[:,0]
            #blend_train[test_index, j] = binarize(clf.predict(x_tst), threshold=0.5)[0]
            #blend_test_nfolds[:, i] = clf.predict(CKSAAP_test_X)[:,0]

    #blend_test[:, j] = blend_test_nfolds.mean(1)
    #y_pred_prob = blend_test[:, j].flatten()
    # binarize方法根据指定的阈值将特征二值化，小于等于阈值的，将特征值赋予0，大于特征值的赋予1，其阈值threshold默认为0
    #blend_test[:, j] = binarize([y_pred_prob], threshold=0.5)[0]
# 使用决策树作为我们的次级分类器

with open('blend_train.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in blend_train:
		writer.writerow(row)