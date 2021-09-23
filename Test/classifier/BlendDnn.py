import csv

import keras
from keras import Input, Model, Sequential
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import metrics
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import binarize

traindata = pd.read_csv(r'blend_train.csv',header=None)
traindata = np.array(traindata)
#LSTMFpr = data[:,:]
print(traindata.shape)
label = pd.read_csv(r'train_Y.csv',header=None)
label = np.array(label)
print(label.shape)

testdata = pd.read_csv(r'blendTest.csv',header=None)
testdata = np.array(testdata)
testtrain = testdata[:,0:5]
print(testtrain.shape)
y_test = testdata[:,5]
print(y_test.shape)
# wtr = csv.writer(open ('yLabel.csv', 'w'), delimiter='\n', lineterminator='\n')
# for x in label :
#     wtr.writerows([x])

model = Sequential()
model.add(Dense(1, input_dim=5))
model.add(Activation('sigmoid'))
# 二分类问题
#adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(traindata, label,epochs=30, batch_size=8,verbose=2)

y_pred = model.predict(testtrain)
y_pred_prob = y_pred.flatten()
# binarize方法根据指定的阈值将特征二值化，小于等于阈值的，将特征值赋予0，大于特征值的赋予1，其阈值threshold默认为0
y_pred_class = binarize([y_pred_prob], threshold=0.5)[0]
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred_class).ravel()
print('TN:', TN, 'FP:', FP, 'FN:', FN, 'TP:', TP)

ACC = metrics.accuracy_score(y_test, y_pred_class)
print('Accuracy:', ACC)

if (TP + FN != 0):
    Sens = metrics.recall_score(y_test, y_pred_class)
else:
    Sens = 1
print('Sensitivity:', Sens)

if (TN + FP != 0):
    Spec = TN / float(TN + FP)
else:
    Spec = 1
print('Specificity:', Spec)

if (TP + FP == 0 or TP + FN == 0 or TN + FP == 0 or TN + FN == 0):
    MCC = 1
else:
    MCC = metrics.matthews_corrcoef(y_test, y_pred_class)
print('Matthews correlation coefficient:', MCC)

if (TP + FN != 0 and TN + FP != 0):
    AUC = metrics.roc_auc_score(y_test, y_pred_prob)
else:
    AUC = 1
print('ROC Curves and Area Under the Curve (AUC):', AUC)



# model = Sequential()
# model.add(Dense(64, input_dim=5, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#
# model.fit(traindata, label,
#           epochs=20,
#           batch_size=128)
# score = model.evaluate(testdata, y_test, batch_size=128)

# plot history
# fig = pyplot.figure( figsize = (8,4) )   #定义整个画布
# ax1 = fig.add_subplot(121)                     #第一个子图
# ax1.plot(history.history['loss'], label='train')
# ax1.plot(history.history['val_loss'], label='test') #在子图上作图
# ax1.legend()
# ax2 = fig.add_subplot(122)
# ax2.plot(history.history['accuracy'], label='train')
# ax2.plot(history.history['val_accuracy'], label='test')
# ax2.legend()
# pyplot.show()