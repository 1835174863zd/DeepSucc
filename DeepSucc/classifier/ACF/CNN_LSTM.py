import pandas as pd
import numpy as np
from keras import backend
from matplotlib import pyplot
from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale, StandardScaler, binarize, MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, Dropout, Bidirectional, Conv1D, BatchNormalization, Reshape, \
    LeakyReLU, MaxPooling1D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.layers.recurrent import LSTM
from tensorflow import keras

from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn import metrics
"""GPU设置为按需增长"""
import os
import tensorflow as tf
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
#config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

seed = 10
np.random.seed(seed)
def get_shuffle(dataset,label):
    #This function disorganize the data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label

# 1 一层CNN 一层LSTM
def get_CNN_LSTM_model(train_X):
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')
    #inputDrop = Dropout(0.3)(inputs)
    conv1 = Conv1D(filters=32, strides=1, kernel_size=20, padding='valid', kernel_initializer='he_normal')(inputs)
    maxpool_pm = MaxPooling1D(6)(conv1)

    maxpool_pm = Dropout(0.5)(maxpool_pm)
    lstm1 = LSTM(32)(maxpool_pm)
    #lstm1 = Dropout(0.5)(lstm1)
    dense1 = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(lstm1)
    model = Model(inputs=inputs, outputs=dense1)

    #opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, name="SGD")
    # opt=keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

# 2 两层CNN 一层LSTM
def get_CNN_CNN_LSTM_model(train_X):
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')
    cnninputs = Dropout(0.5)(inputs)
    conv1 = Conv1D(filters=32, strides=20, kernel_size=20, padding='valid', kernel_initializer='random_normal')(cnninputs)
    convout = Dropout(0.2)(conv1)
    conv2 = Conv1D(filters=32, strides=1, kernel_size=5, padding='valid', kernel_initializer='random_normal')(conv1)
    maxpool_pm = MaxPooling1D(6)(conv2)
    lstm1 = LSTM(32)(maxpool_pm)
    dense1 = Dense(1, activation='sigmoid')(lstm1)

    # model = Model(inputs=[inputs,cnninputs], outputs=conv1)
    model = Model(inputs=inputs, outputs=dense1)

    opt = keras.optimizers.SGD(learning_rate=0.002, momentum=0.0, decay=0.0, nesterov=False, name="SGD")
    # opt=keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
# 3 两层CNN
def get_CNN_CNN_model(train_X):
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')

    conv1 = Conv1D(filters=32, strides=20, kernel_size=20, padding='valid', kernel_initializer='random_normal')(inputs)

    conv2 = Conv1D(filters=32, strides=1, kernel_size=5, padding='valid', kernel_initializer='random_normal')(conv1)
    maxpool_pm = MaxPooling1D(6)(conv2)
    conv_ot = Flatten()(maxpool_pm)
    dense1 = Dense(1, activation='sigmoid')(conv_ot)

    model = Model(inputs=inputs, outputs=dense1)

    opt = keras.optimizers.SGD(learning_rate=0.002, momentum=0.0, decay=0.0, nesterov=False, name="SGD")
    # opt=keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
# 4 一层CNN
def get_CNN_model(train_X):
    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')

    conv1 = Conv1D(filters=32, strides=5, kernel_size=5, padding='valid', kernel_initializer='random_normal')(inputs)

    maxpool_pm = MaxPooling1D(3)(conv1)
    conv_ot = Flatten()(maxpool_pm)
    conv_ot = Dropout(0.5)(conv_ot)
    dense1 = Dense(1, activation='sigmoid')(conv_ot)

    model = Model(inputs=inputs, outputs=dense1)

    opt = keras.optimizers.SGD(learning_rate=0.002, momentum=0.0, decay=0.0, nesterov=False, name="SGD")
    # opt=keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


# 2、数据处理(准备训练数据)
data_=pd.read_csv(r'../../feature/ACF/ACF_Train.csv',header=None)
data=np.array(data_)
print(data.shape)
#[m1,n1]=np.shape(data)
#print(m1,n1)
label1=np.ones((21767,1))#Value can be changed
label2=np.zeros((21767,1))
label=np.append(label1,label2)
X_=scale(data)
y_= label
train_X,train_y=get_shuffle(X_,y_)
print(train_X,train_y)

# reshape input to be 3D [samples, timesteps, features]
#train_X = train_X.reshape((train_X.shape[0], 31, 30))
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
print(train_X.shape, train_y.shape)

'''
#3、训练模型
lstm_model = get_CNN_LSTM_model(train_X)
history = lstm_model.fit(train_X,train_y,batch_size=512,epochs=300,validation_split=0.3,verbose=2)
# plot history
fig = pyplot.figure( figsize = (8,4) )   #定义整个画布
ax1 = fig.add_subplot(121)                     #第一个子图
ax1.plot(history.history['loss'], label='train')
ax1.plot(history.history['val_loss'], label='test') #在子图上作图
ax1.legend()
ax2 = fig.add_subplot(122)
ax2.plot(history.history['accuracy'], label='train')
ax2.plot(history.history['val_accuracy'], label='test')
ax2.legend()
pyplot.show()
'''

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
count = 0
sens_cv = np.array([])
sepc_cv = np.array([])
acc_cv = np.array([])
mcc_cv = np.array([])
auc_cv = np.array([])
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)
for train, test in kfold.split(train_X, train_y):
    # create model
    lstm_model = get_CNN_LSTM_model(train_X)
    x_cv_tarin, y_cv_train = train_X[train], train_y[train]
    x_cv_test, y_cv_test = train_X[test], train_y[test]
    lstm_model.fit(train_X[train], train_y[train], batch_size=512,epochs=300, verbose=0)
    # evaluate the model
    y_pred = lstm_model.predict(x_cv_test)
    print(y_pred)
    del lstm_model
    backend.clear_session()

    count = count + 1
    print('--------------', 'CV  ', count, '---------------------')

    y_pred_prob = y_pred.flatten()
    #binarize方法根据指定的阈值将特征二值化，小于等于阈值的，将特征值赋予0，大于特征值的赋予1，其阈值threshold默认为0
    y_pred_class = binarize([y_pred_prob], threshold=0.5)[0]

    confusion = metrics.confusion_matrix(y_cv_test, y_pred_class)
    print(confusion)
    TN, FP, FN, TP = metrics.confusion_matrix(y_cv_test, y_pred_class).ravel()
    print('TN:', TN, 'FP:', FP, 'FN:', FN, 'TP:', TP)

    ACC = metrics.accuracy_score(y_cv_test, y_pred_class)
    print('Accuracy:', ACC)
    acc_cv = np.append(acc_cv, ACC)

    if (TP + FN != 0):
        Sens = metrics.recall_score(y_cv_test, y_pred_class)
    else:
        Sens = 1
    print('Sensitivity:', Sens)
    sens_cv = np.append(sens_cv, Sens)

    if (TN + FP != 0):
        Spec = TN / float(TN + FP)
    else:
        Spec = 1
    print('Specificity:', Spec)
    sepc_cv = np.append(sepc_cv, Spec)

    if (TP + FP == 0 or TP + FN == 0 or TN + FP == 0 or TN + FN == 0):
        MCC = 1
    else:
        MCC = metrics.matthews_corrcoef(y_cv_test, y_pred_class)
    print('Matthews correlation coefficient:', MCC)
    mcc_cv = np.append(mcc_cv, MCC)

    if (TP + FN != 0 and TN + FP != 0):
        AUC = metrics.roc_auc_score(y_cv_test, y_pred_prob)
    else:
        AUC = 1
    print('ROC Curves and Area Under the Curve (AUC):', AUC)
    auc_cv = np.append(auc_cv, AUC)
    print(AUC)

    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_cv_test, y_pred_prob)
    # interp:插值 把结果添加到tprs列表中
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print(roc_auc)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
    pyplot.plot(fpr, tpr, lw=1, alpha=0.5, label='ROC fold %d(area=%0.2f)' % (count, roc_auc))

    # if (count % 5 == 0):
    #   time.sleep(600)

print()
print('---------------END------------')
print('Sensitivity:', '   ', sens_cv.mean(), '(', sens_cv.std(), ')')
print('Specificity:', '   ', sepc_cv.mean(), '(', sepc_cv.std(), ')')
print('ACC:', '   ', acc_cv.mean(), '(', acc_cv.std(), ')')
print('MCC:', '   ', mcc_cv.mean(), '(', mcc_cv.std(), ')')
print('AUC:', '   ', auc_cv.mean(), '(', auc_cv.std(), ')')
print('\n\n\n')

# 平均ROC曲线
# 画对角线
pyplot.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'FPR':mean_fpr,'TPR':mean_tpr})
#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("ACF_CNN_LSTM.csv",index=False,sep=',')

mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
std_auc = np.std(tprs, axis=0)
pyplot.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
pyplot.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
pyplot.xlim([-0.05, 1.05])
pyplot.ylim([-0.05, 1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('ROC')
pyplot.legend(loc='lower right')
pyplot.show()

