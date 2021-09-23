import csv
import keras
from keras import Input, Model, Sequential, backend
import pandas as pd
import numpy as np
from matplotlib import pyplot
from numpy import interp
from sklearn import metrics
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import binarize
from tensorflow import optimizers

seed = 10
np.random.seed(seed)
traindata = pd.read_csv(r'blend_train.csv',header=None)
traindata = np.array(traindata)
#LSTMFpr = data[:,:]
print(traindata.shape)
label = pd.read_csv(r'train_Y.csv',header=None)
label = np.array(label)
print(label.shape)
def get_DNN_model():
    model = Sequential()
    model.add(Dense(1, input_dim=5))
    model.add(Activation('sigmoid'))
    # 二分类问题
    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # sgd = optimizers.SGD(learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, name="SGD")
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

'''
model = get_DNN_model()
history = model.fit(traindata, label,batch_size=8,epochs=30, validation_split=0.2, verbose=2)
# y_pred = model.predict(testdata)
# y_pred_prob = y_pred.flatten()
# # binarize方法根据指定的阈值将特征二值化，小于等于阈值的，将特征值赋予0，大于特征值的赋予1，其阈值threshold默认为0
# y_pred_class = binarize([y_pred_prob], threshold=0.5)[0]
# confusion = metrics.confusion_matrix(y_test, y_pred_class)
# print(confusion)
# TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred_class).ravel()
# print('TN:', TN, 'FP:', FP, 'FN:', FN, 'TP:', TP)
#
# ACC = metrics.accuracy_score(y_test, y_pred_class)
# print('Accuracy:', ACC)
#
# if (TP + FN != 0):
#     Sens = metrics.recall_score(y_test, y_pred_class)
# else:
#     Sens = 1
# print('Sensitivity:', Sens)
#
# if (TN + FP != 0):
#     Spec = TN / float(TN + FP)
# else:
#     Spec = 1
# print('Specificity:', Spec)
#
# if (TP + FP == 0 or TP + FN == 0 or TN + FP == 0 or TN + FN == 0):
#     MCC = 1
# else:
#     MCC = metrics.matthews_corrcoef(y_test, y_pred_class)
# print('Matthews correlation coefficient:', MCC)
#
# if (TP + FN != 0 and TN + FP != 0):
#     AUC = metrics.roc_auc_score(y_test, y_pred_prob)
# else:
#     AUC = 1
# print('ROC Curves and Area Under the Curve (AUC):', AUC)
#
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
for train, test in kfold.split(traindata, label):
    x_cv_tarin, y_cv_train = traindata[train], label[train]
    x_cv_test, y_cv_test = traindata[test], label[test]
    model = get_DNN_model()
    model.fit(traindata[train], label[train], batch_size=8, epochs=30, verbose=0)
    # evaluate the model
    y_pred = model.predict(x_cv_test)
    del model
    backend.clear_session()

    count = count + 1
    print('--------------', 'CV  ', count, '---------------------')

    y_pred_prob = y_pred.flatten()
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
    pyplot.plot(fpr, tpr, lw=1, alpha=0.5, label='ROC fold %d(area=%0.4f)' % (count, roc_auc))

print()
print('---------------END------------')
print('Sensitivity:', '   ', sens_cv.mean(), '(', sens_cv.std(), ')')
print('Specificity:', '   ', sepc_cv.mean(), '(', sepc_cv.std(), ')')
print('ACC:', '   ', acc_cv.mean(), '(', acc_cv.std(), ')')
print('MCC:', '   ', mcc_cv.mean(), '(', mcc_cv.std(), ')')
print('AUC:', '   ', auc_cv.mean(), '(', auc_cv.std(), ')')
print('\n\n\n')
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'FPR':mean_fpr,'TPR':mean_tpr})
#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("hybrid-4folds.csv",index=False,sep=',')

mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
print(mean_auc)
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

