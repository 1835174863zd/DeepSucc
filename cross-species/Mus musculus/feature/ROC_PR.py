import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve

data = pd.read_csv(r'test_result.csv',header=None)
data = np.array(data)
print(data.shape)
y_cv_test = data[:,0]
y_pred_class = data[:,1]
y_pred_prob = data[:,2]
# ROC曲线
fpr, tpr, thresholds = roc_curve(y_cv_test, y_pred_prob)
precision, recall, thresholds = precision_recall_curve(y_cv_test, y_pred_prob)

# 计算auc
roc_auc = auc(fpr, tpr)
#aucs.append(roc_auc)
print(roc_auc)
#pyplot.plot(fpr, tpr, lw=1, alpha=0.5, label='ROC fold %d(area=%0.4f)' % (6, roc_auc))
pyplot.plot(recall, precision, lw=1, alpha=0.5, label='P-R fold %d' % (6))
pyplot.show()

confusion = metrics.confusion_matrix(y_cv_test, y_pred_class)
print(confusion)
TN, FP, FN, TP = metrics.confusion_matrix(y_cv_test, y_pred_class).ravel()
print('TN:', TN, 'FP:', FP, 'FN:', FN, 'TP:', TP)

ACC = metrics.accuracy_score(y_cv_test, y_pred_class)
print('Accuracy:', ACC)

if (TP + FN != 0):
    Sens = metrics.recall_score(y_cv_test, y_pred_class)
else:
    Sens = 1
print('Sensitivity:', Sens)

if (TN + FP != 0):
    Spec = TN / float(TN + FP)
else:
    Spec = 1
print('Specificity:', Spec)

if (TP + FP != 0):
    Precision = metrics.precision_score(y_cv_test, y_pred_class)
else:
    Precision = 1
print('Precision:', Precision)

if (TP + FP == 0 or TP + FN == 0 or TN + FP == 0 or TN + FN == 0):
    MCC = 1
else:
    MCC = metrics.matthews_corrcoef(y_cv_test, y_pred_class)
print('Matthews correlation coefficient:', MCC)

if (TP + FN != 0 and TN + FP != 0):
    AUC = metrics.roc_auc_score(y_cv_test, y_pred_prob)
else:
    AUC = 1
print('ROC Curves and Area Under the Curve (AUC):', AUC)
