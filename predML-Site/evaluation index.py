import numpy as np
import pandas as pd
from sklearn import metrics
data = pd.read_csv(r'test-result.csv')
data = np.array(data)
print(data.shape)
y_test = data[:,2]
print(y_test.shape)
y_prediction = data[:,3]
print(y_prediction.shape)
#y_prediction = y_prediction.astype(int)
y_test = y_test.astype(int)
y_prediction = y_prediction.astype(int)

print(y_test)
print(y_prediction)

confusion = metrics.confusion_matrix(y_test, y_prediction)
print(confusion)
TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_prediction).ravel()
print('TN:', TN, 'FP:', FP, 'FN:', FN, 'TP:', TP)

ACC = metrics.accuracy_score(y_test, y_prediction)
print('Accuracy:', ACC)

if (TP + FN != 0):
    Sens = metrics.recall_score(y_test, y_prediction)
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
    MCC = metrics.matthews_corrcoef(y_test, y_prediction)
print('Matthews correlation coefficient:', MCC)

if (TP + FN != 0 and TN + FP != 0):
    AUC = metrics.roc_auc_score(y_test, y_prediction)
else:
    AUC = 1
print('ROC Curves and Area Under the Curve (AUC):', AUC)