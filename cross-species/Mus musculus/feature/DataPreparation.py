import csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

def get_shuffle(BLOSUM62_data,ACF_data,onehot_data,aaindex_data,CKSAAP_data,label):
    #This function disorganize the data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    BLOSUM62_dataset = BLOSUM62_data[index]
    ACF_dataset = ACF_data[index]
    onehot_dataset = onehot_data[index]
    aaindex_dataset = aaindex_data[index]
    CKSAAP_dataset = CKSAAP_data[index]
    label = label[index]
    return BLOSUM62_dataset,ACF_dataset,onehot_dataset,aaindex_dataset,CKSAAP_dataset,label,index


BLOSUM62_data = pd.read_csv(r'BLOSUM62/BLOSUM62.csv',header=None)
BLOSUM62_data = np.array(BLOSUM62_data)
BLOSUM62_data = scale(BLOSUM62_data)#标准化


ACF_data = pd.read_csv(r'ACF/ACF.csv',header=None)
ACF_data = np.array(ACF_data)
ACF_data = scale(ACF_data)

onehot_data = pd.read_csv(r'onehot/onehot.csv')
onehot_data = np.array(onehot_data)
onehot_data = onehot_data[:,1:]
onehot_data = onehot_data.astype('float64')

aaindex_data = pd.read_csv(r'AAindex/aaindex.csv')
aaindex_data = np.array(aaindex_data)
aaindex_data = aaindex_data[:,1:]
aaindex_data = aaindex_data.astype('float64')

CKSAAP_data = pd.read_csv(r'CKSAAP/CKSAAP.csv',header=None)
CKSAAP_data = np.array(CKSAAP_data)


print(BLOSUM62_data.shape)
print(ACF_data.shape)
print(onehot_data.shape)
print(aaindex_data.shape)
print(CKSAAP_data.shape)

label1=np.ones((6322,1))#Value can be changed
label2=np.zeros((6323,1))
y_=np.append(label1,label2)

BLOSUM62_X,ACF_X,onehot_X,aaindex_X,CKSAAP_X,train_Y,shuffle_index=get_shuffle(BLOSUM62_data,ACF_data,onehot_data,aaindex_data,CKSAAP_data,y_)

with open('BLOSUM62_X.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in BLOSUM62_X:
		writer.writerow(row)

with open('ACF_X.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in ACF_X:
		writer.writerow(row)

with open('onehot_X.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in onehot_X:
		writer.writerow(row)

with open('aaindex_X.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in aaindex_X:
		writer.writerow(row)

with open('CKSAAP_X.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in CKSAAP_X:
		writer.writerow(row)

with open('train_Y.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in map(lambda x:[x],train_Y):
		writer.writerow(row)

with open('shuffle_index.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in map(lambda x:[x],shuffle_index):
		writer.writerow(row)
