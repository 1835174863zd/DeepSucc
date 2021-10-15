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

BLOSUM62_train = pd.read_csv(r'../../feature/BLOSUM62/BLOSUM62_TrainData.csv',header=None)
BLOSUM62_train = np.array(BLOSUM62_train)
BLOSUM62_test = pd.read_csv(r'../../feature/BLOSUM62/BLOSUM62_TestData.csv',header=None)
BLOSUM62_test = np.array(BLOSUM62_test)

ACF_train = pd.read_csv(r'../../feature/ACF/ACF_TrainData.csv',header=None)
ACF_train = np.array(ACF_train)
ACF_test = pd.read_csv(r'../../feature/ACF/ACF_TestData.csv',header=None)
ACF_test = np.array(ACF_test)

onehot_train = pd.read_csv(r'../../feature/onehot/onehot_Traindata.csv')
onehot_train = np.array(onehot_train)
onehot_train = onehot_train[:,1:]
onehot_train = onehot_train.astype('float64')

onehot_test = pd.read_csv(r'../../feature/onehot/onehot_Testdata.csv')
onehot_test = np.array(onehot_test)
onehot_test = onehot_test[:,1:]
onehot_test = onehot_test.astype('float64')

aaindex_train = pd.read_csv(r'../../feature/AAindex/aaindex_Traindata.csv')
aaindex_train = np.array(aaindex_train)
aaindex_train = aaindex_train[:,1:]
aaindex_train = aaindex_train.astype('float64')

aaindex_test = pd.read_csv(r'../../feature/AAindex/aaindex_Testdata.csv')
aaindex_test = np.array(aaindex_test)
aaindex_test = aaindex_test[:,1:]
aaindex_test = aaindex_test.astype('float64')

CKSAAP_train = pd.read_csv(r'../../feature/CKSAAP/CKSAAP_TrainData.csv',header=None)
CKSAAP_train = np.array(CKSAAP_train)
CKSAAP_test = pd.read_csv(r'../../feature/CKSAAP/CKSAAP_TestData.csv',header=None)
CKSAAP_test = np.array(CKSAAP_test)

label1=np.ones((17414,1))
label2=np.zeros((17414,1))
train_Y=np.append(label1,label2)
label1=np.ones((4353,1))
label2=np.zeros((4353,1))
test_Y=np.append(label1,label2)
print(BLOSUM62_train.shape,BLOSUM62_test.shape)
print(ACF_train.shape,ACF_test.shape)
print(onehot_train.shape,onehot_test.shape)
print(aaindex_train.shape,aaindex_test.shape)
print(CKSAAP_train.shape,CKSAAP_test.shape)
print(train_Y.shape,test_Y.shape)


BLOSUM62_train,ACF_train,onehot_train,aaindex_train,CKSAAP_train,train_Y,shuffle_train=get_shuffle(BLOSUM62_train,ACF_train,onehot_train,aaindex_train,CKSAAP_train,train_Y)
BLOSUM62_test,ACF_test,onehot_test,aaindex_test,CKSAAP_test,test_Y,shuffle_test=get_shuffle(BLOSUM62_test,ACF_test,onehot_test,aaindex_test,CKSAAP_test,test_Y)

with open('BLOSUM62_train.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in BLOSUM62_train:
		writer.writerow(row)
with open('BLOSUM62_test.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in BLOSUM62_test:
		writer.writerow(row)

with open('ACF_train.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in ACF_train:
		writer.writerow(row)
with open('ACF_test.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in ACF_test:
		writer.writerow(row)

with open('onehot_train.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in onehot_train:
		writer.writerow(row)
with open('onehot_test.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in onehot_test:
		writer.writerow(row)

with open('aaindex_train.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in aaindex_train:
		writer.writerow(row)
with open('aaindex_test.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in aaindex_test:
		writer.writerow(row)

with open('CKSAAP_train.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in CKSAAP_train:
		writer.writerow(row)
with open('CKSAAP_test.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in CKSAAP_test:
		writer.writerow(row)

with open('train_Y.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in map(lambda x:[x],train_Y):
		writer.writerow(row)
with open('test_Y.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in map(lambda x:[x],test_Y):
		writer.writerow(row)

with open('shuffle_train.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in map(lambda x:[x],shuffle_train):
		writer.writerow(row)
with open('shuffle_test.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in map(lambda x:[x],shuffle_test):
		writer.writerow(row)
