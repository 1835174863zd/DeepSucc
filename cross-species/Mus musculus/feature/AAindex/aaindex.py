import csv

import pandas as pd
import numpy as np

def aaindex(file):
    index=pd.read_table('aaindex31',sep='\s+',header=None)
    index=index.subtract(index.min(axis=1),axis=0).divide((index.max(axis=1)-index.min(axis=1)),axis=0)
    index=index.to_numpy().T
    index={x:y for x,y in zip('ARNDCQEGHILKMFPSTWYV',index.tolist())}
    index['X']=np.zeros(31).tolist()
    #encoding=[]
    #label=[]

    encodings = []
    header = ['#']
    for i in range(1, 30 * 31 + 1):
        header.append('AAindex.F' + str(i))
    encodings.append(header)

    with open(file, 'r') as fp2:
        for (m, line1) in enumerate(fp2):
            now = line1.split(',')
            name = now[1]
            sequence = now[0]
            code = [name]
            for aa in (sequence[0:15]+sequence[16:]):
                code = code + index[aa]
            encodings.append(code)
            print(code)
    return encodings


result=aaindex('../../Mus_dataset.csv')
with open('aaindex.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in result:
		writer.writerow(row)