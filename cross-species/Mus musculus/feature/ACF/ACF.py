'''
Given a physicochemical property h, each KSP (15, 15) peptide i was encoded into a 31-dimensional numerical vector:
pi = [h1,h2,h3,……,h21]
where h,is the AAindex physicochemical property for the i-th residue (i = 1, 2…, 31),

Then, the ACF was defined as:
r(n) = [ h(1)*h(1+n) + h(2)*h(2+n) + … + h(L-n)*h(L) ] / (L-n)

where L is the length of the KSP(15,15) and equal to 31, and n = 1,2,3,...,m.
m = 1,2,3,...,L-1, eg: m=1 denotes the correlation of two  neighboring amino acid residues.

for 31 aaindex value, each KSP (10, 10) peptide i, an n*31-dimensional vector was generated:
[ [r1,...rn]1 , [r1,....rn]2 , ... , [r1,...,rn]31 ]

'''
import csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

index=pd.read_table('aaindex31',sep='\s+',header=None)
index=index.to_numpy().T
index={x:y for x,y in zip('ARNDCQEGHILKMFPSTWYV',index.tolist())}
#print(index['A'][0], index['A'][1], index['A'][2])

result = []
with open("../../Mus_dataset.csv", 'r') as fp2:
    for (m, line1) in enumerate(fp2):#sequence
        now = line1.split(',')
        protein_name = now[1]
        sequence = now[0]
        L = len(sequence)  # KSP(15,15)
        aaindex = 0
        ACF = []
        while aaindex <= 30:#aaindex
            encodings = []
            for aa in sequence:
                encodings.append(index[aa][aaindex])
            for n in range(1, L):
                i = 0
                sum = 0
                while i < L - n:
                    sum = sum + encodings[i] * encodings[i + n]
                    # print(sum)
                    i = i + 1
                rn = sum / (L - n)
                ACF.append(rn)
            print(ACF)
            aaindex = aaindex + 1
        result.append(ACF)

result = np.array(result)
ACF_data = scale(result)
with open('ACF.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in ACF_data:
		writer.writerow(row)
