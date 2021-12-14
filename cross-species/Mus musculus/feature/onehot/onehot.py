import csv
import pandas as pd
import numpy as np

def binary_encode(file):
    aminoacids = 'ARNDCQEGHILKMFPSTWYV'
    index = {x: y for x, y in zip(aminoacids, np.eye(20, 20).tolist())}
    encodings = []
    header = ['#']
    for i in range(1, 30 * 20 + 1):
        header.append('onehot.F' + str(i))
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


result=binary_encode('../../Mus_dataset.csv')
with open('onehot.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in result:
		writer.writerow(row)