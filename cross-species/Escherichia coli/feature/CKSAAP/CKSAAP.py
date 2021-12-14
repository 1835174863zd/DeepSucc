# Author: Thomas Dorfer <thomas.a.dorfer@gmail.com>
import csv
import re
import numpy as np
from itertools import product

from feature.CKSAAP.validation import check_input, check_alpha, check_natural


def cksaap(X, *, k=1, remove_zero_cols=False, start=1, end=None):
    """Composition of k-spaced amino acid pairs.

    This function returns the k-spaced amino acid pair composition of each
    sequence in the dataset. Since there are 20 natural amino acids, there are
    400 possible amino acid pairs. The parameter 'k' represents the gap between
    the amino acid pair. An example for k=1 would be AxY, where 'x' can be any
    amino acid. Similary, an example for k=2 would be AxxY. If k=0, the function
    returns the dipeptide composition of each sequence.

    Parameters
    ----------

    X : string, fasta, or a list thereof
        Dataset of amino acid sequences.

    lambda_ : int, default=1
        Counted rank (tier) of the correlation along an amino acid sequence.

    k : int, default=1
        Space between two amino acid pairs.

    remove_zero_cols : bool, default=False
        If true, columns containing only zeros will be deleted.

    start : int, default=1
        Determines the starting point of the amino acid sequence. This number is
        based on one-based indexing.

    end : int, default=None
        Determines the end point of the amino acid sequence. Similarly to start,
        this number is based on one-based indexing.

    Returns
    -------

    arr :  ndarray of shape (n_samples, 400)
        Array containing k-spaced amino acid pair composition.

    patterns : list of length 400
        Amino acid pairs with k gaps corresponding to columns in arr.

    References
    ----------

    Chen, K., Kurgan, L.A. & Ruan, J. Prediction of flexible/rigid regions from
    protein sequences using k-spaced amino acid pairs. BMC Struct Biol 7, 25
    (2007). https://doi.org/10.1186/1472-6807-7-25

    """

    # input handling
    X = check_input(X)
    #print(X)
    # initialize empty array
    arr = np.empty((len(X), 400), dtype=float)

    # list of amino acids (IUPAC standard)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    doublets = sorted([c[0] + c[1] for c in product(amino_acids, repeat=2)])
    patterns = [doublets[i][0] + '.' * k + doublets[i][1] for i in range(len(doublets))]

    # compute CKSAAP
    for i, seq in enumerate(X):
        #print(seq)
        check_alpha(seq)  # check if alphabetical
        check_natural(seq)  # check for unnatural amino acids
        seq = seq[start - 1:end]  # positional information
        for j, pattern in enumerate(patterns):
            cnt_pattern = len(re.findall(r'(?=(' + pattern + '))', seq))

            #print(cnt_pattern)
            arr[i, j] = cnt_pattern/(31-k-1)

    # delete zero columns
    if remove_zero_cols:
        cols_zeros = np.where(~arr.any(axis=0))[0]
        arr = np.delete(arr, cols_zeros, axis=1)
        patterns = [i for j, i in enumerate(patterns) if j not in cols_zeros]

    return arr, patterns

seqs = []
protein = []

with open("../../Esc_dataset.csv", 'r') as fp2:
    for (m, line1) in enumerate(fp2):
        now = line1.split(',')
        protein_name = now[1]
        sequence = now[0]
        #seq.append([protein_name, sequence])
        seqs.append(sequence)
        protein.append(protein_name)

print(seqs)

# K=0
k0, pairs0 = cksaap(seqs, k=0, remove_zero_cols=False)
k0 = k0.tolist()
#print(k0)
print(pairs0)

# K=1
k1, pairs1 = cksaap(seqs, remove_zero_cols=False)
k1 = np.array(k1)
#print(k1)
#print(pairs1)

# K=2
k2, pairs2 = cksaap(seqs, k=2, remove_zero_cols=False)
k2 = np.array(k2)
#print(k2)
#print(pairs2)

# K=3
k3, pairs3 = cksaap(seqs, k=3, remove_zero_cols=False)
k3 = np.array(k3)
#print(k3)
#print(pairs3)

# K=4
k4, pairs4 = cksaap(seqs, k=4, remove_zero_cols=False)
k4 = np.array(k4)
#print(k4)
#print(pairs4)

encodings = np.concatenate((k0,k1,k2,k3,k4),axis=1)
'''
encodings = []
for i in range(len(k0)):
    temp = []
    temp.append(k0[i])
    temp.append(k1[i])
    temp.append(k2[i])
    temp.append(k3[i])
    temp.append(k4[i])
    encodings.append(temp)
print(temp)
'''
'''
header = ['#', pairs0, pairs1, pairs2, pairs3, pairs4]

encodings = [header, k0,]
result = protein + encodings
'''

with open('CKSAAP.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in encodings:
		writer.writerow(row)

