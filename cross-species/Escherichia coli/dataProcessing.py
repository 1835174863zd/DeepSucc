import csv
import re
import random
def get_posNeg_data(input, encodedCsv):
    protein_name = []
    final_data = []
    pos_data = []
    neg_data = []
    windowSize = 15
    pos_number = 0
    neg_number = 0

    with open(input, 'r') as fp1:
        for (i, line) in enumerate(fp1):
            if i % 2 == 0:
                protein_name.append(line[1:-1])
        # print(protein_name)


    with open(encodedCsv, 'r') as fp2:
        for (m, line1) in enumerate(fp2):
            if m > 0:
                now = line1.split(',')
                # print(now)

                if now[1] in protein_name:
                    index_list = [i.start() for i in re.finditer('K', now[2])]
                    # print(index_list)

                    sequence=now[2]
                    math_seq=now[3]
                    # print(sequence)
                    # print(math_seq)

                    for j in index_list:
                        # print(j)
                        # print(sequence[j])
                        # print(math_seq[j])
                        k=j+1
                        if j<=windowSize-1:
                            piptide_a=sequence[k-1:k+windowSize]
                            piptide_b=sequence[2*k-1:k+windowSize]
                            piptide_c=piptide_b[::-1] + sequence[0:k-1]
                            piptide_data=piptide_c+piptide_a
                            final_data.append(piptide_data)
                            # print(math_seq[j])
                            if math_seq[j] == '1':
                                pos_number = pos_number + 1
                                code = [piptide_data,now[1], k, 1]
                                pos_data.append(code)
                            if math_seq[j] == '0':
                                code = [piptide_data, now[1], k, 0]
                                neg_data.append(code)
                                neg_number = neg_number + 1
                        elif j+1 > (len(sequence) - windowSize):
                            piptide_d=sequence[j-windowSize:2*j-len(sequence)+1]
                            piptide_e=sequence[j:]+piptide_d[::-1]
                            piptide_f=sequence[j-windowSize:j]
                            final_data.append(piptide_f+piptide_e)
                            if math_seq[j] == '1':
                                code = [piptide_f+piptide_e, now[1], k, 1]
                                pos_data.append(code)
                                pos_number = pos_number + 1
                            if math_seq[j] == '0':
                                code = [piptide_f + piptide_e, now[1], k, 0]
                                neg_data.append(code)
                                neg_number = neg_number + 1
                        else:
                            final_data.append(sequence[j-windowSize:j+windowSize+1])
                            if math_seq[j] == '1':
                                code = [sequence[j-windowSize:j+windowSize+1], now[1], k, 1]
                                pos_data.append(code)
                                pos_number = pos_number + 1
                            if math_seq[j] == '0':
                                code = [sequence[j - windowSize:j + windowSize + 1], now[1], k, 0]
                                neg_data.append(code)
                                neg_number = neg_number + 1
                        # print(final_data)

    print(pos_number)
    print(len(pos_data))
    print(len(neg_data))
    print(neg_number)
    print(len(final_data))
    neg_train = random.sample(neg_data,pos_number)
    print(len(neg_train))
    General_Train = pos_data + neg_train
    #print(pos_data)
    print(len(General_Train))
    with open('Esc_dataset.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in General_Train:
            writer.writerow(row)

get_posNeg_data('Escherichia_coli.fasta','encoded.csv')
