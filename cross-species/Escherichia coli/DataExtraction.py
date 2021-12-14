import pandas as pd

def elm_to_csv(input, output_m):
    needed_data = ['UniProt Accession', 'Position', 'Sequence']
    mice = {}

    with open(input) as fp:
        lines = fp.readlines()
        header = lines[0].split('\t')
        print(header)
        for head in header:
            if head in needed_data:
                mice[head] = []

        for (i, line) in enumerate(lines):
            if i > 0:
                splitedData = line.split('\t')
                for (j, val) in enumerate(splitedData):
                    if header[j] in needed_data:
                            mice[header[j]].append(val)


        mice_df = pd.DataFrame(mice)
        mice_df.to_csv(output_m, sep=",")

        print("General ", mice_df.shape)


elm_to_csv('Escherichia_coli.txt', 'Escherichia_coli_succ.csv')