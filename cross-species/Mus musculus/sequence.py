import pandas as pd


def get_mathematical_sequence(input_url, output_url):
    data = {}
    sequence = {}
    with open(input_url) as fp:
        for (i, line) in enumerate(fp):
            if i > 0:
                infos = line.split(',')
                print(infos[0])
                if len(infos) < 2:
                    print(line, i)
                    continue
                if infos[1] not in data:
                    data[infos[1]] = []
                    sequence[infos[1]] = infos[3].strip()
                data[infos[1]].append(int(infos[2]))

    excel = {
        "UniProt Accession": [],
        "Sequence": [],
        "Mathematical Sequence": []
    }
    for key in data.keys():
        current_sequence = sequence[key]
        math_sequence = ""

        for (ind, val) in enumerate(current_sequence):
            if val == 'K':
                if ind + 1 in data[key]:
                    math_sequence += '1'
                else:
                    math_sequence += '0'
            else:
                math_sequence += '2'

        excel["UniProt Accession"].append(key)
        excel["Sequence"].append(current_sequence)
        excel["Mathematical Sequence"].append(math_sequence)

    df = pd.DataFrame(excel)
    df.to_csv(output_url)


get_mathematical_sequence('Mus_musculus_succ.csv', 'encoded.csv')
