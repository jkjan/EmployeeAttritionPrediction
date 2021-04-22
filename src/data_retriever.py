import pandas as pd
from random import shuffle
from codes import codes


def get_original_data(path):
    original_data = pd.read_csv(path)
    return original_data


def get_data(original_data, features_not_to_use):
    data = []
    label = []

    for i in range(len(original_data)):
        data.append([])
        for col in original_data.columns:
            if col in features_not_to_use:
                continue
            put = data[-1]
            value = original_data[col][i]
            if isinstance(value, str):
                value = codes[col][value]
                if col == "Attrition":
                    put = label
            put.append(value)

    return data, label


def split_data(data, label):
    indices = [i for i in range(len(data))]
    shuffle(indices)

    split_1 = (len(data) // 10) * 7
    split_2 = split_1 + (len(data) - split_1) // 2

    data = {
        "train": {"input": data[:split_1], "label": label[:split_1]},
        "valid": {"input": data[split_1 + 1:split_2], "label": label[split_1 + 1:split_2]},
        "test": {"input": data[split_2 + 1:], "label": label[split_2 + 1:]},
    }

    return data


def data_analyze(data):
    print("There are %d columns." % len(data.columns))
    types = {}

    for col in data.columns:
        unique = set()
        for d in data[col]:
            unique.add(d)
            if isinstance(d, str):
                types[col] = "str"
            else:
                types[col] = "int"

        if types[col] == "str":
            print(col)
            print("%d unique values in column %s which are:" % (len(unique), col))
            print(unique)
            print()