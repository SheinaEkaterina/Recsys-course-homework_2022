import csv

path = '/opt/downloads/ml/recsys/data.csv'

train = '/opt/downloads/ml/recsys/data_train.csv'
val = '/opt/downloads/ml/recsys/data_val.csv'

with open(path, 'r') as d, open(train, 'w') as tr, open(val, 'w') as va:
    i = 0
    for row in d:
        if i == 0:
            tr.write(row + '\n')
            va.write(row + '\n')
            i += 1
        if row[:10] == "2021-10-02":
            va.write(row + '\n')
        else:
            tr.write(row + '\n')
