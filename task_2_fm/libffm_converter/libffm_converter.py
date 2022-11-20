import argparse, sys
import csv
from utils import hashstr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', dest='bins', default=int(10e+6), type=int)
    parser.add_argument('csvfile')
    parser.add_argument('ffmfile')
    args = vars(parser.parse_args())

    return args


args = parse_args()

to_write = [
    'zone_id',
    'banner_id',
    'oaid_hash',
    'os_id',
    'country_id',
]
with open(args['ffmfile'], 'w') as f:
    for row in csv.DictReader(open(args['csvfile'])):
        # row is dict
        row_to_write = [row['clicks'], ]
        field = 0
        for feat in row.keys():
            if feat not in to_write:
                continue
            items = str(row[feat]).split(" ")
            for item in items:
                row_to_write.append(":".join([str(field), hashstr(str(field) + '=' + item, args['bins']), '1']))
            field += 1
        row_to_write = " ".join(row_to_write)
        f.write(row_to_write + '\n')
