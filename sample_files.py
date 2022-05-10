from os import listdir
from os.path import isfile, join

import re, random
import csv
import argparse


'''
Simple uility to sample from dataset
Genereate csv file for the C binary to load datasets

To-do: add selection criteria
'''

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('-a', action='store_true') # sample all dataset
parser.add_argument('-n', type=int, default=1, help='number of samples')
args = parser.parse_args()
sample_all = args.a

# number of samples drawn for the csv, however if sample_all is true, we ignore this
num_samples = args.n

data_path="../data/schur_dataset/"
csv_name="filelist_sample.csv"
dataset_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

seqeunce_set = set()
for f in dataset_files:
    match = re.search("([0-9]+_[0-9]+_[0-9]+)", f)
    if (match):
        seqeunce_set.add(match.group(1))

if sample_all:
    sampled_set = seqeunce_set
    print("Sample all available data into ", data_path + csv_name)
else:
    sampled_set = random.sample(seqeunce_set, num_samples)
    print("Sample", num_samples, "samples into ", data_path + csv_name)

# export CSV in these format /app/data/Hll.oct /app/data/Hpl.oct Hpp.oct B.oct Hschur.oct Bschur.oct
matrix_names = [['Hll', '.oct'], ['Hpl', '.oct'], ['Hpp', '.oct'], ['b', '.out'], ['Hschur', '.oct'], ['bschur', '.out']]


with open(csv_name, 'w', newline='') as csvfile:
    matrix_writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for sample_id in sampled_set:
        matrix_writer.writerow([data_path + matrix_name[0] + '_' + sample_id +  matrix_name[1] for matrix_name in matrix_names])
