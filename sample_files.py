from os import listdir
from os.path import isfile, join

import re, random
import csv

'''
Simple uility to sample from dataset
Genereate csv file for the C binary to load datasets

To-do: add selection criteria
'''

random.seed(1)

num_samples = 10

data_path="../data/schur_dataset"
dataset_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

seqeunce_set = set()
for f in dataset_files:
    match = re.search("([0-9]+_[0-9]+_[0-9]+)", f)
    if (match):
        seqeunce_set.add(match.group(1))

sampled_set = random.sample(seqeunce_set, num_samples)

# export CSV in these format /app/data/Hll.oct /app/data/Hpl.oct Hpp.oct B.oct Hschur.oct Bschur.oct
matrix_names = ['Hll.oct', 'Hpl.oct', 'Hpp.oct', 'B.out', 'Hschur.oct', 'BSchur.out']

with open('filelist_sample.csv', 'w', newline='') as csvfile:
    matrix_writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for sample_id in sampled_set:
        matrix_writer.writerow([sample_id + '_' + matrix_name for matrix_name in matrix_names])

