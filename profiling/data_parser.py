import re
# import pandas as pd

def build_dict_from_line(line):
    decomposed = line.split()
    
    data_dict = {}
    while len(decomposed) > 0:
        curr = decomposed.pop(0)
        if (len(curr)>1 and curr.endswith('=')):
            try:
                data_dict[curr[:-1]] = float(decomposed.pop(0))
            except: # handle misformed data
                return {}

    return data_dict


def generate_pd_from_src(perf_data, data_file_path):
    data_file = open(data_file_path, 'r')
    lines = data_file.readlines()
    last_dataset_name = ''
    for line in lines:
        if line.startswith("[DATASET]"):
            decomposed = line.split(" ")
            last_dataset_name = decomposed[-1].strip()
            # print(dataset_name)
        if line.startswith("[STATS]"):
            decomposed = line.split(" ")
            # dataset_id, number_of_threads, schur_time
            # print(last_dataset_name, int(decomposed[2]), float(decomposed[4]))

            if last_dataset_name not in perf_data:
                perf_data[last_dataset_name] = {}
            
            curr_dict = perf_data[last_dataset_name]
            curr_dict[int(decomposed[2])] = float(decomposed[4])


    # print(global_stats)
    # global_stats_pd = pd.DataFrame(global_stats)

    # return global_stats_pd
    return None

perf_dict = {} # dict of dicts
data_file_path='strongscaling.txt'
generate_pd_from_src(perf_dict, data_file_path)

print(perf_dict)