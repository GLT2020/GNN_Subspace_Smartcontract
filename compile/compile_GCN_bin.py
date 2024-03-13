from solc import compile_files
import os
import pickle
import joblib
import json
import csv
from evm_cfg_builder.cfg import CFG
import numpy as np
from CFGFeature import cfgFeature



# TODO: Modify file address

# PER = "ren_D73"
PER = "ren_505_D73"

def read_pkl():
    with open(PER+"/test_gcn.pkl", "rb") as f:
        s = joblib.load(f)
    print(s)

def store_feature(data,istrain):


    if istrain == True:
        # with open(PER+"/train_gcn_lle20.pkl", "wb") as f:
        # with open(PER+"/train_gcn_cnn500.pkl", "wb") as f:
        with open(PER+"/train_gcn_cnn300_2.pkl", "wb") as f:
            pickle.dump(data, f)


    if istrain == False:
        # with open(PER+"/test_gcn_cnn500.pkl", "wb") as f:
        with open(PER+"/test_gcn_cnn300_2.pkl", "wb") as f:
            pickle.dump(data, f)
    # with open(PER+"/train_gcn.pkl", "rb") as f:
    #     s = joblib.load(f)
    # print(s)


def store_bytecode(data: dict):
    info_json = json.dumps(data)

    # with open("./compile/source_pattern_bin.json", "a+") as f:
    #     f.write(info_json + '\n')
    with open("source.json", "a+") as f:
        f.write(info_json + '\n')


def read_label_bin(istrain):

    if istrain == True:
        path = "./"+PER+"/train_label.csv"
    else:
        path = "./"+PER+"/test_label.csv"
    data = []
    data_dict = {}
    with open(path) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row)
    for i in range(len(data)):
        data_dict[data[i][0]] = data[i][1]
    return  data,data_dict


def read_source_bin():
    info_data = []
    data_dict = {}

    # with open("source.json", "r") as f:
    with open("source_505.json", "r") as f:
        for line in f:
            try:
                info_data.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))
    for i in range(len(info_data)):
        data_dict[info_data[i]["address"]] = [info_data[i]["address"],info_data[i]["fn_name"],
                                              info_data[i]["label"],info_data[i]["runtime_bytecode"],info_data[i]["pattern"]]
    return info_data,data_dict

def read_label():

    path = "ren_D55/all_label.csv"
    data = []
    data_dict = {}
    addr_dict = {}
    with open(path) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row)
    for i in range(len(data)):
        key = data[i][0]
        # id,address,name,label
        data_dict[key] = data[i][1]
    return  data,data_dict


def read_pattern():
    info_data = []
    data_dict = {}

    with open("reentrancy.json", "r") as f:
        for line in f:
            try:
                info_data.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))
    for i in range(len(info_data)):
        data_dict[info_data[i]["name"]] = info_data[i]["pattern"]
    return info_data,data_dict


def compile_bincode():

    path = "../source_file"
    cfg_feature_list = []
    list_dict = []

    label_list,label_dict = read_label()

    pattern_list,pattern_dict = read_pattern()
    all_class = []
    for root,dirs,files in os.walk(path):
        print("Current directory path:",root)
        print("All subdirectories under the current directory:",dirs)
        print("All non directory sub files in the current directory:",files)
        num = 0
        for i in range(len(files)):
            fname = files[i]
            print(root+files[i])

            if files[i] in label_dict.keys():
                file = compile_files([root+'/'+files[i]])
                print(file)
                max_bin_len = 0
                max_file_value = object
                max_file_key = ""
                for key,value in file.items():
                    print(file)
                    fn_name = key.split(':')[1]

                    if fn_name == "Log" or fn_name == "LogFile":
                        continue
                    if (len(value['bin-runtime'])) > max_bin_len:
                        max_file_value = value
                        max_file_key = key
                        max_bin_len = len(value['bin-runtime'])
                print(key)
                max_file_key = max_file_key.split(':')[1]
                runtime_bytecode = max_file_value['bin-runtime']
                num += 1
                store_dict = {"address": fname,"fn_name": max_file_key, "label": label_dict[fname] ,"runtime_bytecode": runtime_bytecode, "pattern":pattern_dict[fname]}
                store_bytecode(store_dict)
    print(num)


def create_feature(istrain:bool):

    label_list, label_dict = read_label_bin(istrain)

    source_list, source_dict = read_source_bin()
    all_class = []
    num = 0
    for key, value in source_dict.items():
        if key in label_dict.keys():
            cfg = CFG(value[3])
            store = cfgFeature(key=value[0], cfg_instructions=cfg.instructions, cfg_basic_blocks=cfg.basic_blocks, pattern=value[4], label=value[2])
            store_dict = {"key": store.name, "label": store.label, "pattern": store.pattern,
                          "block_feature": store.block_feature, "basicBlock_len": store.basicBlock_len,
                          "edge_src": store.edge_src, "edge_dst": store.edge_dst}
            all_class.append(store_dict)

            num += 1
    print(num)
    store_feature(all_class,istrain)


if __name__ == '__main__':
    # Need compile all source file to get the all bincodes. If the "source.json" file had exist, don't use that again.
    # compile_bincode()

    # use bincode and pattern to create feature. "True" means creating train dataset.
    create_feature(True)
    create_feature(False)
