
import os
import pickle
import joblib
import json
import csv
from evm_cfg_builder.cfg import CFG
import numpy as np
from BlockFeature import blockFeatureclass




PER = "ren_D73"

def read_pkl():
    with open(PER+"/block_np.pkl", "rb") as f:
        s = joblib.load(f)
    print(s)

def storeFeature(data):


    with open(PER+"/block_15000.pkl", "wb") as f:
        pickle.dump(data, f)
    # with open(PER+"/block.pkl", "rb") as f:
    #     s = joblib.load(f)
    # print(s)



def read_source_bin():
    info_data = []
    data_dict = {}
    # TODO： 需要更改对应的pattern路径
    with open("../compile/source.json", "r") as f:
        for line in f:
            try:
                info_data.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))
    for i in range(len(info_data)):
        data_dict[info_data[i]["address"]] = [info_data[i]["address"],info_data[i]["fn_name"],
                                              info_data[i]["label"],info_data[i]["runtime_bytecode"],info_data[i]["pattern"]]
    return info_data,data_dict


def create_feature():

    source_list, source_dict = read_source_bin()
    all_block_feature = []
    num = 0
    for key, value in source_dict.items():
        cfg = CFG(value[3])
        blockfeature = blockFeatureclass(key=value[0], cfg_instructions=cfg.instructions, cfg_basic_blocks=cfg.basic_blocks)
        np_block_feature = np.array(blockfeature.block_feature).reshape(-1,15000)
        all_block_feature.append(np_block_feature)

        num += 1
    print(num)
    np_store_feture = np.vstack(all_block_feature)
    # np_store_feture = np_store_feture.reshape(-1,30000)
    storeFeature(np_store_feture)


if __name__ == '__main__':
    # compile_bincode()
    create_feature()
