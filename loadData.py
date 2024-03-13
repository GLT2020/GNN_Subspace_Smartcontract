import torch
import torch.utils
import numpy as np


# 操作码序列数据
class OpCodeData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, index):
        return [torch.FloatTensor(np.array(self.data[index].allinstructions_feature).reshape(-1,300)), # 操作码序列
                int(self.data[index].label), # 标签
                ]

    def __len__(self):
        return len(self.data)

def collate_opcode_batch(batch):  # 这里的输入参数便是__getitem__的返回值
    # print(batch)
    B = len(batch)
    seq_list =[len(batch[b][0]) for b in range(B)]
    seq_list_tensor =torch.IntTensor([sl for sl in seq_list])
    C = batch[0][0].shape[1] # 自定义的特征维度
    seq_max = int(np.max(seq_list)) # 获取最大的序列长度

    x = torch.zeros(B, seq_max, C) # opcode序列矩阵
    for b in range(B):
        x[b, :seq_list[b]] = batch[b][0]

    labels = torch.from_numpy(np.array([batch[b][1] for b in range(B)])).long()

    return [x, seq_list_tensor ,labels]


# 专家模式数据
class PatternData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list


    def __getitem__(self, index):
        return [torch.FloatTensor(np.array(self.data[index].pattern[0])),
                torch.FloatTensor(np.array(self.data[index].pattern[1])),
                torch.FloatTensor(np.array(self.data[index].pattern[2])),
                int(self.data[index].label)  # 标签
                ]

    def __len__(self):
        return len(self.data)



class CSCOData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, index):
        return [torch.FloatTensor(np.array(self.data[index].allinstructions_feature).reshape(-1,300)), # 操作码序列
                torch.FloatTensor(np.array(self.data[index].pattern[0])),
                torch.FloatTensor(np.array(self.data[index].pattern[1])),
                torch.FloatTensor(np.array(self.data[index].pattern[2])),
                int(self.data[index].label), # 标签
                ]

    def __len__(self):
        return len(self.data)


def collate_csco_batch(batch):  # 这里的输入参数便是__getitem__的返回值
    # print(batch)
    B = len(batch)
    seq_list =[len(batch[b][0]) for b in range(B)]
    seq_list_tensor =torch.IntTensor([sl for sl in seq_list])
    C = batch[0][0].shape[1] # 自定义的特征维度
    seq_max = int(np.max(seq_list)) # 获取最大的序列长度

    x = torch.zeros(B, seq_max, C) # opcode序列矩阵
    for b in range(B):
        x[b, :seq_list[b]] = batch[b][0]

    labels = torch.from_numpy(np.array([batch[b][4] for b in range(B)])).long()

    pattern1 = torch.stack([batch[b][1] for b in range(B)])
    pattern2 = torch.stack([batch[b][2] for b in range(B)])
    pattern3 = torch.stack([batch[b][3] for b in range(B)])

    return [x, seq_list_tensor ,labels, pattern1, pattern2, pattern3]