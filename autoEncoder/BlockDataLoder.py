import torch
import copy
import torch.utils
import numpy as np


class BlockDataloder(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index])

    def __len__(self):
        return len(self.data)

