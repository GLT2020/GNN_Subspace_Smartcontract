import torch
import torch.utils
import numpy as np
import dgl
from dgl.data import DGLDataset

class DGLData(DGLDataset):

    def __init__(self, data_list):
        super().__init__(name='gnn')
        self.data_list = data_list
        grap_list = []
        label_list = []
        pattern1 = []
        pattern2 = []
        pattern3 = []
        for i in range(len(self.data_list)):
            node_features = torch.from_numpy(self.data_list[i]["block_feature"]).float()
            edges_src = self.data_list[i]["edge_src"]
            edges_dst = self.data_list[i]["edge_dst"]
            graph = dgl.graph((edges_src, edges_dst), num_nodes=self.data_list[i]["basicBlock_len"])
            graph.ndata['feat'] = node_features
            label_list.append(self.data_list[i]["label"])
            pattern1.append(self.data_list[i]["pattern"][0])
            pattern2.append(self.data_list[i]["pattern"][1])
            pattern3.append(self.data_list[i]["pattern"][2])

            graph= dgl.add_self_loop(graph)
            grap_list.append(graph)
        self.graph = grap_list
        self.label = label_list
        self.pattern1 = pattern1
        self.pattern2 = pattern2
        self.pattern3 = pattern3

    def __getitem__(self, idx):
        return  self.graph[idx], self.label[idx], self.pattern1[idx], self.pattern2[idx], self.pattern3[idx]


    def __len__(self):
        return len(self.data_list)


def collate_dgl(samples):

    graphs, labels, pattern1, pattern2, pattern3 = map(list, zip(*samples))


    batched_graph = dgl.batch(graphs)
    return [batched_graph, torch.LongTensor(labels) , torch.FloatTensor(pattern1), torch.FloatTensor(pattern2), torch.FloatTensor(pattern3)]


