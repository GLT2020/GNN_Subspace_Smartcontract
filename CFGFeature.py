import numpy as np
from gensim.models import KeyedVectors,word2vec,Word2Vec
import torch
from sklearn.decomposition import PCA
from sklearn import datasets,manifold

from autoEncoder.AE_model import AE
from autoEncoder.CNNAE_model import CNNAE


'''
Input:
key: contract_name 
value : the value of compiled contract
cfg: Control Flow Graph of contract 
'''
class cfgFeature:
    def __init__(self,key,cfg_instructions,cfg_basic_blocks,pattern,label=0):
        self.name = key
        self.cfg_basic_blocks = cfg_basic_blocks
        self.label = int(label)
        self.pattern = np.array(pattern)
        # self.AE = AE(15000, 12000, 300)

        # set the AE latent size and load the AE parameter
        # self.AE = CNNAE(100)
        # self.AE.model.load_state_dict(torch.load('../autoEncoder/dict_CnnautoEncoder_30000_100_3cnn.pth'))

        self.AE = CNNAE(300)
        # self.AE.model.load_state_dict(torch.load('../autoEncoder/dict_CnnautoEncoder_30000_300_3cnn.pth'))
        self.AE.model.load_state_dict(torch.load('../autoEncoder/dict_CnnautoEncoder_30000_300_3cnn_2.pth'))

        # self.AE = CNNAE(500)
        # self.AE.model.load_state_dict(torch.load('../autoEncoder/dict_CnnautoEncoder_30000_500_3cnn.pth'))




        # self.AE.model.load_state_dict(torch.load('../autoEncoder/dict_autoEncoder_15000_100.pth'))
        self.hidden_size = self.AE.latent_size

        self.basicBlock = []
        self.basicBlock_pc = []
        self.allInstructions = cfg_instructions # 所有的opcode序列
        self.allinstructions_feature = []
        self.basicBlock_len = len(cfg_basic_blocks)
        self.block_feature = []
        self.edge_src = []
        self.edge_dst = []


        self.initBasicBlock()
        self.init_Degree_adjacency()
        self.opcode_vec()

        # TODO: use EncoderDecoder or PCA
        self.create_block_feature()  # use AE
        # self.create_block_feature_pca()





    def initBasicBlock(self):
        for basic_block in sorted(self.cfg_basic_blocks, key=lambda x: x.start.pc):
            # print(f"{basic_block} -> {sorted(basic_block.all_outgoing_basic_blocks, key=lambda x: x.start.pc)}")
            # print(f"{basic_block} <- {sorted(basic_block.all_incoming_basic_blocks, key=lambda x: x.start.pc)}")
            self.basicBlock.append(basic_block)
            self.basicBlock_pc.append(np.arange(basic_block.start.pc, basic_block.end.pc+1))



    def get_BlockIndex_pc(self, pc):
        for idx in range(self.basicBlock_len):
            if pc in self.basicBlock_pc[idx]:
                return idx
        assert ("pc does not in block!")


    def init_Degree_adjacency(self):
        for i in range(self.basicBlock_len):
            for j in range(len(self.basicBlock[i].all_outgoing_basic_blocks)):
                block_idx = self.get_BlockIndex_pc(self.basicBlock[i].all_outgoing_basic_blocks[j].start.pc)
                self.edge_src.append(i)
                self.edge_dst.append(block_idx)

            for j in range(len(self.basicBlock[i].all_incoming_basic_blocks)):
                block_idx = self.get_BlockIndex_pc(self.basicBlock[i].all_incoming_basic_blocks[j].start.pc)
                self.edge_src.append(block_idx)
                self.edge_dst.append(i)



    def create_block_feature_pca(self):
        pc_idx = 0
        pca1 = PCA(n_components=1)
        # 遍历block
        for i in range(self.basicBlock_len):
            temp_feature = []

            block_op_len = len(self.basicBlock[i].instructions)
            temp_op_feature = self.allinstructions_feature[pc_idx: pc_idx+block_op_len]
            pc_idx += block_op_len
            temp_feature.append(temp_op_feature)
            # （x,300）-> （1，300）
            temp_feature = np.array(temp_feature).reshape((-1,300))

            pca_feature = temp_feature.T
            pca_feature = pca1.fit_transform(pca_feature)
            pca_feature = pca_feature.T
            self.block_feature.append(pca_feature.reshape(1, -1))
        self.block_feature = np.array(self.block_feature).reshape((-1,300))

    def create_block_feature(self):
        pc_idx = 0
        # target_length = 15000
        target_length = 30000

        for i in range(self.basicBlock_len):
            temp_feature = []

            block_op_len = len(self.basicBlock[i].instructions)
            temp_op_feature = self.allinstructions_feature[pc_idx: pc_idx + block_op_len]
            pc_idx += block_op_len
            temp_feature.append(temp_op_feature)
            # （x,300）-> （1，300）
            np_feature = np.array(temp_feature).flatten()
            if len(np_feature) < target_length:
                padded_vector = np.pad(np_feature, (0, target_length - len(np_feature)), 'constant')
            else:
                padded_vector = np_feature[:target_length]
            AE_block_feature = self.AE.get_latent(torch.FloatTensor(padded_vector))
            self.block_feature.append(AE_block_feature.reshape(1, -1))
        self.block_feature = np.array(self.block_feature).reshape((-1, self.hidden_size))




    def opcode_vec(self):
        model = Word2Vec.load('word2vec_3.model')
        for i in range(len(self.allInstructions)):
            temp_op_feature = []
            temp = np.zeros((1,600))
            opcode_seq = str(self.allInstructions[i]).split(" ")
            for k in range(len(opcode_seq)):
                if temp_op_feature == []:
                    temp_op_feature = model.wv[opcode_seq[k]]
                else:
                    vec = model.wv[opcode_seq[k]]
                    temp_op_feature = np.concatenate([temp_op_feature, vec], axis=0)
            #(x*300)-> (300)
            temp_op_feature = np.interp(np.linspace(0, len(temp_op_feature) - 1, 300), np.arange(len(temp_op_feature)),
                                        temp_op_feature)
            self.allinstructions_feature.append(temp_op_feature.reshape(1,-1))
