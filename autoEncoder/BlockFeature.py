import numpy as np
from gensim.models import KeyedVectors,word2vec,Word2Vec
from sklearn.decomposition import PCA


class blockFeatureclass:
    def __init__(self,key,cfg_instructions,cfg_basic_blocks):
        self.name = key
        self.cfg_basic_blocks = cfg_basic_blocks

        self.basicBlock = []
        self.basicBlock_pc = []
        self.allInstructions = cfg_instructions # 所有的opcode序列
        self.allinstructions_feature = []
        self.basicBlock_len = len(cfg_basic_blocks)
        self.block_feature = []



        self.initBasicBlock()
        # TODO:修改是否不转化address操作码，_address（）表示不转化
        self.opcode_vec()
        # self.opcode_vec_address()

        self.create_block_feature()




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



    def opcode_vec(self):
        model = Word2Vec.load('../compile/word2vec_2.model')
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
            # (x*300)-> (300)
            temp_op_feature = np.interp(np.linspace(0, len(temp_op_feature) - 1, 300), np.arange(len(temp_op_feature)),
                                        temp_op_feature)

            self.allinstructions_feature.append(temp_op_feature.reshape(1,-1))




    def create_block_feature(self):
        pc_idx = 0
        target_length = 15000
        # 遍历block
        for i in range(self.basicBlock_len):
            temp_feature = []

            block_op_len = len(self.basicBlock[i].instructions)
            temp_op_feature = self.allinstructions_feature[pc_idx: pc_idx+block_op_len]
            pc_idx += block_op_len
            temp_feature.append(temp_op_feature)
            # （x,300）-> （1，300）
            np_feature = np.array(temp_feature).flatten()
            if len(np_feature) < target_length:

                padded_vector = np.pad(np_feature, (0, target_length - len(np_feature)), 'constant')
            else:

                padded_vector = np_feature[:target_length]


            self.block_feature.append(padded_vector)