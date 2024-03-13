from parser1 import parameter_parser
from BlockDataLoder import BlockDataloder
from AE_model import AE
from CNNAE_model import CNNAE
import matplotlib.pyplot as plt
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np

args = parameter_parser()
PER = args.D

use_cuda = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)

def read_feature():
    path_train = "./ren_" + PER + "/block.pkl"
    # path_train = "./ren_" + PER + "/block_15000.pkl"
    with open(path_train, "rb") as f:
        info_data = joblib.load(f)
    return info_data

if __name__ == '__main__':
    data_list = read_feature()
    train_size = int(0.8 * len(data_list))
    test_size = len(data_list) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data_list, [train_size, test_size])
    train_loder = BlockDataloder(train_dataset)
    test_loder = BlockDataloder(test_dataset)

    if args.mode == 'train':
        loader_data_train = DataLoader(dataset=train_loder, batch_size=32)
        loader_data_test = DataLoader(dataset=test_loder, batch_size=32)
        # AutoEncoder = AE(15000, 12000, 100)
        AutoEncoder = CNNAE(300)
        AutoEncoder.train(loader_data_train,3000)
        AutoEncoder.test(loader_data_test)

    if args.mode == 'test':
        # AutoEncoder = AE(15000, 12000, 100)
        AutoEncoder = CNNAE(300)
        AutoEncoder.model = torch.load('CnnautoEncoder_30000_300_3cnn_2.pth')

        torch.save(AutoEncoder.model.state_dict(),'dict_CnnautoEncoder_30000_300_3cnn_2.pth')
        test = torch.FloatTensor(data_list[3]).to(device)
        encoder_feature = AutoEncoder.get_test(test)
        latent_feature = AutoEncoder.get_latent(test)
        encoder_feature = encoder_feature.detach().cpu().flatten().numpy()

        print(latent_feature)
        print(encoder_feature)
        print(data_list[3])

        mse =  mean_squared_error(test.detach().cpu().numpy(), encoder_feature)
        print(mse)