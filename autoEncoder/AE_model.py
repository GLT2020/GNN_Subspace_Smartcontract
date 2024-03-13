import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import os
import torch.optim.lr_scheduler as lr_scheduler


use_cuda = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)
LR = 0.002

class EncoderDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(EncoderDecoder, self).__init__()
        self.Enlinear1 = nn.Linear(input_size, hidden_size)
        self.Enlinear2 = nn.Linear(hidden_size, 8000)
        self.Enlinear3 = nn.Linear(8000, 3000)
        self.Enlinear4 = nn.Linear(3000, 1000)
        self.Enlinear5 = nn.Linear(1000, 512)
        self.Enlinear6 = nn.Linear(512, latent_size)

        self.Delinear1 = torch.nn.Linear(latent_size, 512)
        self.Delinear2 = torch.nn.Linear(512, 1000)
        self.Delinear3 = torch.nn.Linear(1000, 3000)
        self.Delinear4 = torch.nn.Linear(3000, 8000)
        self.Delinear5 = torch.nn.Linear(8000, hidden_size)
        self.Delinear6 = torch.nn.Linear(hidden_size, input_size)

        # input_size = 18000  hidden_size = 20000时
        # self.Enlinear1 = nn.Linear(input_size, hidden_size)
        # self.Enlinear2 = nn.Linear(hidden_size, 12000)
        # self.Enlinear3 = nn.Linear(15000, 10000)
        # self.Enlinear4 = nn.Linear(10000, 5000)
        # self.Enlinear5 = nn.Linear(5000, 1000)
        # self.Enlinear6 = nn.Linear(1000, 512)
        # self.Enlinear7 = nn.Linear(512, latent_size)
        #
        # self.Delinear1 = torch.nn.Linear(latent_size, 512)
        # self.Delinear2 = torch.nn.Linear(512, 1000)
        # self.Delinear3 = torch.nn.Linear(1000, 5000)
        # self.Delinear4 = torch.nn.Linear(5000, 10000)
        # self.Delinear5 = torch.nn.Linear(10000, 15000)
        # self.Delinear6 = torch.nn.Linear(15000, hidden_size)
        # self.Delinear7 = torch.nn.Linear(hidden_size, input_size)

    def forward(self, x):# x: bs,input_size
        x = F.relu(self.Enlinear1(x)) #-> bs,hidden_size
        x = F.relu(self.Enlinear2(x)) #-> bs,hidden_size
        x = F.relu(self.Enlinear3(x)) #-> bs,hidden_size
        x = F.relu(self.Enlinear4(x)) #-> bs,hidden_size
        x = F.relu(self.Enlinear5(x)) #-> bs,hidden_size
        feat = self.Enlinear6(x) #-> bs,latent_size

        x = F.relu(self.Delinear1(feat))  # ->bs,hidden_size
        x = F.relu(self.Delinear2(x))  # ->bs,hidden_size
        x = F.relu(self.Delinear3(x))  # ->bs,hidden_size
        x = F.relu(self.Delinear4(x))  # ->bs,hidden_size
        x = F.relu(self.Delinear5(x))  # ->bs,hidden_size
        x = self.Delinear6(x)  # ->bs,output_size
        # x = torch.sigmoid(self.Delinear4(x))  # ->bs,output_size
        return feat, x


class AE:
    def __init__(self, input_size,hidden_size, latent_size):
        super(AE, self).__init__()
        self.latent_size = latent_size
        self.model = EncoderDecoder(input_size, hidden_size, latent_size).to(device)
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, (10,30), gamma=0.1)  # dynamic adjustment lr
        self.loss_fn =  nn.MSELoss(reduction = 'sum')

    def train(self,train_loader,epochs):
        for epoch in range(epochs):
            t = tqdm(train_loader, desc=f'[train]epoch:{epoch}')
            self.model.train()

            train_loss = 0
            train_nsample = 0
            loss_history = {'train': [], 'eval': []}
            for idx, data in enumerate(t):
                data = data.to(device)
                hidden, output = self.model(data)
                loss = self.loss_fn(output, data)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()  # 调整lr
                self.optimizer.zero_grad()  # 梯度置零
                # 计算平均损失，设置进度条
                train_loss += loss.item()
                train_nsample += data.shape[0]
                t.set_postfix({'loss': train_loss / train_nsample})

        torch.save(self.model, './autoEncoder_15000_300.pth')

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        test_nsample = 0
        e = tqdm(test_loader, desc='[eval]:')
        for idx, data in enumerate(e):
            data = data.to(device)
            hidden, output = self.model(data)
            loss = self.loss_fn(output, data)
            # 计算平均损失，设置进度条
            test_loss += loss.item()
            test_nsample += data.shape[0]
            e.set_postfix({'loss': test_loss / test_nsample})

    def get_latent(self,data):
        with torch.no_grad():
            self.model.eval()
            latent,_ = self.model(data.to(device))
            return np.array(latent.cpu())

    def get_test(self,data):
        with torch.no_grad():
            self.model.eval()
            _,out = self.model(data.to(device))
            return out

