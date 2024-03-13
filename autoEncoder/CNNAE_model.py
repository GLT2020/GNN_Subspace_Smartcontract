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
    def __init__(self, latent_size):
        super(EncoderDecoder, self).__init__()
        self.input_dim = latent_size

        self.relu = nn.LeakyReLU()
        # self.encoder1 = nn.Conv1d(self.input_dim, 256, 3, stride=2, padding=1)
        # self.encoder2 = nn.Conv1d(256, 128, 3, stride=2, padding=1)
        # self.encoder3 = nn.Conv1d(128, 64, 3, stride=2, padding=1)
        # self.encoder4 = nn.Conv1d(64, 32, 3, stride=2, padding=1)
        self.encoder1 = nn.Conv1d(self.input_dim, 256, 3, stride=2, padding=1)
        self.encoder2 = nn.Conv1d(256, 128, 3, stride=2, padding=1)
        self.encoder3 = nn.Conv1d(128, 64, 3, stride=2, padding=1)
        # self.encoder4 = nn.Conv1d(64, 32, 3, stride=2, padding=1)

        # when:laten_size =300
        if latent_size == 300:
            self.encoder_lin = nn.Sequential(
                nn.Flatten(),
                # nn.Linear(32 * 7, latent_size),
                nn.Linear(64 * 13, 600),
                nn.LeakyReLU(),
                nn.Linear(600, 400),
                nn.LeakyReLU(),
                nn.Linear(400, 300),
                nn.LeakyReLU(),
            )

            self.decoder_lin = nn.Sequential(
                # nn.Linear(latent_size, 32 * 7),
                nn.Linear(300, 400),
                nn.LeakyReLU(),
                nn.Linear(400, 600),
                nn.LeakyReLU(),
                nn.Linear(600, 64 * 13),
                nn.LeakyReLU(),
                nn.Unflatten(dim=1, unflattened_size=(64, 13))
            )

        elif latent_size == 100:
        # when:laten_size = 100
            self.encoder_lin = nn.Sequential(
                nn.Flatten(),
                # nn.Linear(32 * 7, latent_size),
                nn.Linear(64 * 13, 600),
                nn.ReLU(),
                nn.Linear(600, 300),
                nn.ReLU(),
                nn.Linear(300, latent_size),
                nn.ReLU(),
            )

            self.decoder_lin = nn.Sequential(
                # nn.Linear(latent_size, 32 * 7),
                nn.Linear(latent_size, 300),
                nn.ReLU(),
                nn.Linear(300, 600),
                nn.ReLU(),
                nn.Linear(600, 64 * 13),
                nn.ReLU(),
                nn.Unflatten(dim=1, unflattened_size=(64, 13))
            )
        elif latent_size == 500:
        #when:laten_size = 500
            self.encoder_lin = nn.Sequential(
                nn.Flatten(),
                # nn.Linear(32 * 7, latent_size),
                nn.Linear(64 * 13, 650),
                nn.ReLU(),
                nn.Linear(650, 500),
                nn.ReLU(),
            )

            self.decoder_lin = nn.Sequential(
                nn.Linear(500, 650),
                nn.ReLU(),
                nn.Linear(650, 64 * 13),
                nn.ReLU(),
                nn.Unflatten(dim=1, unflattened_size=(64, 13))
            )

        elif latent_size == 700:
            self.encoder_lin = nn.Sequential(
                nn.Flatten(),
                # nn.Linear(32 * 7, latent_size),
                nn.Linear(64 * 13, 700),
                nn.ReLU(),
                # nn.Linear(650, 800),
                # nn.ReLU(),
            )

            self.decoder_lin = nn.Sequential(
                # nn.Linear(500, 650),
                # nn.ReLU(),
                nn.Linear(700, 64 * 13),
                nn.ReLU(),
                nn.Unflatten(dim=1, unflattened_size=(64, 13))
            )

        # self.decoder1 = nn.ConvTranspose1d(32, 64, 3, stride=2, padding=1, output_padding=0)
        self.decoder2 = nn.ConvTranspose1d(64, 128, 3, stride=2, padding=1, output_padding=0)
        self.decoder3 = nn.ConvTranspose1d(128, 256, 3, stride=2, padding=1, output_padding=1)
        self.decoder4 = nn.ConvTranspose1d(256, self.input_dim, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.encoder1(x)
        x = self.relu(x)
        x = self.encoder2(x)
        x = self.relu(x)
        x = self.encoder3(x)
        x = self.relu(x)
        # x = self.encoder4(x)
        # x = self.relu(x)
        feat = self.encoder_lin(x)

        x = self.decoder_lin(feat)
        # x = self.decoder1(x)
        # x = self.relu(x)
        x = self.decoder2(x)
        x = self.relu(x)
        x = self.decoder3(x)
        x = self.relu(x)
        x = self.decoder4(x)
        x = self.relu(x)

        x = x.permute(0,2,1)
        return feat, x


class CNNAE:
    def __init__(self, latent_size):
        super(CNNAE, self).__init__()
        self.latent_size = latent_size
        self.model = EncoderDecoder( latent_size).to(device)
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
                data = data.reshape(data.shape[0], 100, 300)
                hidden, output = self.model(data)
                loss = self.loss_fn(output, data)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                train_loss += loss.item()
                train_nsample += data.shape[0]
                t.set_postfix({'loss': train_loss / train_nsample})

        torch.save(self.model, './CnnautoEncoder_30000_300_3cnn_2.pth')

    def test(self, test_loader):
        self.model.eval()

        test_loss = 0
        test_nsample = 0
        e = tqdm(test_loader, desc='[eval]:')
        for idx, data in enumerate(e):
            data = data.to(device)
            data = data.reshape(data.shape[0],100, 300)
            hidden, output = self.model(data)
            loss = self.loss_fn(output, data)
            # 计算平均损失，设置进度条
            test_loss += loss.item()
            test_nsample += data.shape[0]
            e.set_postfix({'loss': test_loss / test_nsample})

    def get_latent(self,data):
        with torch.no_grad():
            self.model.eval()
            data = data.reshape(1, 100, 300).to(device)
            latent,_ = self.model(data)
            return np.array(latent.cpu())

    def get_test(self,data):
        with torch.no_grad():
            self.model.eval()
            data = data.reshape(1, 100, 300).to(device)
            _,out = self.model(data)
            return out

