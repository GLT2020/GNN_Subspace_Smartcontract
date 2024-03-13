import dgl
import dgl.function as fn
from dgl.utils import check_eq_shape
import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import GraphConvolution
from parser1 import parameter_parser
from torch.nn.parameter import Parameter
import torch.optim.lr_scheduler as lr_scheduler
import math
import time
import numpy as np
from sklearn import metrics
from dgl import DGLGraph
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle

args = parameter_parser()
use_cuda = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)

LR = 0.001

PER = args.D

from dgl.utils import expand_as_pair




class GNNClassifier(nn.Module):
    def __init__(self, gnn_input_size, gnn_hidden_size, num_head=3):
        super(GNNClassifier, self).__init__()
        self.conv1 = dgl.nn.GATv2Conv(gnn_input_size, gnn_hidden_size, num_heads=num_head)
        self.conv2 = dgl.nn.GATv2Conv(gnn_hidden_size*num_head, gnn_hidden_size, num_heads=num_head)
        self.conv3 = dgl.nn.GATv2Conv(gnn_hidden_size*num_head, gnn_hidden_size, num_heads=num_head)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()

        # output
        self.class1 = nn.Linear(gnn_hidden_size, 64)
        self.class2 = nn.Linear(64, 32)
        self.class3 = nn.Linear(32, 2)

    def forward(self, g, inputs):
        gnn_x = inputs
        # gnn_x = self.conv1(g, gnn_x).mean(1)
        gnn_x = self.conv1(g, gnn_x).flatten(1)

        gnn_x = self.conv2(g, gnn_x).flatten(1)

        gnn_x = self.conv3(g, gnn_x).mean(1)

        g.ndata['h'] = gnn_x

        gnn_output = dgl.mean_nodes(g, 'h')


        # classify
        x = self.class1(gnn_output)
        x = self.relu(x)
        x = self.class2(x)
        x = self.relu(x)
        x = self.class3(x)
        x = self.relu(x)

        return x



class Gnn_model():
    def __init__(self, input_dim, hidden_dim):
        super(Gnn_model, self).__init__()
        self.model = GNNClassifier(gnn_input_size=input_dim, gnn_hidden_size=hidden_dim).to(device)
        self.input_dim = str(args.input_dim)

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, args.lr_decay_steps, gamma=0.1)  # dynamic adjustment lr
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self,train_loader,epoch):
        self.model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        pre_loss = 0.9
        for batch_idx,data in enumerate(train_loader):

            batched_graph = data[0]
            labels = data[1]
            graph = batched_graph.to(device)
            feats = batched_graph.ndata['feat'].to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()
            output = self.model(graph, feats)
            loss = self.loss_fn(output, labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)


        torch.save(self.model, './model/pth/' + args.type + '/' + PER + "/" + args.model + self.input_dim + '.pth')
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f})  sec/iter: {:.4f}'.format(
            epoch + 1, n_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1)))
        return  train_loss / n_samples

    def test(self,test_loader,epoch):
        self.model.eval()
        start = time.time()
        test_loss, n_samples, count = 0, 0, 0
        tn, fp, fn, tp = 0, 0, 0, 0  # calculate recall, precision, F1 score
        accuracy, recall, precision, F1 = 0.0, 0.0, 0.0, 0.0
        fn_list = []  # Store the contract id corresponding to the fn
        fp_list = []  # Store the contract id corresponding to the fp

        pro_list = []
        y_label = []

        for batch_idx, data in enumerate(test_loader):
            # for i in range(len(batched_graph)):
            #     graph = batched_graph[i].to(device)
            #     feats = batched_graph[i].ndata['feat'].to(device)

            batched_graph = data[0]
            labels = data[1]
            graph = batched_graph.to(device)
            feats = batched_graph.ndata['feat'].to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()

            output = self.model(graph, feats)
            loss = self.loss_fn(output, labels)
            test_loss += loss.item()
            n_samples += len(output)
            count += 1
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            pro = nn.functional.softmax(output.detach()).cpu()
            pro_list.extend(pro[:, 1])
            y_label.extend(data[1].detach().cpu()[:])

            for k in range(len(pred)):
                if (np.array(pred.view_as(labels)[k]).tolist() == 1) & (
                        np.array(labels.detach().cpu()[k]).tolist() == 1):
                    # TP predict == 1 & label == 1
                    tp += 1
                    continue
                elif (np.array(pred.view_as(labels)[k]).tolist() == 0) & (
                        np.array(labels.detach().cpu()[k]).tolist() == 0):
                    # TN predict == 0 & label == 0
                    tn += 1
                    continue
                elif (np.array(pred.view_as(labels)[k]).tolist() == 0) & (
                        np.array(labels.detach().cpu()[k]).tolist() == 1):
                    # FN predict == 0 & label == 1
                    fn += 1
                    continue
                elif (np.array(pred.view_as(labels)[k]).tolist() == 1) & (
                        np.array(labels.detach().cpu()[k]).tolist() == 0):
                    # FP predict == 1 & label == 0
                    fp += 1
                    continue

        print(tp, fp, tn, fn)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) != 0 else 0

        print("Test:",epoch + 1)
        print(
            'Test set (epoch {}): Average loss: {:.4f}, Accuracy: ({:.4f}%), Recall: ({:.4f}%), Precision: ({:.4f}%), '
            'F1-Score: ({:.4f}%), FPR: ({:.4f}%)  sec/iter: {:.4f}\n'.format(
                epoch + 1, test_loss / n_samples, accuracy, recall, precision, F1, FPR,
                (time.time() - start) / len(test_loader))
        )


        pro_list = np.array(pro_list)
        y_label = np.array(y_label)
        fpr_1, tpr_1, threshold_1 = roc_curve(y_label, pro_list)
        roc_auc_1 = auc(fpr_1, tpr_1)
        print("auc:", roc_auc_1)

        sava_auc = {"fpr": fpr_1, "tpr": tpr_1, "roc_auc": roc_auc_1, "accuracy": (tp + tn) / (tp + tn + fp + fn)}
        path_auc = "./result/auc_record/" + args.type + "/" + PER + "/"+args.model+ self.input_dim + "_aucroc.txt"
        f = open(path_auc, 'wb')
        pickle.dump(sava_auc, f)
        f.close()

        plt.figure(figsize=(8, 7))
        parameters = {"axes.labelsize": 26, "legend.fontsize": 26, "xtick.labelsize": 20, "ytick.labelsize": 20}
        plt.rcParams.update(parameters)
        plt.plot(fpr_1, tpr_1, color='darkorange',
                 lw=2, label=args.model, linestyle='-')

        plt.text(0.75, 0.4, 'AUC = %0.3f' % roc_auc_1, fontdict={'size': 28, 'color': 'blue'}, ha='center', va='center')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.02, 1.05])
        plt.ylim([-0.02, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        return [tp, fp, tn, fn, accuracy, recall, precision, F1, FPR, roc_auc_1, test_loss / n_samples]

    def get_hidden(self,dataset):
        self.model.eval()
        hidden_all = []
        label_all = []
        with torch.no_grad():
            for batch_idx, (batched_graph, labels)  in enumerate(dataset):
                graph = batched_graph.to(device)
                feats = batched_graph.ndata['feat'].to(device)
                output, output_hidden = self.model(graph, feats)
                hidden_all.append(output_hidden.cpu())
                label_all.append(labels.cpu())
        hidden = torch.cat(hidden_all, 0)
        labels = torch.cat(label_all, 0)
        torch.cuda.empty_cache()
        return [hidden, labels]

