from loadData import OpCodeData, PatternData, collate_opcode_batch, CSCOData, collate_csco_batch
from DGLloadData import DGLData, collate_dgl
from model.dgl_gnnpattern import AttGnn_model
from model.dgl_gnn import Gnn_model
from model.dgl_gcn import GCN_model
from model.dgl_gcnpattern import GCNPattern_model
from torch.utils.data import DataLoader
import torch
from parser1 import parameter_parser
import json
import joblib

args = parameter_parser()
PER = args.D

# TODO:use in gcn and gnn
input_dim = args.input_dim


def read_feature():
    # if args.model == "gnn" or args.model == "gnnpattern" or args.model == "gcn" or args.model == "gcnpattern":
    if "pca" in args.model:
        path_train = "./compile/ren_" + PER + "/train_gcn_pca.pkl"
        path_test = "./compile/ren_" + PER + "/test_gcn_pca.pkl"
    else:
        path_train = "./compile/ren_" + PER + "/train_gcn_cnn"+str(input_dim)+".pkl"
        path_test = "./compile/ren_" + PER + "/test_gcn_cnn"+str(input_dim)+".pkl"


    print(path_train)
    print(path_test)
    with open(path_test, "rb") as f:
        info_data_test = joblib.load(f)
    with open(path_train, "rb") as f:
        info_data_train = joblib.load(f)
    return info_data_train,info_data_test


def save_modle_result(data, name, model, input_dim):
    dic = {}
    key = ["tp", "fp", "tn", "fn", "accuracy", "recall", "precision", "F1", "FPR", "roc", "test_loss"]
    for index, value in enumerate(data):
        dic[key[index]] = value
    info_json = json.dumps(dic)
    if "pca" in args.model :
        path = "result/" + args.type + "/" + name + "/" + model + ".json"
    elif ("gnn" in args.model) or ("gcn" in args.model) or ("lle" in args.model):
        path = "result/" + args.type + "/" + name + "/" + model + input_dim + ".json"
    else:
        path = "result/"+ args.type + "/" + name +"/"+ model +".json"
    with open(path, "a+") as f:
        # pickle.dump(data, my_file)
        f.write(info_json + "\n")


def save_modle_loss_result(data, name, model, input_dim):
    dic = {}
    key = ["train_loss"]
    dic["train_loss"] = data
    info_json = json.dumps(dic)
    if "pca" in args.model:
        path = "result/" + args.type + "/" + name + "/" + model + "_loss.json"
    elif ("gnn" in args.model) or ("gcn" in args.model) or ("lle" in args.model):
        path = "result/" + args.type + "/" + name + "/" + model + input_dim + "_loss.json"
    else:
        path = "result/"+ args.type + "/" + name +"/"+ model +"_loss.json"
    with open(path, "a+") as f:
        # pickle.dump(data, my_file)
        f.write(info_json + "\n")


if __name__ == '__main__':
    # load data
    cfg_feature_list_train = []
    cfg_feature_list_test = []
    feature_list_train, feature_list_test = read_feature()
    num = 0



    # set fixed seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # rnd_state = np.random.RandomState(args.seed)

    # create dataloader
    # if args.model == "gnn" or args.model == "gnnpattern"or args.model == "gcn" or args.model == "gcnpattern":
    if ("gnn" in args.model) or ("gcn" in args.model):
        dataloader_train = DGLData(feature_list_train)
        dataloader_test = DGLData(feature_list_test)
        # DataLoder
        loader_data_train = DataLoader(dataset=dataloader_train, batch_size=10, shuffle=True, drop_last=True, collate_fn=collate_dgl)
        loader_data_test = DataLoader(dataset=dataloader_test, batch_size=10, collate_fn=collate_dgl)

    # definition model
    if args.model == 'gnnpattern':
        ClassifierModel = AttGnn_model(input_dim=input_dim, hidden_dim=64)
    elif args.model == 'gnn':
        ClassifierModel = Gnn_model(input_dim=input_dim, hidden_dim=64)
    elif args.model == 'gcn':
        ClassifierModel = GCN_model(input_dim=input_dim, hidden_dim=64)
    elif args.model == 'gcnpattern':
        ClassifierModel = GCNPattern_model(input_dim=input_dim, hidden_dim=64)

    elif args.model == 'gnnpattern_pca':
        ClassifierModel = AttGnn_model(input_dim=300, hidden_dim=64)
    elif args.model == 'gnn_pca':
        ClassifierModel = Gnn_model(input_dim=300, hidden_dim=64)
    elif args.model == 'gcn_pca':
        ClassifierModel = GCN_model(input_dim=300, hidden_dim=64)
    elif args.model == 'gcnpattern_pca':
        ClassifierModel = GCNPattern_model(input_dim=300, hidden_dim=64)



    # train
    loss_list = []
    if args.mode == 'train':
        for i in range(args.epochs):
            loss = ClassifierModel.train(loader_data_train, i)
            loss_list.append(loss)

    save_modle_loss_result(loss_list, PER, args.model, str(input_dim))

    # test
    if "pca" in args.model:
        read_pth = './model/pth/' + args.type + '/' + PER + "/" + args.model + str(300) + '.pth'
    elif ("gnn" in args.model) or ("gcn" in args.model) or ("lle" in args.model):
        read_pth = './model/pth/' + args.type + '/' + PER + "/" + args.model + str(input_dim) + '.pth'
        # read_pth = './model/pth/' + args.type + '/' + args.model + PER + str(input_dim) + '.pth'
    else:
        read_pth = './model/pth/' + args.type + '/' + PER + "/" + args.model + '.pth'
        # read_pth = './model/pth/'+ args.type +'/'+ args.model + PER +'.pth'
    print(read_pth)
    ClassifierModel.model = torch.load(read_pth)

    # ClassifierModel.test(loader_data_train, args.epochs - 1)
    result = ClassifierModel.test(loader_data_test, args.epochs - 1)
    save_modle_result(result, PER, args.model, str(input_dim))


