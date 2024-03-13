import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart contract vulnerability detection')
    parser.add_argument('-M', '--model', type=str, default='gnnpattern_pca',
                        choices=['gcn','gnn','gnnpattern','gcnpattern',
                                 'gcn_pca','gnn_pca','gnnpattern_pca', 'gcnpattern_pca',
                                 'pattern'],help="Select the model to use. gcn, gnn are processed using CNN_AE by default.")
    parser.add_argument('--mode', dest='mode', type=str, default='test')
    parser.add_argument('--type', dest='type', type=str, default='ren')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--input_dim',dest='input_dim',type=int,default=300,help='Control graph node feature dimension, pca only 300, gcn has 100, 300, 500 optional')
    parser.add_argument('--D',dest='D',type=str,help='Selecting a dataset')

    parser.add_argument('--lr_decay_steps', type=str, default='10,30', help='learning rate')

    return parser.parse_args()