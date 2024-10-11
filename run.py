import os
import torch
import numpy as np
import argparse
import optuna
from exp.model_train_val import Train_Val_Test
from exp.model_test import Test
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set random seed
random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)

parser = argparse.ArgumentParser()

# Data Loader
parser.add_argument('--With_augmentation', type=bool, default=False,
                    help='Decide using augmentation dataset or not')
# (1) original
parser.add_argument('--un_aug_dataset_path', type=str, default='data_provider\\dataset\\Example_WTIL\\11classes_imp',
                    help='path of the original data file')
parser.add_argument('--y_train_path', type=str, default='y_train.npy', help='original train set label')
parser.add_argument('--y_test_path', type=str, default='y_test.npy', help='original test set label')
parser.add_argument('--X_train_series_path', type=str, default='X_train_series.npy', help='original series train set')
parser.add_argument('--X_test_series_path', type=str, default='X_test_series.npy', help='original series test set label')
parser.add_argument('--X_train_graph_path', type=str, default='X_train_imp_graph_stft.npy', help='original graph train set')
parser.add_argument('--X_test_graph_path', type=str, default='X_test_imp_graph_stft.npy', help='original graph test set label')
# (2) augmented
parser.add_argument('--aug_dataset_path', type=str, default='data_provider\\dataset\\Example_WTIL\\11classes_imp_aug',
                    help='path of the augmented data file')
parser.add_argument('--y_aug_train_path', type=str, default='y_train.npy', help='augmented train set label')
parser.add_argument('--X_aug_train_series_path', type=str, default='X_train_series.npy', help='augmented series train set')
parser.add_argument('--X_aug_train_graph_path', type=str, default='X_train_augimp_graph_stft_7.npy', help='augmented graph train set')

# Optimization + Model_Set
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--learning_rate', type=float, default=4e-4, help='input learning rate for training (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=15, help='input epochs for training (default: 100)')
parser.add_argument('--mode', type=str, default='ts', help='training model type, options: [multimodal, ts, thu_ts, graph],'
                                                                   'Caution: TimesNet and NonStationary_Transformer should align'
                                                                   'the mode "thu_ts", and other time_series models with mode "ts"')
parser.add_argument('--model', type=str, default='InceptionTime', help='model name, options: [SSFN, TimesNet, Nonstationary_Transformer, '
                                                              'InceptionTime, HydraMultiRocket, MINIROCKET, MultiRocket, MLSTM-FCN, '
                                                              'TST, ResNet2D, WideResNet, MobileNet]')
# Basic Define
parser.add_argument('--enc_in', type=int, default=9, help='encode channel number (default: 9)')
parser.add_argument('--seq_len', type=int, default=200, help='input length (default: 200)')
parser.add_argument('--num_class', type=int, default=11, help='classes total number (default: 11)')
parser.add_argument('--graph_size', type=list, default=[32, 35], help='the width and height of input image (default: [32, 35])')
parser.add_argument('--dropout', type=float, default=0, help='dropout')
# Record
parser.add_argument('--tensorboard_logs_addr', type=str, default="./results/logs", help='checkpoints file')
parser.add_argument('--model_saved_addr', type=str, default="./results/best_model.pth", help='saved model')
parser.add_argument('--test_result_addr', type=str, default="./results/test", help='test result')

# SSFN Model Define
parser.add_argument('--init_channels', type=list, default=[9, 9], help='output channel after first convolution')
parser.add_argument('--graph_kq_dim', type=list, default=[4, 4], help='Spectrogram-block attention query/key dimension list')
parser.add_argument('--graph_attention_head', type=list, default=[2, 2], help='Spectrogram-block attention head')
parser.add_argument('--fusion_kq_dim', type=list, default=[1, 2, 2], help='Fusion-connection attention query/key dimension list, the 1st is static')
parser.add_argument('--fusion_attention_head', type=list, default=[1, 1, 1], help='Fusion-connection attention head, the 1st is static')
parser.add_argument('--growth_rate', type=int, default=4, help='multiple of dimension ascending (default: 4)')
parser.add_argument('--sequence_dilation', type=int, default=1, help='convolution dilation (default: 1)')
parser.add_argument('--blocks', type=list, default=[2], help='block number (default: 2)')

# Other Model Required Hyper-parameter
# __thu_ml__ models from https://github.com/thuml/Time-Series-Library
parser.add_argument('--task_name', type=str, default='classification', help='task type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
parser.add_argument('--label_len', type=int, default=1, help='start token length')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--d_model', type=int, default=108, help='dimension of model')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--num_kernels', type=int, default=5, help='for Inception')
parser.add_argument('--top_k', type=int, default=4, help='for TimesBlock')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
# __tsai__ models from https://github.com/timeseriesAI/tsai
parser.add_argument('--hydra_d', type=tuple, default=None, help='params in HydraMultiROCKET, check in source .py file')
parser.add_argument('--hydra_k', type=int, default=12)
parser.add_argument('--hydra_g', type=int, default=64)
parser.add_argument('--hydra_max_c_in', type=int, default=8)
parser.add_argument('--hydra_clip', type=bool, default=True)
parser.add_argument('--hydra_num_features', type=int, default=10_000)
parser.add_argument('--hydra_max_dilations_per_kernel', type=int, default=36)
parser.add_argument('--hydra_kernel_size', type=int, default=9)
parser.add_argument('--hydra_max_num_channels', type=int, default=None)
parser.add_argument('--hydra_max_num_kernels', type=int, default=84)
parser.add_argument('--hydra_use_bn', type=bool, default=True)
parser.add_argument('--hydra_fc_dropout', type=int, default=0.1)
parser.add_argument('--Mini_num_features', type=int, default=5000, help='for more params in MINIROCKET_pytorch')
parser.add_argument('--Mini_max_dilations_per_kernel', type=int, default=32)
parser.add_argument('--Multi_num_features', type=int, default=8000, help='for more params in MultiPOCKETPlus')
parser.add_argument('--Multi_max_dilations_per_kernel', type=int, default=24)
parser.add_argument('--Multi_max_num_channels', type=int, default=6)
parser.add_argument('--Multi_kernel_size', type=int, default=5)
parser.add_argument('--Multi_max_num_kernels', type=int, default=84)
parser.add_argument('--MLSTM_hidden_size', type=int, default=20, help='for more params in RNN_FCN')
parser.add_argument('--MLSTM_rnn_layers', type=int, default=2)
parser.add_argument('--MLSTM_bias', type=bool, default=True)
parser.add_argument('--MLSTM_cell_dropout', type=float, default=0)
parser.add_argument('--MLSTM_rnn_dropout', type=float, default=0.8)
parser.add_argument('--MLSTM_bidirectional', type=bool, default=False)
parser.add_argument('--MLSTM_shuffle', type=bool, default=True)
parser.add_argument('--MLSTM_fc_dropout', type=float, default=0.)
parser.add_argument('--MLSTM_conv_layers', type=list, default=[128, 256, 128])
parser.add_argument('--MLSTM_kss', type=list, default=[9, 7, 5])
parser.add_argument('--TST_max_seq_len', type=int, default=None, help='for more params in TST')
parser.add_argument('--TST_n_layers', type=int, default=2)
parser.add_argument('--TST_d_model', type=int, default=256)
parser.add_argument('--TST_n_heads', type=int, default=2)
parser.add_argument('--TST_d_k', type=int, default=18)
parser.add_argument('--TST_d_v', type=int, default=18)
parser.add_argument('--TST_d_ff', type=int, default=256)

args = parser.parse_args()

if __name__ == "__main__":

    train_val_exp = Train_Val_Test(args)
    train_val_exp.train()
    test_exp = Test(args)
    test_exp.inference()
