import shutil
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from data_provider.DataLoader import MyDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .regularization import Regularization
from .initialization import initialize_model
from models.time_series import Nonstationary_Transformer, TimesNet
from models.time_series.all import *
from models.multimodal.Sequence_Spectrogram_Fusion_Network import SSFN
from models.spectrogram import ResNet2D, WideResNet, MobileNet
from torchinfo import summary


random_seed = 2024
np.random.seed(random_seed)
torch.manual_seed(random_seed)

class Train_Val_Test():
    def __init__(self, configs):
        super(Train_Val_Test, self).__init__()
        self.configs = configs

    def get_data(self):
        configs = self.configs
        y_test = np.load(os.path.join(configs.un_aug_dataset_path, configs.y_test_path))
        X_test_graph = np.load(os.path.join(configs.un_aug_dataset_path, configs.X_test_graph_path))
        X_test_series = np.load(os.path.join(configs.un_aug_dataset_path, configs.X_test_series_path))
        if configs.With_augmentation:
            y_train = np.load(os.path.join(configs.aug_dataset_path, configs.y_aug_train_path))
            X_train_graph = np.load(os.path.join(configs.aug_dataset_path, configs.X_aug_train_graph_path))
            X_train_series = np.load(os.path.join(configs.aug_dataset_path, configs.X_aug_train_series_path))
            print('-------------Augmentation Mode-------------')
        else:
            y_train = np.load(os.path.join(configs.un_aug_dataset_path, configs.y_train_path))
            X_train_graph = np.load(os.path.join(configs.un_aug_dataset_path, configs.X_train_graph_path))
            X_train_series = np.load(os.path.join(configs.un_aug_dataset_path, configs.X_train_series_path))
            print('-------------Original Mode-------------')
        if configs.mode == 'thu_ts':
            X_train_series = X_train_series.swapaxes(1, 2)
            X_test_series = X_test_series.swapaxes(1, 2)
        train_set = MyDataset(X_train_series, X_train_graph, y_train)
        val_set = MyDataset(X_test_series, X_test_graph, y_test)

        trainLoader = DataLoader(train_set, batch_size=configs.batch_size, shuffle=True)
        validLoader = DataLoader(val_set, batch_size=configs.batch_size, shuffle=True)
        return train_set, val_set, trainLoader, validLoader

    def model_set(self):
        configs = self.configs
        model_dict = {
            'SSFN': SSFN,
            'TimesNet': TimesNet.Model,
            'HydraMultiRocket': HydraMultiRocketPlus,
            'InceptionTime': InceptionTime,
            'MINIROCKET': MiniRocket,
            'MultiRocket': MultiRocketPlus,
            'Nonstationary_Transformer': Nonstationary_Transformer.Model,
            'MLSTM-FCN': MLSTM_FCN,
            'TST': TST,
            'ResNet2D': ResNet2D.ResNet2D,
            'WideResNet': WideResNet.WideResNet,
            'MobileNet': MobileNet.MobileNetV2,
        }
        net = model_dict[configs.model](configs)
        return net

    def train(self):
        configs = self.configs
        train_set, val_set, trainLoader, validLoader = self.get_data()
        print(f'-------------{configs.mode} mode-------------')
        for batch_X1, batch_X2, batch_y in trainLoader:
            print(batch_X1.shape, batch_X2.shape, batch_y.shape, batch_y[0])
        learning_rate, epochs, mode = configs.learning_rate, configs.epochs, configs.mode
        tensorboard_logs_addr = configs.tensorboard_logs_addr
        shutil.rmtree(tensorboard_logs_addr)
        if tensorboard_logs_addr is not None:
            os.makedirs(tensorboard_logs_addr, exist_ok=True)
        writer = SummaryWriter(tensorboard_logs_addr)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # net = initialize_model(net, initialization='xavier')
        net = self.model_set()
        net.to(device)
        # reg_loss = Regularization(net, 0, p=2).to(device)
        cur_epoch = 1
        # Seperately set learning rate for each stream
        # params_path1 = list(net.conv1.parameters()) + list(net.layer1.layers1.parameters()) + list(net.fc1.parameters()) # Specrtrogram
        # params_path2 = list(net.conv2.parameters()) + list(net.layer1.layers2.parameters()) + list(net.fc.parameters()) #  Sequence
        # params_path3 = list(net.layer1.transfer.parameters())
        # learning_rate_path1 = 1e-5
        # learning_rate_path2 = 1e-3
        # learning_rate_path3 = 0

        # Create different optimizers
        # optimizer_path1 = optim.Adam(params_path1, lr=learning_rate_path1)
        # optimizer_path2 = optim.Adam(params_path2, lr=learning_rate_path2)
        # optimizer_path3 = optim.Adam(params_path3, lr=learning_rate_path3)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, mode='min', verbose=True, min_lr= 0.0000001, patience=5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # dynamic learning rate adjustment

        m = nn.Sigmoid()
        # lossfunction = nn.BCELoss()
        # lossfunction = nn.L1Loss()
        # lossfunction = nn.BCEWithLogitsLoss()
        # lossfunction = nn.MultiLabelSoftMarginLoss()
        lossfunction = nn.CrossEntropyLoss()
        lossfunction = lossfunction.to(device)
        loss_val = []
        loss_train = []
        acc_val = []
        acc_train = []
        lowest_val = 100
        best_val_acc = 0.0
        record = 0
        patched_val_acc = 0.0
        for epoch in range(cur_epoch, epochs + 1):
            ###################
            # train the model #
            ###################
            net.train()
            right = 0
            total = 0
            train_loss = 0.0
            t = tqdm(trainLoader, leave=False, total=len(trainLoader))
            for i, (series, graph, targetVar) in enumerate(t):
                series = series.to(device)  # B,S,C,H,W
                graph = graph.to(device)
                label = targetVar.to(device)
                optimizer.zero_grad()
                # optimizer_path1.zero_grad()
                # optimizer_path2.zero_grad()
                net.train()
                if mode == 'multimodal':
                    pred, _ = net(graph, series)  # B,S,C,H,W pred,_
                elif mode == 'ts':
                    pred = net(series)
                elif mode == 'thu_ts':
                    pred = net(series, 1, None, None)
                else:
                    pred = net(graph)
                loss = lossfunction(pred, label)  # +lossfunction(_, label)
                # loss = loss + reg_loss(net) #  regularization
                train_loss = train_loss+loss.item() * series.size(0)
                loss.backward()
                optimizer.step()
                # optimizer_path1.step()
                # optimizer_path2.step()
                t.set_postfix({
                    'trainloss': '{:.6f}'.format(loss.item()),
                    'epoch': '{:02d}'.format(epoch)
                })
                total += label.size(0)
                _, predicted = torch.max(pred, dim=1)
                right += (predicted == label).sum()
            loss_train.append(train_loss / len(train_set))  # len(train_dataloader)
            train_acc = right / total
            acc_train.append(right / total)
            ######################
            # validate the model #
            ######################
            with torch.no_grad():
                net.eval()
                right = 0
                total = 0
                validation_loss = 0.0
                t = tqdm(validLoader, leave=False, total=len(validLoader))
                for i, (series, graph, targetVar) in enumerate(t):
                    # inputVar = torch.unsqueeze(inputVar, dim=-1)
                    # inputVar = torch.permute(inputVar, (0,4,1,2,3))
                    series = series.to(device)
                    graph = graph.to(device)
                    label = targetVar.to(device)
                    if mode == 'multimodal':
                        pred, _ = net(graph, series)  # B,S,C,H,W pred,_
                    elif mode == 'ts':
                        pred = net(series)
                    elif mode == 'thu_ts':
                        pred = net(series, 1, None, None)
                    else:
                        pred = net(graph)
                    loss1 = lossfunction(pred, label)
                    validation_loss = validation_loss + loss1.item() * series.size(0)

                    t.set_postfix({
                        'validloss': '{:.6f}'.format(loss1.item()),
                        'epoch': '{:02d}'.format(epoch)
                    })

                    _, predicted = torch.max(pred, dim=1)
                    right += (predicted == label).sum()
                    total += label.size(0)

            torch.cuda.empty_cache()
            # scheduler.step(validation_loss)
            # scheduler.step()
            loss_val.append(validation_loss / len(val_set))  # len(val_dataloader)
            val_acc = right / total
            acc_val.append(right / total)
            # avg_train_losses.append(train_loss)
            # avg_valid_losses.append(valid_loss)
            if validation_loss/len(val_set) <= lowest_val:
                lowest_val = validation_loss/len(val_set)
                if val_acc >= best_val_acc:
                    patched_val_acc = val_acc
                record = epoch
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                # if validation_loss <= lowest_val:
                # lowest_val = validation_loss
                torch.save(net.state_dict(), configs.model_saved_addr)
            epoch_len = len(str(epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' + f'lr: {optimizer.param_groups[0]["lr"]} '
                         + f'train_accuracy: {train_acc:.6f} ' + f'valid_accuracy: {val_acc:.6f} '
                         + f'train_loss: {train_loss / len(train_set):.6f} ' + f'valid_loss: {validation_loss / len(val_set):.6f} '
                         + f'best_val_acc: {best_val_acc:.6f}' + f'lowest_val/epoch: {lowest_val:.6f} {record: d}'
                         + f' patched_val_acc: {patched_val_acc:.6f} ')
            print(print_msg)

            writer.add_scalars('accuracy',
                               {'train_acc': float(train_acc),
                                'val_acc': float(val_acc),
                                }, epoch)
            writer.add_scalars('loss',
                               {'train_loss': float(train_loss / len(train_set)),
                                'val_loss': float(validation_loss / len(val_set)),
                                }, epoch)
        writer.close()