import torch
import time
import os
from torch.utils.data import DataLoader
from .model_train_val import Train_Val_Test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, f1_score
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, auc
from .model_parameter_compute import compute_model
from torchinfo import summary
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class Test(Train_Val_Test):
    def __init__(self, configs):
        self.configs = configs

    def inference(self):
        configs = self.configs
        y_pred = []
        y_test = []
        y_preprob = []
        net = self.model_set()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(torch.load(configs.model_saved_addr, map_location={'cuda:1':'cuda:0'}))
        net.to(device)
        train_set, val_set, trainLoader, validLoader = self.get_data()
        if configs.mode == 'multimodal':
            summary(net, input_size=((configs.batch_size, configs.enc_in, configs.graph_size[0], configs.graph_size[1]),
                                     (configs.batch_size, configs.enc_in, configs.seq_len)))
            y_pred, y_preprob, y_test, time_elapsed, accuracy = self.multimodal_test(net, validLoader, y_pred, y_preprob, y_test, device)
        if configs.mode == 'ts':
            summary(net, input_size=(configs.batch_size, configs.enc_in, configs.seq_len))
            y_pred, y_preprob, y_test, time_elapsed, accuracy = self.ts_test(net, validLoader, y_pred, y_preprob, y_test, device)
        if configs.mode == 'thu_ts':
            summary(net, input_size=(configs.batch_size, configs.seq_len, configs.enc_in))
            y_pred, y_preprob, y_test, time_elapsed, accuracy = self.thu_ml_test(net, validLoader, y_pred, y_preprob, y_test, device)
        if configs.mode == 'graph':
            summary(net, input_size=(configs.batch_size, configs.enc_in, configs.graph_size[0], configs.graph_size[1]))
            y_pred, y_preprob, y_test, time_elapsed, accuracy = self.graph_test(net, validLoader, y_pred, y_preprob, y_test, device)
        print('Inference complete in {:.0f}m {:.5f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print(compute_model(net, require_grad=False))
        print('Accuracy on the test set: %.5f %%' % (
                100 * accuracy))
        y_pred1 = []
        y_test1 = []
        y_preprob1 = []
        for i in y_pred:
            y_pred1.append(i.cpu().numpy())
        for i in y_test:
            y_test1.append(i.cpu().numpy())
        for i in y_preprob:
            y_preprob1.append(i.cpu().detach().numpy())
        y_pred1 = np.concatenate(y_pred1)
        y_test1 = np.concatenate(y_test1)
        y_preprob1 = np.concatenate(y_preprob1)

        # Confusion Matrix
        self.Confusion_Matrix(y_test1, y_pred1)
        # Macro F1
        macro_f1 = f1_score(y_test1, y_pred1, average='macro')
        print("Macro F1 Score:", macro_f1)

        # To calculate the weighted average F1 score (weighted by the number of samples in each class)
        weighted_f1 = f1_score(y_test1, y_pred1, average='weighted')
        print("Weighted F1 Score:", weighted_f1)

        # ROC: receiver operator characteristic curve
        self.ROC(y_test1, y_preprob1)

        result_save = configs.test_result_addr  # 设定tensorboard文件存放的地址
        if result_save is not None:
            os.makedirs(result_save, exist_ok=True)
        np.save(f'{result_save}' + '\y_preprob1.npy', y_preprob1)
        np.save(f'{result_save}' + '\y_pred1.npy', y_pred1)
        np.save(f'{result_save}' + '\y_test1.npy', y_test1)

    def multimodal_test(self, net, validLoader, y_pred, y_preprob, y_test, device):
        since = time.time()
        with torch.no_grad():
            net.eval()
            right = 0
            total = 0
        for series, graph, targetVar in validLoader:
            series, graph = series.to(device), graph.to(device)
            label = targetVar.to(device)
            pred, _ = net(graph,series)
            probs = torch.softmax(pred, dim=1)
            _, predicted = torch.max(probs, dim=1)
            right += (predicted == label).sum()
            total += label.size(0)
            y_pred.append(predicted)
            y_test.append(targetVar)
            y_preprob.append(probs)
        return y_pred, y_preprob, y_test, time.time() - since, right / total

    def ts_test(self, net, validLoader, y_pred, y_preprob, y_test, device):
        since = time.time()
        with torch.no_grad():
            net.eval()
            right = 0
            total = 0
        for series, graph, targetVar in validLoader:
            series = series.to(device)
            label = targetVar.to(device)
            pred = net(series)
            probs = torch.softmax(pred, dim=1)
            _, predicted = torch.max(probs, dim=1)
            right += (predicted == label).sum()
            total += label.size(0)
            y_pred.append(predicted)
            y_test.append(targetVar)
            y_preprob.append(probs)
        return y_pred, y_preprob, y_test, time.time() - since, right / total

    def thu_ml_test(self, net, validLoader, y_pred, y_preprob, y_test, device):
        since = time.time()
        with torch.no_grad():
            net.eval()
            right = 0
            total = 0
        for series, graph, targetVar in validLoader:
            series = series.to(device)
            label = targetVar.to(device)
            pred = net(series, 1, None, None)
            probs = torch.softmax(pred, dim=1)
            _, predicted = torch.max(probs, dim=1)
            right += (predicted == label).sum()
            total += label.size(0)
            y_pred.append(predicted)
            y_test.append(targetVar)
            y_preprob.append(probs)
        return y_pred, y_preprob, y_test, time.time() - since, right / total

    def graph_test(self, net, validLoader, y_pred, y_preprob, y_test, device):
        since = time.time()
        with torch.no_grad():
            net.eval()
            right = 0
            total = 0
        for series, graph, targetVar in validLoader:
            graph = graph.to(device)
            label = targetVar.to(device)
            pred = net(graph)
            probs = torch.softmax(pred, dim=1)
            _, predicted = torch.max(probs, dim=1)
            right += (predicted == label).sum()
            total += label.size(0)
            y_pred.append(predicted)
            y_test.append(targetVar)
            y_preprob.append(probs)
        return y_pred, y_preprob, y_test, time.time() - since, right / total

    def Confusion_Matrix(self, y_test1, y_pred1):
        cm = confusion_matrix(y_true=y_test1, y_pred=y_pred1)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 将百分比格式化为带有'%'符号的字符串
        cm_formatted = [['{:.1f}'.format(value) if value % 1 != 0 else '{:.1f}'.format(value) for value in row] for row
                        in cm_percentage]

        # 调整图像尺寸和字体大小
        # plt.figure(figsize=(8, 6))
        font_size = 15
        # 创建 ConfusionMatrixDisplay 对象并打印带有百分比的混淆矩阵
        disp = ConfusionMatrixDisplay(confusion_matrix=cm / cm.sum(axis=1) * 100,
                                      display_labels=[' d<0.15 ', '0.15~0.18', '0.18~0.21', '0.21~0.24', '0.24~0.27',
                                                      '0.27~0.30', '0.30~0.33', '0.33~0.36', '0.36~0.39', ' 0.39≤d ',
                                                      'No Impact'])
        disp.plot(cmap='Greens', values_format='{:f}', include_values=False)
        plt.xticks(ticks=np.arange(11),
                   labels=[' d<0.15 ', '0.15~0.18', '0.18~0.21', '0.21~0.24', '0.24~0.27', '0.27~0.30', '0.30~0.33',
                           '0.33~0.36', '0.360.39', ' 0.39≤d ', 'No Impact'], fontsize=10, weight='bold', rotation=45,
                   ha='right')
        plt.yticks(ticks=np.arange(11),
                   labels=[' d<0.15 ', '0.15~0.18', '0.18~0.21', '0.21~0.24', '0.24~0.27', '0.27~0.30', '0.30~0.33',
                           '0.33~0.36', '0.36-0.39', ' 0.39≤d ', 'No Impact'], fontsize=10, weight='bold')
        plt.xlabel("Predicted label", fontsize=16, weight='bold')
        plt.ylabel("True label", fontsize=16, weight='bold')
        plt.title("Confusion Matrix", fontsize=16, weight='bold')
        # 在每个格子中显示百分比
        for i in range(11):
            for j in range(11):
                if cm_percentage[i][j] > 50:
                    disp.ax_.text(j, i, cm_formatted[i][j], va='center', ha='center', color='white', fontsize=11,
                                  weight='bold')

                else:
                    disp.ax_.text(j, i, cm_formatted[i][j], va='center', ha='center', color='black', fontsize=11,
                                  weight='bold')
        colorbar = disp.im_.colorbar
        font_properties = FontProperties(weight='bold', size=10)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontproperties(font_properties)
        # plt.show()

    def ROC(self, y_test1, y_preprob1):
        plt.figure(figsize=(8, 6))
        classes = [' d<0.15 ', '0.15~0.18', '0.18~0.21', '0.21~0.24', '0.24~0.27', '0.27~0.30', '0.30~0.33',
                   '0.33~0.36', '0.36~0.39', ' 0.39≤d ', 'No Impact']
        plt.xlabel('False Positive Rate', fontsize=18, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=18, weight='bold')
        plt.title('ROC Curve', fontsize=18, weight='bold')
        plt.plot([0, 1], [0, 1], 'r--')
        Auc = []
        for n, name in enumerate(classes):
            y_preprob2 = []
            y_test2 = []
            y_preprob2 = np.array(y_preprob1[:, n].tolist())
            for i in y_test1:
                if i == n:
                    y_test2.append(0)

                else:
                    y_test2.append(1)
            fpr, tpr, threshold = roc_curve(y_test2, y_preprob2, pos_label=0)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=1.5, label=f'{name} AUC:{auc_score:.3f}')
            Auc.append(auc_score)
            plt.legend(fontsize=20, loc='lower right')
        MAUC = sum(Auc) / (n + 1)
        plt.plot([0, 1], [MAUC, MAUC], label=f'MAUC:{MAUC:.3f}', color='red')
        plt.legend(fontsize=16, frameon=False)
        plt.xticks(fontsize=16, weight='bold')
        plt.yticks(fontsize=16, weight='bold')
        print('Average Auc: %.4f ' % (sum(Auc) / (n + 1)))
        plt.show()

    def int_to_onehot(self, label):
        return torch.nn.functional.one_hot(torch.tensor(label), num_classes=11)

