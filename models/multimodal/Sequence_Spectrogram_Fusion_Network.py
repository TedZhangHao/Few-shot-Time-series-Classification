import torch
import torch.nn as nn
from .Multihead_Selfattention_Conv import AugmentedConv

'''-------------1. Initial Layers -------------'''
# Spectrogram
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

# Sequence
def Conv2(in_planes, places, stride=2, dilation=4):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3,
                  dilation=dilation,
                  bias=False),
        nn.BatchNorm1d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    )

'''-------------2. Sequence Block and Spectrogram Block -------------'''

'''--- (1) Spectrogram-Block ---'''

# BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv
class _Layergraph(nn.Module):
    def __init__(self, inplace, outplace, growth_rate, drop_rate=0):
        super(_Layergraph, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace, out_channels=growth_rate * outplace, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(growth_rate * outplace),

            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=growth_rate * outplace, out_channels=outplace, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return x + y

'''--- (2) Sequence-Block ---'''

class _Layerseries(nn.Module):
    def __init__(self, inplace, outplace, growth_rate, drop_rate=0, attentionkv=2, attentionhead=1):
        super(_Layerseries, self).__init__()
        self.drop_rate = drop_rate
        self.inplace = inplace
        self.outplace = outplace
        self.dense_layer = nn.Sequential(
            nn.BatchNorm1d(inplace),
            nn.ReLU(inplace=True),
            # outplace: determine the number of generated feature maps of one layer
            nn.Conv1d(in_channels=inplace, out_channels=growth_rate * outplace, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm1d(growth_rate * outplace),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=growth_rate * outplace, out_channels=outplace, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )
        self.attention = AugmentedConv(
            in_channels=inplace,
            out_channels=inplace,
            kernel_size=3,
            dk=attentionkv,  # query and key dim
            dv=attentionkv,  # value dim
            Nh=attentionhead)  # attention head num
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        if self.inplace > 16:
            y = x.unsqueeze(3)
            y = self.attention(y)
            y = y.squeeze()
            y = self.dense_layer(y)
        else:
            y = self.dense_layer(x)
        # if self.drop_rate > 0:
        #     y = self.dropout(y)
        return x + y


class _Transfer(nn.Module):
    def __init__(self, inplace, in_feature, series_length):
        super(_Transfer, self).__init__()
        self.transfer = nn.Sequential(
            nn.Linear(in_feature, series_length),
            nn.BatchNorm1d(inplace),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.transfer(x)
        return y

'''---(3) Fusion Connection + Holistic Model ---'''

class SSFN_Block(nn.Module):
    def __init__(self, num_layers, inplances, outplace, growth_rate, drop_rate, graph_size,
                 series_size, dilation,attentionkv,attention_head,add_attentionkv,add_attentionhead):
        super(SSFN_Block, self).__init__()
        self.num_layers = num_layers
        layers1 = []
        layers2 = []
        transfer = []
        attention = []
        # As the number of layers increases, the input feature map times outplace with each additional layer
        l2 = inplances[1]
        l1 = inplances[0]
        for i in range(num_layers):
            layers2.append(_Layerseries(l2, l2, growth_rate, drop_rate,attentionkv=add_attentionkv[i],attentionhead=add_attentionhead[i]))
            layers1.append(_Layergraph(l1, outplace[0], growth_rate, drop_rate))
            # l1 += outplace[0]
            l2 += l1
            # Attention Insertion
            attention.append(AugmentedConv(
                in_channels=l1,
                out_channels=l1,
                kernel_size=3,
                dk=attentionkv[i],
                dv=attentionkv[i],
                Nh=attention_head[i]
            ))
        layers2.append(_Layerseries(l2, l2, growth_rate, drop_rate, attentionkv=add_attentionkv[-1],
                                    attentionhead=add_attentionhead[-1]))
        self.layers2 = nn.Sequential(*layers2)
        self.layers1 = nn.Sequential(*layers1)
        self.attention1 = nn.Sequential(*attention)

        kernel1 = 7
        padding1 = 3
        kernel2 = 3
        padding2 = 1
        kernel3 = 3
        padding3 = 1
        dilation1 = 1
        # Could be adjusted when tuning kernel size, padding, and dilation
        for i in range(num_layers):
            x1 = int((int((graph_size[0] - kernel1 - (kernel1 - 1) * (
                    dilation1 - 1) + padding1 * 2) / 2 + 1) - kernel3 + 2 * padding3) / 2 + 1)
            x2 = int((int((graph_size[1] - kernel1 - (kernel1 - 1) * (
                    dilation1 - 1) + padding1 * 2) / 2 + 1) - kernel3 + 2 * padding3) / 2 + 1)

            y = int((int((series_size - kernel1 - (kernel1 - 1) * (
                    dilation - 1) + padding1 * 2) / 2 + 1) - kernel2 + 2 * padding2) / 2 + 1)  # int(series_size/4)

            transfer.append(_Transfer((inplances[0]), x1 * x2, y))  # requiring specific parameters
        self.transfer = nn.Sequential(*transfer)

    def forward(self, x1, x2):
        for i in range(self.num_layers):
            y1 = self.layers1[i](x1)
            y2 = self.layers2[i](x2)
            y11 = self.attention1[i](y1)
            m = y11.view(y11.size(0), y11.size(1), -1)  # channel size could be adjusted
            z1 = self.transfer[i](m)
            z2 = torch.cat([z1, y2], 1)  # TS
            x1 = y1
            x2 = z2
        y2 = self.layers2[-1](x2)
        return y1, y2


class SSFN(nn.Module):
    def __init__(self, configs):
        super(SSFN, self).__init__()
        variant = configs.enc_in
        init_channels = configs.init_channels
        growth_rate = configs.growth_rate
        blocks = configs.blocks
        num_classes = configs.num_class
        graph_size = configs.graph_size
        series_size = configs.seq_len
        dilation = configs.sequence_dilation
        attentionkv = configs.graph_kq_dim
        attention_head = configs.graph_attention_head
        add_attentionkv = configs.fusion_kq_dim
        add_attentionhead = configs.fusion_attention_head
        drop_rate = configs.dropout

        self.conv1 = Conv1(in_planes=variant, places=init_channels[0])
        self.conv2 = Conv2(in_planes=variant, places=init_channels[1], dilation=dilation)
        blocks * 2
        # The dimensions of the first execution feature are derived from the previous feature extraction
        num_features = outplace = init_channels  # 64
        self.layer1 = SSFN_Block(num_layers=blocks[0], inplances=num_features, outplace=num_features, growth_rate=growth_rate,
                                 drop_rate=drop_rate, graph_size=graph_size, series_size=series_size, dilation=dilation,
                                 attentionkv=attentionkv, attention_head=attention_head, add_attentionkv=add_attentionkv,
                                 add_attentionhead=add_attentionhead)
        l2 = init_channels[1]
        l1 = init_channels[0]
        for i in range(blocks[0]):
            l1 = outplace[0]
            l2 += outplace[0]
        num_features = l2  # l2

        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_features, num_classes)
        self.fc1 = nn.Linear(l1, num_classes)  # not used

    def forward(self, x1, x2):
        x1 = self.conv1(x1)  # Spectrogram
        x2 = self.conv2(x2)  # Sequence
        x1, x2 = self.layer1(x1, x2)
        x2 = self.avgpool2(x2)
        x1 = self.avgpool1(x1)
        x2 = x2.squeeze()
        x1 = x1.squeeze()
        x = x2.view(x2.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x = self.fc(x)
        x1 = self.fc1(x1)
        return x, x1