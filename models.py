import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Level-1 Attention Layer
class Attention_Alpha (nn.Module) :
    def __init__(self) :
        super(Attention_Alpha, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,  8,   kernel_size = 3, stride = 1, padding = (1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(8,  16,  kernel_size = 3, stride = 1, padding = (1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 32,  kernel_size = 3, stride = 1, padding = (1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 1,   kernel_size = 3, stride = 1, padding = (1, 1)))

    def forward(self, x) :
        out = self.layers(x)
        out = out.mean(1).mean(0)
        return out

# Level-2 Attention Layer
class Attention_Beta (nn.Module) :
    def __init__(self) :
        super(Attention_Beta, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,  8,  kernel_size = 3, stride = 1, padding = (1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(8,  16, kernel_size = 3, stride = 1, padding = (1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = (1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 1,  kernel_size = 3, stride = 1, padding = (1, 1)))

    def forward(self, x) :
        out = self.layers(x)
        out = out.mean(1)
        return out

# Single-Head Transformer
class Basic_Transformer (nn.Module) :
    def __init__(self, out_chs = 64) :
        super(Basic_Transformer, self).__init__()
        self.out_chs = out_chs
        self.Wq = nn.Linear(in_features = 128, out_features = out_chs, bias = False)
        self.Wk = nn.Linear(in_features = 128, out_features = out_chs, bias = False)
        self.Wv = nn.Linear(in_features = 128, out_features = out_chs, bias = False)

        # self.upscale = nn.Sequential(
        #     nn.Dropout(p = 0.4, inplace = True),
        #     nn.ReLU(inplace = True),
        #     nn.Linear(in_features = out_chs, out_features = 128, bias = True))

    def forward(self, x) :
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        q_k = torch.mm(Q, K.T)
        q_k = q_k / np.sqrt(self.out_chs)
        Q_K = F.softmax(q_k, dim = 1)
        out = torch.mm(Q_K, V)

        #############################################
        # No upscaling when modifying the classifier
        # out = self.upscale(out)
        #############################################

        return Q_K, out

class ResBlock (nn.Module) :
    def __init__(self, in_chs, out_chs, strides) :
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = in_chs, out_channels = out_chs,
                      kernel_size = 3, stride = strides, padding = 1),
            nn.BatchNorm2d(num_features = out_chs),
            nn.ReLU(inplace = True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = out_chs, out_channels = out_chs,
                      kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = out_chs))

        if in_chs != out_chs :
            self.id_mapping = nn.Sequential(
                nn.Conv2d(in_channels = in_chs, out_channels = out_chs,
                          stride = strides, padding = 0, kernel_size = 1),
                nn.BatchNorm2d(num_features = out_chs))
        else :
            self.id_mapping = None

        self.final_activation = nn.ReLU(inplace = True)

    def forward(self, x) :
        out = self.conv1(x)
        out = self.conv2(out)
        if self.id_mapping is not None :
            x_ = self.id_mapping(x)
        else :
            x_ = x
        result = self.final_activation(x_ + out)
        return result

# ResNet19 Encoder
class ResNet_Encoder (nn.Module) :
    def __init__(self, num_layers = 20, num_stem_conv = 32, config = (32, 64, 128)) :
        super(ResNet_Encoder, self).__init__()

        self.num_layers = num_layers

        self.head_conv = nn.Sequential(
            nn.Conv2d(3, num_stem_conv, stride = 1, padding = 1, kernel_size = 3),
            nn.BatchNorm2d(num_stem_conv),
            nn.ReLU(True))

        num_layers_per_stage = (num_layers - 2) // 6

        self.body_op = []

        num_inputs = num_stem_conv

        for i in range(0, len(config)) :
            for j in range(num_layers_per_stage) :
                if j == 0 and i != 0 :
                    strides = 2
                else :
                    strides = 1
                self.body_op.append(ResBlock(num_inputs, config[i], strides))
                num_inputs = config[i]

        self.body_op = nn.Sequential(*self.body_op)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size = (1, 1))

    def forward(self, x) :
        out = self.head_conv(x)
        out = self.body_op(out)
        out = self.avg_pool(out)
        out_linear = out.mean(3).mean(2)
        return out_linear

# Classifier
class Classifier (nn.Module) :
    def __init__(self, in_channels = 128, cut_epochs = 50) :
        super(Classifier, self).__init__()

        ################################################################
        # Baseline
        # self.layers = nn.Sequential(
        #     nn.Linear(in_features = in_channels, out_features = 1),
        #     nn.Sigmoid())
        ################################################################

        self.transformer_replace = nn.Sequential(
            nn.Linear(in_features = in_channels, out_features = 32),
            # nn.Dropout(p = 0.5, inplace = True))
            nn.ReLU(inplace = True))

        self.classifier = nn.Sequential(
            nn.Linear(in_features = 32, out_features = 1),
            nn.Sigmoid())
        self.cut_epochs = cut_epochs

    def forward(self, x, curr_epoch) :

        ################################################################
        # Baseline
        # out = self.layers(x)
        ################################################################

        if curr_epoch >= self.cut_epochs :
            out = self.classifier(x)
        else :
            out = self.transformer_replace(x)
            out = self.classifier(out)
        return out
