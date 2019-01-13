#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2019/1/11
Author: Xu Yucheng
Abstract: Code for build VGG model
'''
import math
import torch.nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#trainning parameters
CLASS_NUM = 62
IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCH_NUM = 100
LR = 0.01

#structs of VGG models
#usually use VGG16
#以下每一个VGG模型中，卷积层padding均为SAME， max_pooling层padding均为same，所以图片尺寸变化仅仅发生在max_pooling层
#提取到的特征为512*(IMAGE_SIZE/2^5)*(IMAGE_SIZE/2^5), 此数作为分类器第一层Flatten层的参数
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#Class vgg inheritance from torch.nn.Module
class VGG(torch.nn.Module):
    def __init__(self, vgg_name):
        #调用父类的构造函数
        super(VGG, self).__init__()
        #定义网络中的特征提取层，即卷积层
        self.features = self._make_layers(cfg[vgg_name])
        #定义网络中的分类层，即全链接层
        #使用dropout层来缓解过拟合现象
        self.classifier = torch.nn.Sequential(
            #fc6
            torch.nn.Linear(512*2*2, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            #fc7
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            #fc8
            torch.nn.Linear(4096, CLASS_NUM)
        )
        self._initialize_weight()
    
    #生成网络层
    def _make_layers(self, cfg):
        layers = []
        input_channels = 3
        for layer in cfg:
            if layer == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [torch.nn.Conv2d(input_channels, layer, kernel_size=3, padding=1),
                            torch.nn.BatchNorm2d(layer),
                            torch.nn.ReLU(inplace=True)]
                input_channels = layer
        #在网络层最后加上一个均值池化层
        layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
        #将储存网络层的列表转化为torch中的Sequential模型
        return torch.nn.Sequential(*layers)
    
    #初始化网络层中的权重
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    #定义训练中的前向传播过程
    def forward(self, input_batch):
        features_op = self.features(input_batch)
        features_op = features_op.view(features_op.size(0), -1)
        classify_op = self.classifier(features_op)
        return classify_op



if __name__ == '__main__':
    net = VGG('VGG16')
    x = torch.randn(64, 3, 64, 64)
    print (net.modules())