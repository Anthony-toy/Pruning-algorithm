import torch
import torch.nn as nn
from torch.autograd import Variable
import math  # init
from torchinfo import summary



class SEAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEAttention, self).__init__()
        self.input_channels = input_channels
        self.reduction_ratio = reduction_ratio
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // reduction_ratio, input_channels),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        batch_size, channels, _, _ = inputs.size()

        squeezed = self.squeeze(inputs).view(batch_size, channels)
        weights = self.excitation(squeezed).view(batch_size, channels, 1, 1)
        attended_inputs = inputs * weights

        return attended_inputs


class vgg(nn.Module):

    def __init__(self, init_weights=True, cfg=None,num_classes=2):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]    #基本的vgg网路架构
        self.feature = self.make_layers(cfg, True)
        # #判断是十分类还是一百分类
        # if dataset == 'cifar100':
        #     num_classes = 100
        # elif dataset == 'cifar10':
        #     num_classes = 10
        self.SE = SEAttention(512) #SE注意力机制
        self.classifier = nn.Linear(cfg[-1], num_classes) #加上一个全连接层
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:#基于配置文件构件一个网络架构
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                #print (in_channels,'   ',v)
                if batch_norm:  #如果加入了BN层 结构就是（卷积+BN+RELU）
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v #下一层输入就是这一层的输出
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.SE(x)
        # print(x.size())
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# if __name__ == '__main__':
#     net = vgg()
#     x = Variable(torch.FloatTensor(16, 3, 40, 40))
#     y = net(x)
#     print(y.data.shape)
#
data = torch.ones(size=(16,3,40,40))
net = vgg()
net(data)
summary(net,input_size=(16,3,40,40))

