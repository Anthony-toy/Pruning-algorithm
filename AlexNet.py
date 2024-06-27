import torch
import torch.nn as nn

data=torch.ones(size=(10,3,227,227))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #为了处理尺寸较大的图片，先使用11*11的卷积核和较大的步长来降低特征图的尺寸
        #同时使用较多的输出通道来弥补降低尺寸带来的损失
        self.conv1=nn.Conv2d(3,96,11,stride=4)  #(55*55*96)
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)  #(27*27*96) AlexNet的特点  重叠池化

        #卷积核和步长恢复常用的大小，进一步扩大通道来提取数据
        self.conv2=nn.Conv2d(96,256,5,padding=2)   #(27*27*256)
        self.pool2=nn.MaxPool2d(kernel_size=3,stride=2)  #(13*13*256)

        #连续用多个卷积层来提取特征 并维持特征图的大小
        self.conv3=nn.Conv2d(256,384,3,padding=1)   #(13*13*384)
        self.conv4=nn.Conv2d(384,384,3,padding=1)   #(13*13*384)
        self.conv5=nn.Conv2d(384,256,3,padding=1)   #(13*13*256)
        self.pool3=nn.MaxPool2d(kernel_size=3,stride=2)  #(6*6*256)

        #进入全连接层，进行信息汇总
        self.fc1=nn.Linear(6*6*256,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,1000)

    def forward(self,x):
        x=torch.relu(self.conv1(x))
        x=self.pool1(x)
        x=torch.relu(self.conv2(x))
        x=self.pool2(x)
        x=torch.relu(self.conv3(x))
        x=torch.relu(self.conv4(x))
        x=torch.relu(self.conv5(x))
        x=self.pool3(x)
        x=x.reshape(-1,6*6*256) #拉平

        #随机让50%的权重为0 dropout防止过拟合
        x=torch.dropout(x,p=0.5,train=False)  #以p的概率（50%）来沉默掉权重
        x=torch.relu(torch.dropout(self.fc1(x),p=0.5,train=False))
        x=torch.relu(self.fc2(x))
        output=torch.softmax(self.fc3(x),dim=1)

net=Model()
model=Model().cuda()
net(data)

import torchinfo
torchinfo.summary(net,(10,3,227,227))




