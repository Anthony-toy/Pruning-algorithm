#导入ku
import torch
import torch.nn as nn
import torchinfo
from typing import Type,Union,Optional,List

#定义3*3的conv
def conv3x3(in_,out_,stride=1,initialzer0=False): #大多数情况下步长为1，默认stride=1;默认大多数情况下是不需要0初始化
    bn=nn.BatchNorm2d(out_)
    #需要判断是否对BN进行0初始化
    if initialzer0==True:
        nn.init.constant_(bn.weight,0)  #对BN层的γ初始化，等于0
    return nn.Sequential(nn.Conv2d(in_,out_,
                                   kernel_size=3,padding=1,stride=stride,bias=False), #对于第一个3*3的网络，卷积核和填充是固定的
                         bn) #为了适应不同的层，将RuLu放在卷积的外面

#定义1*1的conv
def conv1x1(in_,out_,stride=1,initialzer0=False): #大多数情况下步长为1，默认stride=1;默认大多数情况下是不需要0初始化
    bn=nn.BatchNorm2d(out_)
    #需要判断是否对BN进行0初始化
    if initialzer0==True:
        nn.init.constant_(bn.weight,0)  #对BN层的γ初始化，等于0
    return nn.Sequential(nn.Conv2d(in_,out_,
                                   kernel_size=1,padding=0,stride=stride,bias=False), #对于第一个1*1的网络，卷积核和填充是固定的
                         bn) #为了适应不同的层，将RuLu放在卷积的外面

#定义残差单元ResidualUnit
class ResidualUnit(nn.Module):
    def __init__(self,out_:int,
                 stride1:int=1, #对于第一个卷积层，第一个步长默认值为1
                 in_:Optional[int]=None
                 ):
        super().__init__()
        self.stride1=stride1    #给残差单元类定义属性
        #由两段结构并联

        #当特征图尺寸需要缩小时，卷积层的特征输出out_等于in_*2
        #当特征图尺寸不需要缩小时，卷积层的特征输出out_等于in_
        if stride1!=1:
            in_=int(out_/2)
        else:
            in_=out_

        #拟合部分F(x)
        self.fit_=nn.Sequential(conv3x3(in_,out_,stride=stride1),
                                nn.ReLU(inplace=True),
                                conv3x3(out_,out_,initialzer0=True)) #由两段卷积串联,对于残差来说，两段的输出都相同,最后一个卷积的结果BN值需要0初始化
        #跳跃层上的1*1卷积层
        self.skipconv=conv1x1(in_,out_,stride=stride1)

        #单独定义放在H(x)之后的ReLU
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        #对于forward来说，有两条线路：拟合结果、x本身
        fx=self.fit_(x) #拟合结果
        # 判断stride1是否为2，若为2，特征图尺寸会发生变化，需要在跳跃层上加上1*1的卷积层来调整特征尺寸
        # 如果为1，不需要操作
        if self.stride1!=1:
            x=self.skipconv(x)  #跳跃连接
        hx=self.relu(x+fx)
        return hx

#定义瓶颈单元BottleneckUnit
class Bottleneck(nn.Module):
    def __init__(self,middle_out,
                 stride1:int=1,
                 in_:Optional[int]=None): #选填参数用来判别这个架构是否跟在conv1后，如若是，in_=64;不是就不填
        super().__init__()

        out_=4*middle_out   #输出是中间层的四倍

        #conv1-conv2的情况特殊，上一层的输出等于下一层的输入
        if in_==None:
            #是否需要将特征图缩小
            #当层与层推进的时候，需要将特征图尺寸减半，同时卷积层上的middle_out=in_/2
            #当瓶颈内部循环堆叠的时候，卷积层上的middle_out=in_/4
            if stride1!=1:
                in_=middle_out*2
            else:
                in_=middle_out*4
        else:
            in_=64

        #拟合部分
        self.fit_=nn.Sequential(conv1x1(in_,middle_out,stride=stride1),
                                nn.ReLU(inplace=True),
                                conv3x3(middle_out,middle_out,),
                                nn.ReLU(inplace=True),
                                conv1x1(middle_out,out_,initialzer0=True))
        # 跳跃层上的1*1卷积层
        self.skipconv = conv1x1(in_, out_, stride=stride1)
        # 单独定义放在H(x)之后的ReLU
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        fx=self.fit_(x)
        x=self.skipconv(x)
        hx=self.relu(fx+x)
        return hx

# #测试
# #50layer-conv4_x  包含6个瓶颈结构
# num_blocks_conv4x=6
# conv4_x_50=[]
# #在列表中添加第0个瓶颈架构块
# conv4_x_50.append(Bottleneck(middle_out=256,stride1=2))
# #往列表中append剩下的块
# for i in range(num_blocks_conv4x-1):
#     conv4_x_50.append(Bottleneck(middle_out=256))
# print(len(conv4_x_50))  #包含了六个块
#
# #建立一个瓶颈单元空列表
# layers=[]
# afterconv1=True #为conv1之后第一个块
# num_blocks=6 #在整个layers里面有6个块
# if afterconv1==True:
#     layers.append(Bottleneck(middle_out=64,in_=64))
# else:
#     layers.append(Bottleneck(middle_out=128,stride1=2))
# for i in range(num_blocks-1):
#     layers.append(Bottleneck(middle_out=128))
# print(len(layers))
#
# #建立一个残差单元空列表
# layers=[]
# afterconv1=True #为conv1之后第一个块
# num_blocks=6 #在整个layers里面有6个块
# if afterconv1==True:
#     layers.append(ResidualUnit(out_=64,in_=64))
# else:
#     layers.append(ResidualUnit(out_=64,stride1=2))
# for i in range(num_blocks-1):
#     layers.append(ResidualUnit(out_=64))
# print(len(layers))

#专门用来生成ResNet中每一个Layers的函数
def make_layers(block:Type[Union[ResidualUnit,Bottleneck]],    #在block中，只能填写类
                middle_out:int,
                num_blocks:int,
                afterconv1:bool=False,
                ):
    layers=[]
    if afterconv1 == True:
        layers.append(block(middle_out, in_=64))    #不需要写参数=， 而是直接写参数即可
    else:
        layers.append(block(middle_out, stride1=2))
    for i in range(num_blocks-1):
        layers.append(block(middle_out))

    return nn.Sequential(*layers)   #nn.Sequential只能放类，不能放列表。可以加*来解析列表/储存器

##对残差块和瓶颈架构进行测试，需要对Conv1后的首个构架，以及中间构架进行测试
# #34层网络，conv2x
# layer_34_conv2x=make_layers(ResidualUnit,
#                             middle_out=64,
#                             num_blocks=3,
#                             afterconv1=True)
# # print(layer_34_conv2x)
# datashape=(10,64,56,56)
# torchinfo.summary(layer_34_conv2x,datashape,depth=1,device='cpu')
#
# #101层网络，conv2x
# layer_101_conv2x=make_layers(Bottleneck,
#                             middle_out=64,
#                             num_blocks=3,
#                             afterconv1=True)
# datashape=(10,64,56,56)
# torchinfo.summary(layer_101_conv2x,datashape,depth=3,device='cpu')
#
# #101层网络，conv4x
# layer_101_conv4x=make_layers(Bottleneck,
#                             middle_out=256,
#                             num_blocks=23,
#                             afterconv1=False)
# datashape=(10,512,28,28)
# torchinfo.summary(layer_101_conv4x,datashape,depth=3,device='cpu')

#定义残差网络
class ResNet(nn.Module):
    def __init__(self,block:Type[Union[ResidualUnit,Bottleneck]],
                 layers:List[int],num_classes:int): #num_classes 输出不一定是1000，可以是任意
        super().__init__()

        #block:用来加深深度的基本构架
        #layers:每个层里有多少块
        #num_classes真实标签有多少类别

        #Layer1: 卷积+池化
        self.layer1=nn.Sequential(nn.Conv2d(3,64,
                                            kernel_size=7,stride=2,
                                            padding=3,bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        #Layer2-Layer5:残差块/瓶颈结构
        self.layer2_x = make_layers(block, 64, layers[0], afterconv1=True)
        self.layer3_x = make_layers(block, 128, layers[1], afterconv1=False)
        self.layer4_x = make_layers(block, 256, layers[2], afterconv1=False)
        self.layer5_x = make_layers(block, 512, layers[3], afterconv1=False)

        #全局平均池化
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))

        #分类
        if block==ResidualUnit:
            self.fc=nn.Linear(512, num_classes)
        else:
            self.fc=nn.Linear(2048, num_classes)

    def forward(self,x):
        x=self.layer1(x)    #普通卷积+池化 layer1
        x=self.layer5_x(self.layer4_x(self.layer3_x(self.layer2_x(x))))
        x=self.avgpool(x)
        x=torch.flatten(x,1)    #将x拉平至一维
        x=self.fc(x)

datashape=(10,3,224,224)    #数据集结构
res34=ResNet(ResidualUnit,[3,4,6,3],num_classes=1000)
# res101=ResNet(Bottleneck,[3,4,23,3],num_classes=1000)
torchinfo.summary(res34,datashape,depth=3,device="cpu")
