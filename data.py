import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms, utils,datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
import numpy as np
# from sklearn.metrics import roc_curve
# import torchvision
#过滤警告信息
import warnings
import torch.optim as optim
import shutil
from torch.autograd import Variable
# from resnet_18 import Resnet
from model import vgg
# from my_model import model
# from model_12 import model
# from networkc import model
# from get_lr import get_lr_scheduler,set_optimizer_lr
#
# from sklearn import metrics
warnings.filterwarnings("ignore")


# 64x64
# (mean=[0.535, 0.484, 0.469], std=[0.263, 0.261, 0.260])
# mean=[0.532, 0.482, 0.465], std=[0.262, 0.260, 0.259]


# test_224
# [0.5311469, 0.48099273, 0.4640799]
# normStd = [0.26059636, 0.2588301, 0.2580747]
# train
# normMean = [0.5343799, 0.48375282, 0.46799666]
# normStd = [0.26113355, 0.25940776, 0.25886327]
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):#正则化的过程，并将梯度更新加入。
            m.weight.grad.data.add_(0.0001*torch.sign(m.weight.data))  # L1 大于0为1 小于0为-1 0还是0



train_transform = transforms.Compose([
 transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
# transforms.RandomGrayscale(),
transforms.Resize((40,40)),
# transforms.RandomResizedCrop(96, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)), 儿童vs成人使用
transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
transforms.RandomHorizontalFlip(p=0.5),
 #nuaa的差transforms.Normalize(mean=[0.534, 0.484, 0.468], std=[0.261, 0.259, 0.259]) # 标准化至[-1, 1]，规定均值和标准差
transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

test_transform = transforms.Compose([
 transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
transforms.Resize((40,40)),
 #NUAA的差transforms.Normalize(mean=[0.531, 0.481, 0.464], std=[0.261, 0.259, 0.258]) # 标准化至[-1, 1]，规定均值和标准差
transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
# D:/my_data/face/light/train
# D:/my_data/face/rgb/train
# E:\NUAA\train\light
# E:/CASIA-fasd/train/light
train_dataset = datasets.ImageFolder(root="D:/FireFit/Flame/Training/Training",
           transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True,drop_last=False)


test_dataset = datasets.ImageFolder(root="D:/FireFit/Flame/Test/Test",
           transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=16,shuffle=True,drop_last=False)

# torch.cuda.manual_seed(2023)
# torch.cuda.manual_seed_all(2023)
# 加载模型
model = vgg()   # 加载模型时应先实例化模型
# load_state_dict()函数接收一个字典，所以不能直接将'./model/model_state_dict.pth'传入，而是先使用load函数将保存的模型参数反序列化
# model = torch.load('E:/model/r18_CA_light.pth')
# model.eval()    # 模型推理时设置


if torch.cuda.is_available():
    model.cuda()

# nbs             = 32
# lr_limit_max    =  5e-2
# lr_limit_min    =  5e-5
# Init_lr             = 2e-4
# Min_lr              = Init_lr * 0.01
# batch_size= 16
# Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
# Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
# UnFreeze_Epoch=100
# lr_decay_type       = 'step'
# lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

# 原0.00001
opt = optim.Adam(model.parameters(),lr = 0.001,betas=(0.9, 0.999),eps=1e-09,weight_decay=0,amsgrad=False)
# opt = optim.Adam(model.parameters(),Init_lr_fit,betas=(0.9, 0.999),eps=1e-09,weight_decay=5e-4,amsgrad=False)
# opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# opt = optim.SGD(model.parameters(), Init_lr_fit, momentum=0.9)
# T_max = 20
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max, eta_min=0, last_epoch=-1, verbose=False)

# scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=40,gamma = 0.1)

def train(net,epoch):
    net.train()
    loss = nn.CrossEntropyLoss()
    loss_train = 0
    trainsamples = 0
    correct = 0
    for batch_idx,(data,target) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            data,target = data.cuda(), target.cuda()
            data,target = Variable(data), Variable(target)
        target = target.view(data.shape[0])
        x = net(data)
        # print('x------',torch.max(x,1)[0])
        # print('target------', target)
        ss = loss(x, target)
        loss_train += ss
        yhat = torch.max(x, 1)[1]
        # print('yhat------', yhat)
        correct += torch.sum(yhat == target)
        # eer=calculate_eer(target,yhat)
        #output=torch.max(output.data,1)[1]
        ss.backward()
        updateBN()
        opt.step()
        opt.zero_grad()
        trainsamples += data.shape[0]

        if (batch_idx+1) % 50  == 0:
            print('Train Epoch{}:[{}/{}({:.0f})%],Loss:{:.6f},Accuracy: {}/{}({:.3f}%)'.format(epoch,trainsamples,train_dataset.__len__(),
                                                                               100*trainsamples/train_dataset.__len__(),ss.item(),
                                                                                               correct,train_dataset.__len__(),
                                                                                               100.*correct/train_dataset.__len__()))
    print("\nTrain set: Average loss: {:.4f}".format(loss_train.item() / 50.))
    print('Accuracy:{}/{}({:.3f}%)'.format(train_dataset.__len__(), correct,
                                                   100. * correct / train_dataset.__len__()))

            # print("EER: ",eer)
    del data,target
    gc.collect()
    torch.cuda.empty_cache()

def test(net):
    net.eval()
    correct = 0
    loss = nn.CrossEntropyLoss()
    test_loss=0
    # FP = 0
    # TN = 0
    # FN = 0
    # TP = 0
    a = np.array([])
    b = np.array([])

    for data,target in test_dataloader:
        with torch.no_grad():
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
            target = target.view(data.shape[0])
            x = net.forward(data)
            ss = loss(x,target)
            test_loss += ss
            yhat = torch.max(x, 1)[1]
            #xx = torch.max(x,1)[0]
            correct += torch.sum(yhat == target)

            a = np.concatenate((a, yhat.cpu().numpy()))
            b = np.concatenate((b, target.cpu()))

            # print('yhat----',yhat)
            # print('target----',target)


            # eer = calculate_eer(b,a)
            # FP += torch.sum((yhat == 0) & (target == 1))
            # TN += torch.sum((yhat == 1) & (target == 1))
            # FN += torch.sum((yhat == 1) & (target == 0))
            # TP += torch.sum((yhat == 0) & (target == 0))
            #
            # FAR = FP / (FP + TN)
            # FRR = FN / (FN + TP)
            # HTER = (FAR + FRR) / 2



    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.3f}%\n".format(test_loss.item() / 50., correct,
                                                                               test_dataset.__len__(),
                                                                               100. * correct / test_dataset.__len__()))
    return correct / float(len(test_dataloader.dataset))
    # print('EER:{:.6f}'.format(eer))
    # print("HTER------", HTER)
            # a = np.concatenate(target)
            # b = np.concatenate(yhat)

            #print('FP-----',FP)

            # FAR = FP / (FP + TN)
            # FRR = FN / (FN + TP)
            # TPR = TP / (TP + FN)
            # FPR = FP / (FP + TN)
            # print("FPR------",FPR)
            # print('TPR------',TPR)

# #绘图函数
# def plotloss(trainloss,testloss):
#     plt.figure(figsize=(10,7))
#     plt.plot(trainloss,color="red")
#     plt.plot(testloss, color="orange")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend(["Trainloss","Testloss"],loc='upper right')
#     plt.show()

def save_checkpoint(state, is_best, filename='D:/FireFit/pytorch-slimming/pytorch-slimming/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

best_prec1 = 0.
for epoch in range(1, 151):
    train(model,epoch)
    prec1 = test(model)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': opt.state_dict(),
    }, is_best)

