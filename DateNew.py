from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from vgg import vgg
import shutil
import gc
import numpy as np

# 1：训练，并且加入l1正则化 保证稀疏性 -sr --s 0.0001
# 2：执行剪枝操作 --model model_best.pth.tar（到该文件下去进行剪枝） --save pruned.pth.tar（当前剪枝好的文件） --percent 0.7（剪枝阈值）
# 3：再次进行微调操作 --refine pruned.pth.tar（再训练剪枝后的模型） --epochs 40

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming FIRE prune')
parser.add_argument('--dataset', type=str, default='Test',
                    help='training dataset (default: Test)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
#                     help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 加载数据集
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
# #下载数据集及预处理
# kwargs = {'num_workers': 0, 'pin_memory': False} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('./data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.Pad(4),
#                        transforms.RandomCrop(32),
#                        transforms.RandomHorizontalFlip(),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=args.test_batch_size, shuffle=True, **kwargs)


#微调再训练
if args.refine:
    checkpoint = torch.load(args.refine)
    model = vgg(cfg=checkpoint['cfg'])
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print('PruneModel Loading Successful!')
else:
    model = vgg()   #首先 构建基本的vgg模型
if args.cuda:
    model.cuda()    #用GPU来运行

opt = optim.Adam(model.parameters(),lr = 0.001,betas=(0.9, 0.999),eps=1e-09,weight_decay=0,amsgrad=False)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)   #指定一个优化器来跑当前的模型

# #当前位置继续训练
# if args.resume:
#     if os.path.isfile(args.resume):
#         print("=> loading checkpoint '{}'".format(args.resume))
#         checkpoint = torch.load(args.resume)
#         args.start_epoch = checkpoint['epoch']
#         best_prec1 = checkpoint['best_prec1']
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
#               .format(args.resume, checkpoint['epoch'], best_prec1))
#     else:
#         print("=> no checkpoint found at '{}'".format(args.resume))

# 更新BN时加入了L1正则化
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):#正则化的过程，并将梯度更新加入。
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1 大于0为1 小于0为-1 0还是0

def train(epoch):
    model.train()
    loss = nn.CrossEntropyLoss()
    loss_train = 0
    trainsamples = 0
    correct = 0
    for batch_idx,(data,target) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            data,target = data.cuda(), target.cuda()
            data,target = Variable(data), Variable(target)
        target = target.view(data.shape[0])
        x = model(data)
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
        opt.step()
        opt.zero_grad()
        trainsamples += data.shape[0]

        if (batch_idx+1) % 50  == 0:
            print('Train Epoch{}:[{}/{}({:.0f})%],Loss:{:.6f},Accuracy: {}/{}({:.3f}%)'.format(epoch,trainsamples,train_dataset.__len__(),
                                                                               100*trainsamples/train_dataset.__len__(),ss.item(),
                                                                                               correct,train_dataset.__len__(),
                                                                                               100.*correct/train_dataset.__len__()))
    print("\nTrain set: Average loss: {:.4f}".format(loss_train.item()/50.))
    print('Accuracy:{}/{}({:.3f}%)'.format(train_dataset.__len__(), correct,
                                                   100. * correct / train_dataset.__len__()))

            # print("EER: ",eer)
    del data,target
    gc.collect()
    torch.cuda.empty_cache()

# def train(epoch):
#     model.train()   #读取模型
#     for batch_idx, (data, target) in enumerate(train_dataloader):   #遍历数据
#         if args.cuda:   #传到CUDA中
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)    #向前传播
#         loss = F.cross_entropy(output, target)  #计算损失
#         loss.backward() #反向传播
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:  #过了几个epoch后的可视化
#             print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_dataloader.dataset),
#                 100. * batch_idx / len(train_dataloader), loss.item()))

def test():
    model.eval()
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
            x = model.forward(data)
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



    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.3f}%\n".format(test_loss.item()/50.,correct,test_dataset.__len__(),
                                                                                    100.*correct/test_dataset.__len__()))
    return correct / float(len(test_dataloader.dataset))
# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_dataloader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     test_loss /= len(test_dataloader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
#         test_loss, correct, len(test_dataloader.dataset),
#         100. * correct / len(test_dataloader.dataset)))
#     return correct / float(len(test_dataloader.dataset))

#保存当前的模型
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

best_prec1 = 0.
for epoch in range(1, 3):
    train(epoch)
    prec1 = test()
# for epoch in range(args.start_epoch, args.epochs):
#     if epoch in [args.epochs*0.5, args.epochs*0.75]:
#         for param_group in opt.param_groups:
#             param_group['lr'] *= 0.1
#     train(epoch)
#     prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': opt.state_dict(),
    }, is_best)
