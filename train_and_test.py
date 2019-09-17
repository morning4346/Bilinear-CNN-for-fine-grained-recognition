# coding:utf-8
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from data import *
from torch.autograd import Variable
import torch
import time
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from HBP_fc import HBP
import visdom
vis = visdom.Visdom(env=u'bilinear_4',use_incoming_socket=False)
from bili import  BCNN
device = torch.device("cuda:0,1,2" if torch.cuda.is_available() else "cpu")
from torchvision.models import vgg16

def train1(Net, LR, dataset):
        net = Net.to(device)
        print("Start Training!")
        with open('result.txt', 'a') as acc_f:
            for epoch in range(pre_epoch, EPOCH):
                start = time.time()
                criterion = nn.CrossEntropyLoss().to(device)
                optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-8)
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.1,patience=3
                                                    ,verbose=True,threshold=1e-4)
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                print("LR = %f" % LR)
                for i, data in enumerate(dataset[0]):
                    imgs, lables = data
                    imgs, lables = imgs.to(device), lables.to(device)
                    optimizer.zero_grad()
                    outputs = net(imgs)
                    loss = criterion(outputs,lables)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item()
                    _, predicteds = torch.max(outputs.data, 1)
                    total += lables.size(0)
                    correct += float((predicteds == lables).sum())
                    if i % 10 == 9:
                        print(
                            '[epoch:%d] Loss: %.06f ,accuracy: %.3f%%,total_num = %d, accurate_num= %d'
                            % (epoch + 1, sum_loss / (i + 1),
                               100.0 * correct / total, total, correct))
                    x=torch.Tensor([epoch+1])
                    y=torch.Tensor([loss.item()])
                    vis.line(X=x, Y=y, win='polynomial', opts={'title':'epoch/loss','xlabel':'epoch','ylabel':'loss'},update='append' if epoch+1 > 0 else None)
                if epoch >= 0 :
                    print("Waiting Test!")
                    with torch.no_grad():
                        net.eval()
                        correct = 0.0
                        total = 0
                        for data in dataset[1]:
                            images, lables = data
                            images, lables = images.to(device), lables.to(device)
                            outputs = net(images)
                            _, predicteds = torch.max(outputs.data, 1)
                            total += lables.size(0)
                            correct += float((predicteds == lables).sum())
                        test_accuracy=100.0*correct/total
                        print('测试分类准确率为: %.3f%%,' %(test_accuracy))
                        acc_f.write("EPOCH=%03d,  accuracy: %.3f%%,accurate_num=%03d,total_num=%03d"
                                    % (epoch + 1, 100.0 * correct / total,correct,total))
                        acc_f.write('\n')
                        acc_f.flush()
                        print('save model.....')
                        if epoch>= 10 :
                            torch.save(net.state_dict(), "model_09_02/" + '%d.pkl' %(epoch + 1))
                scheduler.step(test_accuracy)  # change lr,by epoch
                end = time.time()
                print("每epoch花费时间：%s s" % (-start + end))

CUB_Birds="/data/dataset/CUB_Birds_200/CUB_200_2011/"

if __name__ == "__main__":

    lr = 1.0
    pre_epoch = 100
    EPOCH = 120
    batch_size = 64
    size = [448, 448]
    data = images(batch_size, size, CUB_Birds)
    net=BCNN(200,True)
    state_dict="/home/lyh2017/code/bili/bilinear_09_02/model_09_02/70.pkl"
    pretrained_dict = torch.load(state_dict)
    model_dict = net.state_dict()
    pretrained_dict = {k[7: ]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print(net)
    net = nn.DataParallel(net, device_ids=[0,1,2])
    train1(net, lr, data)

