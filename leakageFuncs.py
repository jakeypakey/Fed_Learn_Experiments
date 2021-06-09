import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
class Network(nn.Module):
    """
    Simple CNN model
    """
    def __init__(self):
        super(Network,self).__init__()
        self.layers = nn.Sequential(
        nn.Conv2d(1,32,kernel_size=5,stride=1,padding=2),nn.Sigmoid(),
        nn.MaxPool2d(2,1,padding=1),
        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),nn.Sigmoid(),
        nn.MaxPool2d(2,1,padding=1))
        self.fc = nn.Sequential(nn.Linear(57600,10))
    def forward(self,x):
        out = self.layers(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


def trainAndReleaseGrads(model,trainData,stopIter,crit,opt):
    """
    Trains the model for a small number of iterations, then returns
    the gradient and model
    """
    model.train()
    for idx, (features,y) in trainData:
        opt.zero_grad()
        out = model(features)
        #turn to one hot
        labels = torch.zeros(10)
        labels[y[0]] = 1
        loss = crit(out,labels)
        #if we are done, release gradient
        if idx > stopIter:
            gradient = autograd.grad(loss,model.parameters())
            trueGrad = list((_.detach().clone() for _ in gradient))
            return model,trueGrad
        lossVal = loss.item()
        loss.backward()
        opt.step()
        if idx%1000==0:
            print("Example: {}, loss: {}".format(idx,lossVal))

#https://github.com/mit-han-lab/dlg
def closure():
    optimizer.zero_grad()
    pred = model(x)
    oneHotY = F.softmax(y,dim=-1)
    modelLoss = crit(pred,oneHotY)
    leakGrad = autograd.grad(modelLoss,model.parameters(),create_graph=True)   
    distance = 0
    for gx,gy in zip(leakGrad,grad):
        distance += ((gx - gy)**2).sum()
    distance.backward()
    return distance

#must use this as pytorch does not support one hot for cat crosentropy
def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
