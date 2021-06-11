import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
import torchvision
from torchvision import transforms
toImage = transforms.ToPILImage()
from PIL import Image
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


def trainAndReleaseGrads(model,trainData,stopIter,lossCutoff,crit,opt):
    """
    Trains the model for a small number of iterations, then returns
    the gradient and model
    """
    model.train()
    for idx, (features,y) in enumerate(trainData):
        opt.zero_grad()
        out = model(features)
        #turn to one hot
        labels = torch.zeros(10)
        labels[y[0]] = 1
        loss = crit(out,labels)
        #if we are done, release gradient
        if idx >= stopIter and loss.item() <= lossCutoff:
            gradient = autograd.grad(loss,model.parameters())
            trueGrad = list((_.detach().clone() for _ in gradient))
            finLoss = loss.item()
            return finLoss,model,trueGrad
        lossVal = loss.item()
        loss.backward()
        opt.step()

#must use this as pytorch does not support one hot for cat crosentropy
def cross_entropy_for_onehot(pred, target):
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def runTrial(model,trainData,stopIter,lossCutoff,crit,opt,nextIter):
    finLoss,model,grad = trainAndReleaseGrads(model,trainData,stopIter,lossCutoff,crit,opt)
    x = torch.randn(1,1,28,28).requires_grad_(True)
    y = torch.randn(1,10).requires_grad_(True)
    res = []
    optimizer = torch.optim.LBFGS([x,y])
    for i in range(nextIter):
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
        optimizer.step(closure)
        cur = closure()
        res.append((cur.item(),x))
    return {"originalLoss":finLoss, "leakResults":res}


def normalizeImage(img):
    mi = min(img.flatten()).item()
    img = torch.add(img,-1*mi)
    ma = max(img.flatten()).item()
    img = img/ma
    return img

def displayImage(reg):
    im = toImage(torch.reshape(reg,(28,28)))
    display(im.resize((5 * 28, 5 * 28), Image.NEAREST))
