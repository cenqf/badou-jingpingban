# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.linear=nn.Linear(input_size,1)
        self.activation=torch.sigmoid
        self.loss=nn.functional.mse_loss
    def forward(self,x,y=None):
        x=self.linear(x)
        y_pred=self.activation(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

def build_sample():
    x=np.random.random(5)
    if x[0]>x[1] :
        return x,0
    elif x[2]>x[3]:
        return x,1
    else:
        return x,2

def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x,y =build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X),torch.FloatTensor(Y)


def evaluate(model):
    model.eval()
    test_smaple_num=100
    x,y=build_dataset(test_smaple_num)
    correct,wrong=0,0
    with torch.no_grad():
        y_pred=model(x)
        for y_p,y_t in zip(y_pred,y):
            if int(y_t)in [0,1] and int(y_p)<=0.5:
                correct+=1
            else: wrong+=1
    print("正确的个数为：%d,正确率为%f"%(correct,correct/(correct+wrong)))
    return correct/(correct+wrong)



def main():
    epoch_num=20
    train_sample=100
    batch_size=20
    input_size=5
    learning_rate=0.0001
    model=TorchModel(input_size)
    optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
    log=[]
    train_x,train_y=build_dataset(train_sample)
    # print(train_x,train_y)
    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch_index in range(train_sample//batch_size):
            x=train_x[batch_index*batch_index:(batch_index+1)*batch_size]
            y=train_y[batch_index*batch_index:(batch_index+1)*batch_size]
            loss=model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
            print("watch_loss:",watch_loss)
        acc=evaluate(model)
        log.append([acc,np.mean(watch_loss)])
    print(log)
if __name__=="__main__":
    main()





