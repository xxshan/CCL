import os
import sys
import torch
import numpy as np
from torch import nn
    
def calculate_single_prediction(data, net, device, alpha=None): 
    cal_prob = nn.Softmax()
    #data = torch.unsqueeze(data, 0)
    data = data.to(device)
    if(alpha is None):
        outputs = net(data)
        prediction = outputs.argmax(dim=1)
    else:
        outputs = net(data, alpha=alpha)[0]
        prediction = outputs.argmax(dim=1)
    prob = cal_prob(outputs)
                 
    del data
    del outputs
    return prediction, prob

def voting_prediction(use_gpu, labels, pre_ff, prob_ff, pre_rc, prob_rc):
    acc_sum, n, cnt0, cnt1 = 0.0, 0, 0, 0
    num = len(labels)
    prediction = []
    pseudo = []
    for i in range(num):
        if(pre_ff[i] == pre_rc[i]):
            pre_fix = pre_ff[i]
        else:
            prob_list = np.array(torch.cat((prob_ff[i], prob_rc[i]),0))
            prob_0 = (np.sum(prob_list, 0))[0]
            prob_1 = (np.sum(prob_list, 0))[1]
            pre_fix = 0 if prob_0>prob_1 else 1
        prediction.append(pre_fix) 

    prediction = torch.tensor(prediction)
    comp = (prediction == torch.tensor(labels))
    acc_sum += comp.float().sum().cpu().item()
    index = torch.nonzero(comp==True)
    index = torch.squeeze(index)         
    label0 = torch.tensor([0 for x in range(labels[index].size(0))])
    label1 = torch.tensor([1 for x in range(labels[index].size(0))])         
    cnt0 += (labels[index] == label0).float().sum().cpu().item()
    cnt1 += (labels[index] == label1).float().sum().cpu().item()             
    n += labels.shape[0]
    
    del prediction
    del pre_ff
    del pre_rc
    del comp
    del label0
    del label1
    return acc_sum / n, cnt0, cnt1
    