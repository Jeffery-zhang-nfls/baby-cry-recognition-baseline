#! python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import math
import numpy as np
import torch.nn.functional as F


class AMSoftmax(nn.Module):
    def __init__(self):
        super(AMSoftmax, self).__init__()

    # s = 10.0  m = 0.5
    # s = 10.0  m = 0.2
    def forward(self, input, target, scale=20.0, margin=0.0):  # original s = 10.0 m = 0.35 paper s = 30.0 m = 0.35

        # print("input: ", input.type(), input.dtype, input.size(), input.requires_grad)
        # self.it += 1
        cos_theta = input        
        target = target.view(-1, 1)  # size=(B,1)
        # print("target: ", target.type(), target.dtype, target.size(), target.requires_grad)         
                
        index = cos_theta.data * 0.0  # size=(B,Classnum)        

        index.scatter_(1, target.data.view(-1, 1), 1)        
        index = index.byte()
        # index = index.bool()
        index = Variable(index)
        # print("index: ", type(index), index.dtype, index.size(), index.requires_grad, index.size(0))
       
        output = cos_theta * 1.0  # size=(B,Classnum) 
        # print("output: ", type(output), output.dtype, output.size(), output.requires_grad)
        # print("output: ", type(output), output.dtype, output.size(), output.requires_grad)
        # print("output[index]: ", type(output[index]), output[index].dtype, output[index].size(), output[index].requires_grad)
        # output[index] -= margin
        output = output - margin
        output = output * scale

        # print("output: ", output.size())
        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean() 

        return loss

