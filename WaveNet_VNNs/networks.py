import torch.nn.functional as F
import os
import logging
import numpy as np
import pandas as pd
import utils as utils
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm

class Causal_Conv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,bias,dilation=1):
        super(Causal_Conv1d,self).__init__()
        self.kernel_size=kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=0,bias = bias,dilation=dilation)
    def forward(self,x):
        x = F.pad(x,(self.dilation*(self.kernel_size-1),0),mode='constant',value=0)
        
        x = self.conv(x)
        return x 

class VNN2(nn.Module):
    def __init__(self, config):
        super(VNN2, self).__init__()
        self.config = config
        self.Q2 = config['VNN2']['Q2']
        self.out_channel = config['VNN2']['conv1d']['out'][0]
        self.conv1 = Causal_Conv1d(self.config['VNN2']['conv1d']['input'][0], self.config['VNN2']['conv1d']['out'][0], 
                                   self.config['VNN2']['conv1d']['kernel'][0], stride=1, bias = False)
        self.conv2 = Causal_Conv1d(self.config['VNN2']['conv1d']['input'][1], 2*self.Q2 * self.config['VNN2']['conv1d']['out'][1], 
                                   self.config['VNN2']['conv1d']['kernel'][1], stride=1, bias = False)
    def forward(self, x):
        linear_term = self.conv1(x) # first-order

        x2 = self.conv2(x)
        x2_mul = torch.mul(x2[:,0:self.Q2*self.out_channel,:],x2[:,self.Q2*self.out_channel:2*self.Q2*self.out_channel,:])
        x2_add = torch.zeros_like(linear_term)
        for q in range(self.Q2):
            x2_add = torch.add(x2_add, x2_mul[:,(q*self.out_channel):((q*self.out_channel)+(self.out_channel)),:])
        quad_term  = x2_add         # second-order
        x = torch.add(linear_term, quad_term )
        return x.squeeze()


"Residual_block"
class dilated_residual_block(nn.Module):
    
    def __init__(self, dilation, config):
        super().__init__()
        self.dilation =  dilation
        self.config = config
        self.conv1 = Causal_Conv1d(self.config['WaveNet']['Resblock']['conv1d']['res'], 2*self.config['WaveNet']['Resblock']['conv1d']['res'],
                               kernel_size = self.config['WaveNet']['Resblock']['conv1d']['kernel'][0], stride=1, bias = False, dilation = self.dilation)
        self.conv2 = Causal_Conv1d(self.config['WaveNet']['Resblock']['conv1d']['res'], 
                               self.config['WaveNet']['Resblock']['conv1d']['res'] + self.config['WaveNet']['Resblock']['conv1d']['skip'],
                               self.config['WaveNet']['Resblock']['conv1d']['kernel'][1], stride=1, bias = False)

    def forward(self, data_x ):
        original_x = data_x
        data_out = self.conv1(data_x)
        data_out_1 = utils.slicing(data_out, slice(0, self.config['WaveNet']['Resblock']['conv1d']['res'],1), 1) 
        data_out_2 = utils.slicing(data_out, slice(self.config['WaveNet']['Resblock']['conv1d']['res'], 
                                                    2*self.config['WaveNet']['Resblock']['conv1d']['res'],1), 1)
        tanh_out = torch.tanh(data_out_1)
        sigm_out = torch.sigmoid(data_out_2)
        data_x = tanh_out*sigm_out
        data_x = self.conv2(data_x)
        res_x = utils.slicing(data_x, slice(0, self.config['WaveNet']['Resblock']['conv1d']['res'],1), 1)
        skip_x = utils.slicing(data_x, slice(self.config['WaveNet']['Resblock']['conv1d']['res'], 
                                self.config['WaveNet']['Resblock']['conv1d']['res']+self.config['WaveNet']['Resblock']['conv1d']['skip'],1), 1)
        res_x = res_x + original_x
        return res_x, skip_x

# Wavenet-VNNs for ANC
class WaveNet_VNNs(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_stacks = self.config['WaveNet']['num_stacks']
        if type(self.config['WaveNet']['dilations']) is int:
            self.dilations = [2 ** i for i in range(0, self.config['WaveNet']['dilations'] + 1)]
        elif type(self.config['WaveNet']['dilations']) is list:
            self.dilations = self.config['WaveNet']['dilations']
        
        self.num_residual_blocks = len(self.dilations) * self.num_stacks
        
        # Layers in the model
        "class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)"
        self.conv1 = Causal_Conv1d(self.config['WaveNet']['conv']['input'][0], self.config['WaveNet']['conv']['out'][0], 
                                   self.config['WaveNet']['conv']['kernel'][0], stride=1, bias = False)
        
        self.conv2 = Causal_Conv1d(self.config['WaveNet']['conv']['input'][1], self.config['WaveNet']['conv']['out'][1],
                                   self.config['WaveNet']['conv']['kernel'][1], stride=1, bias = False)
        
        self.conv3 = Causal_Conv1d(self.config['WaveNet']['conv']['input'][2], self.config['WaveNet']['conv']['out'][2],
                                   self.config['WaveNet']['conv']['kernel'][2], stride=1, bias = False)
        
        self.conv4 = Causal_Conv1d(self.config['WaveNet']['conv']['input'][3], self.config['WaveNet']['conv']['out'][3], 
                                   self.config['WaveNet']['conv']['kernel'][3], stride=1, bias = False)
        self.dilated_layers = nn.ModuleList([dilated_residual_block(dilation, self.config) for dilation in self.dilations])
        self.VNN = VNN2(self.config)

    def forward(self, x):
        data_input = x
        data_expanded = data_input
        data_out = self.conv1(data_expanded)
        skip_connections = []
        for _ in range(self.num_stacks):
            for layer in self.dilated_layers:
                data_out, skip_out = layer(data_out)
                if skip_out is not None:
                    skip_connections.append(skip_out)
        
        data_out = torch.stack( skip_connections, dim = 0).sum(dim = 0)
        data_out = F.tanh(data_out)
        data_out = self.conv2(data_out)
        data_out = F.tanh(data_out)
        data_out = self.conv3(data_out)
        data_out = F.tanh(data_out)
        data_out = self.conv4(data_out)
        data_out = F.tanh(data_out)
        data_out = self.VNN(data_out).squeeze()
        return data_out



