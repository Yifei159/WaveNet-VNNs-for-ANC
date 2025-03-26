import torch
from networks import WaveNet_VNNs
import json
from utils import fir_filter ,SEF
from loss_function import dBA_Loss, NMSE_Loss
import json
from scipy.io import loadmat
from Dataloader import TestDataset
from torch.utils.data import DataLoader
import numpy as np

## load Pri and Sec path
Pri, Sec = (torch.tensor(loadmat(f, mat_dtype=True)[k]).squeeze() for f, k in [('pri_channel.mat', 'pri_channel'), ('sec_channel.mat', 'sec_channel')])

## load model
config = json.load(open('config.json', 'r'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveNet_VNNs(config).eval().to(device) 
model.load_state_dict(torch.load("trained_model/WaveNetVNNs_linear.pth"))

## load data
testdata = TestDataset()
print(len(testdata))    
test_loader = DataLoader(testdata, batch_size=1, shuffle=False, drop_last=True)

## inference
total_test_dBA = []  
total_test_NMSE = []  
test_dBA = []
test_NMSE = []
for i,datas in enumerate(test_loader,0):
    test_input=datas.to(device)
    test_target=fir_filter(Pri.to(device),test_input) 
    with torch.no_grad():
      test_outputs = model(test_input)
      test_outputs1 = SEF(test_outputs, etafang=0)
    test_dn = fir_filter(Sec.to(device),test_outputs1)
    test_en = test_dn + test_target
    dBA = dBA_Loss(fs = 16000, nfft = 512, f_up=16000/2)(test_en.squeeze(0)) - dBA_Loss(fs = 16000, nfft = 512, f_up=16000/2)(test_target.squeeze(0))
    NMSE = 10*torch.log10(torch.sum((test_en.squeeze())**2)/torch.sum((test_target.squeeze())**2))
    total_test_dBA .append(dBA.item())
    total_test_NMSE .append(NMSE.item())

test_dBA.append(np.mean(total_test_dBA ))
test_NMSE.append(np.mean(total_test_NMSE ))
print(test_dBA,test_NMSE) 
