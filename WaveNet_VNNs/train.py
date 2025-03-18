import os
import toml
import torch
import random
import argparse
import json
import numpy as np
import torch.distributed as dist

from trainer import Trainer
from networks import WaveNet_VNNs
from Dataloader import TrainDataset
from loss_function import dBA_Loss, NMSE_Loss

net_config = json.load(open('config.json', 'r'))
seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic =True
from scipy.io import loadmat
Pri, Sec = (torch.tensor(loadmat(f, mat_dtype=True)[k]).squeeze() for f, k in [('pri_channel.mat', 'pri_channel'), ('sec_channel.mat', 'sec_channel')])

def run(rank, config, args):
    
    args.rank = rank
    args.device = torch.device(rank)
    
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12354'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

        train_dataset = TrainDataset()
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                                        **config['train_dataloader'], shuffle=False)
        
    else:
        train_dataset = TrainDataset()
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **config['train_dataloader'], shuffle=True)
        
    model = WaveNet_VNNs(net_config)   
    model.to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])

    trainer = Trainer(config=config, model=model,optimizer=optimizer, loss_func1=dBA_Loss(fs = 16000, nfft = 512, f_up=16000/2), loss_func2=NMSE_Loss(),
                      train_dataloader=train_dataloader, pri_channel=Pri, sec_channel=Sec, train_sampler=train_sampler, args=args)

    trainer.train()
    if args.world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='cfg_train.toml')
    parser.add_argument('-D', '--device', default='0', help='The index of the available devices, e.g. 0,1,2,3')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))
    config = toml.load(args.config)
    
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)