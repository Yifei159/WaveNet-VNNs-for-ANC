import os
import torch
from torch.utils.data import  Dataset
import torchaudio

#TrainDataset
class TrainDataset(Dataset):
    def __init__(self, dimension=16000*3, stride=16000*3, data_path="/data/ssd0/lu.bai/", cache_file="/data/ssd0/lu.bai/prepare_train_data.pth"):
        super(TrainDataset, self).__init__()
        self.dimension = dimension
        self.stride = stride
        self.data_path = data_path
        self.cache_file = cache_file
        
        if os.path.exists(cache_file):
            # Load from cache if it exists
            self.wb_list = torch.load(cache_file)
            print("1")
        else:
            self.wb_list = []
            self.split()
            # Save the processed data to cache
            torch.save(self.wb_list, cache_file)
            print("2")
    
    def split(self):
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    wav, _ = torchaudio.load(os.path.join(root, file))
                    wav = wav.to(torch.float)
                    wav_length = wav.size(1)
                    if wav_length >= self.stride:
                        start_index = 0
                        while start_index + self.dimension <= wav_length:
                            self.wb_list.append(wav[:, start_index:start_index + self.dimension])
                            start_index += self.stride

    def __len__(self):
        return len(self.wb_list)

    def __getitem__(self, index):
        return self.wb_list[index]

#TestDataset
class TestDataset(Dataset):
    def __init__(self, dimension=16000*10, stride=16000*10, data_path="/data/ssd0/lu.bai/WaveNet_VNNs/test_dataset"):
        super(TestDataset, self).__init__()
        self.dimension = dimension
        self.stride = stride
        self.data_path = data_path
        self.wb_list = []
        self.split()

    def split(self):
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    wav, fs = torchaudio.load(os.path.join(root, file))
                    print(fs)
                    wav = wav.to(torch.float)
                    wav_length = wav.size(1)
                    if wav_length >= self.stride:
                        start_index = 0
                        while start_index + self.dimension <= wav_length:
                            self.wb_list.append(wav[:, start_index:start_index + self.dimension])
                            start_index += self.stride

    def __len__(self):
        return len(self.wb_list)

    def __getitem__(self, index):
        return self.wb_list[index]
