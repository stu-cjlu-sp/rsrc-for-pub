import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from data import STFTDataset, Radioml_18_Dataset, Radioml_16_Dataset


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def get_dataframe(snr):
    csv_file_path = 'snr_data/output_data_snr_{}.csv'.format(str(snr))
    data_frame = pd.read_csv(csv_file_path)
    return data_frame

def get_dataloader_2018a_IQ(batch_size=64):
    dataset_path = "/home/song/Datasets/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
    dataset2018a = Radioml_18_Dataset(dataset_path)
    data_loader_train = DataLoader(dataset2018a, batch_size=batch_size, sampler=dataset2018a.train_sampler)
    data_loader_val = DataLoader(dataset2018a, batch_size=batch_size, sampler=dataset2018a.val_sampler)
    data_loader_test = DataLoader(dataset2018a, batch_size=batch_size, sampler=dataset2018a.test_sampler)
    return data_loader_train, data_loader_val, data_loader_test

def datalodaer_2016a_IQ():
    data_path = "/home/song/Datasets/RML2016.10a_dict.pkl"
    dataset2016a = Radioml_16_Dataset(dataset_path=data_path)
    dataloader_train = dataset2016a.train_loader
    dataloader_val = dataset2016a.val_loader
    dataloader_test = dataset2016a.test_loader
    return dataloader_train, dataloader_val, dataloader_test

if __name__ == '__main__':
    data_frame = get_dataframe(snr=0)
    print(data_frame['Mod_Type'].astype('category').cat.categories)
    label_names = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']