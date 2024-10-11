import torch
from torch.utils.data import Dataset

# Load Data
class MyDataset(Dataset):
    def __init__(self, time_series_data, graph_data, labels):
        self.time_series_data = time_series_data
        self.graph_data = graph_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        time_series_sample = torch.tensor(self.time_series_data[index].astype(float), dtype=torch.float32)
        graph_sample = torch.tensor(self.graph_data[index].astype(float), dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return time_series_sample, graph_sample, label