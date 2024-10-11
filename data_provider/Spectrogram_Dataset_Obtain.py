import os, sys
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform

        self.samples = []
        self.labels = []
        class_names = os.listdir(data_root)
        class_names.sort()  # Sort following the file name
        for label, class_name in enumerate(class_names):
            class_path = os.path.join(data_root, class_name)
            sample_names = os.listdir(class_path)
            sample_names.sort(key=lambda x: int(x.split('_')[0]))  # Sort following the file name (number)
            for sample_name in sample_names:
                sample_path = os.path.join(class_path, sample_name)
                self.samples.append(sample_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        label = self.labels[idx]
        # Load 9 images (for one sample), resize, and convert to grayscale
        images = [Image.open(os.path.join(sample_path, f)) for f in os.listdir(sample_path)]
        images = [self.transform(image) for image in images]

        # fold into a tensor(9，32，35), eliminate the extra dim
        sample = np.stack(images)
        return sample