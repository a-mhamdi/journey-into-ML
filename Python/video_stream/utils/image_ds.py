from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

# ===================== TFDS ===================== #
class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        label = data_sample['labels']
        image = self.transform(data_sample['image'])
        return image, label
        
    def __len__(self):
        return len(self.dataset)
