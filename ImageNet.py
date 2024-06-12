import os
from torch.utils.data import Dataset

class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        super().__init__()
        self.samples = []
        self.targets = []
        self.class_indices = {}

