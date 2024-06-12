import json
import os
from torch.utils.data import Dataset


class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        super().__init__()
        self.samples = []  # list of images
        self.targets = []  # list of labels of images
        self.class_indices = {}  # dictionary mapping class_ids and a class index
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as json_file:
            class_index_file = json.load(json_file)
            for class_id, class_items in class_index_file.items():
                self.class_indices[class_items[0]] = int(class_id)

