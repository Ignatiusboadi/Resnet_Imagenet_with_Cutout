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
        with open(os.path.join(root, "ILSVRC/Data/CLS-LOC", split)) as json_file:
            self.val_to_syn = json.load(json_file)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            # looping through images folders for training images set
            if split == "train" and entry != ".DS_Store":  # skipping the .DS_Store file in mac folders
                class_index = entry
                target = self.class_indices[class_index]
                syn_folder = os.path.join(samples_dir, class_index)
                for sample in os.lisdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)

