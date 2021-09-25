import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

def read_txt(path):
    
    ims, labels = [], []
    root_path='dataset/'
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split("\t")
            ims.append(os.path.join(root_path,im))
            labels.append(os.path.join(root_path,label))
    return ims, labels

class Dataset(Dataset):
    def __init__(self, txtpath, transform):
        super().__init__()
        self.ims, self.labels = read_txt(txtpath)
        self.transform = transform

    def __getitem__(self, index):
        
        im_path = self.ims[index]
        label_path = self.labels[index]

        image = Image.open(im_path)
        image = self.transform(image).float().cuda()
        label = torch.from_numpy(np.asarray(Image.open(label_path), dtype=np.int32)).long().cuda()

        return image, label

    def __len__(self):
        return len(self.ims)