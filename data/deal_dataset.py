import os
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from torch.utils import data
from torchvision import transforms

class DealDataset(data.Dataset):
    def __init__(self, realA_data, realA_labels, realB_data, realB_labels):
        self.A = realA_data
        self.B = realB_data
        self.A_labels = realA_labels
        self.B_labels = realB_labels
        self.A_size = len(realA_data)  # get the size of dataset A
        self.B_size = len(realB_data)  # get the size of dataset B
       
    def __getitem__(self, index):
        A_img = self.A[index]
        B_img = self.B[index]
        A_labels = self.A_labels[index]
        B_labels = self.B_labels[index]
        
        return {'A': A_img, 'B': B_img, 'A_labels': A_labels, 'B_labels': B_labels}

    def __len__(self):
        return max(self.A_size, self.B_size)

class FakeDataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.size = len(data)  # get the size of dataset
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([
          transforms.Resize([256,256]),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          self.normalize
        ])
       
    def __getitem__(self, index):
        img_tensor = self.data[index]
        #img_tensor = self.preprocess(img)
        labels = self.labels[index]
        
        return img_tensor, labels

    def __len__(self):
        return self.size
