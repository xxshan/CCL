"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torchvision
import numpy as np
from PIL import Image
import os
import sys
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data.deal_dataset import DealDataset 
from data.deal_dataset import FakeDataset
from torchvision import transforms
import torchvision.datasets
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler
import logging
import datetime
import time
import random
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value.
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_all_dataloader(opt, size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augs = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    data = ImageFolder(os.path.join(opt.data_root, 'train'), transform=augs)

    num_class0 = [data.imgs[x][1] for x in range(len(data))].count(0)
    num_class1 = [data.imgs[x][1] for x in range(len(data))].count(1)
    train_targets = [data.imgs[x][1] for x in range(len(data))]
    class_sample_counts = [num_class0, num_class1]
    weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[train_targets]
    weights_sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    realA_loader = torch.utils.data.DataLoader(dataset=data, batch_size=opt.batch_size, shuffle=False, sampler = weights_sampler, drop_last=True)
    
    data_test = ImageFolder(os.path.join(opt.data_root, 'test'), transform=augs)
    testA_loader = torch.utils.data.DataLoader(data_test, batch_size=len(data_test), drop_last=True) 
    print((torchvision.datasets.ImageFolder(os.path.join(opt.data_root, 'train'), transform=augs)).classes)
    
    cross_data = ImageFolder(os.path.join(opt.cross_data_root, 'test'), transform=augs)
    cross_arr = np.arange(len(cross_data))
    np.random.shuffle(cross_arr)
    cross_data_test = torch.utils.data.Subset(cross_data, list(cross_arr))
    realB_loader = torch.utils.data.DataLoader(cross_data_test, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
    del cross_arr
    del augs
    del data_test
    del cross_data
    return realA_loader, testA_loader, realB_loader

def get_two_dataloader(opt, size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augs = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    data = ImageFolder(os.path.join(opt.data_root, 'test'), transform=augs)

    num_class0 = [data.imgs[x][1] for x in range(len(data))].count(0)
    num_class1 = [data.imgs[x][1] for x in range(len(data))].count(1)
    train_targets = [data.imgs[x][1] for x in range(len(data))]
    class_sample_counts = [num_class0, num_class1]
    weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[train_targets]
    weights_sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    realA_loader = torch.utils.data.DataLoader(dataset=data, batch_size=opt.batch_size, shuffle=False, sampler = weights_sampler, drop_last=True)

    print((torchvision.datasets.ImageFolder(os.path.join(opt.data_root, 'test'), transform=augs)).classes)
    
    cross_data = ImageFolder(os.path.join(opt.cross_data_root, 'test'), transform=augs)
    cross_arr = np.arange(len(cross_data))
    np.random.shuffle(cross_arr)
    cross_data_test = torch.utils.data.Subset(cross_data, list(cross_arr))
    realB_loader = torch.utils.data.DataLoader(cross_data_test, batch_size=opt.batch_size, shuffle=True, drop_last=True)#, drop_last=True

    del cross_arr
    del augs
    del cross_data
    return realA_loader, realB_loader
    
def save_model(savepath, best_acc, epoch, net, optimizer_model, name):
        state = {
            "epoch": epoch + 1,
            "model_state": net.state_dict(),
            "optimizer_state": optimizer_model.state_dict(),
            "best_acc": best_acc,
        }
        ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
        ts = ts.replace(":", "_").replace("-", "_")
        ts = ts + '_acc=' + str(best_acc)
        ts = ts + name
        save_path = os.path.join(
            os.getcwd(), savepath, "best_FINETUNE_{}.pkl".format(ts),
        )
        torch.save(state, save_path)    

def generate_gandataset_realAB(realA_data, realA_labels, realB_data, realB_labels):
    data_loader = DealDataset(realA_data, realA_labels, realB_data, realB_labels)
    return data_loader

def get_cross_dataloader(data, labels):
    data_loader = FakeDataset(data, labels)
    return data_loader

def generate_finetune_dataset(realA_data, realA_labels, rec_A, idt_B):# rec_A and idt_B are from real_A
    #deal_dataset = TensorDataset()
    rec_A = rec_A.data
    idt_B = idt_B.data
    cnt0_A, cnt1_A = 0, 0
    label0 = torch.tensor([0 for x in range(realA_labels.size(0))])
    label1 = torch.tensor([1 for x in range(realA_labels.size(0))])             
    cnt0_A = (realA_labels == label0).float().sum().cpu().item()
    cnt1_A = (realA_labels == label1).float().sum().cpu().item()
    cnt0_A = int(cnt0_A)
    cnt1_A = int(cnt1_A)
    #random.seed(10) # set the seeds to make the results of each sample all the same
    index_1 = torch.nonzero(realA_labels) # find the index of label_1 
    index_1 = torch.squeeze(index_1)
    index_0 = (realA_labels==0).nonzero() # find the index of label_0
    index_0 = torch.squeeze(index_0) 
    if(cnt0_A < cnt1_A): # need to supplement 0_A
      Sr = int((cnt1_A - cnt0_A)/2)
      Si = int(cnt1_A - cnt0_A - Sr)
      if (Sr > 0 and index_0.numel() > 0):
          if(index_0.size() == torch.Size([])):
              index_0 = torch.unsqueeze(index_0, 0) 
          Sr_data_index = random.choices(index_0, k = Sr)
      else:
          Sr_data_index = torch.Tensor()
      if (Si > 0 and index_0.numel() > 0):
          if(index_0.size() == torch.Size([])):
              index_0 = torch.unsqueeze(index_0, 0)
          Si_data_index = random.choices(index_0, k = Si)
      else:
          Si_data_index = torch.Tensor()
      Sr_labels = torch.tensor([0 for i in range(Sr)])
      Si_labels = torch.tensor([0 for i in range(Si)])
    elif(cnt0_A > cnt1_A): # may need to supplement 1_A
      Sr = int((cnt0_A - cnt1_A)/2)
      Si = int(cnt0_A - cnt1_A - Sr)
      if (Sr > 0 and index_1.numel() > 0):
          if(index_1.size() == torch.Size([])):
              index_1 = torch.unsqueeze(index_1, 0)
          Sr_data_index = random.choices(index_1, k = Sr)
      else:
          Sr_data_index = torch.Tensor()
      if (Si > 0 and index_1.numel() > 0):
          if(index_1.size() == torch.Size([])):
              index_1 = torch.unsqueeze(index_1, 0)
          Si_data_index = random.choices(index_1, k = Si)
      else:
          Si_data_index = torch.Tensor()
      Sr_labels = torch.tensor([1 for i in range(Sr)])
      Si_labels = torch.tensor([1 for i in range(Si)])
    if(cnt0_A != cnt1_A):
      Sr_data_index = torch.tensor(Sr_data_index).type(torch.long)
      Si_data_index = torch.tensor(Si_data_index).type(torch.long)
      Sr_data = rec_A[Sr_data_index]
      Si_data = idt_B[Si_data_index]
      Sr_data = torch.tensor(Sr_data) 
      Si_data = torch.tensor(Si_data)
      Sr_data = torch.squeeze(Sr_data)
      Si_data = torch.squeeze(Si_data)
      if(len(list(Sr_data.size()))==3): # when there is only one element to be added
        Sr_data = torch.unsqueeze(Sr_data, 0)
      if(len(list(Si_data.size()))==3): # when there is only one element to be added
        Si_data = torch.unsqueeze(Si_data, 0)
      S_A_data = torch.cat((Sr_data, Si_data), 0)
      S_A_data = torch.tensor(S_A_data)  
      S_A_data = torch.squeeze(S_A_data)
      if(len(list(S_A_data.size()))==3): # when there is only one element to be added
        S_A_data = torch.unsqueeze(S_A_data, 0)
      A_data = torch.cat((realA_data, S_A_data), 0)
      S_A_labels = torch.cat((Sr_labels, Si_labels), 0)
      A_labels = torch.cat((realA_labels, S_A_labels), 0)
      del Sr_data
      del Si_data
      del S_A_data
    else:
      A_data = realA_data
      A_labels = realA_labels
    del rec_A
    del idt_B
    return A_data, A_labels
    
def evaluate_accuracy(data_loader, net, alpha=None, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    cnt0 = 0
    cnt1 = 0
    with torch.no_grad():
        for X, y in data_loader:
            net.eval() 
            if(alpha is None):
                prediction = net(X.to(device)).argmax(dim=1)
            else:
                prediction = net(X.to(device), alpha=alpha)[0].argmax(dim=1)
            comp = (prediction == y.to(device))
            acc_sum += comp.float().sum().cpu().item()
            index = torch.nonzero(comp==True)
            index = torch.squeeze(index)
                
            label0 = torch.tensor([0 for x in range(y[index].size(0))])
            label1 = torch.tensor([1 for x in range(y[index].size(0))])
                 
            cnt0 += (y[index] == label0).float().sum().cpu().item()
            cnt1 += (y[index] == label1).float().sum().cpu().item()

            net.train()                
            n += y.shape[0]
    del prediction       
    return acc_sum / n, cnt0, cnt1
    
def evaluate_fake_accuracy(data, labels, net, device): 
    acc_sum, n = 0.0, 0
    cnt0 = 0
    cnt1 = 0
    data = data.to(device)
    labels = labels.to(device)
    outputs = net(data)
    
    prediction = outputs.argmax(dim=1)
    comp = (prediction == labels)
    acc_sum += comp.float().sum().cpu().item()
    index = torch.nonzero(comp==True)
    index = torch.squeeze(index)
                
    label0 = torch.tensor([0 for x in range(labels[index].size(0))])
    label1 = torch.tensor([1 for x in range(labels[index].size(0))])
                 
    cnt0 += (labels[index] == label0).float().sum().cpu().item()
    cnt1 += (labels[index] == label1).float().sum().cpu().item()

    net.train()              
    n += labels.shape[0]
    
    del prediction
    del data
    del labels
    del outputs
    return acc_sum / n, cnt0, cnt1
    
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_logger(logdir):
    logger = logging.getLogger("ptsxx")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict 

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
