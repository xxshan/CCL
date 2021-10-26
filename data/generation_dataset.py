import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def getfilename(outfile, img_root, phase):
    plist = glob.glob(os.path.join(img_root, phase) + '/PNEUMONIA/*.*') 
    nlist = glob.glob(os.path.join(img_root, phase) + '/NORMAL/*.*') 
    file = open(outfile,"w")
    for i in range(len(plist)):
        name = plist[i].split('/')
        name = name[-1]
        file.write(name+ ",")
        file.write('1' + "\n")
    for i in range(len(nlist)):
        name = nlist[i].split('/')
        name = name[-1]
        file.write(name+ ",")
        file.write('0' + "\n")
    file.close()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def get_transform(params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    preprocess = 'resize_and_crop'
    no_flip = False
    load_size = 286
    crop_size = 256
    
    transform_list = []
    
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)))

    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class GenDataset(data.Dataset):
    def __init__(self, img_root, phase, file_root):
        super(GenDataset, self).__init__()

        self.dir_img = os.path.join(img_root, phase)  
        self.dir_name = os.path.join(os.getcwd(), file_root+phase+'.txt') 
        self.transform = get_transform()
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        flag = os.path.exists(self.dir_name)
        if(flag == False):
            getfilename(self.dir_name, img_root, phase)
            
    def __getitem__(self, index):
        
        
        with open(self.dir_name) as file:
            data_list = file.readlines()
            data_list = [x.strip() for x in data_list]

        img_name, label = data_list[index].strip().split(',')
        
        if(label == '1'):
            img_path = os.path.join(self.dir_img, 'PNEUMONIA', img_name)
        elif(label == '0'):
            img_path = os.path.join(self.dir_img, 'NORMAL', img_name)
        img = Image.open(img_path).convert('RGB')
        #img = self.transform(img)
        label = int(label)
        #label = [label]
        '''label = torch.IntTensor(label)
        label = label.long()'''
        return img, label

    def __len__(self):
        myfile = open(self.dir_name)
        img_num = len(myfile.readlines())
        return img_num




