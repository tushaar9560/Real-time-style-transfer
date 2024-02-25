import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import *
from PIL import Image, ImageOps, ImageFile
import random
from glob import glob
from torchvision import transforms as transforms


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, ref_dir, finesize=256):
        super(DatasetFromFolder, self).__init__()

        self.data_dir = data_dir
        self.ref_dir = ref_dir

        self.transform = transforms.Compose([
            transforms.Resize((288,288)),
            transforms.RandomCrop(finesize),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

        self.style_transform = transforms.Compose([
            transforms.Resize((finesize, finesize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        self.input_filenames = sorted(glob(join(data_dir, '*.jpg')))
        self.ref_filenames = sorted(glob(join(ref_dir, '*/*.jpg')))
        self.ref_len = len(self.ref_filenames)
        self.input_len = len(self.input_filenames)

    def __getitem__(self,index):
        img = Image.open(self.input_filenames[index]).conver('RGB')
        rand_no = torch.randint(0,self.ref_len)
        ref = Image.open(self.ref_filenames[rand_no]).convert('RGB')

        input = self.transform(input)
        ref = self.style_transform(ref)

        return input, ref
    
    def __len__(self):
        return self.input_len