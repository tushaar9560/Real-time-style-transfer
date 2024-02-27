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

import sys
from src.logger import logging
from src.exception import CustomException

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class DatasetLoader(data.Dataset):
    def __init__(self, data_dir, ref_dir, finesize=256):
        super(DatasetLoader, self).__init__()

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
        try:
            img = Image.open(self.input_filenames[index]).conver('RGB')
            rand_no = torch.randint(0,self.ref_len)
            ref = Image.open(self.ref_filenames[rand_no]).convert('RGB')

            input = self.transform(img)
            ref = self.style_transform(ref)

            return input, ref
        
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
    def __len__(self):
        return self.input_len


class Getdata(DatasetLoader):

    def transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.455,0.456,0.406), (0.229,0.224,0.225))
        ])
    
    def get_training_set(self):
        try:
            content_dir = self.data_dir + '/content'
            ref_dir = self.data_dir + '/style'
            train_set = DatasetLoader(content_dir, ref_dir)
            logging.info("Training dataset loaded")
            return train_set
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
    def get_testing_set(self, test_dir, data_augment):
        test_set = DatasetLoader(test_dir, data_augment, transforms=transforms())
        logging.info("Testing data loaded")
        return test_set