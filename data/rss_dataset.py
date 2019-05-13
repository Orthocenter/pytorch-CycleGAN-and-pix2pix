"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import random
import pickle
import torch
import glob
import os
import re
import numpy as np

class RSSDataset(BaseDataset):
    """RSS Dataset, synthetic data"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # self.dir = os.path.join(opt.dataroot, opt.phase)
        self.dir = opt.dataroot

        # get the image paths of your dataset;
        self.A_paths = self.make_dataset(os.path.join(self.dir, "A"))
        self.A_paths = sorted(self.A_paths)
        self.B_paths = self.make_dataset(os.path.join(self.dir, "B"))
        random.shuffle(self.B_paths)
        self.B_paths = self.B_paths[:min(opt.max_B_size, len(self.B_paths))]

        self.A_size = len(self.A_paths)
        print("A size: ", self.A_size)
        self.B_size = len(self.B_paths)
        print("B size: ", self.B_size)

        self.min_rss = -85
        self.max_rss = 30
    
    def is_image_file(self, filename):
        IMG_EXTENSIONS = ['.pickle']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(self, dir, max_dataset_size=float("inf")):
        images = []
        if not os.path.isdir(dir):
            print('%s is not a valid directory' % dir)
            return []

        for root, dirs, fnames in sorted(os.walk(dir)):
            for d in dirs:
                images += self.make_dataset(os.path.join(root, d))

            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    loc = path.split('/')[-1].split('_')[1:3]
                    loc = np.array([float(x) for x in loc])
                    x = float(loc[0])
                    y = float(loc[1])
                    if x < -3.2 or y < -3.2 or x > 3.2 or y > 3.2:
                        continue

                    images.append(path)
        return images[:min(max_dataset_size, len(images))]

    def normalize_data(self, data):
        return (data - self.min_rss) / (self.max_rss - self.min_rss) * 2 - 1

    def transform(self, data_A):
        data_A = self.normalize_data(data_A)
        data_A = torch.tensor(data_A, dtype=torch.float32)
        data_A = data_A[18:18+64, 18:18+64]
        data_A = data_A.view((1, data_A.size()[0], -1))
        return data_A

    def normalize_loc(self, loc):
        return loc / 5.

    def get_loc_from_path(self, path):
        loc = path.split('/')[-1].split('_')[1:3]
        loc = np.array([float(x) for x in loc])
        return self.normalize_loc(loc)
    
    def get_pwr_from_path(self, path):
        pwr = np.array(
            [float(path.split('/')[-1].split('_')[-1].split('.')[0])]
            )
        return self.normalize_data(pwr)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        A_path = self.A_paths[index]
        with open(A_path, 'rb') as f:
            data_A = pickle.load(f)
        
        # randomly pick a rss map as B
        indexB = random.randint(0, self.B_size-1)
        B_path = self.B_paths[indexB]
        with open(B_path, 'rb') as f:
            data_B = pickle.load(f)

        data_A = self.transform(data_A)
        data_B = self.transform(data_B)

        tx_loc = torch.tensor(self.get_loc_from_path(A_path)).float()
        tx_pwr = torch.tensor(self.get_pwr_from_path(A_path)).float()

        tx_loc_pwr = torch.cat((tx_loc, tx_pwr))

        return {'A': data_A, 'B': data_B, 'A_paths': A_path, 'B_paths': B_path,
                'tx_loc_pwr': tx_loc_pwr}

    def __len__(self):
        """Return the total number of images."""
        return self.A_size
